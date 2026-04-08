import os
# reduce MPS upper watermark to avoid premature OOM (recommended in PyTorch message)
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')

import json
import torch
import argparse
import time
import statistics
import random
from collections import Counter
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
import dataset as dataset_module
from dataset import SpanDataset, collate_fn, LABELS
from model import SpanClassifier
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, f1_score
import numpy as np
import traceback
import glob
import shutil
import multiprocessing

# Helper: robust tokenizer load with diagnostics and cache cleanup
MODEL_NAME = "microsoft/deberta-v3-base"

# transformers ≥5.x a un bug : il tente de lire spm.model via tiktoken (≠ SentencePiece)
# → ValueError: Error parsing line b'\x0e' in spm.model
# Workaround : utiliser le tokenizer.json pré-converti (fast tokenizer) depuis le repo local.
# Le modèle lui-même (weights) continue d'être chargé depuis HF / cache.
# Peut être surchargé via --tokenizer-path ou la variable d'environnement TOKENIZER_PATH.
_DEFAULT_TOKENIZER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "debertav3-ner", "tokenizer_from_hf"
)
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", _DEFAULT_TOKENIZER_PATH)

ID2LABEL = {v: k for k, v in LABELS.items()}


def _cleanup_hf_cache(model_name):
    """Attempt to remove cached snapshots for the given HF model to force a fresh download."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    pattern = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
    matches = glob.glob(pattern + "*")
    removed = []
    for m in matches:
        try:
            shutil.rmtree(m)
            removed.append(m)
        except Exception:
            pass
    return removed


def load_tokenizer_with_retries(model_name, use_fast_prefer=False):
    """Try several tokenizer load strategies and provide diagnostics if all fail.

    Returns: tokenizer
    Raises RuntimeError with actionable message if it cannot be loaded.
    """
    last_exc = None

    strategies = []
    if use_fast_prefer:
        strategies.append({'use_fast': True, 'note': 'fast tokenizer preferred'})
    strategies.append({'use_fast': False, 'note': 'slow sentencepiece tokenizer (recommended for this model)'})
    strategies.append({'use_fast': False, 'cleanup': True, 'note': 'clean HF cache and retry slow tokenizer'})
    strategies.append({'use_fast': True, 'note': 'fallback to fast tokenizer (may be less accurate)'})

    for strat in strategies:
        try:
            if strat.get('cleanup'):
                removed = _cleanup_hf_cache(model_name)
                if removed:
                    print(f"Removed HF cache dirs: {removed}")
            print(f"Trying tokenizer load: use_fast={strat.get('use_fast')} ({strat.get('note')})")
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=bool(strat.get('use_fast')))
            print("Tokenizer loaded successfully with use_fast=", bool(strat.get('use_fast')))
            return tok
        except Exception as e:
            last_exc = e
            print(f"Tokenizer load failed for strategy use_fast={strat.get('use_fast')}, cleanup={strat.get('cleanup', False)}")
            traceback.print_exc()

    msg_lines = [
        "Failed to load tokenizer for model: " + model_name,
        "Last exception: " + repr(last_exc),
        "--- Diagnostics: common fixes ---",
        "1) Install missing packages (in your venv):",
        "   pip install sentencepiece protobuf tokenizers tiktoken transformers",
        "2) Clear HF cache for the model and retry (the script attempted this already):",
        "   rm -rf ~/.cache/huggingface/hub/models--" + model_name.replace('/', '--') + "*",
        "3) Ensure you run the script in the same Python environment where packages are installed (activate venv).",
        "4) If using a container, avoid mounting an old host HF cache; let container download fresh files.",
        "5) As a last resort you can try use_fast=True to fall back to the fast tokenizer (already attempted).",
        "--- End diagnostics ---",
        ]
    raise RuntimeError("\n".join(msg_lines)) from last_exc


def _summary_stats(lst):
    if not lst:
        return {'mean': 0.0, 'median': 0.0, 'p95': 0.0}
    return {
        'mean': statistics.mean(lst),
        'median': statistics.median(lst),
        'p95': statistics.quantiles(lst, n=100)[94]
    }


def compute_label_metrics(y_true, y_pred):
    """
    Retourne:
      - macro_f1
      - micro_f1
      - per_label_f1: dict[label_name] = f1
      - report_dict: classification_report(..., output_dict=True)
      - report_text: classification_report(..., string)
    """
    if not y_true or not y_pred:
        empty_f1 = {label: 0.0 for label in LABELS.keys()}
        return 0.0, 0.0, empty_f1, {}, "Aucune prédiction / aucun label."

    label_ids = list(range(len(LABELS)))
    label_names = [ID2LABEL[i] for i in label_ids]

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        digits=3,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        digits=3,
        zero_division=0
    )

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    per_label_f1 = {
        label: report_dict.get(label, {}).get("f1-score", 0.0)
        for label in label_names
    }

    return macro_f1, micro_f1, per_label_f1, report_dict, report_text


def run_epoch(loader, model, optimizer, device, args, training=False, log_interval=50, epoch_num=None):
    """
    Run one epoch and collect detailed timing logs.

    Returns:
      (mean_loss, metrics, timing_stats)

    metrics = {
        "macro_f1": float,
        "micro_f1": float,
        "per_label_f1": dict,
        "report_dict": dict,
        "report_text": str,
        "all_labels": list[int],
        "all_preds": list[int],
    }
    """
    accum_steps = max(1, int(getattr(args, 'accum_steps', 1)))

    if training:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    all_preds = []
    all_labels = []
    losses = []

    # timing collectors
    data_times = []
    prep_times = []
    forward_times = []
    loss_times = []
    backward_times = []
    optim_times = []
    total_times = []

    it = iter(loader)
    batch_idx = 0
    t_epoch_start = time.perf_counter()

    # optional per-batch debug for first few batches
    debug_per_batch = []

    while True:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break

        t_after_load = time.perf_counter()
        data_times.append(t_after_load - t0)

        # move tensors and ensure dtypes
        t_prep_start = time.perf_counter()
        input_ids = batch["input_ids"].to(device)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        att = batch["attention_mask"].to(device)
        if att.dtype != torch.long:
            att = att.long()

        spans = batch["spans"]
        labels = batch.get("labels")

        # ── Coarse noise injection (training only) ────────────────────────────
        # Simule les ~20% d'erreurs du modèle NER coarse à l'inférence.
        # 10% des spans reçoivent un coarse_id aléatoire → le modèle apprend
        # à ne pas dépendre aveuglément du signal coarse.
        coarse_noise_rate = getattr(args, 'coarse_noise', 0.0)
        if training and coarse_noise_rate > 0.0:
            for sample_spans in spans:
                for sp in sample_spans:
                    if random.random() < coarse_noise_rate:
                        sp['coarse_id'] = random.randint(0, 5)

        if labels is not None and labels.numel() > 0:
            labels = labels.to(device)
            if labels.dtype != torch.long:
                labels = labels.long()

        t_prep_end = time.perf_counter()
        prep_times.append(t_prep_end - t_prep_start)

        with torch.set_grad_enabled(training):
            # forward
            t_fwd_start = time.perf_counter()
            logits = model({
                "input_ids": input_ids,
                "attention_mask": att,
                "spans": spans
            })
            t_fwd_end = time.perf_counter()
            forward_times.append(t_fwd_end - t_fwd_start)

            # loss
            t_loss_start = time.perf_counter()
            loss_fn = batch.get("loss_fn")

            if callable(loss_fn) and labels is not None and labels.numel() > 0:
                # S'assurer que le weight de la loss est sur le même device que les logits
                if hasattr(loss_fn, 'weight') and loss_fn.weight is not None:
                    if loss_fn.weight.device != logits.device:
                        loss_fn.weight = loss_fn.weight.to(logits.device)

                if logits.numel() == 0:
                    loss = torch.tensor(0.0, device=device)
                else:
                    num_logits = logits.size(0)
                    num_labels = labels.numel()
                    if num_logits != num_labels:
                        print(f"⚠️ MISMATCH logits ({num_logits}) vs labels ({num_labels}) - truncating to min and logging")
                        m = min(num_logits, num_labels)
                        logits = logits[:m]
                        labels = labels[:m]

                    # apply gradient accumulation: scale loss
                    loss = loss_fn(logits, labels) / accum_steps
            else:
                loss = torch.tensor(0.0, device=device)

            t_loss_end = time.perf_counter()
            loss_times.append(t_loss_end - t_loss_start)

            # backward + optim (with accumulation)
            t_back_start = time.perf_counter()
            if training:
                loss.backward()
            t_back_end = time.perf_counter()
            backward_times.append(t_back_end - t_back_start)

        if training and ((batch_idx + 1) % accum_steps == 0):
            t_opt_start = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad()
            t_opt_end = time.perf_counter()
            optim_times.append(t_opt_end - t_opt_start)
        else:
            optim_times.append(0.0)

        t_end = time.perf_counter()
        total_times.append(t_end - t0)

        # collect preds/labels
        if logits.numel() > 0:
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())

        if labels is not None and labels.numel() > 0:
            all_labels.extend(labels.detach().cpu().numpy().tolist())

        losses.append(loss.item() * accum_steps if isinstance(loss, torch.Tensor) else float(loss))

        # per-batch debug info (first 5 batches)
        if batch_idx < 5:
            debug_per_batch.append({
                'batch': batch_idx + 1,
                'data': data_times[-1],
                'prep': prep_times[-1],
                'forward': forward_times[-1],
                'loss': loss_times[-1],
                'backward': backward_times[-1],
                'optim': optim_times[-1],
                'total': total_times[-1]
            })

        batch_idx += 1

        if batch_idx % log_interval == 0:
            s_data = _summary_stats(data_times)
            s_prep = _summary_stats(prep_times)
            s_fwd = _summary_stats(forward_times)
            s_loss = _summary_stats(loss_times)
            s_back = _summary_stats(backward_times)
            s_opt = _summary_stats(optim_times)
            s_tot = _summary_stats(total_times)
            print(
                f"[{'TRAIN' if training else 'EVAL'}] batch {batch_idx} | batches so far {batch_idx} | "
                f"data mean {s_data['mean']:.3f}s prep {s_prep['mean']:.3f}s fwd {s_fwd['mean']:.3f}s "
                f"loss {s_loss['mean']:.3f}s back {s_back['mean']:.3f}s opt {s_opt['mean']:.3f}s total {s_tot['mean']:.3f}s"
            )

    # final step if leftover grads
    if training and (batch_idx % accum_steps != 0):
        t_opt_start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad()
        t_opt_end = time.perf_counter()
        optim_times.append(t_opt_end - t_opt_start)

    t_epoch_end = time.perf_counter()
    epoch_time = t_epoch_end - t_epoch_start

    timing_stats = {
        'data_times': data_times,
        'prep_times': prep_times,
        'forward_times': forward_times,
        'loss_times': loss_times,
        'backward_times': backward_times,
        'optim_times': optim_times,
        'total_times': total_times,
        'epoch_time': epoch_time,
        'batches': batch_idx,
        'per_batch_debug': debug_per_batch
    }

    try:
        macro_f1, micro_f1, per_label_f1, report_dict, report_text = compute_label_metrics(all_labels, all_preds)
    except Exception:
        macro_f1, micro_f1 = 0.0, 0.0
        per_label_f1 = {label: 0.0 for label in LABELS.keys()}
        report_dict = {}
        report_text = "Erreur lors du calcul du classification_report."

    metrics = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_label_f1": per_label_f1,
        "report_dict": report_dict,
        "report_text": report_text,
        "all_labels": all_labels,
        "all_preds": all_preds,
    }

    # optionally write timings to a file for later analysis
    if args.timings_out:
        outdir = args.timings_out
        os.makedirs(outdir, exist_ok=True)

        fname = os.path.join(outdir, f"timings_epoch_{epoch_num if epoch_num is not None else 'unknown'}.json")
        with open(fname, 'w', encoding='utf-8') as fh:
            json.dump(timing_stats, fh, ensure_ascii=False, indent=2)

        metrics_fname = os.path.join(outdir, f"metrics_epoch_{epoch_num if epoch_num is not None else 'unknown'}.json")
        with open(metrics_fname, 'w', encoding='utf-8') as fh:
            json.dump({
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "per_label_f1": per_label_f1,
                "report_dict": report_dict
            }, fh, ensure_ascii=False, indent=2)

    return np.mean(losses) if losses else 0.0, metrics, timing_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Chemin vers le tokenizer local (fast tokenizer DeBERTa). "
             "Défaut : variable d'env TOKENIZER_PATH ou chemin relatif ../../debertav3-ner/tokenizer_from_hf"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--log-interval", type=int, default=50, help="Number of batches between intermediate timing logs")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers (set >0 for parallel data loading)")
    parser.add_argument("--timings-out", type=str, default=None, help="If set, write per-epoch timing JSON files to this directory")
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps to lower peak memory usage")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None, help="Force device: 'cpu', 'mps' or 'cuda'. If absent, auto-detect.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint file to resume training from")
    parser.add_argument("--start-epoch", type=int, default=None, help="Force start epoch when resuming from a plain model state_dict")
    parser.add_argument(
        "--coarse-noise",
        type=float,
        default=0.10,
        help="Taux de bruit sur le coarse_id pendant l'entraînement (0.0 = désactivé, 0.10 = 10%%). "
             "Simule les erreurs du modèle NER coarse à l'inférence pour rendre le SpanClassifier robuste. "
             "Uniquement appliqué pendant le training, pas l'évaluation."
    )
    parser.add_argument(
        "--class-weights",
        choices=["none", "auto"],
        default="auto",
        help="Pondération des classes dans la CrossEntropyLoss. "
             "'auto' (défaut) : poids inverse-fréquence calculés depuis --train. "
             "'none' : pas de pondération (comportement historique)."
    )
    parser.add_argument(
        "--qty-weight",
        type=float,
        default=0.60,
        help="Facteur multiplicatif appliqué au poids de hint_quantity APRÈS calcul inverse-fréquence "
             "(défaut: 0.35). Réduit la sur-représentation de hint_quantity dans la famille OBJECT. "
             "Ignoré si --class-weights=none."
    )
    args = parser.parse_args()

    # allow overriding device from CLI to force CPU (useful when MPS leaks memory)
    if args.device:
        device = args.device
    else:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    print("✅ Using device:", device)

    try:
        tok_path = args.tokenizer_path if args.tokenizer_path else TOKENIZER_PATH
        print(f"Loading tokenizer from: {tok_path}")
        tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
        print("Tokenizer loaded successfully from local path.")
    except Exception as e:
        raise RuntimeError(
            f"Échec du chargement du tokenizer depuis {tok_path}.\n"
            "Vérifier que le fichier tokenizer.json est présent et non vide dans ce répertoire.\n"
            "Vous pouvez surcharger le chemin via --tokenizer-path ou la variable TOKENIZER_PATH.\n"
            f"Erreur : {e}"
        ) from e

    train_data = SpanDataset(args.train, tokenizer)
    val_data = SpanDataset(args.val, tokenizer)
    test_data = SpanDataset(args.test, tokenizer)

    # ── Class weights ──────────────────────────────────────────────────────────
    if args.class_weights == "auto":
        print("\n⚖️  Calcul des class weights (inverse-fréquence) depuis le training set…")
        raw_train = [json.loads(l) for l in Path(args.train).open(encoding="utf-8") if l.strip()]
        counts: Counter = Counter()
        for row in raw_train:
            for sp in row.get("spans", []):
                lbl = sp.get("label")
                if lbl in LABELS:
                    counts[lbl] += 1

        N = sum(counts.values())
        C = len(LABELS)  # 22
        weights = torch.ones(C, dtype=torch.float32)

        for label, idx in LABELS.items():
            n_i = counts.get(label, 0)
            if n_i > 0:
                weights[idx] = N / (C * n_i)
            # labels absents du training → poids 1.0 (neutre)

        # Normalisation : moyenne = 1.0 (conserve l'échelle globale de la loss)
        weights = weights / weights.mean()

        # Réduction supplémentaire de hint_quantity
        qty_idx = LABELS["hint_quantity"]   # 21
        weights[qty_idx] *= args.qty_weight

        # Re-normalisation finale
        weights = weights / weights.mean()

        dataset_module.CLASS_WEIGHTS = weights

        # Affichage pour traçabilité
        idx_to_label = {v: k for k, v in LABELS.items()}
        print(f"  {'label':<25} {'count':>7}  {'weight':>8}")
        print(f"  {'-'*45}")
        for idx in range(C):
            lbl = idx_to_label[idx]
            cnt = counts.get(lbl, 0)
            w = weights[idx].item()
            marker = "  ← qty_weight applied" if idx == qty_idx else ""
            print(f"  {lbl:<25} {cnt:>7}  {w:>8.4f}{marker}")
        print()
    else:
        print("⚖️  --class-weights=none : CrossEntropyLoss sans pondération.")
        dataset_module.CLASS_WEIGHTS = None

    # DataLoader: allow controlling workers and pin_memory
    pin_mem = True if device == "cuda" else False
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_mem
    )

    model = SpanClassifier(MODEL_NAME, num_labels=22, num_coarse=6, coarse_embed_dim=128).to(device)
    # Ensure model parameters are float32 to avoid dtype mismatch with inputs
    model = model.float()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # checkpointing / resume support
    best_val_f1 = 0.0
    start_epoch = 1

    # make sure device mapping for checkpoint load is correct
    torch_device = torch.device(device)

    if getattr(args, 'resume', None):
        resume_path = args.resume
    else:
        resume_path = None

    # If user passed --resume via environment or CLI arg, attempt to load
    if resume_path:
        if os.path.exists(resume_path):
            print(f"⤴️  Loading checkpoint from {resume_path}")
            ckpt = torch.load(resume_path, map_location=torch_device)

            # ckpt may be a state_dict or a dict containing keys
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
                try:
                    optimizer.load_state_dict(ckpt.get('optim_state', {}))
                except Exception:
                    print("⚠️ Unable to restore optimizer state fully; continuing with fresh optimizer.")

                best_val_f1 = ckpt.get('best_val_f1', 0.0)
                start_epoch = int(ckpt.get('epoch', 0)) + 1
                print(f"Resuming from epoch {start_epoch} with best_val_f1={best_val_f1}")
            else:
                # assume plain state_dict
                try:
                    model.load_state_dict(ckpt)
                    print("Loaded model.state_dict from resume file")
                    # if user provided a forced start epoch, use it; otherwise warn that training will restart from epoch 1
                    if args.start_epoch:
                        start_epoch = args.start_epoch
                        print(f"Resuming training from forced start epoch {start_epoch} (optimizer state not restored)")
                    else:
                        print("⚠️ Resume file appears to be a plain state_dict; optimizer state and epoch not restored. Use --start-epoch to continue from a specific epoch if desired.")
                except Exception as e:
                    print("⚠️ Failed to load resume checkpoint:", e)
        else:
            print(f"⚠️ Resume file not found: {resume_path} — starting from scratch.")

    print("🚀 Starting training...")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_metrics, train_timing = run_epoch(
            train_loader,
            model,
            optimizer,
            device,
            args,
            training=True,
            log_interval=args.log_interval,
            epoch_num=epoch
        )

        val_loss, val_metrics, val_timing = run_epoch(
            val_loader,
            model,
            optimizer,
            device,
            args,
            training=False,
            log_interval=args.log_interval,
            epoch_num=epoch
        )

        print(f"\n📅 Epoch {epoch}")
        print(
            f"   Train loss = {train_loss:.4f} | "
            f"Train Macro F1 = {train_metrics['macro_f1']:.4f} | "
            f"Train Micro F1 = {train_metrics['micro_f1']:.4f} | "
            f"epoch time {train_timing['epoch_time']:.2f}s | batches {train_timing['batches']}"
        )
        print(
            f"   Val   loss = {val_loss:.4f} | "
            f"Val   Macro F1 = {val_metrics['macro_f1']:.4f} | "
            f"Val   Micro F1 = {val_metrics['micro_f1']:.4f} | "
            f"epoch time {val_timing['epoch_time']:.2f}s | batches {val_timing['batches']}"
        )

        print("\n[Métriques par label - TRAIN]")
        print(train_metrics["report_text"])

        print("[Métriques par label - VAL]")
        print(val_metrics["report_text"])

        # save last checkpoint
        ckpt_last = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'best_val_f1': best_val_f1
        }
        torch.save(ckpt_last, 'checkpoint_last.pt')

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]

            # save best full checkpoint
            ckpt_best = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'best_val_f1': best_val_f1
            }
            torch.save(ckpt_best, 'checkpoint_best.pt')

            # also save legacy best_model.pt for compatibility
            torch.save(model.state_dict(), 'best_model.pt')
            print("✅ New best model saved as checkpoint_best.pt / best_model.pt")

    print("\n✅ Training finished. Loading best model for test evaluation...")

    # load best checkpoint if exists
    best_path = 'checkpoint_best.pt' if os.path.exists('checkpoint_best.pt') else ('best_model.pt' if os.path.exists('best_model.pt') else None)
    if best_path:
        ckpt = torch.load(best_path, map_location=torch_device)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
        else:
            model.load_state_dict(ckpt)
    else:
        print('⚠️ No best checkpoint found; using current model weights')

    # ---- Final test evaluation ----
    test_preds = []
    test_gold = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()

            att = batch["attention_mask"].to(device)
            if att.dtype != torch.long:
                att = att.long()

            spans = batch["spans"]
            labels = batch["labels"].to(device)
            if labels.dtype != torch.long:
                labels = labels.long()

            logits = model({
                "input_ids": input_ids,
                "attention_mask": att,
                "spans": spans
            })

            preds = logits.argmax(dim=-1).cpu().numpy()
            test_preds.extend(preds.tolist())
            test_gold.extend(labels.cpu().numpy().tolist())

    test_macro_f1, test_micro_f1, test_per_label_f1, test_report_dict, test_report_text = compute_label_metrics(
        test_gold, test_preds
    )

    print("\n🎯 FINAL TEST RESULTS")
    print(test_report_text)
    print(f"Macro F1: {test_macro_f1:.4f}")
    print(f"Micro F1: {test_micro_f1:.4f}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()