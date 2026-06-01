"""
train_multi_task_hn.py
─────────────────────
Version avec Hard Negative Mining en ligne.

Différences avec train_multi_task.py :
  - Après chaque forward, détecte les spans "hard negative" :
    spans gold négatifs que le modèle prédit positifs avec haute confiance
  - Booste dynamiquement le sample_weight de ces spans dans la loss
  - Le boost augmente progressivement au fil des epochs (curriculum)
  - Fonctionne sur boundary ET coarse (NONE prédit non-NONE avec confiance)
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import argparse
import json
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, classification_report

from multitask_dataset import MultiTaskSpanDataset, make_collate_fn
from multitask_model import SpanMultiTaskModel
from labels import COARSE_LABELS, FINE_LABELS, COARSE_NONE_ID, SVO_LABELS, VOICE_LABELS
from labels import NUM_SVO, NUM_VOICE


# ─── Réutilisé depuis train_multi_task.py ──────────────────────

def compute_class_weights_from_multitask_jsonl(path: str, power: float = 0.5):
    boundary_counts = Counter()
    coarse_counts = Counter()
    fine_counts = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for c in row["candidates"]:
                boundary_counts[c["boundary_label"]] += 1
                coarse_counts[c["coarse_label_id"]] += 1
                fine_counts[c["fine_label_id"]] += 1

    def make_weights(counts, num_classes, power=0.5):
        total = sum(counts.values())
        weights = torch.ones(num_classes, dtype=torch.float32)
        if total == 0:
            return weights
        for i in range(num_classes):
            n_i = counts.get(i, 0)
            if n_i > 0:
                inv_freq = total / (num_classes * n_i)
                weights[i] = float(inv_freq) ** float(power)
            else:
                weights[i] = 1.0
        weights = weights / weights.mean()
        return weights

    return (
        make_weights(boundary_counts, 2, power=power),
        make_weights(coarse_counts, len(COARSE_LABELS), power=power),
        make_weights(fine_counts, len(FINE_LABELS), power=power),
        boundary_counts, coarse_counts, fine_counts,
    )


def apply_class_weight_floor(weights, class_idx, min_value):
    weights = weights.clone()
    weights[class_idx] = max(weights[class_idx].item(), float(min_value))
    return weights


# ─── Hard Negative Mining ──────────────────────────────────────

def compute_hard_negative_boost(
    boundary_logits: torch.Tensor,
    coarse_logits: torch.Tensor,
    boundary_labels: torch.Tensor,
    coarse_labels: torch.Tensor,
    sample_weights: torch.Tensor,
    hn_conf_threshold: float = 0.8,
    hn_boost_factor: float = 3.0,
    hn_ramp: float = 1.0,  # 0→1 sur les epochs
) -> tuple[torch.Tensor, dict]:
    """
    Identifie les hard negatives et booste leur sample_weight.

    Hard negative boundary : gold boundary=0, mais P(boundary=1) > threshold
    Hard negative coarse   : gold coarse=NONE, mais P(coarse!=NONE) > threshold

    Args:
        boundary_logits: [N, 2]
        coarse_logits: [N, C]
        boundary_labels: [N]
        coarse_labels: [N]
        sample_weights: [N]
        hn_conf_threshold: seuil de confiance pour considérer un hard negative
        hn_boost_factor: facteur multiplicatif max sur le sample_weight
        hn_ramp: facteur de progression (0=pas de boost, 1=boost complet)

    Returns:
        boosted_weights: [N]
        stats: dict avec compteurs
    """
    device = boundary_logits.device
    N = boundary_logits.size(0)
    boosted = sample_weights.clone()

    # Probabilités
    b_probs = F.softmax(boundary_logits, dim=-1)  # [N, 2]
    c_probs = F.softmax(coarse_logits, dim=-1)    # [N, C]

    # ── Hard negative boundary ──
    # gold=0 (négatif) mais modèle dit positif avec haute confiance
    is_gold_neg = (boundary_labels == 0)
    conf_pos = b_probs[:, 1]  # P(boundary=1)
    hn_boundary_mask = is_gold_neg & (conf_pos > hn_conf_threshold)

    # ── Hard negative coarse ──
    # gold=NONE mais modèle confiant sur un coarse != NONE
    is_gold_none = (coarse_labels == COARSE_NONE_ID)
    # Probabilité max sur les classes non-NONE
    c_probs_no_none = c_probs.clone()
    c_probs_no_none[:, COARSE_NONE_ID] = 0.0
    conf_non_none = c_probs_no_none.max(dim=-1).values
    hn_coarse_mask = is_gold_none & (conf_non_none > hn_conf_threshold)

    # Union des deux
    hn_mask = hn_boundary_mask | hn_coarse_mask

    # Calculer le boost effectif (progressif)
    effective_boost = 1.0 + (hn_boost_factor - 1.0) * hn_ramp

    # Appliquer le boost proportionnel à la confiance
    if hn_mask.any():
        # Boost proportionnel : plus le modèle est confiant, plus on booste
        max_conf = torch.where(
            hn_boundary_mask,
            conf_pos,
            torch.zeros_like(conf_pos),
        )
        max_conf = torch.max(max_conf, torch.where(
            hn_coarse_mask,
            conf_non_none,
            torch.zeros_like(conf_non_none),
        ))
        # Normaliser entre [1, effective_boost]
        boost = 1.0 + (effective_boost - 1.0) * max_conf
        boosted[hn_mask] = boosted[hn_mask] * boost[hn_mask]

    stats = {
        "hn_boundary_count": int(hn_boundary_mask.sum().item()),
        "hn_coarse_count": int(hn_coarse_mask.sum().item()),
        "hn_total": int(hn_mask.sum().item()),
        "hn_mean_boost": float(boosted[hn_mask].mean().item()) if hn_mask.any() else 0.0,
        "hn_ramp": hn_ramp,
    }

    return boosted, stats


# ─── Run epoch avec HN mining ─────────────────────────────────

def run_epoch(
    loader, model, optimizer, device, train: bool,
    boundary_class_weights=None,
    coarse_class_weights=None,
    fine_class_weights=None,
    lambda_boundary=1.0,
    lambda_coarse=1.0,
    lambda_fine=1.2,
    lambda_svo=1.0,
    lambda_voice=0.5,
    accum_steps=1,
    log_every=50,
    # Hard negative params
    hn_enabled=False,
    hn_conf_threshold=0.8,
    hn_boost_factor=3.0,
    hn_ramp=1.0,
):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    losses = []
    all_b_true, all_b_pred = [], []
    all_c_true, all_c_pred = [], []
    all_f_true_pos, all_f_pred_pos = [], []
    all_svo_true, all_svo_pred = [], []
    all_voice_true, all_voice_pred = [], []
    hn_stats_accum = Counter()

    coarse_fine_mask = model.coarse_fine_mask.to(device)

    def masked_fine_predictions(fine_logits, coarse_preds, cfm):
        if fine_logits.numel() == 0:
            return []
        d = fine_logits.device
        cp = torch.as_tensor(coarse_preds, dtype=torch.long, device=d)
        allowed = cfm[cp]
        no_valid = ~allowed.any(dim=-1)
        masked = fine_logits.clone().masked_fill(~allowed, -1e9)
        pred = masked.argmax(dim=-1).masked_fill(no_valid, -1)
        return pred.detach().cpu().tolist()

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        spans = batch["spans"]

        boundary_labels = batch["boundary_labels"].to(device)
        coarse_labels = batch["coarse_labels"].to(device)
        fine_labels = batch["fine_labels"].to(device)
        svo_labels_batch = batch["svo_labels"].to(device)
        voice_labels_batch = batch["voice_labels"].to(device)
        sample_weights = batch["sample_weights"].to(device)

        with torch.set_grad_enabled(train):
            outputs = model({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "spans": spans,
            })

            span_indices = outputs.get("span_indices", None)
            if span_indices is not None:
                si = span_indices.to(device=device, dtype=torch.long)
                boundary_labels_loss = boundary_labels[si]
                coarse_labels_loss = coarse_labels[si]
                fine_labels_loss = fine_labels[si]
                svo_labels_loss = svo_labels_batch[si]
                voice_labels_loss = voice_labels_batch[si]
                sample_weights_loss = sample_weights[si]
            else:
                boundary_labels_loss = boundary_labels
                coarse_labels_loss = coarse_labels
                fine_labels_loss = fine_labels
                svo_labels_loss = svo_labels_batch
                voice_labels_loss = voice_labels_batch
                sample_weights_loss = sample_weights

            # ── Hard Negative Boost ──
            if train and hn_enabled:
                sample_weights_loss, hn_stats = compute_hard_negative_boost(
                    boundary_logits=outputs["boundary_logits"].detach(),
                    coarse_logits=outputs["coarse_logits"].detach(),
                    boundary_labels=boundary_labels_loss,
                    coarse_labels=coarse_labels_loss,
                    sample_weights=sample_weights_loss,
                    hn_conf_threshold=hn_conf_threshold,
                    hn_boost_factor=hn_boost_factor,
                    hn_ramp=hn_ramp,
                )
                for k, v in hn_stats.items():
                    if k == "hn_ramp":
                        continue
                    hn_stats_accum[k] += v
            # ── /Hard Negative Boost ──

            loss_dict = model.compute_loss(
                outputs=outputs,
                boundary_labels=boundary_labels_loss,
                coarse_labels=coarse_labels_loss,
                fine_labels=fine_labels_loss,
                svo_labels=svo_labels_loss,
                voice_labels=voice_labels_loss,
                sample_weights=sample_weights_loss,
                boundary_class_weights=boundary_class_weights,
                coarse_class_weights=coarse_class_weights,
                lambda_boundary=lambda_boundary,
                lambda_coarse=lambda_coarse,
                lambda_fine=lambda_fine,
                lambda_svo=lambda_svo,
                lambda_voice=lambda_voice,
            )

            loss = loss_dict["loss"] / accum_steps
            if train:
                loss.backward()

        if train and (step % accum_steps == 0):
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss_dict["loss"].item())

        if (step % log_every == 0) or (step == 1):
            mode = "TRAIN" if train else "EVAL"
            avg_loss = sum(losses) / max(1, len(losses))
            hn_info = ""
            if train and hn_enabled:
                hn_info = f" hn_total={hn_stats_accum['hn_total']} hn_boost={hn_stats_accum.get('hn_mean_boost', 0):.2f}"
            print(
                f"[{mode}] step={step}/{len(loader)} "
                f"loss={loss_dict['loss'].item():.4f} "
                f"avg_loss={avg_loss:.4f}{hn_info}"
            )

        # Predictions
        b_pred = outputs["boundary_logits"].argmax(dim=-1).detach().cpu().tolist()
        c_pred = outputs["coarse_logits"].argmax(dim=-1).detach().cpu().tolist()
        f_pred = masked_fine_predictions(outputs["fine_logits"], c_pred, coarse_fine_mask)
        svo_pred = outputs["svo_logits"].argmax(dim=-1).detach().cpu().tolist()
        voice_pred = outputs["voice_logits"].argmax(dim=-1).detach().cpu().tolist()

        if span_indices is not None:
            si_cpu = span_indices.detach().cpu().to(dtype=torch.long)
            b_true = boundary_labels.detach().cpu()[si_cpu].tolist()
            c_true = coarse_labels.detach().cpu()[si_cpu].tolist()
            f_true = fine_labels.detach().cpu()[si_cpu].tolist()
            svo_true = svo_labels_batch.detach().cpu()[si_cpu].tolist()
            voice_true = voice_labels_batch.detach().cpu()[si_cpu].tolist()
        else:
            b_true = boundary_labels.detach().cpu().tolist()
            c_true = coarse_labels.detach().cpu().tolist()
            f_true = fine_labels.detach().cpu().tolist()
            svo_true = svo_labels_batch.detach().cpu().tolist()
            voice_true = voice_labels_batch.detach().cpu().tolist()

        all_b_true.extend(b_true)
        all_b_pred.extend(b_pred)
        all_c_true.extend(c_true)
        all_c_pred.extend(c_pred)

        for bt, ft, fp in zip(b_true, f_true, f_pred):
            if bt == 1:
                all_f_true_pos.append(ft)
                all_f_pred_pos.append(fp)

        for st, sp_pred in zip(svo_true, svo_pred):
            if st < NUM_SVO:  # span avec un vrai rôle SVO
                all_svo_true.append(st)
                all_svo_pred.append(sp_pred)
        for vt, vp in zip(voice_true, voice_pred):
            if vt < NUM_VOICE:  # svo_verb uniquement
                all_voice_true.append(vt)
                all_voice_pred.append(vp)

    if train and (len(loader) % accum_steps != 0):
        optimizer.step()
        optimizer.zero_grad()

    metrics = {
        "loss": sum(losses) / max(1, len(losses)),
        "boundary_f1": f1_score(all_b_true, all_b_pred, average="macro", zero_division=0) if all_b_true else 0.0,
        "coarse_macro_f1": f1_score(all_c_true, all_c_pred, average="macro", labels=list(range(len(COARSE_LABELS))), zero_division=0) if all_c_true else 0.0,
        "fine_macro_f1": f1_score(all_f_true_pos, all_f_pred_pos, average="macro", labels=list(range(len(FINE_LABELS))), zero_division=0) if all_f_true_pos else 0.0,
        "svo_macro_f1": f1_score(all_svo_true, all_svo_pred, average="macro", labels=list(range(len(SVO_LABELS))), zero_division=0) if all_svo_true else 0.0,
        "voice_macro_f1": f1_score(all_voice_true, all_voice_pred, average="macro", labels=list(range(len(VOICE_LABELS))), zero_division=0) if all_voice_true else 0.0,
        "boundary_report": classification_report(all_b_true, all_b_pred, digits=3, zero_division=0) if all_b_true else "N/A",
        "coarse_report": classification_report(all_c_true, all_c_pred, labels=list(range(len(COARSE_LABELS))), target_names=COARSE_LABELS, digits=3, zero_division=0) if all_c_true else "N/A",
        "fine_report": classification_report(all_f_true_pos, all_f_pred_pos, labels=list(range(len(FINE_LABELS))), target_names=FINE_LABELS, digits=3, zero_division=0) if all_f_true_pos else "N/A",
        "svo_report": classification_report(all_svo_true, all_svo_pred, labels=list(range(len(SVO_LABELS))), target_names=SVO_LABELS, digits=3, zero_division=0) if all_svo_true else "N/A",
        "voice_report": classification_report(all_voice_true, all_voice_pred, labels=list(range(len(VOICE_LABELS))), target_names=VOICE_LABELS, digits=3, zero_division=0) if all_voice_true else "N/A",
    }

    if train and hn_enabled:
        metrics["hn_boundary_count"] = hn_stats_accum["hn_boundary_count"]
        metrics["hn_coarse_count"] = hn_stats_accum["hn_coarse_count"]
        metrics["hn_total"] = hn_stats_accum["hn_total"]

    return metrics


# ─── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)

    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--max-length", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)

    parser.add_argument("--lambda-boundary", type=float, default=1.0)
    parser.add_argument("--lambda-coarse", type=float, default=1.0)
    parser.add_argument("--lambda-fine", type=float, default=1.2)
    parser.add_argument("--lambda-svo", type=float, default=1.0)
    parser.add_argument("--lambda-voice", type=float, default=0.5)

    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--class-weights", choices=["none", "auto"], default="auto")
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--min-coarse-none-weight", type=float, default=0.05)
    parser.add_argument("--min-fine-none-weight", type=float, default=0.05)

    # Hard negative mining
    parser.add_argument("--hn-enabled", action="store_true", default=True,
                        help="Active le hard negative mining")
    parser.add_argument("--no-hn", dest="hn_enabled", action="store_false",
                        help="Désactive le hard negative mining")
    parser.add_argument("--hn-conf-threshold", type=float, default=0.75,
                        help="Seuil de confiance pour considérer un span comme hard negative")
    parser.add_argument("--hn-boost-factor", type=float, default=3.0,
                        help="Facteur multiplicatif max sur le sample_weight des hard negatives")
    parser.add_argument("--hn-warmup-epochs", type=int, default=2,
                        help="Nombre d'epochs avant d'activer le HN mining à pleine puissance (ramp 0→1)")

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start-epoch", type=int, default=None)
    parser.add_argument("--train-on-val-too", action="store_true", default=False,
                        help="Concatène train+val pour l'entraînement (évalue sur test uniquement). "
                             "Utile en phase finale pour maximiser l'exposition du modèle.")

    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ device = {device}")

    tokenizer_source = args.tokenizer_path or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    train_ds = MultiTaskSpanDataset(args.train, tokenizer, max_length=args.max_length)
    val_ds = MultiTaskSpanDataset(args.val, tokenizer, max_length=args.max_length)
    test_ds = MultiTaskSpanDataset(args.test, tokenizer, max_length=args.max_length)

    if args.train_on_val_too:
        from torch.utils.data import ConcatDataset
        combined_ds = ConcatDataset([train_ds, val_ds])
        print(f"🔀 train+val combinés : {len(combined_ds)} exemples (train={len(train_ds)}, val={len(val_ds)})")
        effective_train_ds = combined_ds
    else:
        effective_train_ds = train_ds

    collate_fn = make_collate_fn(tokenizer)
    pin_memory = (device == "cuda")
    train_loader = DataLoader(effective_train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_memory)

    model = SpanMultiTaskModel(model_name=args.model_name, num_coarse=len(COARSE_LABELS)).to(device).float()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    boundary_w = coarse_w = fine_w = None
    if args.class_weights == "auto":
        boundary_w, coarse_w, fine_w, bc, cc, fc = compute_class_weights_from_multitask_jsonl(args.train, power=args.class_weight_power)
        coarse_w = apply_class_weight_floor(coarse_w, len(COARSE_LABELS) - 1, args.min_coarse_none_weight)
        fine_w = apply_class_weight_floor(fine_w, len(FINE_LABELS) - 1, args.min_fine_none_weight)

        print("⚖️ class weights auto activés")
        print(f"   power={args.class_weight_power}")
        print(f"\n[boundary]")
        for i in range(2):
            print(f"  class {i}: count={bc.get(i, 0)} weight={boundary_w[i].item():.6f}")
        print(f"\n[coarse]")
        for i, name in enumerate(COARSE_LABELS):
            print(f"  {name:<10} count={cc.get(i, 0):>8} weight={coarse_w[i].item():.6f}")
        print(f"\n[fine]")
        for i, name in enumerate(FINE_LABELS):
            print(f"  {name:<22} count={fc.get(i, 0):>8} weight={fine_w[i].item():.6f}")

    if args.hn_enabled:
        print(f"\n⛏️  Hard Negative Mining ACTIVÉ")
        print(f"   conf_threshold = {args.hn_conf_threshold}")
        print(f"   boost_factor   = {args.hn_boost_factor}")
        print(f"   warmup_epochs  = {args.hn_warmup_epochs}")

    best_score = -1.0
    start_epoch = 1

    if args.resume is not None:
        print(f"⤴️ Reprise depuis: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optim_state" in ckpt and ckpt["optim_state"] is not None:
            optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = args.start_epoch if args.start_epoch is not None else int(ckpt.get("epoch", 0)) + 1
        best_score = ckpt.get("best_score", -1.0)
        print(f"✅ start_epoch={start_epoch} best_score={best_score:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        # Calculer le ramp HN : 0 pendant les warmup_epochs, puis linéaire vers 1
        if args.hn_warmup_epochs > 0:
            hn_ramp = max(0.0, min(1.0, (epoch - 1) / args.hn_warmup_epochs))
        else:
            hn_ramp = 1.0

        print(f"\n{'='*60}")
        print(f"  Epoch {epoch} | hn_ramp={hn_ramp:.2f}")
        print(f"{'='*60}")

        train_metrics = run_epoch(
            train_loader, model, optimizer, device, train=True,
            boundary_class_weights=boundary_w, coarse_class_weights=coarse_w,
            lambda_boundary=args.lambda_boundary, lambda_coarse=args.lambda_coarse, lambda_fine=args.lambda_fine,
            lambda_svo=args.lambda_svo, lambda_voice=args.lambda_voice,
            accum_steps=args.accum_steps, log_every=args.log_every,
            hn_enabled=args.hn_enabled, hn_conf_threshold=args.hn_conf_threshold,
            hn_boost_factor=args.hn_boost_factor, hn_ramp=hn_ramp,
        )

        val_metrics = run_epoch(
            val_loader, model, optimizer, device, train=False,
            boundary_class_weights=boundary_w, coarse_class_weights=coarse_w,
            lambda_boundary=args.lambda_boundary, lambda_coarse=args.lambda_coarse, lambda_fine=args.lambda_fine,
            lambda_svo=args.lambda_svo, lambda_voice=args.lambda_voice,
            accum_steps=args.accum_steps, log_every=args.log_every,
        )

        score = (val_metrics["boundary_f1"] + val_metrics["coarse_macro_f1"] + val_metrics["fine_macro_f1"]) / 3.0

        print(f"\n📅 Epoch {epoch}")
        print(
            f"Train loss={train_metrics['loss']:.4f} | "
            f"B_F1={train_metrics['boundary_f1']:.4f} | "
            f"C_F1={train_metrics['coarse_macro_f1']:.4f} | "
            f"F_F1={train_metrics['fine_macro_f1']:.4f} | "
            f"SVO_F1={train_metrics['svo_macro_f1']:.4f} | "
            f"Voice_F1={train_metrics['voice_macro_f1']:.4f}"
        )
        if args.hn_enabled:
            print(
                f"  ⛏️  HN: boundary={train_metrics.get('hn_boundary_count', 0)} "
                f"coarse={train_metrics.get('hn_coarse_count', 0)} "
                f"total={train_metrics.get('hn_total', 0)} "
                f"ramp={hn_ramp:.2f}"
            )
        print(
            f"Val   loss={val_metrics['loss']:.4f} | "
            f"B_F1={val_metrics['boundary_f1']:.4f} | "
            f"C_F1={val_metrics['coarse_macro_f1']:.4f} | "
            f"F_F1={val_metrics['fine_macro_f1']:.4f} | "
            f"SVO_F1={val_metrics['svo_macro_f1']:.4f} | "
            f"Voice_F1={val_metrics['voice_macro_f1']:.4f} | "
            f"Score={score:.4f}"
        )

        print("\n[VAL boundary]")
        print(val_metrics["boundary_report"])
        print("[VAL coarse]")
        print(val_metrics["coarse_report"])
        print("[VAL fine]")
        print(val_metrics["fine_report"])
        print("[VAL svo]")
        print(val_metrics["svo_report"])
        print("[VAL voice]")
        print(val_metrics["voice_report"])

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_score": best_score,
        }, "checkpoint_last_multitask.pt")

        if score > best_score:
            best_score = score
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_score": best_score,
            }, "checkpoint_best_multitask.pt")
            torch.save(model.state_dict(), "best_model_multitask.pt")
            print("✅ nouveau best model sauvegardé")

    # ── Test final ──
    print("\n✅ Fin training, évaluation test sur le best model")
    ckpt = torch.load("checkpoint_best_multitask.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = run_epoch(
        test_loader, model, optimizer, device, train=False,
        boundary_class_weights=boundary_w, coarse_class_weights=coarse_w,
        lambda_boundary=args.lambda_boundary, lambda_coarse=args.lambda_coarse, lambda_fine=args.lambda_fine,
        lambda_svo=args.lambda_svo, lambda_voice=args.lambda_voice,
        accum_steps=args.accum_steps, log_every=args.log_every,
    )

    print("\n🎯 TEST")
    print(f"Loss={test_metrics['loss']:.4f}")
    print(f"Boundary F1={test_metrics['boundary_f1']:.4f}")
    print(f"Coarse   F1={test_metrics['coarse_macro_f1']:.4f}")
    print(f"Fine     F1={test_metrics['fine_macro_f1']:.4f}")
    print(f"SVO      F1={test_metrics['svo_macro_f1']:.4f}")
    print(f"Voice    F1={test_metrics['voice_macro_f1']:.4f}")
    print("\n[TEST boundary]")
    print(test_metrics["boundary_report"])
    print("[TEST coarse]")
    print(test_metrics["coarse_report"])
    print("[TEST fine]")
    print(test_metrics["fine_report"])
    print("[TEST svo]")
    print(test_metrics["svo_report"])
    print("[TEST voice]")
    print(test_metrics["voice_report"])


if __name__ == "__main__":
    main()

