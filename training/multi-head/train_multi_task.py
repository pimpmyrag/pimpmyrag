from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import argparse
import json
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, classification_report

from multitask_dataset import MultiTaskSpanDataset, make_collate_fn
from multitask_model import SpanMultiTaskModel
from labels import COARSE_LABELS, FINE_LABELS


def compute_class_weights_from_multitask_jsonl(path: str, power: float = 0.5):
    """
    Calcule des poids de classes à partir du dataset multitask enrichi.

    - power=1.0  -> inverse-fréquence brut
    - power=0.5  -> inverse-fréquence tempéré (recommandé)
    - power=0.0  -> tous les poids = 1.0

    Retourne:
      boundary_w, coarse_w, fine_w, boundary_counts, coarse_counts, fine_counts
    """
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
                # classe absente du training -> poids neutre
                weights[i] = 1.0

        # normalisation: moyenne = 1.0
        weights = weights / weights.mean()
        return weights

    return (
        make_weights(boundary_counts, 2, power=power),
        make_weights(coarse_counts, len(COARSE_LABELS), power=power),
        make_weights(fine_counts, len(FINE_LABELS), power=power),
        boundary_counts,
        coarse_counts,
        fine_counts,
    )


def apply_class_weight_floor(weights: torch.Tensor, class_idx: int, min_value: float) -> torch.Tensor:
    """Applique un plancher minimal à une classe sans re-normaliser ensuite."""
    weights = weights.clone()
    weights[class_idx] = max(weights[class_idx].item(), float(min_value))
    return weights


def safe_macro_f1(y_true, y_pred, labels=None):
    if not y_true:
        return 0.0
    return f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)


def run_epoch(
        loader,
        model,
        optimizer,
        device,
        train: bool,
        boundary_class_weights=None,
        coarse_class_weights=None,
        fine_class_weights=None,   # gardé pour compat API, ignoré si compute_loss ne l'utilise pas
        lambda_boundary=1.0,
        lambda_coarse=1.0,
        lambda_fine=1.2,
        lambda_compat=0.0,         # gardé pour compat API, ignoré ici
        accum_steps=1,
        log_every=50,
        focal_gamma=0.0,
):
    """
    Version adaptée à l'architecture :
      - boundary : binaire
      - coarse   : 6 familles + NONE
      - fine     : 22 labels positifs uniquement
      - fine loss : positive-only
      - fine metrics : positive-only
      - fine prediction : masquage coarse -> fine

    Retourne:
      metrics = {
          "loss": float,
          "boundary_f1": float,
          "coarse_macro_f1": float,
          "fine_macro_f1": float,              # positive-only
          "boundary_report": str,
          "coarse_report": str,
          "fine_report": str,                  # positive-only
      }
    """

    import torch
    from sklearn.metrics import f1_score, classification_report

    def safe_macro_f1_local(y_true, y_pred, labels=None):
        if not y_true:
            return 0.0
        return f1_score(
            y_true,
            y_pred,
            average="macro",
            labels=labels,
            zero_division=0
        )

    def masked_fine_predictions(fine_logits, coarse_preds, coarse_fine_mask):
        """
        Applique un masquage coarse -> fine.

        Args:
            fine_logits: [N, F]
            coarse_preds: [N]
            coarse_fine_mask: [C, F] bool

        Returns:
            pred_fine: list[int] de taille N
                - label fine prédit si coarse != NONE et masque valide
                - -1 si coarse = NONE ou aucun fine autorisé
        """
        if fine_logits.numel() == 0:
            return []

        device_local = fine_logits.device
        coarse_preds_t = torch.as_tensor(coarse_preds, dtype=torch.long, device=device_local)

        # [N, F]
        allowed = coarse_fine_mask[coarse_preds_t]

        # rows sans aucun label fine autorisé (ex: coarse=NONE)
        no_valid = ~allowed.any(dim=-1)

        masked_logits = fine_logits.clone()
        masked_logits = masked_logits.masked_fill(~allowed, -1e9)

        pred = masked_logits.argmax(dim=-1)

        # pour les coarse=NONE / rows invalides, on force une prédiction invalide
        pred = pred.masked_fill(no_valid, -1)

        return pred.detach().cpu().tolist()

    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    losses = []

    all_b_true, all_b_pred = [], []
    all_c_true, all_c_pred = [], []

    # fine positive-only
    all_f_true_pos, all_f_pred_pos = [], []

    coarse_fine_mask = getattr(model, "coarse_fine_mask", None)
    if coarse_fine_mask is None:
        raise ValueError(
            "Le modèle n'expose pas `coarse_fine_mask`. "
            "Assure-toi d'utiliser la version modifiée de multitask_model.py."
        )

    coarse_fine_mask = coarse_fine_mask.to(device)

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        spans = batch["spans"]

        boundary_labels = batch["boundary_labels"].to(device)
        coarse_labels = batch["coarse_labels"].to(device)
        fine_labels = batch["fine_labels"].to(device)
        sample_weights = batch["sample_weights"].to(device)

        # Sanity check avant forward
        num_spans = sum(len(x) for x in spans)
        if not (
                num_spans
                == boundary_labels.size(0)
                == coarse_labels.size(0)
                == fine_labels.size(0)
                == sample_weights.size(0)
        ):
            raise ValueError(
                "Mismatch batch avant forward: "
                f"num_spans={num_spans}, "
                f"boundary={boundary_labels.size(0)}, "
                f"coarse={coarse_labels.size(0)}, "
                f"fine={fine_labels.size(0)}, "
                f"sample_weights={sample_weights.size(0)}, "
                f"ids={batch.get('ids')}"
            )

        with torch.set_grad_enabled(train):
            outputs = model({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "spans": spans,
            })

            # Si le modèle renvoie span_indices, on aligne tout sur les spans gardés
            span_indices = outputs.get("span_indices", None)
            if span_indices is not None:
                si = span_indices.to(device=device, dtype=torch.long)
                boundary_labels_loss = boundary_labels[si]
                coarse_labels_loss = coarse_labels[si]
                fine_labels_loss = fine_labels[si]
                sample_weights_loss = sample_weights[si]
            else:
                boundary_labels_loss = boundary_labels
                coarse_labels_loss = coarse_labels
                fine_labels_loss = fine_labels
                sample_weights_loss = sample_weights

            # Sanity check après forward / avant loss
            num_logits = outputs["fine_logits"].size(0)
            if not (
                    num_logits
                    == boundary_labels_loss.size(0)
                    == coarse_labels_loss.size(0)
                    == fine_labels_loss.size(0)
                    == sample_weights_loss.size(0)
            ):
                raise ValueError(
                    "Mismatch logits/labels après forward: "
                    f"logits={num_logits}, "
                    f"boundary={boundary_labels_loss.size(0)}, "
                    f"coarse={coarse_labels_loss.size(0)}, "
                    f"fine={fine_labels_loss.size(0)}, "
                    f"sample_weights={sample_weights_loss.size(0)}, "
                    f"ids={batch.get('ids')}"
                )

            loss_dict = model.compute_loss(
                outputs=outputs,
                boundary_labels=boundary_labels_loss,
                coarse_labels=coarse_labels_loss,
                fine_labels=fine_labels_loss,
                sample_weights=sample_weights_loss,
                boundary_class_weights=boundary_class_weights,
                coarse_class_weights=coarse_class_weights,
                lambda_boundary=lambda_boundary,
                lambda_coarse=lambda_coarse,
                lambda_fine=lambda_fine,
                focal_gamma=focal_gamma,
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
            try:
                total_steps = len(loader)
            except Exception:
                total_steps = "?"
            avg_loss = sum(losses) / max(1, len(losses))
            print(
                f"[{mode}] step={step}/{total_steps} "
                f"loss={loss_dict['loss'].item():.4f} "
                f"avg_loss={avg_loss:.4f}"
            )

        # Predictions
        b_pred = outputs["boundary_logits"].argmax(dim=-1).detach().cpu().tolist()
        c_pred = outputs["coarse_logits"].argmax(dim=-1).detach().cpu().tolist()

        # Fine prédite avec masquage coarse -> fine
        f_pred = masked_fine_predictions(
            outputs["fine_logits"],
            c_pred,
            coarse_fine_mask,
        )

        # Vérité terrain alignée sur les spans scorés
        if span_indices is not None:
            si_cpu = span_indices.detach().cpu().to(dtype=torch.long)
            b_true = boundary_labels.detach().cpu()[si_cpu].tolist()
            c_true = coarse_labels.detach().cpu()[si_cpu].tolist()
            f_true = fine_labels.detach().cpu()[si_cpu].tolist()
        else:
            b_true = boundary_labels.detach().cpu().tolist()
            c_true = coarse_labels.detach().cpu().tolist()
            f_true = fine_labels.detach().cpu().tolist()

        # Accumulate boundary / coarse
        all_b_true.extend(b_true)
        all_b_pred.extend(b_pred)

        all_c_true.extend(c_true)
        all_c_pred.extend(c_pred)

        # Fine metrics = POSITIVE ONLY
        for bt, ft, fp in zip(b_true, f_true, f_pred):
            if bt == 1:
                all_f_true_pos.append(ft)
                all_f_pred_pos.append(fp)

    if train and (len(loader) % accum_steps != 0):
        optimizer.step()
        optimizer.zero_grad()

    metrics = {
        "loss": sum(losses) / max(1, len(losses)),
        "boundary_f1": safe_macro_f1_local(all_b_true, all_b_pred),
        "coarse_macro_f1": safe_macro_f1_local(
            all_c_true,
            all_c_pred,
            labels=list(range(len(COARSE_LABELS)))
        ),
        "fine_macro_f1": safe_macro_f1_local(
            all_f_true_pos,
            all_f_pred_pos,
            labels=list(range(len(FINE_LABELS)))
        ),
        "boundary_report": classification_report(
            all_b_true,
            all_b_pred,
            digits=3,
            zero_division=0
        ) if all_b_true else "N/A",
        "coarse_report": classification_report(
            all_c_true,
            all_c_pred,
            labels=list(range(len(COARSE_LABELS))),
            target_names=COARSE_LABELS,
            digits=3,
            zero_division=0
        ) if all_c_true else "N/A",
        "fine_report": classification_report(
            all_f_true_pos,
            all_f_pred_pos,
            labels=list(range(len(FINE_LABELS))),
            target_names=FINE_LABELS,
            digits=3,
            zero_division=0
        ) if all_f_true_pos else "N/A",
    }

    return metrics


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
    parser.add_argument("--lambda-compat", type=float, default=0.2)
    parser.add_argument("--focal-gamma", type=float, default=0.0,
                        help="Focal loss gamma pour boundary (0=CE, 2.0=focal)")
    parser.add_argument("--head-lr-multiplier", type=float, default=5.0,
                        help="Multiplicateur LR pour les heads vs encoder")
    parser.add_argument("--warmup-epochs", type=int, default=1,
                        help="Nombre d'epochs de linear warmup LR")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing pour coarse/fine CE")

    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--class-weights", choices=["none", "auto"], default="auto")
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help="Puissance appliquée aux poids inverse-fréquence. 1.0 = brut, 0.5 = tempéré (recommandé)."
    )
    parser.add_argument(
        "--min-coarse-none-weight",
        type=float,
        default=0.05,
        help="Poids minimum autorisé pour la classe NONE de la tête coarse."
    )
    parser.add_argument(
        "--min-fine-none-weight",
        type=float,
        default=0.05,
        help="Poids minimum autorisé pour la classe NONE de la tête fine."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Chemin vers un checkpoint pour reprendre le training."
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="Force l'epoch de départ quand on reprend depuis un checkpoint."
    )

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

    collate_fn = make_collate_fn(tokenizer)

    pin_memory = (device == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    model = SpanMultiTaskModel(model_name=args.model_name).to(device).float()

    # Differential LR : encoder à LR base, heads + MLP à LR * multiplier
    encoder_params = list(model.encoder.parameters())
    head_params = (
        list(model.span_mlp.parameters())
        + list(model.boundary_head.parameters())
        + list(model.coarse_head.parameters())
        + list(model.fine_head.parameters())
        + list(model.width_emb.parameters())
    )
    head_lr = args.lr * args.head_lr_multiplier
    optimizer = AdamW([
        {"params": encoder_params, "lr": args.lr},
        {"params": head_params, "lr": head_lr},
    ])

    # LR scheduler : linear warmup + cosine decay
    total_epochs = args.epochs
    warmup_epochs_count = min(args.warmup_epochs, total_epochs)
    steps_per_epoch = 1  # sera ajusté après DataLoader creation

    print(f"📐 Differential LR: encoder={args.lr}, heads={head_lr}")
    print(f"📐 Focal gamma: {args.focal_gamma}")
    print(f"📐 Warmup: {warmup_epochs_count} epochs, then cosine decay")

    boundary_w = coarse_w = fine_w = None
    if args.class_weights == "auto":
        (
            boundary_w,
            coarse_w,
            fine_w,
            boundary_counts,
            coarse_counts,
            fine_counts,
        ) = compute_class_weights_from_multitask_jsonl(
            args.train,
            power=args.class_weight_power,
        )

        coarse_none_idx = len(COARSE_LABELS) - 1
        fine_none_idx = len(FINE_LABELS) - 1

        coarse_w = apply_class_weight_floor(
            coarse_w,
            coarse_none_idx,
            args.min_coarse_none_weight,
        )
        fine_w = apply_class_weight_floor(
            fine_w,
            fine_none_idx,
            args.min_fine_none_weight,
        )

        print("⚖️ class weights auto activés")
        print(f"   class_weight_power       = {args.class_weight_power}")
        print(f"   min_coarse_none_weight   = {args.min_coarse_none_weight}")
        print(f"   min_fine_none_weight     = {args.min_fine_none_weight}")

        print("\n[boundary counts / weights]")
        for i in range(2):
            print(f"  class {i}: count={boundary_counts.get(i, 0)} weight={boundary_w[i].item():.6f}")

        print("\n[coarse counts / weights]")
        for i, name in enumerate(COARSE_LABELS):
            print(f"  {name:<10} count={coarse_counts.get(i, 0):>8} weight={coarse_w[i].item():.6f}")

        print("\n[fine counts / weights]")
        for i, name in enumerate(FINE_LABELS):
            print(f"  {name:<22} count={fine_counts.get(i, 0):>8} weight={fine_w[i].item():.6f}")
    else:
        print("⚖️ class weights désactivés")

    best_score = -1.0
    start_epoch = 1

    # LR scheduler basé sur les epochs
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs_count)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs_count))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs_count])

    # Reprise éventuelle depuis checkpoint
    if args.resume is not None:
        print(f"⤴️ Reprise depuis checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        if "optim_state" in ckpt and ckpt["optim_state"] is not None:
            optimizer.load_state_dict(ckpt["optim_state"])

        if args.start_epoch is not None:
            start_epoch = args.start_epoch
        else:
            start_epoch = int(ckpt.get("epoch", 0)) + 1

        best_score = ckpt.get("best_score", -1.0)

        print(
            f"✅ checkpoint rechargé | start_epoch={start_epoch} | best_score={best_score:.4f}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        # Log LR
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"\n🔧 Epoch {epoch} | LR encoder={lrs[0]:.2e}, heads={lrs[1]:.2e}")

        train_metrics = run_epoch(
            train_loader,
            model,
            optimizer,
            device,
            train=True,
            boundary_class_weights=boundary_w,
            coarse_class_weights=coarse_w,
            fine_class_weights=fine_w,
            lambda_boundary=args.lambda_boundary,
            lambda_coarse=args.lambda_coarse,
            lambda_fine=args.lambda_fine,
            lambda_compat=args.lambda_compat,
            accum_steps=args.accum_steps,
            log_every=args.log_every,
            focal_gamma=args.focal_gamma,
        )

        val_metrics = run_epoch(
            val_loader,
            model,
            optimizer,
            device,
            train=False,
            boundary_class_weights=boundary_w,
            coarse_class_weights=coarse_w,
            fine_class_weights=fine_w,
            lambda_boundary=args.lambda_boundary,
            lambda_coarse=args.lambda_coarse,
            lambda_fine=args.lambda_fine,
            lambda_compat=args.lambda_compat,
            accum_steps=args.accum_steps,
            log_every=args.log_every,
            focal_gamma=args.focal_gamma,
        )

        # Step scheduler after each epoch
        scheduler.step()

        score = (
            val_metrics["boundary_f1"]
            + val_metrics["coarse_macro_f1"]
            + val_metrics["fine_macro_f1"]
        ) / 3.0

        print(f"\n📅 Epoch {epoch}")
        print(
            f"Train loss={train_metrics['loss']:.4f} | "
            f"Boundary F1={train_metrics['boundary_f1']:.4f} | "
            f"Coarse F1={train_metrics['coarse_macro_f1']:.4f} | "
            f"Fine F1={train_metrics['fine_macro_f1']:.4f}"
        )
        print(
            f"Val   loss={val_metrics['loss']:.4f} | "
            f"Boundary F1={val_metrics['boundary_f1']:.4f} | "
            f"Coarse F1={val_metrics['coarse_macro_f1']:.4f} | "
            f"Fine F1={val_metrics['fine_macro_f1']:.4f} | "
            f"Score={score:.4f}"
        )

        print("\n[VAL boundary]")
        print(val_metrics["boundary_report"])
        print("[VAL coarse]")
        print(val_metrics["coarse_report"])
        print("[VAL fine]")
        print(val_metrics["fine_report"])

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

    print("\n✅ Fin training, évaluation test sur le best model")

    ckpt = torch.load("checkpoint_best_multitask.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = run_epoch(
        test_loader,
        model,
        optimizer,
        device,
        train=False,
        boundary_class_weights=boundary_w,
        coarse_class_weights=coarse_w,
        fine_class_weights=fine_w,
        lambda_boundary=args.lambda_boundary,
        lambda_coarse=args.lambda_coarse,
        lambda_fine=args.lambda_fine,
        lambda_compat=args.lambda_compat,
        accum_steps=args.accum_steps,
        log_every=args.log_every,
        focal_gamma=args.focal_gamma,
    )

    print("\n🎯 TEST")
    print(f"Loss={test_metrics['loss']:.4f}")
    print(f"Boundary F1={test_metrics['boundary_f1']:.4f}")
    print(f"Coarse   F1={test_metrics['coarse_macro_f1']:.4f}")
    print(f"Fine     F1={test_metrics['fine_macro_f1']:.4f}")

    print("\n[TEST boundary]")
    print(test_metrics["boundary_report"])
    print("[TEST coarse]")
    print(test_metrics["coarse_report"])
    print("[TEST fine]")
    print(test_metrics["fine_report"])


if __name__ == "__main__":
    main()
