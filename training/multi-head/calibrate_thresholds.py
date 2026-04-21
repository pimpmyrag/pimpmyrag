from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import argparse
import itertools
import json
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, classification_report

from multitask_dataset import MultiTaskSpanDataset, make_collate_fn
from multitask_model import SpanMultiTaskModel
from labels import COARSE_LABELS, FINE_LABELS


def parse_grid(s: str) -> List[float]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError(f"Grille vide: {s}")
    return vals


def safe_macro_f1(y_true, y_pred, labels=None):
    if not y_true:
        return 0.0
    return f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)


def masked_fine_predictions_from_logits(fine_logits: torch.Tensor, coarse_preds: torch.Tensor, coarse_fine_mask: torch.Tensor):
    """
    fine_logits: [N, F]
    coarse_preds: [N]
    coarse_fine_mask: [C, F] bool

    returns: pred_fine [N] with -1 for invalid rows / coarse NONE
    """
    if fine_logits.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=fine_logits.device)

    allowed = coarse_fine_mask[coarse_preds]  # [N, F]
    no_valid = ~allowed.any(dim=-1)

    masked_logits = fine_logits.masked_fill(~allowed, -1e9)
    pred = masked_logits.argmax(dim=-1)
    pred = pred.masked_fill(no_valid, -1)
    return pred


def collect_val_outputs(loader, model, device):
    model.eval()
    coarse_fine_mask = getattr(model, "coarse_fine_mask", None)
    if coarse_fine_mask is None:
        raise ValueError("Le modèle n'expose pas coarse_fine_mask.")
    coarse_fine_mask = coarse_fine_mask.to(device)

    rows = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            spans = batch["spans"]

            boundary_labels = batch["boundary_labels"].to(device)
            coarse_labels = batch["coarse_labels"].to(device)
            fine_labels = batch["fine_labels"].to(device)

            outputs = model({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "spans": spans,
            })

            span_indices = outputs.get("span_indices", None)
            if span_indices is not None:
                si = span_indices.to(device=device, dtype=torch.long)
                b_true = boundary_labels[si]
                c_true = coarse_labels[si]
                f_true = fine_labels[si]
            else:
                b_true = boundary_labels
                c_true = coarse_labels
                f_true = fine_labels

            b_logits = outputs["boundary_logits"]
            c_logits = outputs["coarse_logits"]
            f_logits = outputs["fine_logits"]

            b_probs = torch.softmax(b_logits, dim=-1)
            c_probs = torch.softmax(c_logits, dim=-1)

            # stockage par span scoré
            for i in range(b_logits.size(0)):
                rows.append({
                    "b_true": int(b_true[i].item()),
                    "c_true": int(c_true[i].item()),
                    "f_true": int(f_true[i].item()),
                    "p_ent": float(b_probs[i, 1].item()),
                    "coarse_probs": c_probs[i].detach().cpu().tolist(),
                    "fine_logits": f_logits[i].detach().cpu().tolist(),
                })

    return rows


def evaluate_thresholds(rows: List[Dict[str, Any]], coarse_fine_mask: torch.Tensor, tau_boundary: float, tau_none: float, tau_coarse: float):
    none_idx = len(COARSE_LABELS) - 1

    all_b_true, all_b_pred = [], []
    all_c_true, all_c_pred = [], []
    all_f_true_pos, all_f_pred_pos = [], []

    device = coarse_fine_mask.device

    for row in rows:
        b_true = row["b_true"]
        c_true = row["c_true"]
        f_true = row["f_true"]
        p_ent = row["p_ent"]
        c_probs = torch.tensor(row["coarse_probs"], dtype=torch.float32, device=device)
        f_logits = torch.tensor(row["fine_logits"], dtype=torch.float32, device=device).unsqueeze(0)

        # 1) boundary gating
        b_pred = 1 if p_ent >= tau_boundary else 0

        # 2) coarse decision avec seuils
        if b_pred == 0:
            c_pred = none_idx
            f_pred = -1
        else:
            coarse_top_idx = int(torch.argmax(c_probs).item())
            coarse_top_prob = float(torch.max(c_probs).item())
            p_none = float(c_probs[none_idx].item())

            if (p_none >= tau_none) or (coarse_top_prob < tau_coarse):
                c_pred = none_idx
                f_pred = -1
            else:
                c_pred = coarse_top_idx
                c_pred_t = torch.tensor([c_pred], dtype=torch.long, device=device)
                f_pred_t = masked_fine_predictions_from_logits(f_logits, c_pred_t, coarse_fine_mask)
                f_pred = int(f_pred_t[0].item())

        all_b_true.append(b_true)
        all_b_pred.append(b_pred)

        all_c_true.append(c_true)
        all_c_pred.append(c_pred)

        if b_true == 1:
            all_f_true_pos.append(f_true)
            all_f_pred_pos.append(f_pred)

    boundary_f1 = safe_macro_f1(all_b_true, all_b_pred)
    coarse_macro_f1 = safe_macro_f1(all_c_true, all_c_pred, labels=list(range(len(COARSE_LABELS))))
    fine_macro_f1 = safe_macro_f1(all_f_true_pos, all_f_pred_pos, labels=list(range(len(FINE_LABELS))))
    fine_micro_f1 = f1_score(all_f_true_pos, all_f_pred_pos, average="micro", labels=list(range(len(FINE_LABELS))), zero_division=0) if all_f_true_pos else 0.0

    metrics = {
        "tau_boundary": tau_boundary,
        "tau_none": tau_none,
        "tau_coarse": tau_coarse,
        "boundary_f1": boundary_f1,
        "coarse_macro_f1": coarse_macro_f1,
        "fine_macro_f1": fine_macro_f1,
        "fine_micro_f1": fine_micro_f1,
        "score": (boundary_f1 + coarse_macro_f1 + fine_macro_f1) / 3.0,
        "boundary_report": classification_report(all_b_true, all_b_pred, digits=3, zero_division=0),
        "coarse_report": classification_report(
            all_c_true, all_c_pred,
            labels=list(range(len(COARSE_LABELS))),
            target_names=COARSE_LABELS,
            digits=3, zero_division=0
        ),
        "fine_report": classification_report(
            all_f_true_pos, all_f_pred_pos,
            labels=list(range(len(FINE_LABELS))),
            target_names=FINE_LABELS,
            digits=3, zero_division=0
        ) if all_f_true_pos else "N/A",
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--tau-boundary-grid", type=str, default="0.40,0.80,0.85,0,90")
    parser.add_argument("--tau-none-grid", type=str, default="0.40,0.80,0.85,0,90")
    parser.add_argument("--tau-coarse-grid", type=str, default="0.40,0.80,0.85,0,90")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--save-json", type=str, default="threshold_search_results.json")
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"✅ device = {device}")

    tokenizer_source = args.tokenizer_path or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    val_ds = MultiTaskSpanDataset(args.val, tokenizer, max_length=args.max_length)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
        pin_memory=(device == "cuda"),
    )

    model = SpanMultiTaskModel(model_name=args.model_name).to(device).float()

    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    print("⏳ Collecte des sorties de validation...")
    rows = collect_val_outputs(val_loader, model, device)
    print(f"✅ {len(rows)} spans scorés collectés")

    coarse_fine_mask = model.coarse_fine_mask.to(device)

    tau_b_grid = parse_grid(args.tau_boundary_grid)
    tau_none_grid = parse_grid(args.tau_none_grid)
    tau_coarse_grid = parse_grid(args.tau_coarse_grid)

    results = []
    total = len(tau_b_grid) * len(tau_none_grid) * len(tau_coarse_grid)
    seen = 0

    for tb, tn, tc in itertools.product(tau_b_grid, tau_none_grid, tau_coarse_grid):
        seen += 1
        metrics = evaluate_thresholds(rows, coarse_fine_mask, tb, tn, tc)
        results.append(metrics)
        if seen % 20 == 0 or seen == total:
            print(f"   ... {seen}/{total} combinaisons évaluées")

    results.sort(key=lambda x: x["score"], reverse=True)

    print("\n🏆 TOP CONFIGS")
    for i, r in enumerate(results[: args.topk], start=1):
        print(
            f"#{i} | tau_boundary={r['tau_boundary']:.2f} | "
            f"tau_none={r['tau_none']:.2f} | tau_coarse={r['tau_coarse']:.2f} | "
            f"boundary={r['boundary_f1']:.4f} | coarse={r['coarse_macro_f1']:.4f} | "
            f"fine={r['fine_macro_f1']:.4f} | fine_micro={r['fine_micro_f1']:.4f} | score={r['score']:.4f}"
        )

    best = results[0]
    print("\n✅ BEST CONFIG")
    print(json.dumps({k: v for k, v in best.items() if not k.endswith("_report")}, ensure_ascii=False, indent=2))

    print("\n[VAL boundary report - BEST]")
    print(best["boundary_report"])
    print("[VAL coarse report - BEST]")
    print(best["coarse_report"])
    print("[VAL fine report - BEST]")
    print(best["fine_report"])

    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Résultats complets sauvés dans {args.save_json}")


if __name__ == "__main__":
    main()
