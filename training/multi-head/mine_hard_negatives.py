#!/usr/bin/env python3
"""
Targeted hard negative mining — v2.

Différencie 4 types d'erreurs et applique des boosts par classe :

  FP_BOUNDARY  : span négatif classé comme entité  → précision coarse faible
  FN_BOUNDARY  : span positif manqué               → recall boundary
  COARSE_ERR   : span positif mais mauvais coarse
  FINE_ERR     : coarse ok mais mauvais fine

Les classes VALUE / EVENT / TIME / ABSTRACT ont des faux positifs massifs
→ boost supplémentaire sur FP_BOUNDARY prédit dans ces classes.

Usage:
    python3 mine_hard_negatives.py \
        --dataset data/train.adaptive.multitask.jsonl \
        --checkpoint checkpoint_best_multitask.pt \
        --output data/train.adaptive.multitask.jsonl \
        --device mps
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from labels import COARSE_LABELS, FINE_LABELS
from multitask_dataset import MultiTaskSpanDataset, make_collate_fn
from multitask_model import SpanMultiTaskModel

# ─────────────────────────────────────────────────────────────────────────────
# Boosts par type d'erreur
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BOOST = {
    "FP_BOUNDARY": 3.5,   # faux positif → le modèle invente une entité
    "FN_BOUNDARY": 2.0,   # faux négatif → le modèle rate une entité réelle
    "COARSE_ERR":  2.5,   # bonne boundary, mauvais coarse
    "FINE_ERR":    2.0,   # bon coarse, mauvais fine
}

# Classes avec précision coarse < 0.65 → FP_BOUNDARY encore plus pénalisé
LOW_PRECISION_COARSE = {"VALUE", "EVENT", "TIME", "ABSTRACT"}
FP_LOW_PREC_EXTRA_BOOST = 1.5

# Labels fine avec F1 < 0.65 → FINE_ERR extra boost
LOW_F1_FINE = {"hint_quantity", "hint_measure", "hint_rate", "hint_infra", "hint_object_generic"}
FINE_ERR_EXTRA_BOOST = 1.4


def mine(
    dataset_path: str,
    checkpoint_path: str,
    output_path: str,
    model_name: str,
    device: str,
    boosts: dict[str, float],
    decay: float,
    max_weight: float,
    min_weight: float,
    batch_size: int,
):
    print(f"🔍 Targeted HN mining v2 sur {dataset_path}")
    print(f"   checkpoint : {checkpoint_path}")
    print(f"   boosts     : {boosts}")
    print(f"   decay={decay}, max_weight={max_weight}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = SpanMultiTaskModel(model_name=model_name).to(device).float()

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "ema_state" in ckpt and ckpt["ema_state"] is not None:
        ema_state = {k: v.to(dtype=model.state_dict()[k].dtype) for k, v in ckpt["ema_state"].items()}
        model.load_state_dict(ema_state)
        print("   ✅ Poids EMA chargés")
    else:
        model.load_state_dict(ckpt["model_state"])
        print("   ✅ Poids model_state chargés")

    model.eval()

    ds = MultiTaskSpanDataset(dataset_path, tokenizer, max_length=128)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
        pin_memory=(device == "cuda"),
    )

    coarse_fine_mask = model.coarse_fine_mask.to(device)

    # id → list of (error_type | None, pred_coarse_name, pred_fine_name)
    results_by_id: dict[str, list] = {}
    error_counts: Counter = Counter()
    fp_by_coarse: Counter = Counter()
    fine_err_by_label: Counter = Counter()

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            spans          = batch["spans"]
            b_labels       = batch["boundary_labels"].to(device)
            c_labels       = batch["coarse_labels"].to(device)
            f_labels       = batch["fine_labels"].to(device)

            outputs = model({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "spans": spans,
            })

            si = outputs.get("span_indices")
            if si is not None:
                si = si.to(device=device, dtype=torch.long)
                b_labels = b_labels[si]
                c_labels = c_labels[si]
                f_labels = f_labels[si]

            b_pred = outputs["boundary_logits"].argmax(dim=-1)
            c_pred = outputs["coarse_logits"].argmax(dim=-1)

            masked_fine = outputs["fine_logits"].clone()
            masked_fine = masked_fine.masked_fill(~coarse_fine_mask[c_pred], -1e9)
            f_pred = masked_fine.argmax(dim=-1)

            # pos_map : position_globale → batch_idx
            pos_map = []
            for bi, sample_spans in enumerate(spans):
                for _ in range(len(sample_spans)):
                    pos_map.append(bi)

            n_global = len(pos_map)
            raw_results: list = [None] * n_global

            si_list = si.cpu().tolist() if si is not None else list(range(n_global))

            b_pred_l = b_pred.cpu().tolist()
            c_pred_l = c_pred.cpu().tolist()
            f_pred_l = f_pred.cpu().tolist()
            b_lab_l  = b_labels.cpu().tolist()
            c_lab_l  = c_labels.cpu().tolist()
            f_lab_l  = f_labels.cpu().tolist()

            for out_idx, in_idx in enumerate(si_list):
                bp = b_pred_l[out_idx]; bl = b_lab_l[out_idx]
                cp = c_pred_l[out_idx]; cl = c_lab_l[out_idx]
                fp_ = f_pred_l[out_idx]; fl = f_lab_l[out_idx]

                pred_coarse = COARSE_LABELS[cp] if cp < len(COARSE_LABELS) else "?"
                pred_fine   = FINE_LABELS[fp_]  if fp_ < len(FINE_LABELS)  else "?"

                if bp != bl:
                    err = "FP_BOUNDARY" if (bl == 0 and bp == 1) else "FN_BOUNDARY"
                    if err == "FP_BOUNDARY":
                        fp_by_coarse[pred_coarse] += 1
                elif bl == 1 and cp != cl:
                    err = "COARSE_ERR"
                elif bl == 1 and fp_ != fl:
                    err = "FINE_ERR"
                    fine_err_by_label[pred_fine] += 1
                else:
                    err = None

                raw_results[in_idx] = (err, pred_coarse, pred_fine)
                if err:
                    error_counts[err] += 1

            per_row: dict[int, list] = defaultdict(list)
            for in_idx, bi in enumerate(pos_map):
                per_row[bi].append(raw_results[in_idx])

            for bi, row_results in per_row.items():
                results_by_id[batch["ids"][bi]] = row_results

    # ── Rapport ──────────────────────────────────────────────────────────────
    print(f"\n📊 Erreurs détectées :")
    for err_type, count in error_counts.most_common():
        print(f"   {err_type:15s} : {count:6d}")

    print(f"\n   FP par coarse prédit (top 8) :")
    for coarse, n in fp_by_coarse.most_common(8):
        marker = "⚠️  " if coarse in LOW_PRECISION_COARSE else "   "
        print(f"   {marker}{coarse:10s} : {n}")

    print(f"\n   FINE_ERR par label prédit (top 8) :")
    for fine, n in fine_err_by_label.most_common(8):
        marker = "⚠️  " if fine in LOW_F1_FINE else "   "
        print(f"   {marker}{fine:30s} : {n}")

    # ── Mise à jour des poids ─────────────────────────────────────────────────
    raw_rows = [
        json.loads(line)
        for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    updated = 0
    decayed = 0

    for row in raw_rows:
        rid = row["id"]
        if rid not in results_by_id:
            continue

        row_results = results_by_id[rid]
        valid_cands = [c for c in row["candidates"] if _is_valid(c)]

        for i, c in enumerate(valid_cands):
            if i >= len(row_results):
                break

            entry = row_results[i]
            if entry is None:
                w = c.get("sample_weight", 1.0)
                c["sample_weight"] = max(min_weight, 1.0 + (w - 1.0) * decay)
                decayed += 1
                continue

            err, pred_coarse, pred_fine = entry

            if err is None:
                w = c.get("sample_weight", 1.0)
                c["sample_weight"] = max(min_weight, 1.0 + (w - 1.0) * decay)
                decayed += 1
            else:
                base = boosts.get(err, 2.0)
                if err == "FP_BOUNDARY" and pred_coarse in LOW_PRECISION_COARSE:
                    base *= FP_LOW_PREC_EXTRA_BOOST
                if err == "FINE_ERR" and pred_fine in LOW_F1_FINE:
                    base *= FINE_ERR_EXTRA_BOOST
                c["sample_weight"] = min(max_weight, c.get("sample_weight", 1.0) * base)
                c["neg_type"] = err
                updated += 1

    print(f"\n   Weights boostés : {updated}")
    print(f"   Weights décrus  : {decayed}")

    out_path = Path(output_path)
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in raw_rows) + "\n",
        encoding="utf-8",
    )
    print(f"   ✅ Dataset mis à jour → {out_path}")


def _is_valid(c: dict) -> bool:
    ts = c.get("tok_start")
    te = c.get("tok_end")
    if not isinstance(ts, int) or not isinstance(te, int):
        return False
    return ts >= 0 and te >= ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      required=True)
    parser.add_argument("--checkpoint",   default="checkpoint_best_multitask.pt")
    parser.add_argument("--output",       default=None)
    parser.add_argument("--model-name",   default="microsoft/deberta-v3-base")
    parser.add_argument("--device",       choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--boost-fp",     type=float, default=DEFAULT_BOOST["FP_BOUNDARY"],
                        help="Boost faux positifs boundary (défaut=3.5)")
    parser.add_argument("--boost-fn",     type=float, default=DEFAULT_BOOST["FN_BOUNDARY"],
                        help="Boost faux négatifs boundary (défaut=2.0)")
    parser.add_argument("--boost-coarse", type=float, default=DEFAULT_BOOST["COARSE_ERR"],
                        help="Boost erreurs coarse (défaut=2.5)")
    parser.add_argument("--boost-fine",   type=float, default=DEFAULT_BOOST["FINE_ERR"],
                        help="Boost erreurs fine (défaut=2.0)")
    parser.add_argument("--decay",        type=float, default=0.85)
    parser.add_argument("--max-weight",   type=float, default=8.0)
    parser.add_argument("--min-weight",   type=float, default=0.3)
    parser.add_argument("--batch-size",   type=int,   default=32)
    args = parser.parse_args()

    if args.device is None:
        args.device = (
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available()         else "cpu"
        )
        print(f"   Device auto-détecté : {args.device}")

    mine(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        output_path=args.output or args.dataset,
        model_name=args.model_name,
        device=args.device,
        boosts={
            "FP_BOUNDARY": args.boost_fp,
            "FN_BOUNDARY": args.boost_fn,
            "COARSE_ERR":  args.boost_coarse,
            "FINE_ERR":    args.boost_fine,
        },
        decay=args.decay,
        max_weight=args.max_weight,
        min_weight=args.min_weight,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

