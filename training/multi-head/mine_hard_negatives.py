#!/usr/bin/env python3
"""
Active hard negative mining.

Passe le modèle courant (checkpoint EMA) sur le dataset train multitask,
identifie les candidats mal prédits, et booste leur sample_weight.

Usage:
    python3 mine_hard_negatives.py \
        --dataset data/train.adaptive.multitask.jsonl \
        --checkpoint checkpoint_best_multitask.pt \
        --output data/train.adaptive.multitask.jsonl \
        --device cuda \
        --boost 2.5 \
        --decay 0.85 \
        --max-weight 6.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from multitask_dataset import MultiTaskSpanDataset, make_collate_fn
from multitask_model import SpanMultiTaskModel


def mine(
    dataset_path: str,
    checkpoint_path: str,
    output_path: str,
    model_name: str,
    device: str,
    boost: float,       # multiplicateur de poids si candidat mal prédit
    decay: float,       # facteur de décroissance si candidat bien prédit
    max_weight: float,  # plafond du sample_weight
    min_weight: float,  # plancher (ne jamais tomber sous)
    batch_size: int,
):
    print(f"🔍 Mining hard negatives sur {dataset_path}")
    print(f"   checkpoint: {checkpoint_path}")
    print(f"   boost={boost}, decay={decay}, max_weight={max_weight}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = SpanMultiTaskModel(model_name=model_name).to(device).float()

    ckpt = torch.load(checkpoint_path, map_location=device)
    # Essayer de charger l'état EMA si disponible, sinon model_state
    if "ema_state" in ckpt and ckpt["ema_state"] is not None:
        ema_state = {k: v.to(dtype=model.state_dict()[k].dtype) for k, v in ckpt["ema_state"].items()}
        model.load_state_dict(ema_state)
        print("   ✅ Poids EMA chargés")
    else:
        model.load_state_dict(ckpt["model_state"])
        print("   ✅ Poids model_state chargés")

    model.eval()

    ds = MultiTaskSpanDataset(dataset_path, tokenizer, max_length=128)
    collate_fn = make_collate_fn(tokenizer)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    # Collecter les prédictions par (row_idx, candidate_idx_in_row)
    # On a besoin de savoir si chaque candidat était bien ou mal prédit.
    # On itère dans l'ordre (shuffle=False) et on trackle les résultats.

    all_results: list[dict] = []  # {row_idx, cand_idx, wrong: bool}
    row_cursor = 0
    cand_cursors: list[int] = [0] * len(ds.rows)  # curseur candidat par row

    # Pré-calculer pour chaque row combien de candidats valides il a
    # (le dataset filtre les invalides, donc on doit s'aligner sur ce que le loader sort)
    # On va collecter résultats en ordre et les associer aux rows via les ids

    coarse_fine_mask = model.coarse_fine_mask.to(device)

    results_by_id: dict[str, list[bool]] = {}  # id -> [wrong_per_candidate]

    total_candidates = 0
    total_wrong = 0

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
                boundary_labels = boundary_labels[si]
                coarse_labels = coarse_labels[si]
                fine_labels = fine_labels[si]

            b_pred = outputs["boundary_logits"].argmax(dim=-1)
            c_pred = outputs["coarse_logits"].argmax(dim=-1)

            # Fine avec masquage coarse -> fine
            allowed = coarse_fine_mask[c_pred]
            masked_fine = outputs["fine_logits"].clone()
            masked_fine = masked_fine.masked_fill(~allowed, -1e9)
            f_pred = masked_fine.argmax(dim=-1)

            # Un candidat est "wrong" si:
            # - boundary mal prédit (FP ou FN), OU
            # - boundary=1 ET coarse mal prédit, OU
            # - boundary=1 ET fine mal prédit
            b_wrong = (b_pred != boundary_labels)
            c_wrong = (boundary_labels == 1) & (c_pred != coarse_labels)
            f_wrong = (boundary_labels == 1) & (f_pred != fine_labels)
            wrong = (b_wrong | c_wrong | f_wrong).cpu().tolist()

            # Aligner sur les spans (en tenant compte de span_indices si présent)
            # On reconstruit le mapping candidat -> row en suivant l'ordre des spans
            # Pour simplifier, on reconstruit depuis le batch original (avant span_indices)
            # car on a besoin de l'index complet pour mettre à jour le jsonl

            # On trace par id, en order: chaque span du batch est dans spans[batch_i]
            # L'ordre global est: spans[0][0], spans[0][1], ..., spans[1][0], ...
            if span_indices is not None:
                si_list = span_indices.cpu().tolist()
            else:
                si_list = list(range(sum(len(s) for s in spans)))

            # Construire mapping position_globale -> (batch_i, cand_i)
            pos_map = []
            for bi, sample_spans in enumerate(spans):
                for ci in range(len(sample_spans)):
                    pos_map.append((bi, ci))

            # Associer les résultats aux ids
            # wrong est aligné sur span_indices si présent
            batch_ids = batch["ids"]
            # Résultats par position globale AVANT span_indices
            raw_wrong = [False] * len(pos_map)
            if span_indices is not None:
                for out_idx, in_idx in enumerate(si_list):
                    raw_wrong[in_idx] = wrong[out_idx]
            else:
                raw_wrong = wrong

            # Grouper par row (batch_i)
            per_row: dict[int, list[bool]] = {}
            for pos, (bi, ci) in enumerate(pos_map):
                if bi not in per_row:
                    per_row[bi] = []
                per_row[bi].append(raw_wrong[pos])

            for bi, row_wrong in per_row.items():
                rid = batch_ids[bi]
                results_by_id[rid] = row_wrong
                total_candidates += len(row_wrong)
                total_wrong += sum(row_wrong)

    print(f"   Candidats: {total_candidates} | Mal prédits: {total_wrong} ({100*total_wrong/max(1,total_candidates):.1f}%)")

    # Mettre à jour les sample_weights dans le jsonl
    raw_rows = [
        json.loads(line)
        for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    updated = 0
    for row in raw_rows:
        rid = row["id"]
        if rid not in results_by_id:
            continue

        row_wrong = results_by_id[rid]
        valid_cands = [c for c in row["candidates"] if _is_valid(c)]

        # Aligner avec les candidats valides (même filtrage que le dataset)
        for i, c in enumerate(valid_cands):
            if i >= len(row_wrong):
                break
            if row_wrong[i]:
                c["sample_weight"] = min(max_weight, c.get("sample_weight", 1.0) * boost)
                updated += 1
            else:
                # Décroissance douce vers 1.0
                w = c.get("sample_weight", 1.0)
                c["sample_weight"] = max(min_weight, 1.0 + (w - 1.0) * decay)

    print(f"   Weights boostés: {updated}")

    out_path = Path(output_path)
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in raw_rows) + "\n",
        encoding="utf-8"
    )
    print(f"   ✅ Dataset mis à jour: {out_path}")


def _is_valid(c: dict) -> bool:
    ts = c.get("tok_start")
    te = c.get("tok_end")
    if not isinstance(ts, int) or not isinstance(te, int):
        return False
    return ts >= 0 and te >= ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", default="checkpoint_best_multitask.pt")
    parser.add_argument("--output", default=None, help="défaut = écrase le dataset source")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--boost", type=float, default=2.5,
                        help="Multiplicateur de poids pour candidats mal prédits")
    parser.add_argument("--decay", type=float, default=0.85,
                        help="Facteur de décroissance vers 1.0 pour candidats bien prédits")
    parser.add_argument("--max-weight", type=float, default=6.0,
                        help="Poids maximum autorisé")
    parser.add_argument("--min-weight", type=float, default=0.3,
                        help="Poids minimum autorisé")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.device is None:
        import torch
        args.device = (
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    mine(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        output_path=args.output or args.dataset,
        model_name=args.model_name,
        device=args.device,
        boost=args.boost,
        decay=args.decay,
        max_weight=args.max_weight,
        min_weight=args.min_weight,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

