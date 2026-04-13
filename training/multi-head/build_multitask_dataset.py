# build_multitask_dataset.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set

from transformers import AutoTokenizer

from labels import (
    FINE2ID, FINE_NONE_ID,
    fine_label_to_coarse_id,
    COARSE_NONE_ID,
)

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def tokenize_with_offsets(tokenizer, text: str):
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    return enc["input_ids"], enc["offset_mapping"]

def char_span_to_token_span(offsets: List[Tuple[int, int]], start: int, end: int):
    """
    Convertit un span char -> span token couvrant la zone [start, end)
    même si les frontières char ne tombent pas exactement sur des offsets token.
    """
    tok_start = None
    tok_end = None

    for i, (s, e) in enumerate(offsets):
        if e <= start:
            continue
        if s >= end:
            break
        if tok_start is None:
            tok_start = i
        tok_end = i

    if tok_start is None or tok_end is None:
        return None
    return tok_start, tok_end

def token_span_to_char_span(offsets: List[Tuple[int, int]], tok_start: int, tok_end: int):
    return offsets[tok_start][0], offsets[tok_end][1]

def spans_overlap(a_start, a_end, b_start, b_end):
    return not (a_end <= b_start or b_end <= a_start)

def token_span_iou(a, b):
    a0, a1 = a
    b0, b1 = b
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return inter / union if union > 0 else 0.0

def build_gold_candidates(row, tokenizer):
    text = row["text"]
    input_ids, offsets = tokenize_with_offsets(tokenizer, text)

    gold_candidates = []
    gold_token_spans = []
    gold_char_spans = set()

    for sp in row.get("spans", []):
        label = sp["label"]
        if label not in FINE2ID:
            raise ValueError(f"Label inconnu: {label}")

        start = sp["start"]
        end = sp["end"]

        tok_span = char_span_to_token_span(offsets, start, end)
        if tok_span is None:
            continue

        tok_start, tok_end = tok_span
        fine_id = FINE2ID[label]
        coarse_id = fine_label_to_coarse_id(label)

        cand = {
            "char_start": start,
            "char_end": end,
            "tok_start": tok_start,
            "tok_end": tok_end,
            "boundary_label": 1,
            "coarse_label_id": coarse_id,
            "fine_label_id": fine_id,
            "neg_type": "gold",
            "sample_weight": 1.0,
            "text": sp.get("text", text[start:end]),
        }
        gold_candidates.append(cand)
        gold_token_spans.append((tok_start, tok_end))
        gold_char_spans.add((start, end))

    return text, input_ids, offsets, gold_candidates, gold_token_spans, gold_char_spans

def generate_hard_negatives(offsets, gold_candidates, gold_char_spans, max_per_gold=6):
    """
    Hard negatives = variantes de frontières autour des spans golds.
    """
    n_tokens = len(offsets)
    out = []
    seen = set()

    for gc in gold_candidates:
        l = gc["tok_start"]
        r = gc["tok_end"]

        proposals = []
        for dl in [-2, -1, 0, 1, 2]:
            for dr in [-2, -1, 0, 1, 2]:
                nl = l + dl
                nr = r + dr
                if nl < 0 or nr < 0 or nl >= n_tokens or nr >= n_tokens or nl > nr:
                    continue
                if nl == l and nr == r:
                    continue

                cstart, cend = token_span_to_char_span(offsets, nl, nr)
                if (cstart, cend) in gold_char_spans:
                    continue

                # On veut plutôt des spans "proches" d'un gold
                iou = token_span_iou((l, r), (nl, nr))
                if iou <= 0.0:
                    continue

                proposals.append((nl, nr, cstart, cend, iou))

        # garder les plus proches
        proposals.sort(key=lambda x: x[-1], reverse=True)
        kept = 0
        for nl, nr, cstart, cend, iou in proposals:
            key = (cstart, cend)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "char_start": cstart,
                "char_end": cend,
                "tok_start": nl,
                "tok_end": nr,
                "boundary_label": 0,
                "coarse_label_id": COARSE_NONE_ID,
                "fine_label_id": FINE_NONE_ID,
                "neg_type": "hard_neg",
                "sample_weight": 1.0,
                "text": None,
            })
            kept += 1
            if kept >= max_per_gold:
                break

    return out

def generate_soft_negatives(offsets, gold_token_spans, gold_char_spans, num_soft=20, max_span_len=8, seed=13):
    """
    Soft negatives = spans aléatoires non-overlap.
    """
    rnd = random.Random(seed)
    n_tokens = len(offsets)
    out = []
    seen = set()

    attempts = 0
    max_attempts = num_soft * 50

    while len(out) < num_soft and attempts < max_attempts:
        attempts += 1
        if n_tokens == 0:
            break

        l = rnd.randint(0, n_tokens - 1)
        span_len = rnd.randint(1, max_span_len)
        r = min(n_tokens - 1, l + span_len - 1)

        cstart, cend = token_span_to_char_span(offsets, l, r)
        key = (cstart, cend)
        if key in seen or key in gold_char_spans:
            continue

        # pas de recouvrement avec les golds
        overlap = False
        for gl, gr in gold_token_spans:
            if token_span_iou((l, r), (gl, gr)) > 0.0:
                overlap = True
                break
        if overlap:
            continue

        seen.add(key)
        out.append({
            "char_start": cstart,
            "char_end": cend,
            "tok_start": l,
            "tok_end": r,
            "boundary_label": 0,
            "coarse_label_id": COARSE_NONE_ID,
            "fine_label_id": FINE_NONE_ID,
            "neg_type": "soft_neg",
            "sample_weight": 0.35,   # moins fort car plus bruité
            "text": None,
        })

    return out

def make_multitask_row(row, tokenizer, hard_per_gold=6, soft_factor=2.0, max_span_len=8, seed=13):
    text, input_ids, offsets, gold_candidates, gold_token_spans, gold_char_spans = build_gold_candidates(row, tokenizer)

    num_soft = max(1, int(len(gold_candidates) * soft_factor))
    hard_negs = generate_hard_negatives(
        offsets,
        gold_candidates,
        gold_char_spans,
        max_per_gold=hard_per_gold
    )
    soft_negs = generate_soft_negatives(
        offsets,
        gold_token_spans,
        gold_char_spans,
        num_soft=num_soft,
        max_span_len=max_span_len,
        seed=seed
    )

    candidates = gold_candidates + hard_negs + soft_negs

    # renseigner le texte du span si absent
    for c in candidates:
        if c["text"] is None:
            c["text"] = text[c["char_start"]:c["char_end"]]

    return {
        "id": row["id"],
        "text": text,
        "candidates": candidates,
        "meta": {
            "num_gold": len(gold_candidates),
            "num_hard_neg": len(hard_negs),
            "num_soft_neg": len(soft_negs),
            "num_tokens": len(input_ids),
        }
    }

def export_head_views(rows, out_prefix: str):
    """
    Exporte des vues séparées si tu veux inspecter / debugger chaque tête.
    """
    boundary_rows = []
    coarse_rows = []
    fine_rows = []

    for row in rows:
        boundary_rows.append({
            "id": row["id"],
            "text": row["text"],
            "spans": [
                {
                    "start": c["char_start"],
                    "end": c["char_end"],
                    "text": c["text"],
                    "label": c["boundary_label"],
                    "neg_type": c["neg_type"],
                    "sample_weight": c["sample_weight"],
                }
                for c in row["candidates"]
            ]
        })
        coarse_rows.append({
            "id": row["id"],
            "text": row["text"],
            "spans": [
                {
                    "start": c["char_start"],
                    "end": c["char_end"],
                    "text": c["text"],
                    "label_id": c["coarse_label_id"],
                    "neg_type": c["neg_type"],
                    "sample_weight": c["sample_weight"],
                }
                for c in row["candidates"]
            ]
        })
        fine_rows.append({
            "id": row["id"],
            "text": row["text"],
            "spans": [
                {
                    "start": c["char_start"],
                    "end": c["char_end"],
                    "text": c["text"],
                    "label_id": c["fine_label_id"],
                    "neg_type": c["neg_type"],
                    "sample_weight": c["sample_weight"],
                }
                for c in row["candidates"]
            ]
        })

    write_jsonl(out_prefix + ".boundary.jsonl", boundary_rows)
    write_jsonl(out_prefix + ".coarse.jsonl", coarse_rows)
    write_jsonl(out_prefix + ".fine.jsonl", fine_rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="dataset JSONL source (positifs uniquement)")
    parser.add_argument("--output", required=True, help="dataset JSONL multitask de sortie")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--tokenizer-path", default=None, help="si tu veux utiliser ton tokenizer local fast")
    parser.add_argument("--hard-per-gold", type=int, default=6)
    parser.add_argument("--soft-factor", type=float, default=2.0, help="nb soft neg = soft_factor * nb gold")
    parser.add_argument("--max-span-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--export-head-views-prefix", default=None, help="si défini, exporte aussi boundary/coarse/fine séparés")
    args = parser.parse_args()

    random.seed(args.seed)

    tokenizer_source = args.tokenizer_path or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    rows = []
    for row in load_jsonl(args.input):
        mt_row = make_multitask_row(
            row,
            tokenizer=tokenizer,
            hard_per_gold=args.hard_per_gold,
            soft_factor=args.soft_factor,
            max_span_len=args.max_span_len,
            seed=args.seed
        )
        rows.append(mt_row)

    write_jsonl(args.output, rows)
    print(f"✅ dataset multitask écrit dans {args.output} ({len(rows)} lignes)")

    if args.export_head_views_prefix:
        export_head_views(rows, args.export_head_views_prefix)
        print(f"✅ vues séparées écrites avec préfixe {args.export_head_views_prefix}")

if __name__ == "__main__":
    main()
