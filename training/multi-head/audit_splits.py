#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data")

def load_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

splits = {name: load_jsonl(DATA_DIR / f"{name}.jsonl") for name in ["train", "val", "test"]}

counts = {name: Counter() for name in splits}
for name, rows in splits.items():
    for row in rows:
        for sp in row.get("spans", []):
            lbl = sp.get("label", "")
            if lbl:
                counts[name][lbl] += 1

all_labels = sorted(set(lbl for c in counts.values() for lbl in c))
grand_total = {lbl: sum(counts[n][lbl] for n in splits) for lbl in all_labels}

print(f"{'LABEL':<30} {'TOTAL':>7}  {'TRAIN%':>7} {'VAL%':>7} {'TEST%':>7}  {'TRAIN':>7} {'VAL':>6} {'TEST':>6}  FLAG")
print("-" * 105)
flagged = []
for lbl in sorted(all_labels, key=lambda l: -grand_total[l]):
    tot = grand_total[lbl]
    if tot == 0:
        continue
    tr = counts["train"][lbl]
    va = counts["val"][lbl]
    te = counts["test"][lbl]
    tr_pct = tr / tot * 100
    va_pct = va / tot * 100
    te_pct = te / tot * 100
    flag = ""
    if va_pct < 7 or va_pct > 13 or te_pct < 7 or te_pct > 13:
        flag = "⚠️"
        flagged.append((lbl, tot, tr_pct, va_pct, te_pct, tr, va, te))
    print(f"  {lbl:<28} {tot:>7}  {tr_pct:>6.1f}% {va_pct:>6.1f}% {te_pct:>6.1f}%  {tr:>7} {va:>6} {te:>6}  {flag}")

print()
print(f"{'='*50}")
print(f"LABELS DÉSÉQUILIBRÉS ({len(flagged)}) :")
print(f"{'='*50}")
for lbl, tot, tr_pct, va_pct, te_pct, tr, va, te in flagged:
    print(f"  {lbl:<28}  val={va_pct:.1f}%({va})  test={te_pct:.1f}%({te})  total={tot}")

