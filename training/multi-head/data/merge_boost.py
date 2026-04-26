#!/usr/bin/env python3
"""Merge boost_weak_classes.jsonl into train/val/test (90/5/5 split)."""
import json, random
from pathlib import Path
from collections import Counter

base = Path(__file__).parent
random.seed(77)

rows = []
with open(base / 'boost_weak_classes.jsonl', encoding='utf-8') as f:
    for line in f:
        rows.append(json.loads(line))

random.shuffle(rows)
n = len(rows)
n_val = max(1, n // 20)   # 5%
n_test = max(1, n // 20)  # 5%

val_rows = rows[:n_val]
test_rows = rows[n_val:n_val+n_test]
train_rows = rows[n_val+n_test:]

for fname, new_rows in [('train.jsonl', train_rows), ('val.jsonl', val_rows), ('test.jsonl', test_rows)]:
    with open(base / fname, 'a', encoding='utf-8') as f:
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

print(f"Merged: train +{len(train_rows)}, val +{len(val_rows)}, test +{len(test_rows)}")

# Final stats
for fname in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
    lc = Counter()
    n_rows = 0
    with open(base / fname, encoding='utf-8') as f:
        for line in f:
            n_rows += 1
            for sp in json.loads(line).get('spans', []):
                lc[sp['label']] += 1
    print(f"\n{'='*60}")
    print(f"  {fname}: {n_rows} rows, {sum(lc.values())} spans")
    print(f"{'='*60}")
    for lab, cnt in sorted(lc.items(), key=lambda x: x[1]):
        print(f"   {lab:30s} {cnt:5d}")

