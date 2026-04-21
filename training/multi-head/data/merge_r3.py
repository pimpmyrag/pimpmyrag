#!/usr/bin/env python3
import json, random
from pathlib import Path
from collections import Counter

base = Path('/Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head/data')
random.seed(456)

rows = []
with open(base / 'boost_weak_r3.jsonl', encoding='utf-8') as f:
    for line in f:
        rows.append(json.loads(line))

random.shuffle(rows)
n = len(rows)
n_val = max(1, n // 20)
n_test = max(1, n // 20)

for fname, new_rows in [('train.jsonl', rows[n_val+n_test:]), ('val.jsonl', rows[:n_val]), ('test.jsonl', rows[n_val:n_val+n_test])]:
    with open(base / fname, 'a', encoding='utf-8') as f:
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

print(f'Merged: train +{n-n_val-n_test}, val +{n_val}, test +{n_test}')

for fname in ['train.jsonl']:
    lc = Counter()
    n_rows = 0
    with open(base / fname, encoding='utf-8') as f:
        for line in f:
            n_rows += 1
            for sp in json.loads(line).get('spans', []):
                lc[sp['label']] += 1
    print(f'\n{fname}: {n_rows} rows, {sum(lc.values())} spans')
    for lab, cnt in sorted(lc.items(), key=lambda x: x[1]):
        print(f'  {lab:30s} {cnt:5d}')

