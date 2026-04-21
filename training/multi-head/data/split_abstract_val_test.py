#!/usr/bin/env python3
"""
Split une partie des données ABSTRACT du train vers val et test
pour permettre l'évaluation des nouveaux labels.
Prend les dernières lignes abstract_* du train, en répartit vers val et test.
"""
import json, random
from pathlib import Path

base = Path('/Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head/data')
random.seed(42)

# Lire le train et séparer les lignes abstract des autres
train_rows = []
abstract_rows = []
with open(base / 'train.jsonl', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        if row['id'].startswith('abstract_'):
            abstract_rows.append(row)
        else:
            train_rows.append(row)

print(f"Train original (non-abstract): {len(train_rows)}")
print(f"Abstract total: {len(abstract_rows)}")

# Shuffle et split: 80% train, 10% val, 10% test
random.shuffle(abstract_rows)
n = len(abstract_rows)
n_val = max(1, n // 10)
n_test = max(1, n // 10)
n_train = n - n_val - n_test

abs_train = abstract_rows[:n_train]
abs_val = abstract_rows[n_train:n_train + n_val]
abs_test = abstract_rows[n_train + n_val:]

print(f"\nSplit abstract: train={len(abs_train)}, val={len(abs_val)}, test={len(abs_test)}")

# Écrire
train_rows.extend(abs_train)
random.shuffle(train_rows)
with open(base / 'train.jsonl', 'w', encoding='utf-8') as f:
    for row in train_rows:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

# Append to val/test
for fname, rows in [('val.jsonl', abs_val), ('test.jsonl', abs_test)]:
    with open(base / fname, 'a', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

print(f"\n✅ train.jsonl: {len(train_rows)} rows")
print(f"✅ val.jsonl: +{len(abs_val)} abstract rows")
print(f"✅ test.jsonl: +{len(abs_test)} abstract rows")

# Stats des nouveaux labels dans chaque split
from collections import Counter
abstract_labels = {'hint_law', 'hint_work_of_art', 'hint_concept', 'hint_disease', 'hint_language'}
for fname in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
    lc = Counter()
    with open(base / fname, encoding='utf-8') as f:
        for line in f:
            for sp in json.loads(line).get('spans', []):
                if sp['label'] in abstract_labels:
                    lc[sp['label']] += 1
    print(f"\n  {fname} ABSTRACT labels:")
    for lab, cnt in sorted(lc.items(), key=lambda x: -x[1]):
        print(f"    {lab:25s} {cnt:4d}")

