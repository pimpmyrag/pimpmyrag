#!/usr/bin/env python3
"""
Fusionne abstract_sentences.jsonl + abstract_sentences_extra.jsonl dans train.jsonl
et affiche les stats finales par label sur tout le dataset.
"""
import json
from collections import Counter
from pathlib import Path

base = Path(__file__).parent

# 1. Fusionner les nouveaux fichiers dans train.jsonl
new_files = ['abstract_sentences.jsonl', 'abstract_sentences_extra.jsonl']
added = 0
with open(base / 'train.jsonl', 'a', encoding='utf-8') as out:
    for fname in new_files:
        p = base / fname
        if not p.exists():
            print(f"  ⚠️  {fname} not found")
            continue
        with open(p, encoding='utf-8') as f:
            for line in f:
                out.write(line)
                added += 1
        print(f"  ✅ {fname}: merged")

print(f"\n  Total lignes ajoutées au train: {added}")

# 2. Stats globales
for split in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
    label_counts = Counter()
    n_rows = 0
    with open(base / split, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            n_rows += 1
            for sp in row.get('spans', []):
                label_counts[sp['label']] += 1

    print(f"\n{'='*60}")
    print(f"  {split}: {n_rows} rows, {sum(label_counts.values())} spans")
    print(f"{'='*60}")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"   {label:30s} {count:5d}")

