#!/usr/bin/env python3
"""Audit élargi pour hint_concept dans le dataset existant."""
import json
from collections import Counter

concept_kw = [
    'isme', 'iste',
    'doctrine', 'philosophie', 'mouvement', 'courant',
    'monarchie', 'dictature',
    'islam', 'christianisme', 'bouddhisme', 'hindouisme',
    'protestantisme', 'catholicisme',
    'inflation', 'mondialisation', 'globalisation',
    'croissance',
]

results = []
for split in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
    path = f'/Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head/data/{split}'
    with open(path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            for sp in row.get('spans', []):
                txt = sp['text'].lower()
                lab = sp['label']
                if lab.startswith('hint_concept'):
                    continue
                for kw in concept_kw:
                    if kw in txt:
                        results.append({
                            'split': split,
                            'label': lab,
                            'text': sp['text'][:80],
                            'kw': kw,
                            'sentence': row['text'][:140],
                        })
                        break

print(f"TOTAL: {len(results)} suspects\n")

label_counts = Counter(r['label'] for r in results)
for lab, cnt in label_counts.most_common():
    print(f"  {lab:30s} {cnt:4d}")

kw_counts = Counter(r['kw'] for r in results)
print(f"\nPar mot-cle:")
for kw, cnt in kw_counts.most_common(30):
    print(f"  {kw:30s} {cnt:4d}")

print(f"\nExemples (tous):")
for r in results[:60]:
    print(f"  [{r['kw']:15s}] {r['label']:25s} -> \"{r['text']}\"")

