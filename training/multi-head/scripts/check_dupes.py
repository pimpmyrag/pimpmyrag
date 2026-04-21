#!/usr/bin/env python3
import json
from collections import Counter

existing_texts = set()
for fname in ['data/train.jsonl', 'data/val.jsonl', 'data/test.jsonl']:
    try:
        for line in open(fname):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get('text', '').strip()
            if t:
                existing_texts.add(t)
    except FileNotFoundError:
        print(f'  ⚠️  {fname} introuvable')

print(f'Textes existants (train+val+test): {len(existing_texts)}')

candidates = [json.loads(l) for l in open('data/rare_candidates_clean.jsonl') if l.strip()]
print(f'rare_candidates_clean: {len(candidates)} phrases')

dupes = [c for c in candidates if c['text'].strip() in existing_texts]
print(f'Doublons détectés: {len(dupes)}')

clean = [c for c in candidates if c['text'].strip() not in existing_texts]
print(f'Après déduplication: {len(clean)} phrases')

final_counts = Counter()
for item in clean:
    for lbl in item.get('rare_labels', []):
        final_counts[lbl] += 1

RARE_LABELS_ORDER = [
    'hint_concept', 'hint_count', 'hint_disease', 'hint_fac_name',
    'hint_food', 'hint_infra', 'hint_language', 'hint_law',
    'hint_measure', 'hint_money', 'hint_object_name', 'hint_percentage',
    'hint_rate', 'hint_substance', 'hint_time_clock', 'hint_tool',
    'hint_weapon', 'hint_work_of_art'
]
print()
print('Distribution après déduplication:')
for lbl in RARE_LABELS_ORDER:
    cnt = final_counts.get(lbl, 0)
    bar = '█' * min(cnt // 20, 50)
    status = '✅' if cnt >= 1000 else '⚠️ '
    print(f'  {status} {lbl:<25} {cnt:>5}  {bar}')

with open('data/rare_candidates_deduped.jsonl', 'w') as f:
    for item in clean:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f'\n✅ Sauvegardé: data/rare_candidates_deduped.jsonl ({len(clean)} phrases)')

# Estimation coût Claude batch (5 phrases/requête, ~600 tokens input/req, ~300 output)
n_req = len(clean) // 5 + 1
tokens_in = n_req * 600
tokens_out = n_req * 300
cost = (tokens_in / 1e6 * 3 + tokens_out / 1e6 * 15) * 0.5
print(f'💰 Estimation coût batch Claude-3.5-sonnet: ~${cost:.2f} USD')

