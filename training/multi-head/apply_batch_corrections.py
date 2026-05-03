"""
Applique les corrections du batch-review Mistral au dataset v6.1 → v6.2.
Seuls les vrais changements de label (suggested != current) sont appliqués.
"""
import json
from pathlib import Path
from collections import defaultdict, Counter

RESULTS_FILE = Path('data/mistral_batch_review.jsonl')
SPLITS = ['train', 'val', 'test']

# Charge les corrections : (current_label, span_text) -> new_label
corrections = {}
with open(RESULTS_FILE, encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r['verdict'] != 'SUSPECT':
            continue
        sugg = r.get('label_suggested')
        if not sugg or sugg == r['label'] or sugg in ('non-NER', 'hint_norp', 'hint_other'):
            continue
        corrections[(r['label'], r['span'])] = sugg

print(f'{len(corrections)} corrections chargées\n')

# Résumé par paire
by_pair = Counter()
for (src, _), dst in corrections.items():
    by_pair[(src, dst)] += 1
for (src, dst), n in by_pair.most_common():
    print(f'  {src} → {dst}: {n}')

print()

# Applique au dataset
for split in SPLITS:
    in_path  = Path(f'data/{split}_v6.1.jsonl')
    out_path = Path(f'data/{split}_v6.2.jsonl')
    if not in_path.exists():
        print(f'SKIP {in_path}')
        continue

    changed_spans = changed_sents = total = 0
    stats = Counter()

    with open(in_path, encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            d = json.loads(line)
            total += 1
            modified = False
            for span in d.get('spans', []):
                key = (span['label'], span.get('text', ''))
                if key in corrections:
                    stats[(span['label'], corrections[key])] += 1
                    span['label'] = corrections[key]
                    changed_spans += 1
                    modified = True
            if modified:
                changed_sents += 1
            fout.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(f'[{split}] {total} phrases | {changed_sents} modifiées | {changed_spans} spans → {out_path}')

# Distribution finale
print('\n=== Distribution labels ORG (train) ===')
TARGET = {'hint_group_role','hint_inst_name','hint_inst_role','hint_org_name'}
for version, fname in [('v6.1', 'train_v6.1.jsonl'), ('v6.2', 'train_v6.2.jsonl')]:
    p = Path(f'data/{fname}')
    if not p.exists():
        continue
    c = Counter()
    with open(p, encoding='utf-8') as f:
        for line in f:
            for span in json.loads(line).get('spans', []):
                if span['label'] in TARGET:
                    c[span['label']] += 1
    print(f'\n{version}:')
    for lbl, n in sorted(c.items()):
        print(f'  {lbl}: {n}')

