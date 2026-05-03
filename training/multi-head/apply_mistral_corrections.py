"""
Applique les corrections Mistral au dataset v6 → v6.1.
Pour chaque span dont le texte correspond à un span_key corrigé,
on remplace l'ancien label par le nouveau.

Format span_key : "hint_xxx::texte_du_span"
"""
import json
from pathlib import Path
from collections import defaultdict

RESULTS = Path('data/mistral_review_results.jsonl')
SPLITS = ['train', 'val', 'test']

# Charge les corrections (seulement les vrais changements)
corrections = {}  # (current_label, span_text) -> new_label
skipped = []

with open(RESULTS, encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if 'error' in r:
            continue
        current = r['current_label']
        new = r['mistral_label']
        if current == new or new in ('PARSE_ERROR', None):
            continue
        # span_key = "hint_xxx::texte"
        span_text = r['span_key'].split('::', 1)[1]
        corrections[(current, span_text)] = new
        print(f"  {current} → {new}  [{span_text}]  (×{r['count']})")

print(f"\n{len(corrections)} corrections chargées\n")

# Appliquer au dataset
stats = defaultdict(lambda: defaultdict(int))

for split in SPLITS:
    in_path = Path(f'data/{split}_v6.jsonl')
    out_path = Path(f'data/{split}_v6.1.jsonl')
    if not in_path.exists():
        print(f"SKIP {in_path} (absent)")
        continue

    changed_sentences = 0
    changed_spans = 0
    total = 0

    with open(in_path, encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            d = json.loads(line)
            total += 1
            modified = False
            for span in d.get('spans', []):
                key = (span['label'], span['text'])
                if key in corrections:
                    old_label = span['label']
                    new_label = corrections[key]
                    span['label'] = new_label
                    stats[split][(old_label, new_label)] += 1
                    changed_spans += 1
                    modified = True
            if modified:
                changed_sentences += 1
            fout.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(f"[{split}] {total} phrases, {changed_sentences} modifiées, {changed_spans} spans corrigés → {out_path}")

print("\n" + "="*60)
print("DÉTAIL DES CORRECTIONS PAR SPLIT")
print("="*60)
for split in SPLITS:
    if not stats[split]:
        continue
    print(f"\n[{split}]")
    for (old, new), count in sorted(stats[split].items()):
        print(f"  {old} → {new} : {count} occurrences")

# Vérification rapide : comptage labels avant/après
print("\n" + "="*60)
print("VÉRIFICATION DISTRIBUTION LABELS (train)")
print("="*60)
from collections import Counter

def count_labels(path):
    c = Counter()
    with open(path, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            for span in d.get('spans', []):
                lbl = span['label']
                if 'inst' in lbl or 'group' in lbl:
                    c[lbl] += 1
    return c

for version, fname in [('v6', 'train_v6.jsonl'), ('v6.1', 'train_v6.1.jsonl')]:
    p = Path(f'data/{fname}')
    if p.exists():
        c = count_labels(p)
        print(f"\n{version}:")
        for lbl, cnt in sorted(c.items()):
            print(f"  {lbl}: {cnt}")

