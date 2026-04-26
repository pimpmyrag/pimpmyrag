import json
from collections import Counter

splits = ['train.jsonl', 'val.jsonl', 'test.jsonl']
label_counts = {split: Counter() for split in splits}
sentence_counts = {split: 0 for split in splits}
total_spans = {split: 0 for split in splits}

for split in splits:
    try:
        with open(split, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sentence_counts[split] += 1
                for sp in obj.get('spans', []):
                    label_counts[split][sp.get('label', '?')] += 1
                    total_spans[split] += 1
    except FileNotFoundError:
        print(split + ': non trouvé')

print('Nombre de phrases par split:')
for split in splits:
    print(f'  {split}: {sentence_counts[split]} phrases')
print('\nNombre de spans par split:')
for split in splits:
    print(f'  {split}: {total_spans[split]} spans')
print('\nDistribution des labels par split:')
all_labels = set()
for split in splits:
    all_labels.update(label_counts[split].keys())
all_labels = sorted(all_labels)

for split in splits:
    print(f'\n--- {split} ---')
    total = total_spans[split]
    for label in all_labels:
        count = label_counts[split][label]
        pct = (count * 100.0 / total) if total else 0
        bar = chr(9608) * int(pct / 1.5)
        print(f'  {label:25s} {count:5d}  ({pct:5.1f}%)  {bar}')
