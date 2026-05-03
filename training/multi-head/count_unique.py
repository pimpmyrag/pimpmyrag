import json
from pathlib import Path
from collections import defaultdict, Counter

TARGET = {'hint_group_role','hint_inst_name','hint_inst_role','hint_org_name'}
uniq = defaultdict(list)

for split in ['train','val','test']:
    p = Path(f'data/{split}_v6.1.jsonl')
    if not p.exists():
        continue
    with open(p, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            text = d['text']
            for span in d.get('spans', []):
                if span['label'] not in TARGET:
                    continue
                key = (span['label'], span.get('text', text[span['start']:span['end']]))
                if len(uniq[key]) < 2:
                    start, end = span['start'], span['end']
                    ctx = text[max(0,start-60):start] + '[[' + text[start:end] + ']]' + text[end:min(len(text),end+60)]
                    uniq[key].append(ctx)

print(f'Paires uniques (label, span_text): {len(uniq)}')
c = Counter(k[0] for k in uniq)
for lbl, n in sorted(c.items()):
    print(f'  {lbl}: {n}')
print(f'Batches de 30: {(len(uniq)+29)//30}')

