import json
from collections import Counter

results = []
with open('data/mistral_batch_review.jsonl', encoding='utf-8') as f:
    for line in f:
        results.append(json.loads(line))

# Vrais changements de label (label suggéré différent du label actuel)
real_changes = [
    (r['label'], r['label_suggested'], r['span'], r.get('raison','')[:80])
    for r in results
    if r['verdict'] == 'SUSPECT'
    and r.get('label_suggested')
    and r['label_suggested'] != r['label']
    and r['label_suggested'] not in ('non-NER', 'hint_norp', 'hint_other')
]

real_changes.sort()
by_pair = Counter((src, dst) for src, dst, _, _ in real_changes)

print('=== VRAIS CHANGEMENTS SUGGERES ===')
print(f'Total: {len(real_changes)}')
print()
for (src, dst), n in by_pair.most_common():
    print(f'  {src} -> {dst}: {n} spans')

print()
print('=== SUSPECTS MEME LABEL (qualite douteuse) ===')
same_label = [(r['label'], r['span']) for r in results
              if r['verdict'] == 'SUSPECT' and r.get('label_suggested') == r['label']]
by_lbl = Counter(lbl for lbl, _ in same_label)
for lbl, n in by_lbl.most_common():
    print(f'  {lbl}: {n}')

print()
print('=== EXEMPLES org_name -> inst_name (top 20) ===')
for src, dst, span, raison in real_changes:
    if src == 'hint_org_name' and dst == 'hint_inst_name':
        print(f'  "{span}" | {raison}')
        n = sum(1 for s,d,_,_ in real_changes if s==src and d==dst and _ == span)

