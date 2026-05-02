"""Distribution des labels hint_ sur train/val/test v6."""
import json, collections

FILES = {
    'train': 'data/train_v6.jsonl',
    'val':   'data/val_v6.jsonl',
    'test':  'data/test_v6.jsonl',
}

counts = {}
for split, path in FILES.items():
    c = collections.Counter()
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            for sp in ex.get('spans', []):
                lbl = sp['label']
                if lbl.startswith('hint_'):
                    c[lbl] += 1
    counts[split] = c

all_labels = sorted(set(k for c in counts.values() for k in c.keys()))
train_total = sum(counts['train'].values())
val_total   = sum(counts['val'].values())
test_total  = sum(counts['test'].values())

print()
print(f"{'Label':<28}  {'TRAIN':>6}  {'%':>5}  {'VAL':>5}  {'%':>5}  {'TEST':>5}  {'%':>5}  {'TOTAL':>6}")
print("-" * 84)

rows = []
for lbl in all_labels:
    tr = counts['train'].get(lbl, 0)
    vl = counts['val'].get(lbl, 0)
    te = counts['test'].get(lbl, 0)
    rows.append((lbl, tr, vl, te, tr + vl + te))

for lbl, tr, vl, te, total in sorted(rows, key=lambda x: x[1]):
    tr_pct = 100 * tr / train_total if train_total else 0
    vl_pct = 100 * vl / val_total   if val_total   else 0
    te_pct = 100 * te / test_total  if test_total  else 0
    flag = "  <<< FAIBLE" if tr < 200 else ("  < 500" if tr < 500 else "")
    print(f"{lbl:<28}  {tr:>6}  {tr_pct:>4.1f}%  {vl:>5}  {vl_pct:>4.1f}%  {te:>5}  {te_pct:>4.1f}%  {total:>6}{flag}")

print("-" * 84)
print(f"{'TOTAL HINT_*':<28}  {train_total:>6}         {val_total:>5}         {test_total:>5}")
print(f"\n  Labels < 200 spans train = candidats prioritaires pour boost dans v7")

