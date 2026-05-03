"""
Applique les corrections du batch review (CHANGE uniquement, pas REMOVE)
pour construire le dataset v6.4 à partir de v6.3.

- CHANGE où new_label != old_label : change le label du span
- CHANGE où new_label == old_label : no-op (Haiku confirme en signalant "CHANGE" vers même label)
- REMOVE : ignoré (on garde les spans tels quels)
- hint_concept_named est ajouté comme nouveau label

Sortie : data/{split}_v6.4.jsonl
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

REVIEW_FILE = Path("data/work_labels_mistral_review.jsonl")
IN_VERSION  = "v6.3"
OUT_VERSION = "v6.4"

# ── Charger le mapping (label_orig, span_lower) -> new_label ──────────────────
changes = {}   # (label_orig, span_lower) -> new_label
stats = Counter()

for line in REVIEW_FILE.read_text(encoding='utf-8').splitlines():
    r = json.loads(line)
    old_label = r['label']
    key = (old_label, r['span'].lower().strip())

    if r['verdict'] == 'REMOVE' and old_label == 'hint_work_of_art':
        # Catégorie générique sans titre → hint_concept (pas de suppression)
        changes[key] = 'hint_concept'
        continue

    if r['verdict'] != 'CHANGE':
        continue
    new_label = r.get('label_suggested')
    if not new_label or new_label == 'REMOVE':
        continue
    if new_label != old_label:
        changes[key] = new_label

print(f"{len(changes)} corrections effectives à appliquer ({sum(1 for l in REVIEW_FILE.read_text().splitlines() if json.loads(l)['verdict']=='CHANGE')} CHANGE total, self-changes ignorés)")

# Par label source
change_from = defaultdict(Counter)
for (src, _), tgt in changes.items():
    change_from[src][tgt] += 1
for src, targets in sorted(change_from.items()):
    print(f"  {src}:")
    for tgt, n in sorted(targets.items(), key=lambda x: -x[1]):
        print(f"    -> {tgt} : {n}")

# ── Appliquer sur chaque split ────────────────────────────────────────────────
SPLITS = ['train', 'val', 'test']

total_spans = total_changed = 0

for split in SPLITS:
    in_path  = Path(f"data/{split}_{IN_VERSION}.jsonl")
    out_path = Path(f"data/{split}_{OUT_VERSION}.jsonl")
    if not in_path.exists():
        print(f"⚠ Absent : {in_path}")
        continue

    changed_in_split = 0
    out_lines = []

    for line in in_path.read_text(encoding='utf-8').splitlines():
        d = json.loads(line)
        text = d['text']
        new_spans = []
        for span in d.get('spans', []):
            total_spans += 1
            s, e = span['start'], span['end']
            span_text = text[s:e]
            key = (span['label'], span_text.lower().strip())
            if key in changes:
                new_label = changes[key]
                span = dict(span)
                span['label'] = new_label
                changed_in_split += 1
                total_changed += 1
            new_spans.append(span)
        d['spans'] = new_spans
        out_lines.append(json.dumps(d, ensure_ascii=False))

    out_path.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
    print(f"✅ {split}: {changed_in_split} spans modifiés → {out_path}")

print(f"\nTotal : {total_changed}/{total_spans} spans modifiés ({total_changed/total_spans*100:.1f}%)")

# ── Vérification : distribution des labels WORK/ABSTRACT ─────────────────────
print("\n=== Distribution labels WORK/ABSTRACT dans v6.4 (train) ===")
label_count = Counter()
target = {'hint_law', 'hint_document', 'hint_concept', 'hint_work_of_art', 'hint_concept_named'}
train64 = Path(f"data/train_{OUT_VERSION}.jsonl")
if train64.exists():
    for line in train64.read_text(encoding='utf-8').splitlines():
        d = json.loads(line)
        text = d['text']
        for span in d.get('spans', []):
            if span['label'] in target:
                label_count[span['label']] += 1
for lbl, n in sorted(label_count.items(), key=lambda x: -x[1]):
    print(f"  {lbl:25s} : {n}")

