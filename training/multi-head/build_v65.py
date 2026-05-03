"""
Construit le dataset v6.5 :

  train_v6.5 = train_v6.4
               + corrections inst_name/inst_role/org_name (inst_labels_review.jsonl)
               + 4377 phrases wikinews re-annotées (wikinews_new_annotated_v64.jsonl)

  val_v6.5   = val_v6.4   (inchangé)
  test_v6.5  = test_v6.4  (inchangé)
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

DATA = Path("data")

# ── 1. Charger les corrections inst depuis le batch review ───────────────────
INST_REVIEW = DATA / "inst_labels_review.jsonl"
INST_LABELS = {'hint_inst_name', 'hint_inst_role', 'hint_org_name'}

inst_changes = {}   # (old_label, span_lower) -> new_label
inst_stats   = Counter()

if not INST_REVIEW.exists():
    print(f"⚠  {INST_REVIEW} absent — skip corrections inst")
else:
    for line in INST_REVIEW.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r['verdict'] != 'CHANGE':
            continue
        new_label = r.get('label_suggested')
        if not new_label or new_label == 'REMOVE' or new_label not in INST_LABELS:
            continue
        old_label = r['label']
        if new_label == old_label:
            continue
        key = (old_label, r['span'].lower().strip())
        inst_changes[key] = new_label

    print(f"Corrections inst chargées : {len(inst_changes)}")
    change_from = defaultdict(Counter)
    for (src, _), tgt in inst_changes.items():
        change_from[src][tgt] += 1
    for src, targets in sorted(change_from.items()):
        for tgt, n in sorted(targets.items(), key=lambda x: -x[1]):
            print(f"  {src} → {tgt} : {n}")

# ── 2. Appliquer corrections inst sur v6.4 train ────────────────────────────
print("\nApplication des corrections inst sur train_v6.4...")
base_train = []
n_inst_changed = 0

for line in (DATA / "train_v6.4.jsonl").read_text(encoding='utf-8').splitlines():
    if not line.strip():
        continue
    d = json.loads(line)
    text = d['text']
    new_spans = []
    for span in d.get('spans', []):
        if span['label'] in INST_LABELS:
            key = (span['label'], text[span['start']:span['end']].lower().strip())
            if key in inst_changes:
                span = dict(span)
                span['label'] = inst_changes[key]
                n_inst_changed += 1
        new_spans.append(span)
    d['spans'] = new_spans
    base_train.append(d)

print(f"  {len(base_train)} phrases, {n_inst_changed} spans inst corrigés")

# ── 3. Charger les phrases wikinews re-annotées ──────────────────────────────
WIKI_ANNOTATED = DATA / "wikinews_new_annotated_v64.jsonl"

wiki_phrases = []
if not WIKI_ANNOTATED.exists():
    print(f"⚠  {WIKI_ANNOTATED} absent — skip wikinews")
else:
    for line in WIKI_ANNOTATED.read_text(encoding='utf-8').splitlines():
        if line.strip():
            d = json.loads(line)
            if not d.get('_fallback'):  # skip les fallbacks sans SVO
                wiki_phrases.append(d)
            else:
                # Garder quand même les fallbacks (ils ont au moins les NER hints)
                wiki_phrases.append(d)
    print(f"Wikinews annotés chargés : {len(wiki_phrases)} phrases")

# ── 4. Construire et écrire train_v6.5 ──────────────────────────────────────
train_v65 = base_train + wiki_phrases
print(f"\ntrain_v6.5 : {len(base_train)} (v6.4) + {len(wiki_phrases)} (wikinews) = {len(train_v65)} phrases")

out_train = DATA / "train_v6.5.jsonl"
with open(out_train, 'w', encoding='utf-8') as f:
    for d in train_v65:
        # Retirer la clé _fallback du JSON final
        d.pop('_fallback', None)
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
print(f"✅ {out_train}")

# ── 5. Copier val/test (corrections inst appliquées aussi) ───────────────────
for split in ['val', 'test']:
    src = DATA / f"{split}_v6.4.jsonl"
    dst = DATA / f"{split}_v6.5.jsonl"
    if not src.exists():
        print(f"⚠  {src} absent")
        continue
    # Appliquer corrections inst sur val/test aussi
    out_lines = []
    n_changed = 0
    for line in src.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        text = d['text']
        new_spans = []
        for span in d.get('spans', []):
            if span['label'] in INST_LABELS:
                key = (span['label'], text[span['start']:span['end']].lower().strip())
                if key in inst_changes:
                    span = dict(span)
                    span['label'] = inst_changes[key]
                    n_changed += 1
            new_spans.append(span)
        d['spans'] = new_spans
        out_lines.append(json.dumps(d, ensure_ascii=False))
    dst.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
    print(f"✅ {dst}  ({len(out_lines)} phrases, {n_changed} spans inst corrigés)")

# ── 6. Résumé distribution labels ───────────────────────────────────────────
print("\n=== Distribution labels WORK/ABSTRACT/ORG dans v6.5 train ===")
label_count = Counter()
for d in train_v65:
    for span in d.get('spans', []):
        label_count[span['label']] += 1

groups = {
    'ORG':      ['hint_org_name', 'hint_inst_name', 'hint_inst_role'],
    'WORK':     ['hint_law', 'hint_document', 'hint_work_of_art'],
    'ABSTRACT': ['hint_concept', 'hint_concept_named', 'hint_disease', 'hint_language'],
}
for grp, labels in groups.items():
    total = sum(label_count[l] for l in labels)
    print(f"\n  [{grp}] total={total}")
    for l in labels:
        print(f"    {l:25s} : {label_count[l]:6d}")

print(f"\n  Total spans train : {sum(label_count.values())}")
print(f"  Total phrases     : {len(train_v65)}")

