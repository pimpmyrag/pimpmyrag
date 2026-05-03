"""
Prépare les phrases wikinews non intégrées en v6.3 pour re-annotation avec preannotate_claude_batch.py.

Actions :
- Déduplique wikinews_claude_annotated_valid.jsonl + train_wiki_claude_annotated.jsonl
- Retire les phrases déjà dans v6.3/v6.4
- Supprime les anciens labels SVO Stanza (svo_verb, svo_subject, svo_object, svo_iobj)
- Mappe hint_quantity → hint_count (label renommé)
- Garde tous les hint_* comme pré-annotations pour Claude
- Assigne un ID stable
- Sortie : data/wikinews_new_for_reannotation.jsonl
"""
import json
from pathlib import Path
from collections import Counter

DATA = Path("data")

# Labels SVO ancienne génération à supprimer (remplacés par verb_trigger + svo_role attribut)
OLD_SVO = {'svo_verb', 'svo_subject', 'svo_object', 'svo_iobj',
           'svo_oblique', 'svo_agent', 'svo_cause'}

# Remapping labels renommés
LABEL_REMAP = {
    'hint_quantity': 'hint_count',   # renommé entre v4 et v5
}

# Labels hint_* valides en v6.4 (les autres seront gardés quand même comme hint pour Claude)
VALID_V64 = {
    "hint_person_name", "hint_person_role", "hint_norp", "hint_group_role",
    "hint_org_name", "hint_inst_name", "hint_inst_role",
    "hint_gpe", "hint_fac_name", "hint_loc_generic", "hint_infra",
    "hint_weapon", "hint_vehicle", "hint_substance", "hint_food",
    "hint_tool", "hint_object_generic", "hint_object_name",
    "hint_event_nominal", "hint_event_named",
    "hint_time_date", "hint_time_clock", "hint_time_duration",
    "hint_measure", "hint_percentage", "hint_count", "hint_money", "hint_rate",
    "hint_law", "hint_document", "hint_work_of_art",
    "hint_concept", "hint_concept_named", "hint_disease", "hint_language",
    # labels SVO actuels (gardés tels quels)
    "verb_trigger", "pron_subj", "pron_obj",
}

# ── Charger v6 pour dédupliquer ──────────────────────────────────────────────
in_v6 = set()
for split in ['train', 'val', 'test']:
    p = DATA / f'{split}_v6.3.jsonl'
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                in_v6.add(json.loads(line)['text'].strip().lower())
print(f"v6.3 total : {len(in_v6)} phrases")

# ── Charger et fusionner les sources wikinews ────────────────────────────────
sources = [
    DATA / 'wikinews_claude_annotated_valid.jsonl',
    DATA / 'train_wiki_claude_annotated.jsonl',
]

seen_texts = set()
candidates = []
for src in sources:
    if not src.exists():
        print(f"⚠ Absent : {src}")
        continue
    n_loaded = n_dup = n_in_v6 = 0
    for line in src.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        text = d.get('text', '').strip()
        if not text:
            continue
        n_loaded += 1
        key = text.lower()
        if key in in_v6:
            n_in_v6 += 1
            continue
        if key in seen_texts:
            n_dup += 1
            continue
        seen_texts.add(key)
        candidates.append(d)
    print(f"{src.name}: {n_loaded} chargées, {n_in_v6} déjà en v6, {n_dup} dups → {n_loaded-n_in_v6-n_dup} retenues")

print(f"\nTotal nouvelles phrases : {len(candidates)}")

# ── Nettoyer les spans ───────────────────────────────────────────────────────
label_removed = Counter()
label_remapped = Counter()
label_kept = Counter()

out_records = []
for i, d in enumerate(candidates):
    clean_spans = []
    for span in d.get('spans', []):
        lbl = span.get('label', '')

        # Supprimer anciens SVO Stanza
        if lbl in OLD_SVO:
            label_removed[lbl] += 1
            continue

        # Remap labels renommés
        if lbl in LABEL_REMAP:
            label_remapped[lbl] += 1
            span = dict(span)
            span['label'] = LABEL_REMAP[lbl]
            lbl = span['label']

        # Garder tout le reste (hint_* + verb_trigger + pron_*)
        label_kept[lbl] += 1
        # Nettoyer les anciens champs SVO encodés en attributs de span
        clean = {k: v for k, v in span.items()
                 if k not in ('svo_role_silver', 'head', 'dep', 'voice_silver')}
        clean_spans.append(clean)

    record = {
        'id': d.get('id') or f'wiki_new_{i:05d}',
        'text': d['text'].strip(),
        'spans': clean_spans,
    }
    out_records.append(record)

# ── Sauvegarder ─────────────────────────────────────────────────────────────
out_path = DATA / 'wikinews_new_for_reannotation.jsonl'
with open(out_path, 'w', encoding='utf-8') as f:
    for r in out_records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"\n=== Spans nettoyés ===")
print(f"Supprimés (anciens SVO Stanza) : {sum(label_removed.values())}")
for lbl, n in label_removed.most_common():
    print(f"  {lbl}: {n}")
print(f"\nRemappés : {sum(label_remapped.values())}")
for lbl, n in label_remapped.most_common():
    print(f"  {lbl} → {LABEL_REMAP[lbl]}: {n}")
print(f"\nGardés (pré-annotations pour Claude) : {sum(label_kept.values())}")
for lbl, n in label_kept.most_common(15):
    print(f"  {lbl}: {n}")

print(f"\n✅ {len(out_records)} phrases → {out_path}")
print(f"\nCommande de re-annotation :")
print(f"  ./venv/bin/python3 scripts/preannotate_claude_batch.py \\")
print(f"    --input {out_path} \\")
print(f"    --output data/wikinews_new_annotated_v64.jsonl \\")
print(f"    --batch-size 5 \\")
print(f"    --model claude-sonnet-4-6")

