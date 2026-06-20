# enrich_span_nominal_groups.py

Enrichit chaque span existant avec un groupe nominal minimal basé sur Stanza (UD deps).

## Champs ajoutés par span

- `nominal_group`
- `nominal_group_start`
- `nominal_group_end`
- `nominal_group_head`

## Principes

- Ne modifie pas les offsets/labels du span original.
- Ignore les labels `verb_trigger`, `pron_subj`, `pron_obj`.
- Expansion conservative sur dépendances nominales (`det`, `amod`, `nmod`, `appos`, etc.).

## Exemples

```bash
cd /Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head
source venv/bin/activate
python3 scripts/enrich_span_nominal_groups.py \
  --input /tmp/test_10_val.jsonl \
  --output /tmp/test_10_val_ng.jsonl \
  --max-records 10
```

```bash
cd /Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head
source venv/bin/activate
python3 scripts/enrich_span_nominal_groups.py \
  --input data/val_v8.21_verbfam.jsonl \
  --output data/val_v8.21_verbfam_ng.jsonl \
  --download-models
```

