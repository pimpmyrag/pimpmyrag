# Haiku Span Repair (Haiku-first)

Script: `training/multi-head/scripts/repair_spans_haiku_batch.py`

## Source canonique Stanza

Le script `training/multi-head/scripts/stanza_inject_nominal_parents.py` peut écrire un cache
persistant `*_stanza_cache.jsonl` via `--cache-output`. C'est ce snapshot qu'il faut réutiliser
pour les passes ultérieures au lieu de recalculer Stanza.

## Objectif

Nettoyer un dataset JSONL en faisant la correction des spans uniquement avec Haiku (Batch API):
- ajout de `verb_trigger` manquants
- correction des frontières (groupe nominal minimal)
- suppression de doublons/faux positifs
- correction des rôles SVO (réduction des `OBLIQUE` non justifiés)

## Pré-requis

- `ANTHROPIC_API_KEY` dans `training/multi-head/.secrets.env` (chargé automatiquement)
- venv activé

## Commandes

```bash
cd /Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head
source venv/bin/activate
python3 scripts/repair_spans_haiku_batch.py \
  --input data/train_v8.18.jsonl \
  --output data/train_v8.19_haiku_repair.jsonl \
  --passes 2 \
  --poll-interval 30
```

Reprendre un batch existant (1 passe):

```bash
cd /Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head
source venv/bin/activate
python3 scripts/repair_spans_haiku_batch.py \
  --input data/train_v8.18.jsonl \
  --output data/train_v8.19_haiku_repair.jsonl \
  --batch-id msgbatch_... \
  --passes 1
```

## Notes

- Le script remplace la liste `spans` de chaque phrase par la sortie Haiku validée.
- En cas de réponse invalide pour une phrase, fallback sur la phrase d'entrée (sans crash global).
- Les requêtes sont enregistrées dans `*_passN_requests.jsonl` pour audit/reprise.
- Pour `v8.21_verbfam`, produire un cache par split : `train`, `val`, `test`.

