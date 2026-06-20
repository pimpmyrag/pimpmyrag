# fill_nominal_edges_with_stanza.py

Remplit `edges` (relations nominales) dans un fichier graph JSONL à partir de Stanza.

## Inputs

- `--graph-input`: JSONL graph (`id`, `text`, `nodes`, ...)
- `--spans-input`: JSONL spans (`id`, `text`, `spans`) avec mêmes ids

## Output

- `--output`: graph JSONL avec `edges` enrichis (ou remplacés)
- chaque record contient aussi `pending_nominal_links` pour les relations nominales
  où la cible/source pointe vers un groupe nominal sans node NER associé.

## Règles

- Relations nominales uniquement: `APPOS`, `NMOD`, `AMOD`, `COMPOUND`
- Ne touche pas aux `nodes`, `events`, `discourse`
- En mode défaut, merge avec les edges existants
- Les liens non résolus (sans node NER mappable) sont conservés pour une passe NER ultérieure

## Exemples

```bash
cd /Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head
source venv/bin/activate
python3 scripts/fill_nominal_edges_with_stanza.py \
  --graph-input /tmp/test_10_val_graph.jsonl \
  --spans-input /tmp/test_10_val.jsonl \
  --output /tmp/test_10_val_graph_stanza_edges.jsonl \
  --max-records 10
```

```bash
cd /Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head
source venv/bin/activate
python3 scripts/fill_nominal_edges_with_stanza.py \
  --graph-input /tmp/test_10_val_graph.jsonl \
  --spans-input /tmp/test_10_val.jsonl \
  --output /tmp/test_10_val_graph_stanza_edges_replace.jsonl \
  --replace-edges
```

