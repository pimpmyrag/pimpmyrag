# build_nominal_head_trees.py

Construit une couche auxiliaire de têtes nominales à partir de Stanza.

## Idée

On garde tous les spans existants, puis on ajoute:
- `nominal_nodes`: têtes nominales uniques (liées à un span ou synthétiques)
- `nominal_edges`: arbre/forêt nominale issue des dépendances Stanza
- pour chaque span: `nominal_head_id`, `nominal_head_text`, offsets de la tête

## Pourquoi

Cela permet de:
- récupérer les relations nominales même quand il manque un span NER sur la tête,
- ajouter des têtes synthétiques comme `information`, `somme`, etc.,
- différer ensuite le matching NER sur ces têtes si nécessaire.

## Exemple

```bash
cd /Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head
source venv/bin/activate
python3 scripts/build_nominal_head_trees.py \
  --input /tmp/one_phrase_spans.jsonl \
  --output /tmp/one_phrase_nominal_trees.jsonl
```

