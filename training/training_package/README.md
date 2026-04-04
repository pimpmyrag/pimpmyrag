
# DeBERTa v3 Span Classification Training Package

Ce package contient :
- `train.py` : script d'entraînement
- `dataset.py` : loader JSONL pour spans
- `model.py` : architecture DeBERTa + classification spans
- `evaluate.py` : évaluation basique F1

## Installation
```
pip install torch transformers datasets scikit-learn
```

## Lancement
```
python train.py --data data.jsonl --epochs 5 --batch 16 --lr 2e-5
```
