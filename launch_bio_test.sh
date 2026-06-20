#!/bin/bash
# Script de test rapide pour valider les changements BIO
# 5% du dataset, CPU seulement, 3 epochs

cd /Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head

# Chemin vers le venv
VENV="venv/bin/activate"

# Fichiers 5pct avec BIO
TRAIN="data/train_v8.21_verbfam_5pct_bio_test.jsonl"
VAL="data/val_v8.21_verbfam_5pct_bio_test.jsonl"
TEST="data/test_v8.21_verbfam_5pct_bio_test.jsonl"

# Nom du run
RUN_NAME="bio_validation_5pct_cpu_$(date +%Y%m%d_%H%M%S)"

# Dossier de sortie (les checkpoints seront sauvés ici)
OUTPUT_DIR="logs/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"

echo "Lancement du training de validation BIO..."
echo "Run: $RUN_NAME"
echo "Output: $OUTPUT_DIR"
echo "Dataset: 5% de v8.21_verbfam avec famille BIO"
echo ""

# Utiliser le venv
source "$VENV"

# Changer de répertoire pour sauvegarder les checkpoints
cd "$OUTPUT_DIR"

# Lancer le training avec des paramètres optimisés pour CPU
time python ../../train_multi_task.py \
    --train "../../$TRAIN" \
    --val "../../$VAL" \
    --test "../../$TEST" \
    --model-name microsoft/deberta-v3-base \
    --batch-size 4 \
    --epochs 3 \
    --lr 2e-5 \
    --max-length 128 \
    --log-every 20 \
    --lambda-verb-family 0.0 \
    --lambda-verb-family-fine 0.0 \
    --lambda-verb-polarity 0.0 \
    --lambda-verb-aspect 0.0 \
    --lambda-verb-source 0.0 \
    --device cpu \
    --num-workers 0 \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "Training terminé !"
echo "Dossier des résultats : $OUTPUT_DIR"
