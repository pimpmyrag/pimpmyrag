#!/bin/bash
# Script de validation BIO - training local CPU sur 5% du dataset

cd /Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head

# Activer le venv
source venv/bin/activate

# Nom du run
RUN_NAME="bio_validation_$(date +%Y%m%d_%H%M%S)"
CONFIG="configs/bio-validation-5pct.json"

# Créer le dossier de logs
mkdir -p "logs/${RUN_NAME}"

echo "═══════════════════════════════════════════════════════════"
echo "  VALIDATION FAMILLE COARSE BIO"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Config: $CONFIG"
echo "  Run: $RUN_NAME"
echo "  Device: CPU"
echo "  Dataset: 5% v8.21_verbfam avec famille BIO"
echo "  Épochs: 3"
echo ""

# Lancer le training (pas de --wandb-project = wandb désactivé)
python run_training.py \
    --config "$CONFIG" \
    --start-epoch 0 \
    --device cpu \
    2>&1 | tee "logs/${RUN_NAME}/training.log"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  VALIDATION TERMINÉE"
echo "  Résultats dans: logs/${RUN_NAME}/"
echo "═══════════════════════════════════════════════════════════"
