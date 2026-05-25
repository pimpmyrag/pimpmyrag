#!/bin/bash
# run_adaptive_training.sh — Détection HW + lancement run_training.py
# Toute la logique adaptative est dans run_training.py + configs/*.json
set -e

cd /workspace/pimpmyrag/training/multi-head

CONFIG="${CONFIG:-configs/bndwarm-oblique.json}"
GOLD_VERSION="${GOLD_VERSION:-v8.18}"
export TOKENIZERS_PARALLELISM="false"

# ── Détection device ────────────────────────────────────────────────────────
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
    VRAM_GB=$(python3 -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory/1024**3))")
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    echo "🖥️  CUDA détecté — VRAM=${VRAM_GB}GB"
elif python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    DEVICE="mps"
    echo "🍎 Device MPS"
else
    DEVICE="cpu"
    echo "💻 Device CPU"
fi

# ── Lancement ───────────────────────────────────────────────────────────────
echo "🚀 config=$CONFIG  dataset=$GOLD_VERSION  device=$DEVICE"
mkdir -p logs

python3 run_training.py \
    --config        "$CONFIG" \
    --gold-version  "$GOLD_VERSION" \
    --device        "$DEVICE" \
    ${START_EPOCH:+--start-epoch $START_EPOCH} \
    ${START_LEVEL:+--start-level $START_LEVEL} \
    ${KEEP_CHECKPOINT:+--keep-checkpoint} \
    ${NER_ONLY_BENCH:+--ner-only-bench} \
    2>&1 | tee logs/training.log
