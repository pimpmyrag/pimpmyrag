#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_local.sh — Lance le training NER directement sur Mac (sans Docker)
# ~8× plus rapide que Docker grâce à Apple Accelerate Framework (CPU natif)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv_clean"
TRAIN_SCRIPT="$SCRIPT_DIR/train_ner.py"
DATA_DIR="$SCRIPT_DIR/../../data"
OUTPUT_DIR="$SCRIPT_DIR/training_output"

# Vérifier que le venv existe
if [ ! -f "$VENV/bin/activate" ]; then
  echo "❌  venv introuvable : $VENV"
  echo "   Lance d'abord : python3.11 -m venv $VENV && source $VENV/bin/activate && pip install torch transformers datasets accelerate sentencepiece tokenizers scikit-learn seqeval numpy protobuf"
  exit 1
fi

source "$VENV/bin/activate"

mkdir -p "$OUTPUT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Training NER — Mac natif (Apple Accelerate, sans Docker)"
python3 -c "import torch; print(f'  torch {torch.__version__} | MPS: {torch.backends.mps.is_available()} | CPU threads: {torch.get_num_threads()}')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Paramètres training complet (modifiables en argument)
BILOU="${1:-$DATA_DIR/data.bilou.fixed}"
EPOCHS="${2:-6}"
LR="${3:-3e-5}"
BATCH="${4:-16}"
EVAL_STEPS="${5:-300}"

echo "  Data    : $BILOU"
echo "  Epochs  : $EPOCHS | LR : $LR | Batch : $BATCH | Eval steps : $EVAL_STEPS"
echo "  Output  : $OUTPUT_DIR"
echo ""

python3 "$TRAIN_SCRIPT" \
  --bilou        "$BILOU" \
  --output_dir   "$OUTPUT_DIR" \
  --model        "FacebookAI/xlm-roberta-base" \
  --epochs       "$EPOCHS" \
  --batch_size   "$BATCH" \
  --lr           "$LR" \
  --eval_steps   "$EVAL_STEPS" \
  --logging_steps 50 \
  --bilou_boost         3.0 \
  --class_weight_o      0.4 \
  --entity_boost_object 1.0 \
  --entity_boost_event  1.0 \
  --weight_clamp_max    10.0 \
  --label_smoothing     0.05 \
  2>&1 | tee "$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅  Training terminé. Modèle dans : $OUTPUT_DIR"

