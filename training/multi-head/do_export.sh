#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Export ONNX — toutes les 8 têtes (NER + SVO + voice + morpho)
#  Vectorisé, compatible axes dynamiques, prêt pour Kotlin/ORT
#
#  Étape 1 : export float32  (export_vec_full.py)
#  Étape 2 : quantification int8 (quantize_onnx.py) → 2-4× plus rapide
# ═══════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")"
source venv/bin/activate

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHECKPOINT="${NER_CHECKPOINT:-$REPO_ROOT/models/deberta/fine-tuning-29042026/best_model_multitask.pt}"
OUT_DIR="${NER_OUT_DIR:-$REPO_ROOT/models/deberta/fine-tuning-29042026}"
OUT_ONNX="$OUT_DIR/best_model_multitask_full.onnx"
OUT_Q8="$OUT_DIR/best_model_multitask_q8.onnx"
TOKENIZER="$REPO_ROOT/tokenizer_export_clean"
LOG="/tmp/onnx_export_full.log"

echo "📦 Checkpoint : $CHECKPOINT"
echo "📤 Output f32 : $OUT_ONNX"
echo "📤 Output q8  : $OUT_Q8"
echo "🔤 Tokenizer  : $TOKENIZER"

# ── Étape 1 : Export float32 ────────────────────────────────
echo ""
echo "══ Étape 1/2 : Export ONNX float32 ══"
python export_vec_full.py \
  --checkpoint     "$CHECKPOINT" \
  --output         "$OUT_ONNX" \
  --model-name     microsoft/deberta-v3-base \
  --tokenizer-path "$TOKENIZER" \
  --opset 17 \
  2>&1 | tee "$LOG"

# ── Étape 2 : Quantification int8 ───────────────────────────
echo ""
echo "══ Étape 2/2 : Quantification int8 ══"
python quantize_onnx.py \
  --input  "$OUT_ONNX" \
  --output "$OUT_Q8" \
  2>&1 | tee -a "$LOG"

echo ""
echo "✅ Pipeline complet"
echo "   Float32 : $OUT_ONNX"
echo "   Int8 q8 : $OUT_Q8   ← utiliser ce modèle en Java/Kotlin"
echo "   Log     : $LOG"
echo ""
echo "Dans OnnxMultiHeadEntityExtractor, passer le chemin vers :"
echo "  $OUT_Q8"
