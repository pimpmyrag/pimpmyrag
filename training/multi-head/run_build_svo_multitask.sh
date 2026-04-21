#!/usr/bin/env bash
# run_build_svo_multitask.sh
# ==========================
# Construit les datasets multitask à partir des fichiers silver SVO.
#
# Usage (depuis training/multi-head/) :
#   bash run_build_svo_multitask.sh [--tokenizer-path /path/to/tokenizer]
#
# Le script suppose que le venv est activé ou que les dépendances sont disponibles.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

TOKENIZER_PATH="${TOKENIZER_PATH:-microsoft/deberta-v3-base}"
HARD_PER_GOLD=6
SOFT_FACTOR=2.0
MAX_SPAN_LEN=12   # élargi pour couvrir les sujets/objets SVO (NP plus longs)
SEED=13

# Permettre un override via arg
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tokenizer-path) TOKENIZER_PATH="$2"; shift 2 ;;
        *) echo "Arg inconnu: $1"; exit 1 ;;
    esac
done

echo "════════════════════════════════════════════════════════"
echo "  Build multitask SVO silver"
echo "  DATA_DIR       : $DATA_DIR"
echo "  TOKENIZER      : $TOKENIZER_PATH"
echo "  max_span_len   : $MAX_SPAN_LEN"
echo "════════════════════════════════════════════════════════"
echo

for SPLIT in train val test; do
    INPUT="$DATA_DIR/${SPLIT}_svo_silver.jsonl"
    OUTPUT="$DATA_DIR/${SPLIT}.svo.multitask.jsonl"

    if [[ ! -f "$INPUT" ]]; then
        echo "⚠️  $INPUT introuvable, skip."
        continue
    fi

    echo "── $SPLIT ──────────────────────────────────────────────"
    python "$SCRIPT_DIR/build_multitask_dataset.py" \
        --input              "$INPUT"           \
        --output             "$OUTPUT"          \
        --tokenizer-path     "$TOKENIZER_PATH"  \
        --hard-per-gold      "$HARD_PER_GOLD"   \
        --soft-factor        "$SOFT_FACTOR"     \
        --max-span-len       "$MAX_SPAN_LEN"    \
        --seed               "$SEED"
    echo
done

echo "✅ Tous les splits SVO multitask générés dans $DATA_DIR"

