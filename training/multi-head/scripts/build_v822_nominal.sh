#!/usr/bin/env bash
# build_v822_nominal.sh — Pipeline complet v8.22 nominal
# ========================================================
# Chaîne : v8.21_verbfam → règles → Stanza → Haiku → v8.22_nominal
#
# Usage :
#   cd /Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head
#   source venv/bin/activate && source .secrets.env
#   bash scripts/build_v822_nominal.sh [train|val|test|all]
#
# Étapes :
#   1. Règles haute précision (annotate_nominal_parents_rules.py)
#   2. Stanza/UD (stanza_inject_nominal_parents.py) — streaming + checkpoint + cache persistant
#   3. Haiku batch API (annotate_nominal_parents_haiku_batch.py) — ~15-20% des phrases
#   4. (optionnel) Rebuild multitask dataset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$BASE_DIR/data"

SPLIT="${1:-all}"
IN_VERSION="v8.21_verbfam"
OUT_VERSION="v8.22_nominal"

echo "🚀 Pipeline nominal v8.22 — split=$SPLIT"
echo "   Base : $IN_VERSION → $OUT_VERSION"
echo ""

process_split() {
    local split="$1"
    echo "══════════════════════════════════════════════════"
    echo "  Split : $split"
    echo "══════════════════════════════════════════════════"

    local IN="$DATA_DIR/${split}_${IN_VERSION}.jsonl"
    local RULES="$DATA_DIR/${split}_${IN_VERSION}_rules.jsonl"
    local STANZA="$DATA_DIR/${split}_${IN_VERSION}_stanza.jsonl"
    local STANZA_CACHE="$DATA_DIR/${split}_${IN_VERSION}_stanza_cache.jsonl"
    local FINAL="$DATA_DIR/${split}_${OUT_VERSION}.jsonl"

    if [ ! -f "$IN" ]; then
        echo "❌ Fichier source manquant : $IN"
        return 1
    fi

    # ── Étape 1 : Règles ────────────────────────────────────────────────
    echo ""
    echo "📐 Étape 1 : Règles haute précision..."
    python3 "$SCRIPT_DIR/annotate_nominal_parents_rules.py" \
        --input  "$IN" \
        --output "$RULES"

    # ── Étape 2 : Stanza ────────────────────────────────────────────────
    echo ""
    echo "🌿 Étape 2 : Stanza/UD injection..."
    python3 "$SCRIPT_DIR/stanza_inject_nominal_parents.py" \
        --input       "$RULES" \
        --output      "$STANZA" \
        --cache-output "$STANZA_CACHE" \
        --skip-existing \
        --batch-size  64

    # ── Étape 3 : Haiku (fallback sur phrases ambiguës) ─────────────────
    echo ""
    echo "🤖 Étape 3 : Haiku batch (phrases ambiguës)..."
#    python3 "$SCRIPT_DIR/annotate_nominal_parents_haiku_batch.py" \
#        --input        "$STANZA" \
#        --output       "$FINAL" \
#        --api-key      "${ANTHROPIC_API_KEY:}" \
#        --poll-interval 60

    echo ""
    echo "✅ $split terminé : $FINAL"
    wc -l "$FINAL"
}

# ── Sélection des splits ────────────────────────────────────────────────────
case "$SPLIT" in
    train) process_split train ;;
    val)   process_split val   ;;
    test)  process_split test  ;;
    all)
        process_split train
        process_split val
        process_split test
        ;;
    *)
        echo "Usage: $0 [train|val|test|all]"
        exit 1
        ;;
esac

echo ""
echo "══════════════════════════════════════════════════"
echo "🎉 Pipeline v8.22 nominal terminé !"
echo ""
echo "Prochaines étapes :"
echo "  1. Vérifier la couverture :"
echo "     python3 /tmp/check_nominal_coverage.py data/train_${OUT_VERSION}.jsonl"
echo ""
echo "  2. Rebuild multitask (si test_local OK) :"
echo "     python3 build_multitask_dataset.py \\"
echo "       --input  data/train_${OUT_VERSION}.jsonl \\"
echo "       --output data/train.${OUT_VERSION}.multitask.jsonl"
echo ""
echo "  3. DVC add/push :"
echo "     dvc add data/train_${OUT_VERSION}.jsonl data/val_${OUT_VERSION}.jsonl data/test_${OUT_VERSION}.jsonl"
echo "     source .secrets.env && dvc push ..."
echo "     git add data/*.dvc && git commit -m 'feat: dataset ${OUT_VERSION}'"
echo ""
echo "  4. Mettre à jour DEFAULT_GOLD_VERSION dans launch_training.py"
echo "══════════════════════════════════════════════════"

