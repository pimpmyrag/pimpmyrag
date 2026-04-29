#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  Annotation Gold v4 via Claude Batch API
#  Enrichit train/val/test_v3.jsonl avec :
#    - verb_trigger (voice, certainty, negated)
#    - svo_role + gov_verb_start sur chaque span NER
#    - mod_of_start pour les modificateurs nominaux
#    - gender/number sur PER/ORG/EVENT
#    - pron_subj/pron_obj avec gender/number/person
#
#  Résumable : les splits déjà traités sont automatiquement sautés.
#  Usage : remplacer ANTHROPIC_API_KEY puis ./run_claude_annotation.sh
# ═══════════════════════════════════════════════════════════════════
# PAS de set -e : on veut continuer même si un split échoue,
# et afficher une erreur claire sans tout arrêter.
cd "$(dirname "$0")"

# ── Clé API Claude ────────────────────────────────────────────────
# La clé est lue depuis l'env — ne jamais la mettre en dur ici
# export ANTHROPIC_API_KEY="sk-ant-..."   ← ne pas faire
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ Variable ANTHROPIC_API_KEY non définie."
    echo "   Lance : export ANTHROPIC_API_KEY='sk-ant-...'"
    exit 1
fi

# ── Config ────────────────────────────────────────────────────────
MODEL="claude-sonnet-4-5"
BATCH_SIZE=8
POLL_INTERVAL=60
SCRIPT="scripts/preannotate_claude_batch.py"

# ── Vérifications ─────────────────────────────────────────────────

for f in data/train_v3.jsonl data/val_v3.jsonl data/test_v3.jsonl; do
    if [ ! -f "$f" ]; then
        echo "❌ Fichier manquant : $f"
        exit 1
    fi
done

mkdir -p data/logs_annotation

echo "🚀 Annotation Claude v4 — modèle=$MODEL batch_size=$BATCH_SIZE"
echo "   Train : 21781 phrases  (~2730 requêtes)"
echo "   Val   :  3000 phrases  (~375 requêtes)"
echo "   Test  :  3000 phrases  (~375 requêtes)"
echo "   Coût estimé : ~30-40€"
echo ""

# Compteur d'erreurs pour le résumé final
ERRORS=0

# ─── Fonction helper ──────────────────────────────────────────────
# Vérifie si un fichier output est déjà complet (non vide)
is_done() {
    local out="$1"
    local min_lines="${2:-100}"
    [ -f "$out" ] && [ "$(wc -l < "$out")" -ge "$min_lines" ]
}

run_split() {
    local label="$1"
    local idx="$2"
    local input="$3"
    local output="$4"
    local requests="$5"

    echo "═══ [$idx/3] $label ════════════════════════════════════════"

    if is_done "$output"; then
        local n
        n=$(wc -l < "$output")
        echo "⏭️  $label déjà traité ($n phrases dans $output) — skip"
        echo ""
        return 0
    fi

    python3 $SCRIPT \
        --input  "$input" \
        --output "$output" \
        --batch-size $BATCH_SIZE \
        --model $MODEL \
        --poll-interval $POLL_INTERVAL \
        --requests-file "$requests" \
        2>&1 | tee "data/logs_annotation/$(echo "$label" | tr '[:upper:]' '[:lower:]').log"

    local exit_code=${PIPESTATUS[0]}
    echo ""
    if [ $exit_code -eq 0 ]; then
        echo "✅ $label terminé → $output"
    else
        echo "❌ $label ÉCHOUÉ (code=$exit_code) — voir data/logs_annotation/$(echo "$label" | tr '[:upper:]' '[:lower:]').log"
        ERRORS=$((ERRORS + 1))
    fi
    echo ""
}

# ─── TRAIN ────────────────────────────────────────────────────────
run_split "TRAIN" 1 \
    data/train_v3.jsonl \
    data/train_v4_claude.jsonl \
    data/_batch_train_requests.jsonl

# ─── VAL ──────────────────────────────────────────────────────────
run_split "VAL" 2 \
    data/val_v3.jsonl \
    data/val_v4_claude.jsonl \
    data/_batch_val_requests.jsonl

# ─── TEST ─────────────────────────────────────────────────────────
run_split "TEST" 3 \
    data/test_v3.jsonl \
    data/test_v4_claude.jsonl \
    data/_batch_test_requests.jsonl

# ─── Résumé ───────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
if [ $ERRORS -eq 0 ]; then
    echo "  ✅ ANNOTATION TERMINÉE"
else
    echo "  ⚠️  ANNOTATION TERMINÉE AVEC $ERRORS ERREUR(S)"
fi

for label_out in "Train:data/train_v4_claude.jsonl" "Val:data/val_v4_claude.jsonl" "Test:data/test_v4_claude.jsonl"; do
    label="${label_out%%:*}"
    out="${label_out##*:}"
    if [ -f "$out" ]; then
        n=$(wc -l < "$out")
        echo "  $label : $n phrases"
    else
        echo "  $label : ❌ absent"
    fi
done

echo ""
echo "  Prochaine étape : adapter labels.py + train_multi_task.py"
echo "  pour le nouveau schéma v4 (verb_trigger, svo_role, morpho)"
echo "═══════════════════════════════════════════════════════════"

[ $ERRORS -eq 0 ] && exit 0 || exit 1

