#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Training adaptatif — hard negatives introduits seulement
#  quand le modèle stagne (plateau détecté sur val score)
# ═══════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"
source venv/bin/activate

MODEL="microsoft/deberta-v3-base"
DATA="data"
BS=32
ACCUM=2
MAX_EPOCHS=15
PATIENCE=2        # nb epochs sans amélioration avant d'augmenter difficulté
MIN_DELTA=0.0005  # amélioration minimale considérée comme progrès

# Niveaux de difficulté progressifs
ENGLOBANT_RATIOS=(0.0 0.4 0.7 1.0)
HARD_NEG_RATIOS=(0.3 0.7 1.0 1.0)
ENGLOBANT_WEIGHTS=(1.0 1.0 1.2 1.5)
FOCAL_GAMMAS=(0.0 0.5 1.0 1.5)
LEVEL_NAMES=("easy" "medium" "hard" "full")

current_level=0
stagnation_count=0
best_score=-1.0
current_epoch=1
resume_arg=""

mkdir -p logs
log_file="logs/adaptive.log"
echo "🚀 Démarrage training adaptatif — $(date)" | tee $log_file

rebuild_dataset() {
    local level=$1
    local ratio=${ENGLOBANT_RATIOS[$level]}
    local hard=${HARD_NEG_RATIOS[$level]}
    local weight=${ENGLOBANT_WEIGHTS[$level]}
    local name=${LEVEL_NAMES[$level]}
    echo "🔧 Build dataset niveau $level ($name) — englobant_ratio=$ratio hard_neg_ratio=$hard weight=$weight" | tee -a $log_file
    python3 data/build_multitask_dataset.py \
        --input $DATA/train.jsonl \
        --output $DATA/train.adaptive.multitask.jsonl \
        --model-name $MODEL \
        --englobant-ratio $ratio \
        --hard-neg-ratio $hard \
        --englobant-weight $weight
}

# Build val/test une seule fois (full hard)
echo "📦 Build val/test datasets..." | tee -a $log_file
python3 data/build_multitask_dataset.py \
    --input $DATA/val.jsonl \
    --output $DATA/val.multitask.jsonl \
    --model-name $MODEL \
    --englobant-ratio 1.0 \
    --hard-neg-ratio 1.0

python3 data/build_multitask_dataset.py \
    --input $DATA/test.jsonl \
    --output $DATA/test.multitask.jsonl \
    --model-name $MODEL \
    --englobant-ratio 1.0 \
    --hard-neg-ratio 1.0

# Build dataset niveau initial
rebuild_dataset $current_level

# ─── BOUCLE PRINCIPALE ───────────────────────────────────
while [ $current_epoch -le $MAX_EPOCHS ]; do
    echo "" | tee -a $log_file
    echo "══════════════════════════════════════════════════" | tee -a $log_file
    echo "  Epoch $current_epoch/$MAX_EPOCHS  |  Niveau ${LEVEL_NAMES[$current_level]} (stagnation=$stagnation_count/$PATIENCE)" | tee -a $log_file
    echo "══════════════════════════════════════════════════" | tee -a $log_file

    focal=${FOCAL_GAMMAS[$current_level]}
    epoch_log="logs/epoch_${current_epoch}.log"

    python3 train_multi_task.py \
        --train $DATA/train.adaptive.multitask.jsonl \
        --val   $DATA/val.multitask.jsonl \
        --test  $DATA/test.multitask.jsonl \
        --model-name $MODEL \
        --epochs $current_epoch \
        --start-epoch $current_epoch \
        --batch-size $BS \
        --accum-steps $ACCUM \
        --lr 5e-6 \
        --head-lr-multiplier 4.0 \
        --warmup-epochs 0 \
        --lambda-boundary 1.5 \
        --lambda-coarse 1.0 \
        --lambda-fine 1.0 \
        --focal-gamma $focal \
        --device cpu \
        $resume_arg \
        2>&1 | tee $epoch_log

    # Extraire le val score depuis le log de l'epoch
    val_score=$(grep "Score=" $epoch_log | tail -1 | grep -oE "Score=[0-9.]+" | cut -d= -f2)

    if [ -z "$val_score" ]; then
        echo "⚠️  Impossible d'extraire le val score — on continue" | tee -a $log_file
        resume_arg="--resume checkpoint_last_multitask.pt"
        current_epoch=$((current_epoch + 1))
        continue
    fi

    echo "📊 Epoch $current_epoch — Val Score=$val_score (best=$best_score)" | tee -a $log_file

    # Vérifier amélioration
    improved=$(python3 -c "print('yes' if float('$val_score') > float('$best_score') + $MIN_DELTA else 'no')")

    if [ "$improved" = "yes" ]; then
        best_score=$val_score
        stagnation_count=0
        echo "✅ Amélioration! best_score=$best_score" | tee -a $log_file
    else
        stagnation_count=$((stagnation_count + 1))
        echo "⏸️  Pas d'amélioration ($stagnation_count/$PATIENCE)" | tee -a $log_file
    fi

    # Augmenter difficulté si plateau atteint
    max_level=$(( ${#ENGLOBANT_RATIOS[@]} - 1 ))
    if [ $stagnation_count -ge $PATIENCE ]; then
        if [ $current_level -lt $max_level ]; then
            current_level=$((current_level + 1))
            stagnation_count=0
            echo "🔥 Plateau détecté → passage au niveau ${LEVEL_NAMES[$current_level]}" | tee -a $log_file
            rebuild_dataset $current_level
        else
            echo "🛑 Plateau au niveau max (${LEVEL_NAMES[$current_level]}) — early stopping" | tee -a $log_file
            break
        fi
    fi

    resume_arg="--resume checkpoint_last_multitask.pt"
    current_epoch=$((current_epoch + 1))
done

echo "" | tee -a $log_file
echo "═══════════════════════════════════════════" | tee -a $log_file
echo "  ✅ TRAINING ADAPTATIF TERMINÉ" | tee -a $log_file
echo "  Best val score : $best_score" | tee -a $log_file
echo "  Niveau final   : ${LEVEL_NAMES[$current_level]}" | tee -a $log_file
echo "═══════════════════════════════════════════" | tee -a $log_file

