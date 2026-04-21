#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Training adaptatif — hard negatives introduits seulement
#  quand le modèle stagne (plateau détecté sur val score)
# ═══════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"

# ── Environnement Python ──────────────────────────────────
if [ -f venv/bin/activate ]; then
    echo "🐍 Activation venv local"
    source venv/bin/activate
else
    echo "🐍 Pas de venv détecté — vérification des dépendances"
    if [ -f requirements.txt ]; then
        pip install -q -r requirements.txt
    fi
fi

# ── Détection device & batch size ────────────────────────
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
    BS=64
    ACCUM=1
    echo "🚀 Device: CUDA (BS=$BS)"
elif python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    DEVICE="mps"
    BS=24
    ACCUM=2
    echo "🍎 Device: MPS (BS=$BS)"
else
    DEVICE="cpu"
    BS=16
    ACCUM=2
    echo "💻 Device: CPU (BS=$BS)"
fi

MODEL="microsoft/deberta-v3-base"
DATA="data"
MAX_EPOCHS=40
PATIENCE=3        # nb epochs sans amélioration avant d'augmenter difficulté
MAX_EPOCHS_PER_LEVEL=5  # forcer niveau suivant même si amélioration continue
MIN_DELTA=0.0005  # amélioration minimale considérée comme progrès

# Niveaux de difficulté progressifs (6 niveaux au lieu de 4)
ENGLOBANT_RATIOS=(0.0  0.2  0.4  0.6  0.8  1.0)
HARD_NEG_RATIOS=( 0.2  0.4  0.6  0.8  1.0  1.0)
ENGLOBANT_WEIGHTS=(1.0 1.0  1.0  1.1  1.3  1.5)
FOCAL_GAMMAS=(    0.0  0.0  0.5  0.5  1.0  1.5)
LEVEL_NAMES=("easy" "easy+" "medium" "medium+" "hard" "full")

# Reprise: START_LEVEL=1 START_EPOCH=13 KEEP_CHECKPOINT=1 ./run_adaptive_training.sh
START_LEVEL=${START_LEVEL:-0}
START_EPOCH=${START_EPOCH:-1}
KEEP_CHECKPOINT=${KEEP_CHECKPOINT:-0}

current_level=$START_LEVEL
stagnation_count=0
epochs_at_level=0
best_score=-1.0
current_epoch=$START_EPOCH
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
        --input $DATA/train_v2.jsonl \
        --output $DATA/train.adaptive.multitask.jsonl \
        --model-name $MODEL \
        --englobant-ratio $ratio \
        --hard-neg-ratio $hard \
        --englobant-weight $weight
}

# Build val/test une seule fois (full hard)
echo "📦 Build val/test datasets..." | tee -a $log_file
python3 data/build_multitask_dataset.py \
    --input $DATA/val_v2.jsonl \
    --output $DATA/val.multitask.jsonl \
    --model-name $MODEL \
    --englobant-ratio 1.0 \
    --hard-neg-ratio 1.0

python3 data/build_multitask_dataset.py \
    --input $DATA/test_v2.jsonl \
    --output $DATA/test.multitask.jsonl \
    --model-name $MODEL \
    --englobant-ratio 1.0 \
    --hard-neg-ratio 1.0

# Checkpoints: supprimer ou reprendre selon KEEP_CHECKPOINT
if [ "$KEEP_CHECKPOINT" = "1" ]; then
    echo "Reprise depuis checkpoint existant" | tee -a $log_file
    if [ -f checkpoint_best_multitask.pt ]; then
        resume_arg="--resume checkpoint_best_multitask.pt"
        best_score=$(python3 -c "import torch; c=torch.load('checkpoint_best_multitask.pt',map_location='cpu'); print(f\"{c.get('best_score',-1.0):.4f}\")" 2>/dev/null || echo "-1.0")
        echo "best_score checkpoint: $best_score" | tee -a $log_file
    fi
elif [ -f checkpoint_best_multitask.pt ] || [ -f checkpoint_last_multitask.pt ]; then
    echo "Suppression anciens checkpoints" | tee -a $log_file
    rm -f checkpoint_best_multitask.pt checkpoint_last_multitask.pt
fi

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
        --device $DEVICE \
        --layer-lr-decay 0.9 \
        --ema-decay 0.999 \
        $resume_arg \
        2>&1 | tee $epoch_log

    # Extraire le val score depuis le log de l'epoch
    val_score=$(grep "Score=" $epoch_log | tail -1 | grep -oE "Score=[0-9.]+" | cut -d= -f2)

    if [ -z "$val_score" ]; then
        echo "⚠️  Impossible d'extraire le val score — on continue" | tee -a $log_file
        resume_arg="--resume checkpoint_best_multitask.pt"
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

    epochs_at_level=$((epochs_at_level + 1))
    max_level=$(( ${#ENGLOBANT_RATIOS[@]} - 1 ))

    should_advance=0
    if [ $stagnation_count -ge $PATIENCE ]; then
        echo "PLATEAU ($stagnation_count epochs)" | tee -a $log_file
        should_advance=1
    fi
    if [ $epochs_at_level -ge $MAX_EPOCHS_PER_LEVEL ]; then
        echo "MAX epochs/level ($epochs_at_level/$MAX_EPOCHS_PER_LEVEL) -> advance" | tee -a $log_file
        should_advance=1
    fi

    if [ $should_advance -eq 1 ]; then
        if [ $current_level -lt $max_level ]; then
            current_level=$((current_level + 1))
            stagnation_count=0
            epochs_at_level=0
            echo "ADVANCE to level ${LEVEL_NAMES[$current_level]}" | tee -a $log_file
            rebuild_dataset $current_level
        else
            echo "EARLY STOP at max level ${LEVEL_NAMES[$current_level]}" | tee -a $log_file
            break
        fi
    fi

    # ── Active hard negative mining ───────────────────────
    # Dès qu'on a un checkpoint, on booste les candidats mal prédits
    if [ -f checkpoint_best_multitask.pt ]; then
        echo "⛏️  Mining hard negatives actifs..." | tee -a $log_file
        python3 mine_hard_negatives.py \
            --dataset $DATA/train.adaptive.multitask.jsonl \
            --checkpoint checkpoint_best_multitask.pt \
            --model-name $MODEL \
            --device $DEVICE \
            --boost 2.5 \
            --decay 0.85 \
            --max-weight 6.0 \
            --batch-size $BS \
            2>&1 | tee -a $log_file
    fi

    resume_arg="--resume checkpoint_best_multitask.pt"
    current_epoch=$((current_epoch + 1))
done

echo "" | tee -a $log_file
echo "═══════════════════════════════════════════" | tee -a $log_file
echo "  ✅ TRAINING ADAPTATIF TERMINÉ" | tee -a $log_file
echo "  Best val score : $best_score" | tee -a $log_file
echo "  Niveau final   : ${LEVEL_NAMES[$current_level]}" | tee -a $log_file
echo "═══════════════════════════════════════════" | tee -a $log_file

