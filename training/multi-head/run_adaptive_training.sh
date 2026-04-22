#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Training adaptatif — hard negatives introduits seulement
#  quand le modèle stagne (plateau détecté sur val score)
#  Inclut les têtes SVO/voice entraînées sur le silver Stanza.
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
PATIENCE=3
MAX_EPOCHS_PER_LEVEL=5
MIN_DELTA=0.0005

# Niveaux de difficulté progressifs (6 niveaux)
LEVEL_NAMES=("easy" "easy+" "medium" "medium+" "hard" "full")
HARD_PER_GOLD=(2    3      4       5        6      6)
SOFT_FACTORS=( 1.0  1.5    2.0     2.0      2.0    2.0)

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

# ── Fusion silver (auto-détection des sources disponibles) ───────────────────
# Dataset v3 = source principale (NER gold/silver + spans UD/SVO Stanza, re-splitté stratifié)
# Format : chemin:weight  (weight=1.0 = poids normal, <1.0 = silver de moindre qualité)
SILVER_SOURCES="$DATA/train_v3.jsonl:1.0"
[ -f "$DATA/train_svo_silver.jsonl" ] && SILVER_SOURCES="$SILVER_SOURCES $DATA/train_svo_silver.jsonl:1.0"
[ -f "$DATA/train_svo_de.jsonl"     ] && SILVER_SOURCES="$SILVER_SOURCES $DATA/train_svo_de.jsonl:0.8"
[ -f "$DATA/train_svo_en.jsonl"     ] && SILVER_SOURCES="$SILVER_SOURCES $DATA/train_svo_en.jsonl:0.8"
# Ajouter d'autres sources ici au fur et à mesure

echo "📦 Fusion silver train (base: train_v3.jsonl)..." | tee -a $log_file
python3 merge_silver.py --sources $SILVER_SOURCES --out $DATA/train_svo_silver_merged.jsonl | tee -a $log_file
TRAIN_SILVER="$DATA/train_svo_silver_merged.jsonl"

# Val/test : silver Stanza si disponible (contient SVO), sinon fallback sur v3 (NER seul)
VAL_SILVER="$DATA/val_v3.jsonl"
TEST_SILVER="$DATA/test_v3.jsonl"
[ -f "$DATA/val_svo_silver.jsonl"  ] && VAL_SILVER="$DATA/val_svo_silver.jsonl"
[ -f "$DATA/test_svo_silver.jsonl" ] && TEST_SILVER="$DATA/test_svo_silver.jsonl"
echo "📊 Val  source : $VAL_SILVER"  | tee -a $log_file
echo "📊 Test source : $TEST_SILVER" | tee -a $log_file

rebuild_dataset() {
    local level=$1
    local hard=${HARD_PER_GOLD[$level]}
    local soft=${SOFT_FACTORS[$level]}
    local name=${LEVEL_NAMES[$level]}
    echo "🔧 Build dataset niveau $level ($name) — hard_per_gold=$hard soft_factor=$soft" | tee -a $log_file
    # Utiliser le silver fusionné (NER + SVO + obliques + langues supplémentaires)
    python3 build_multitask_dataset.py \
        --input  $TRAIN_SILVER \
        --output $DATA/train.adaptive.multitask.jsonl \
        --model-name $MODEL \
        --hard-per-gold $hard \
        --soft-factor $soft \
        --max-span-len 12
}

# Build val/test une seule fois
echo "📦 Build val/test datasets..." | tee -a $log_file
python3 build_multitask_dataset.py \
    --input  $VAL_SILVER \
    --output $DATA/val.multitask.jsonl \
    --model-name $MODEL \
    --hard-per-gold 6 \
    --soft-factor 2.0 \
    --max-span-len 12

python3 build_multitask_dataset.py \
    --input  $TEST_SILVER \
    --output $DATA/test.multitask.jsonl \
    --model-name $MODEL \
    --hard-per-gold 6 \
    --soft-factor 2.0 \
    --max-span-len 12

# Checkpoints
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

rebuild_dataset $current_level

# ─── BOUCLE PRINCIPALE ───────────────────────────────────
while [ $current_epoch -le $MAX_EPOCHS ]; do
    echo "" | tee -a $log_file
    echo "══════════════════════════════════════════════════" | tee -a $log_file
    echo "  Epoch $current_epoch/$MAX_EPOCHS  |  Niveau ${LEVEL_NAMES[$current_level]} (stagnation=$stagnation_count/$PATIENCE)" | tee -a $log_file
    echo "══════════════════════════════════════════════════" | tee -a $log_file

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
        --lambda-boundary 2.0 \
        --lambda-coarse 1.0 \
        --lambda-fine 1.0 \
        --lambda-svo-boundary 1.0 \
        --lambda-svo 0.5 \
        --lambda-voice 0.2 \
        --lambda-morpho 0.3 \
        --focal-gamma 0.5 \
        --device $DEVICE \
        --layer-lr-decay 0.9 \
        --ema-decay 0.999 \
        --hn-every 1 \
        --hn-boost-fp 5.0 \
        --hn-boost-fn 2.0 \
        --hn-boost-coarse 2.5 \
        --hn-boost-fine 2.0 \
        --hn-boost-fp-svo 3.0 \
        --hn-boost-fn-svo 2.0 \
        --hn-decay 0.85 \
        --hn-max-weight 8.0 \
        --hn-min-weight 0.3 \
        $resume_arg \
        2>&1 | tee $epoch_log

    # Score = grep sur la ligne Val avec Score=
    val_score=$(grep "Score=" $epoch_log | tail -1 | grep -oE "Score=[0-9.]+" | cut -d= -f2)

    if [ -z "$val_score" ]; then
        echo "⚠️  Impossible d'extraire le val score — on continue" | tee -a $log_file
        resume_arg="--resume checkpoint_best_multitask.pt"
        current_epoch=$((current_epoch + 1))
        continue
    fi

    # Log résumé SVO depuis le log epoch
    svo_f1=$(grep "SVO F1=" $epoch_log | tail -1 | grep -oE "SVO F1=[0-9.]+" | cut -d= -f2 || echo "?")
    voice_f1=$(grep "Voice F1=" $epoch_log | tail -1 | grep -oE "Voice F1=[0-9.]+" | head -1 | cut -d= -f2 || echo "?")
    gender_f1=$(grep "Gender F1=" $epoch_log | tail -1 | grep -oE "Gender F1=[0-9.]+" | cut -d= -f2 || echo "?")
    number_f1=$(grep "Number F1=" $epoch_log | tail -1 | grep -oE "Number F1=[0-9.]+" | cut -d= -f2 || echo "?")
    echo "📊 Epoch $current_epoch — Val Score=$val_score SVO_F1=$svo_f1 Voice_F1=$voice_f1 Gender_F1=$gender_f1 Number_F1=$number_f1 (best=$best_score)" | tee -a $log_file

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
    max_level=$(( ${#LEVEL_NAMES[@]} - 1 ))

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

    resume_arg=""
    if [ -f checkpoint_best_multitask.pt ]; then
        resume_arg="--resume checkpoint_best_multitask.pt"
    fi
    current_epoch=$((current_epoch + 1))
done

echo "" | tee -a $log_file
echo "═══════════════════════════════════════════" | tee -a $log_file
echo "  ✅ TRAINING ADAPTATIF TERMINÉ" | tee -a $log_file
echo "  Best val score : $best_score" | tee -a $log_file
echo "  Niveau final   : ${LEVEL_NAMES[$current_level]}" | tee -a $log_file
echo "═══════════════════════════════════════════" | tee -a $log_file

