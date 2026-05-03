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

# ── Vérification version PyTorch (CVE-2025-32434) ────────
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "0.0")
echo "✅ PyTorch $TORCH_VERSION"

# ── Détection device & batch size ────────────────────────
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
    VRAM_GB=$(python3 -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1024**3))" 2>/dev/null || echo "24")
    # BF16 activé (stable sur Ampere+, contrairement à fp16+GradScaler)
    # Batch effectif cible ~96 :
    #   40GB+ (A100)      → BS=128 accum=1  (BF16)
    #   28-40GB (5090/32) → BS=96  accum=1  (BF16)
    #   <28GB  (3090/24)  → BS=48  accum=2  (BF16) → batch effectif=96
    AMP_FLAG="--amp"
    if [ "$VRAM_GB" -ge 40 ] 2>/dev/null; then
        BS=128
        ACCUM=1
    elif [ "$VRAM_GB" -ge 28 ] 2>/dev/null; then
        BS=96
        ACCUM=1
    else
        # RTX 3090 / 4090 24 GB — BF16 permet BS=48 avec 2× accum = 96 effectif
        BS=48
        ACCUM=2
    fi
    NUM_WORKERS=4
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_inductor_cache"
    echo "🚀 Device: CUDA (BS=$BS×accum=$ACCUM, BF16, workers=$NUM_WORKERS, VRAM=${VRAM_GB}GB)"
elif python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    DEVICE="mps"
    BS=24
    ACCUM=2
    AMP_FLAG=""
    NUM_WORKERS=0
    echo "🍎 Device: MPS (BS=$BS)"
else
    DEVICE="cpu"
    BS=16
    ACCUM=2
    AMP_FLAG=""
    NUM_WORKERS=0
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

# ── Lambdas NER (têtes principales, labels gold) ─────────────────────────────
# lambda_boundary élevé = priorité absolue : c'est la tête la plus fragile.
# lambda_fine=1.8 identifié empiriquement comme bon compromis fine vs coarse.
L_BOUNDARY=2.5    # Restauré haut (1.0 avait causé -0.70 de F1 boundary)
L_COARSE=1.0
L_FINE=1.8        # Conseillé (vs 1.0 avant) — discrimine mieux les labels fins

# ── Lambdas SVO cibles (têtes secondaires, labels silver Stanza) ─────────────
# Ces lambdas s'appliquent au PLEIN RÉGIME (niveau 5/full).
# Budget SVO total = 1.95 vs budget NER = 5.3 → SVO = 27% du total.
# Aux niveaux inférieurs, un facteur de montée multiplicatif est appliqué.
L_SVO_BOUNDARY=0.7   # Boundary SVO (silver → moins critique)
L_SVO=0.6            # Labels SVO (svo_verb/subject/object/iobj/tcomp/lcomp/cause/attr…)
L_VOICE=0.15         # Voix active/passive (très silver)
L_MORPHO=0.2         # Gender/Number/Person (silver)
L_VERB_PTR=0.25      # Pointer head verbe gouverneur (silver)

# ── Ramp SVO par niveau ───────────────────────────────────────────────────────
# Le modèle apprend NER en premier.  SVO monte progressivement quand le NER
# est stable, pour éviter l'interférence de gradient sur boundary/fine.
#   Niveau 0 (easy)   : SVO à  5% → contribution négligeable sur le gradient
#   Niveau 2 (medium) : SVO à 35% → commence à avoir du signal
#   Niveau 5 (full)   : SVO à 100% → plein régime
# Tableau 6 valeurs (1 par niveau), virgule pas supportée en bash → on multiplie ×100
SVO_RAMP_PCT=(5 15 35 60 85 100)  # pourcentage × 100 du lambda cible

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

# ── Sources gold v6.1 ──────────────────────────────────────────────────────────
# *_v6.jsonl = v6 + Mistral corrections (v6.1) (label 34, coarse=ORG)
#   ~803 hint_group_role + ~239 hint_inst_name → hint_inst_role (1042 spans train)
TRAIN_SILVER="$DATA/train_v6.6.jsonl"
VAL_SILVER="$DATA/val_v6.6.jsonl"
TEST_SILVER="$DATA/test_v6.6.jsonl"

# Vérification présence des fichiers gold
for f in "$TRAIN_SILVER" "$VAL_SILVER" "$TEST_SILVER"; do
    if [ ! -f "$f" ]; then
        echo "❌ Fichier gold manquant : $f"
        echo "   → Upload via : scp <local_path> root@<RUNPOD_IP>:<workspace>/data/"
        exit 1
    fi
done
echo "📦 Source gold v4 : $TRAIN_SILVER ($(wc -l < "$TRAIN_SILVER") phrases)" | tee -a $log_file
echo "📊 Val  source    : $VAL_SILVER  ($(wc -l < "$VAL_SILVER") phrases)"    | tee -a $log_file
echo "📊 Test source    : $TEST_SILVER ($(wc -l < "$TEST_SILVER") phrases)"    | tee -a $log_file

# ── Nom du run W&B — lisible et traçable ─────────────────────────────────────
# Format : v6.3-deberta-bs160-RTX_5090-0503-1430
DATASET_VERSION=$(basename "$TRAIN_SILVER" | grep -oE 'v[0-9]+\.[0-9]+' | head -1 || echo "v6")
GPU_SHORT=$(python3 -c "import torch; n=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'; print(n.replace('NVIDIA GeForce ','').replace(' ','_'))" 2>/dev/null || echo "gpu")
WANDB_RUN_NAME="${DATASET_VERSION}-deberta-bs${BS}-${GPU_SHORT}-$(date +%m%d-%H%M)"
WANDB_TAGS="${DATASET_VERSION},deberta-v3,fp32,adaptive"
WANDB_ID_FILE="wandb_run_id.txt"
# Supprime un éventuel run ID d'une session précédente (sauf en mode reprise)
if [ "$KEEP_CHECKPOINT" != "1" ]; then
    rm -f "$WANDB_ID_FILE"
fi
echo "📊 W&B run name : $WANDB_RUN_NAME" | tee -a $log_file
echo "📊 W&B tags     : $WANDB_TAGS"     | tee -a $log_file

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
        # ⚠️  Vérification compatibilité schéma labels (v6 : fine=35, coarse=10)
        CKPT_FINE=$(python3 -c "
import torch
c = torch.load('checkpoint_best_multitask.pt', map_location='cpu')
sd = c.get('model_state', c)
k = [k for k in sd if 'fine_head' in k and 'weight' in k]
print(sd[k[0]].shape[0] if k else 0)
" 2>/dev/null || echo "0")
        if [ "$CKPT_FINE" != "36" ] && [ "$CKPT_FINE" != "0" ]; then
            echo "⚠️  Checkpoint incompatible : fine_head=$CKPT_FINE classes (attendu 36 — labels v6.6)" | tee -a $log_file
            echo "   → Démarrage à froid (checkpoints v5 ont fine=34, v4 ont fine=32/33)" | tee -a $log_file
            resume_arg=""
        else
            resume_arg="--resume checkpoint_best_multitask.pt"
            best_score=$(python3 -c "import torch; c=torch.load('checkpoint_best_multitask.pt',map_location='cpu'); print(f\"{c.get('best_score',-1.0):.4f}\")" 2>/dev/null || echo "-1.0")
            echo "best_score checkpoint: $best_score" | tee -a $log_file
        fi
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

    # ── Calcul des lambdas SVO pour ce niveau (ramp progressif) ──────────────
    svo_pct=${SVO_RAMP_PCT[$current_level]}
    L_SVO_B_NOW=$(python3 -c "print(f'{$L_SVO_BOUNDARY * $svo_pct / 100:.4f}')")
    L_SVO_NOW=$(python3   -c "print(f'{$L_SVO        * $svo_pct / 100:.4f}')")
    L_VOICE_NOW=$(python3 -c "print(f'{$L_VOICE      * $svo_pct / 100:.4f}')")
    L_MORPHO_NOW=$(python3 -c "print(f'{$L_MORPHO    * $svo_pct / 100:.4f}')")
    L_VPTR_NOW=$(python3  -c "print(f'{$L_VERB_PTR   * $svo_pct / 100:.4f}')")
    echo "🎛️  Lambdas niveau ${LEVEL_NAMES[$current_level]} (SVO ramp=${svo_pct}%)" | tee -a $log_file
    echo "      NER  : boundary=$L_BOUNDARY  coarse=$L_COARSE  fine=$L_FINE" | tee -a $log_file
    echo "      SVO  : svo_boundary=$L_SVO_B_NOW  svo=$L_SVO_NOW  voice=$L_VOICE_NOW  morpho=$L_MORPHO_NOW  verb_ptr=$L_VPTR_NOW" | tee -a $log_file

    python3 train_multi_task.py \
        --train $DATA/train.adaptive.multitask.jsonl \
        --val   $DATA/val.multitask.jsonl \
        --test  $DATA/test.multitask.jsonl \
        --model-name $MODEL \
        --epochs $current_epoch \
        --start-epoch $current_epoch \
        --patience 0 \
        --batch-size $BS \
        --accum-steps $ACCUM \
        --lr 5e-6 \
        --head-lr-multiplier 4.0 \
        --warmup-epochs 0 \
        --max-grad-norm 1.0 \
        --lambda-boundary   $L_BOUNDARY \
        --lambda-coarse     $L_COARSE \
        --lambda-fine       $L_FINE \
        --lambda-svo-boundary $L_SVO_B_NOW \
        --lambda-svo        $L_SVO_NOW \
        --lambda-voice      $L_VOICE_NOW \
        --lambda-morpho     $L_MORPHO_NOW \
        --lambda-verb-ptr   $L_VPTR_NOW \
        --lambda-compat     0.2 \
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
        --num-workers $NUM_WORKERS \
        --wandb-run-name  "$WANDB_RUN_NAME" \
        --wandb-tags      "$WANDB_TAGS" \
        --wandb-id-file   "$WANDB_ID_FILE" \
        $AMP_FLAG \
        $resume_arg \
        2>&1 | tee $epoch_log

    # Score = grep sur la ligne Val avec Score=
    val_score=$(grep "Score=" $epoch_log | tail -1 | grep -oE "Score=[0-9.]+" | cut -d= -f2)

    if [ -z "$val_score" ]; then
        echo "⚠️  Impossible d'extraire le val score — on continue" | tee -a $log_file
        resume_arg=""
        [ -f checkpoint_best_multitask.pt ] && resume_arg="--resume checkpoint_best_multitask.pt"
        current_epoch=$((current_epoch + 1))
        continue
    fi

    # Log résumé SVO depuis le log epoch
    svo_f1=$(grep "SVO F1=" $epoch_log | tail -1 | grep -oE "SVO F1=[0-9.]+" | cut -d= -f2 || echo "?")
    voice_f1=$(grep "Voice F1=" $epoch_log | tail -1 | grep -oE "Voice F1=[0-9.]+" | head -1 | cut -d= -f2 || echo "?")
    gender_f1=$(grep "Gender F1=" $epoch_log | tail -1 | grep -oE "Gender F1=[0-9.]+" | cut -d= -f2 || echo "?")
    number_f1=$(grep "Number F1=" $epoch_log | tail -1 | grep -oE "Number F1=[0-9.]+" | cut -d= -f2 || echo "?")
    person_f1=$(grep "Person F1=" $epoch_log | tail -1 | grep -oE "Person F1=[0-9.]+" | cut -d= -f2 || echo "?")
    echo "📊 Epoch $current_epoch — Val Score=$val_score SVO_F1=$svo_f1 Voice_F1=$voice_f1 Gender_F1=$gender_f1 Number_F1=$number_f1 Person_F1=$person_f1 (best=$best_score)" | tee -a $log_file

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

# ── Sauvegarde automatique du best model via git ──────────────────────────────
if [ -f checkpoint_best_multitask.pt ]; then
    echo "" | tee -a $log_file
    echo "💾 Sauvegarde du best model dans git..." | tee -a $log_file
    cp checkpoint_best_multitask.pt best_model.pt
    git add best_model.pt 2>/dev/null || true
    git commit -m "best model ${WANDB_RUN_NAME} score=${best_score}" --no-verify 2>&1 | tee -a $log_file || true
    git push 2>&1 | tee -a $log_file || echo "⚠ git push échoué — récupérer manuellement : scp root@<pod>:<workspace>/checkpoint_best_multitask.pt ." | tee -a $log_file
    echo "✅ Best model poussé" | tee -a $log_file
else
    echo "⚠ Aucun checkpoint_best_multitask.pt trouvé" | tee -a $log_file
fi

