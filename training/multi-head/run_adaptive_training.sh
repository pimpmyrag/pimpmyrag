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
MAX_EPOCHS=${MAX_EPOCHS:-60}
PATIENCE=${PATIENCE:-5}
MAX_EPOCHS_PER_LEVEL=${MAX_EPOCHS_PER_LEVEL:-12}
MIN_DELTA=${MIN_DELTA:-0.0003}

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
# RETOUR AUX VALEURS v8.0 : les valeurs v8.1 (+17.6%) causaient régression SVO -13%
L_SVO_BOUNDARY=0.595  # Boundary SVO (silver → moins critique) [v8.1: 0.7]
L_SVO=0.51            # Labels SVO (svo_verb/subject/object/iobj/tcomp/lcomp/cause/attr…) [v8.1: 0.6]
L_ROLE=0.6            # Rôles SVO (INCHANGÉ)
L_VOICE=0.1275        # Voix active/passive (très silver) [v8.1: 0.15]
L_CERTAINTY=0.4       # Certainty active/hypo/etc. (silver) (INCHANGÉ)
L_MORPHO=0.17         # Gender/Number/Person (silver) — valeur v8.0 référence pour isoler torch 2.6
L_VERB_PTR=0.2125     # Pointer head verbe gouverneur (silver) [v8.1: 0.25]

# ── Ramp SVO linéaire sur epochs (v8.1) ──────────────────────────────────────
# CHANGEMENT v8.1 : Rampup linéaire fixe au lieu de rampup par phases/niveaux.
# Justification : Le score NER est déjà bon à epoch 10-15, pas besoin d'attendre
# 40+ epochs pour commencer SVO sérieusement. Rampup plus rapide = meilleur SVO.
#   Epochs 1-6   : warmup NER (λ_SVO = 0)
#   Epochs 7-20  : rampup linéaire SVO (0 → 1.0)
#   Epochs 21+   : plein régime (λ_SVO = 1.0)
# Cette logique remplace SVO_RAMP_PCT (désormais ignoré, gardé pour compatibilité)
SVO_RAMP_PCT=(5 15 35 60 85 100)  # DEPRECATED v8.1 — ignoré, rampup linéaire utilisé
SVO_RAMPUP_START=7    # Epoch de début du rampup (après warmup NER)
SVO_RAMPUP_END=20     # Epoch où SVO atteint 100%

# Reprise: START_LEVEL=1 START_EPOCH=13 KEEP_CHECKPOINT=1 ./run_adaptive_training.sh
START_LEVEL=${START_LEVEL:-0}
START_EPOCH=${START_EPOCH:-1}
KEEP_CHECKPOINT=${KEEP_CHECKPOINT:-0}
NER_ONLY_BENCH=${NER_ONLY_BENCH:-0}
# Phase NER-only initiale : stabilise boundary/coarse/fine avant d'introduire les têtes SVO.
# Mettre à 0 pour désactiver (multitask dès le début), ou >0 pour N epochs warmup.
# Empirique : les gains NER-only (+4 pts boundary sur ep 39) se forment surtout dans les
# 5-10 premières epochs — après, le plateau NER est atteint.
# Pendant le warmup : SVO=0, stagnation/level ignorés, checkpoint NER sauvegardé séparément.
NER_WARMUP_EPOCHS=${NER_WARMUP_EPOCHS:-6}

current_level=$START_LEVEL
stagnation_count=0
epochs_at_level=0
best_score=-1.0
current_epoch=$START_EPOCH
resume_arg=""

mkdir -p logs
log_file="logs/adaptive.log"
echo "🚀 Démarrage training adaptatif — $(date)" | tee $log_file
if [ "$NER_ONLY_BENCH" != "1" ] && [ "$NER_WARMUP_EPOCHS" -gt 0 ]; then
    echo "🏋️  Phase NER warmup : $NER_WARMUP_EPOCHS epochs sans SVO avant multitask" | tee -a $log_file
fi

if [ "$NER_ONLY_BENCH" = "1" ]; then
    echo "🧪 Mode NER-only benchmark activé" | tee -a $log_file
    L_SVO_BOUNDARY=0.0
    L_SVO=0.0
    L_ROLE=0.0
    L_VOICE=0.0
    L_CERTAINTY=0.0
    L_MORPHO=0.0
    L_VERB_PTR=0.0
fi

# ── Sources gold v8.1 (TEST: torch 2.6+ avec dataset v8.1) ───────────────────
# v8.0 : hint_quantity supprimé (→ hint_measure comme fallback), 38 fine labels
# v8.1 : morpho Stanza+gender_guesser sur tous les spans, re-annotation hint_person_name,
#         nettoyage hint_measure/count/rate/percentage, fragments adjec. supprimés,
#         hint_object_generic documents → hint_document
# TEST: v8.1 dataset + torch 2.6+ (comme v8.0 réussi) pour isoler cause régression
TRAIN_SILVER="$DATA/train_v8.1.jsonl"
VAL_SILVER="$DATA/val_v8.1.jsonl"
TEST_SILVER="$DATA/test_v8.1.jsonl"

# Vérification présence des fichiers gold
for f in "$TRAIN_SILVER" "$VAL_SILVER" "$TEST_SILVER"; do
    if [ ! -f "$f" ]; then
        echo "❌ Fichier gold manquant : $f"
        echo "   → Upload via : scp <local_path> root@<RUNPOD_IP>:<workspace>/data/"
        exit 1
    fi
done
echo "📦 Source gold (DVC) : $TRAIN_SILVER ($(wc -l < "$TRAIN_SILVER") phrases)" | tee -a $log_file
echo "📊 Val  source    : $VAL_SILVER  ($(wc -l < "$VAL_SILVER") phrases)"    | tee -a $log_file
echo "📊 Test source    : $TEST_SILVER ($(wc -l < "$TEST_SILVER") phrases)"    | tee -a $log_file

# ── Nom du run W&B — lisible et traçable ─────────────────────────────────────
# Format : v6.3-deberta-bs160-RTX_5090-0503-1430
TORCH_SHORT=$(python3 -c "import torch; v=torch.__version__.split('+')[0]; print('t'+''.join(v.split('.')[:2]))" 2>/dev/null || echo "t26")
DATASET_VERSION="v8.1-${TORCH_SHORT}"  # dataset v8.1 + version torch dynamique (test isolation torch)
GPU_SHORT=$(python3 -c "import torch; n=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'; print(n.replace('NVIDIA GeForce ','').replace(' ','_'))" 2>/dev/null || echo "gpu")
WANDB_RUN_NAME="${DATASET_VERSION}-deberta-bs${BS}-${GPU_SHORT}-$(date +%m%d-%H%M)"
WANDB_TAGS="${DATASET_VERSION},deberta-v3,fp32,adaptive"
if [ "$NER_ONLY_BENCH" = "1" ]; then
    WANDB_RUN_NAME="${WANDB_RUN_NAME}-neronly"
    WANDB_TAGS="${WANDB_TAGS},ner-only"
fi
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
        if [ "$CKPT_FINE" != "38" ] && [ "$CKPT_FINE" != "0" ]; then
            echo "⚠️  Checkpoint incompatible : fine_head=$CKPT_FINE classes (attendu 38 — labels v8.0 sans hint_quantity)" | tee -a $log_file
            echo "   → Démarrage à froid (v7.0 ont fine=39, v6.9 ont fine=42, v6.6 ont fine=36)" | tee -a $log_file
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

    # ── Calcul des lambdas SVO (v8.1 : rampup linéaire) ───────────────────────
    # Phase 1 : NER warmup (current_epoch <= NER_WARMUP_EPOCHS)
    #   → SVO = 0, têtes SVO non entraînées, gradient NER pur
    # Phase 2 : rampup linéaire SVO (epochs 7-20)
    #   → SVO progress = (epoch - 6) / 14.0 (0.0 → 1.0)
    # Phase 3 : plein régime (epochs 21+)
    #   → SVO progress = 1.0
    in_warmup=0
    if [ "$NER_ONLY_BENCH" != "1" ] && [ "$NER_WARMUP_EPOCHS" -gt 0 ] \
       && [ $current_epoch -le $NER_WARMUP_EPOCHS ]; then
        in_warmup=1
        L_SVO_B_NOW=0.0000
        L_SVO_NOW=0.0000
        L_ROLE_NOW=0.0000
        L_VOICE_NOW=0.0000
        L_CERTAINTY_NOW=0.0000
        L_MORPHO_NOW=0.0000
        L_VPTR_NOW=0.0000
        echo "🏋️  Warmup NER-only ep $current_epoch/$NER_WARMUP_EPOCHS — SVO=0" | tee -a $log_file
        echo "      NER  : boundary=$L_BOUNDARY  coarse=$L_COARSE  fine=$L_FINE" | tee -a $log_file
    else
        # Phase multitask : rampup SVO linéaire sur epochs (v8.1)
        if [ $in_warmup -eq 0 ] && [ $current_epoch -eq $((NER_WARMUP_EPOCHS + 1)) ] \
           && [ "$NER_WARMUP_EPOCHS" -gt 0 ]; then
            echo "🚀 Fin warmup NER → démarrage multitask avec rampup linéaire SVO" | tee -a $log_file
            # Réinitialiser stagnation/niveau pour repartir proprement en multitask
            stagnation_count=0
            epochs_at_level=0
            best_score=-1.0
        fi

        # Nouveau rampup linéaire v8.1 (remplace le rampup par niveau)
        if [ $current_epoch -le $SVO_RAMPUP_START ]; then
            # Warmup NER terminé mais pas encore rampup SVO (normalement pas possible)
            svo_progress=0.0
        elif [ $current_epoch -le $SVO_RAMPUP_END ]; then
            # Rampup linéaire : epochs 7-20 → 0.0 à 1.0
            svo_progress=$(python3 -c "print(min(1.0, max(0.0, ($current_epoch - $SVO_RAMPUP_START) / ($SVO_RAMPUP_END - $SVO_RAMPUP_START))))")
        else
            # Plein régime : epochs 21+
            svo_progress=1.0
        fi

        svo_pct=$(python3 -c "print(int($svo_progress * 100))")
        L_SVO_B_NOW=$(python3 -c "print(f'{$L_SVO_BOUNDARY * $svo_progress:.4f}')")
        L_SVO_NOW=$(python3   -c "print(f'{$L_SVO        * $svo_progress:.4f}')")
        L_ROLE_NOW=$(python3  -c "print(f'{$L_ROLE       * $svo_progress:.4f}')")
        L_VOICE_NOW=$(python3 -c "print(f'{$L_VOICE      * $svo_progress:.4f}')")
        L_CERTAINTY_NOW=$(python3 -c "print(f'{$L_CERTAINTY * $svo_progress:.4f}')")
        L_MORPHO_NOW=$(python3 -c "print(f'{$L_MORPHO    * $svo_progress:.4f}')")
        L_VPTR_NOW=$(python3  -c "print(f'{$L_VERB_PTR   * $svo_progress:.4f}')")

        echo "🎛️  Lambdas multitask (SVO ramp linéaire: ${svo_pct}% — epoch $current_epoch)" | tee -a $log_file
        echo "      NER  : boundary=$L_BOUNDARY  coarse=$L_COARSE  fine=$L_FINE" | tee -a $log_file
        echo "      SVO  : svo_boundary=$L_SVO_B_NOW  svo=$L_SVO_NOW  role=$L_ROLE_NOW  voice=$L_VOICE_NOW  certainty=$L_CERTAINTY_NOW  morpho=$L_MORPHO_NOW  verb_ptr=$L_VPTR_NOW" | tee -a $log_file
    fi

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
        --lr 8e-6 \
        --head-lr-multiplier 4.0 \
        --warmup-epochs 0 \
        --max-grad-norm 1.0 \
        --lambda-boundary   $L_BOUNDARY \
        --lambda-coarse     $L_COARSE \
        --lambda-fine       $L_FINE \
        --lambda-svo-boundary $L_SVO_B_NOW \
        --lambda-svo        $L_SVO_NOW \
        --lambda-role       $L_ROLE_NOW \
        --lambda-voice      $L_VOICE_NOW \
        --lambda-certainty  $L_CERTAINTY_NOW \
        --lambda-morpho     $L_MORPHO_NOW \
        --lambda-verb-ptr   $L_VPTR_NOW \
        --lambda-compat     $( [ "$NER_ONLY_BENCH" = "1" ] && echo "0.0" || echo "0.2" ) \
        --focal-gamma 0.5 \
        --device $DEVICE \
        --layer-lr-decay 0.9 \
        --ema-decay 0.999 \
        --hn-every 1 \
        --hn-boost-fp 5.0 \
        --hn-boost-fn 2.0 \
        --hn-boost-coarse 2.5 \
        --hn-boost-fine 2.0 \
        --hn-boost-fp-svo $( [ "$NER_ONLY_BENCH" = "1" ] || [ "$in_warmup" = "1" ] && echo "0.0" || echo "3.0" ) \
        --hn-boost-fn-svo $( [ "$NER_ONLY_BENCH" = "1" ] || [ "$in_warmup" = "1" ] && echo "0.0" || echo "2.0" ) \
        --hn-decay 0.85 \
        --hn-max-weight 8.0 \
        --hn-min-weight 0.3 \
        --num-workers $NUM_WORKERS \
        $( [ "$NER_ONLY_BENCH" = "1" ] && echo "--ner-only-score" ) \
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

    # Pendant le warmup NER : ne pas toucher au niveau ni à la stagnation
    # (le score NER-only ne reflète pas encore le régime multitask final)
    if [ "$in_warmup" = "1" ]; then
        resume_arg=""
        [ -f checkpoint_best_multitask.pt ] && resume_arg="--resume checkpoint_best_multitask.pt"
        current_epoch=$((current_epoch + 1))
        continue
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

# ── Sauvegarde automatique du best model (W&B artifact + R2 direct) ──────────
# On n'utilise PAS git push depuis le pod (pas de credentials write).
# Upload binaire direct vers R2 via aws s3 cp, + W&B artifact si clé dispo.
if [ -f checkpoint_best_multitask.pt ]; then
    echo "" | tee -a $log_file
    echo "💾 Upload best model vers R2 + W&B artifact..." | tee -a $log_file
    cp checkpoint_best_multitask.pt best_model_multitask.pt

    # ── Upload R2 via boto3 (préinstallé dans l'image, pas besoin d'awscli) ──────
    R2_ENDPOINT="${DVC_R2_ENDPOINT:-https://07027fdcb4c08fe1418a9595986c3ac8.r2.cloudflarestorage.com}"
    R2_PREFIX="models/${WANDB_RUN_NAME}"
    if [ -n "$AWS_ACCESS_KEY_ID" ]; then
        echo "   → boto3 upload vers R2 : $R2_PREFIX/" | tee -a $log_file
        python3 - <<PYEOF 2>&1 | tee -a $log_file || echo "⚠ R2 upload échoué" | tee -a $log_file
import os, boto3
s3 = boto3.client("s3",
    endpoint_url=os.environ.get("DVC_R2_ENDPOINT","https://07027fdcb4c08fe1418a9595986c3ac8.r2.cloudflarestorage.com"),
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)
for fname, key in [
    ("checkpoint_best_multitask.pt", "$R2_PREFIX/checkpoint_best_multitask.pt"),
    ("best_model_multitask.pt",      "$R2_PREFIX/best_model_multitask.pt"),
]:
    if os.path.exists(fname):
        sz = os.path.getsize(fname)/1024**2
        print(f"  uploading {fname} ({sz:.0f} MB)...")
        s3.upload_file(fname, "pimpmyrag-data", key)
        print(f"  OK s3://pimpmyrag-data/{key}")
PYEOF
        echo "   ✅ Upload R2 OK" | tee -a $log_file
        echo "   Récupération : python3 -c \"import boto3; boto3.client('s3',endpoint_url='$R2_ENDPOINT',...).download_file('pimpmyrag-data','$R2_PREFIX/checkpoint_best_multitask.pt','ckpt.pt')\"" | tee -a $log_file
    else
        echo "⚠ AWS_ACCESS_KEY_ID absent — R2 upload ignoré" | tee -a $log_file
    fi

    # ── W&B artifact ─────────────────────────────────────────────────────────
    if [ -n "$WANDB_API_KEY" ]; then
        python3 - <<PYEOF 2>&1 | tee -a $log_file || echo "⚠ W&B artifact log échoué"
import os, wandb
run_id_file = "wandb_run_id.txt"
run_id = open(run_id_file).read().strip() if os.path.exists(run_id_file) else None
api = wandb.Api(api_key=os.environ["WANDB_API_KEY"])
run = api.run(f"pimpmyrag-ner/{run_id}") if run_id else \
      next((r for r in api.runs("pimpmyrag-ner", order="-created_at")
            if r.state in ("running","finished")), None)
if run is None:
    print("⚠ Aucun run W&B trouvé — artifact non loggé")
else:
    art = wandb.Artifact(
        "pimpmyrag-ner-model", type="model",
        description=f"multitask best model — run=${run.name} score=$best_score",
        metadata={"run_name": "$WANDB_RUN_NAME", "best_score": $best_score},
    )
    for fname in ("checkpoint_best_multitask.pt", "best_model_multitask.pt"):
        if os.path.exists(fname):
            art.add_file(fname)
            print(f"  added {fname}")
    run.log_artifact(art)
    print(f"✅ W&B artifact loggé sur run {run.name}")
PYEOF
    else
        echo "⚠ WANDB_API_KEY absent — W&B artifact ignoré" | tee -a $log_file
    fi

    echo "✅ Best model sauvegardé (R2 + W&B artifact)" | tee -a $log_file
    echo "   Récupération locale : aws s3 cp ${R2_DEST}checkpoint_best_multitask.pt . --endpoint-url $R2_ENDPOINT" | tee -a $log_file
else
    echo "⚠ Aucun checkpoint_best_multitask.pt trouvé" | tee -a $log_file
fi

