#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Training adaptatif — hard negatives introduits seulement
#  quand le modèle stagne (plateau détecté sur val score)
#  Inclut les têtes SVO/voice entraînées sur le silver Stanza.
# ═══════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔑  SEUL ENDROIT À CHANGER POUR UPGRADER LE DATASET
GOLD_VERSION="${GOLD_VERSION:-v8.8}"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
    # Batch effectif cible ~96, optimizer steps/epoch cible ~312 :
    #   40GB+ (A100/A40)  → BS=96  accum=1  (BF16) → 312 steps/ep  (was BS=128 → 234 steps, -25%)
    #   28-40GB (5090/32) → BS=96  accum=1  (BF16)
    #   <28GB  (3090/24)  → BS=48  accum=2  (BF16) → batch effectif=96, 312 steps/ep
    AMP_FLAG="--amp"
    if [ "$VRAM_GB" -ge 40 ] 2>/dev/null; then
        BS=96
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
MAX_EPOCHS_PER_LEVEL=${MAX_EPOCHS_PER_LEVEL:-12}   # hard negatives : adaptatif (boundary-driven)
MIN_DELTA=${MIN_DELTA:-0.0003}
# Détection plateau boundary par fenêtre glissante (évite les faux-reset par micro-améliorations)
BOUNDARY_WINDOW=${BOUNDARY_WINDOW:-5}              # nb epochs de la fenêtre glissante boundary
BOUNDARY_WINDOW_DELTA=${BOUNDARY_WINDOW_DELTA:-0.005}  # progrès net minimal requis sur la fenêtre (0.5%)
SVO_RAMP_EPOCHS=${SVO_RAMP_EPOCHS:-20}             # utilisé uniquement pour la ramp role (ROLE_DELAY → 100% en SVO_RAMP_EPOCHS epochs)
MORPHO_RAMP_EPOCHS=${MORPHO_RAMP_EPOCHS:-20}       # Morpho : ramp sur 20 epochs multitask
MORPHO_DELAY=${MORPHO_DELAY:-8}                    # Morpho démarre 8 epochs après fin warmup NER
# ── Trigger SVO basé sur les métriques (v8.8+) ──────────────────────────────
# SVO ne démarre que quand boundary > thr_bnd ET coarse > thr_coarse → NER stable.
# fine exclu du trigger : monte trop lentement (ep 20-25), ferait attendre inutilement.
# Seuils calibrés sur les bons runs v8.0/v8.1 : trigger naturel vers ep 8-12.
# Pas de délai fixe, pas de ramp linéaire : trigger organique + +20% tous les N epochs.
SVO_TRIGGER_BND=${SVO_TRIGGER_BND:-0.76}               # seuil boundary : reflète stabilité NER de base
SVO_TRIGGER_COARSE=${SVO_TRIGGER_COARSE:-0.87}         # seuil coarse  : confirme que l'encodeur est ancré
SVO_TRIGGER_STEP_EPOCHS=${SVO_TRIGGER_STEP_EPOCHS:-5}  # +20% SVO tous les N epochs après trigger (0→20→40→60→80→100)

# ── NER rescue : si boundary stagne sous la cible, réduit les lambdas concurrents ──
# Déclenché une seule fois quand boundary n'a pas progressé de DELTA sur WINDOW epochs
# ET est encore sous TARGET. Réduit L_VERB_PTR et L_VOICE ×FACTOR pour libérer du gradient NER.
BOUNDARY_RESCUE_WINDOW=${BOUNDARY_RESCUE_WINDOW:-10}    # epochs de fenêtre pour détecter le plateau
BOUNDARY_RESCUE_TARGET=${BOUNDARY_RESCUE_TARGET:-0.90}  # seuil cible : ne rescuer que si boundary < 0.90
BOUNDARY_RESCUE_DELTA=${BOUNDARY_RESCUE_DELTA:-0.003}   # gain minimal requis sur la fenêtre (0.3%)
BOUNDARY_RESCUE_FACTOR=${BOUNDARY_RESCUE_FACTOR:-0.50}  # facteur de réduction (×0.5 sur verb_ptr et voice)

# ── Dynamic loss weighting (uncertainty / gradnorm) ──────────────────
LOSS_WEIGHTING=${LOSS_WEIGHTING:-fixed}  # fixed | uncertainty | gradnorm

# Niveaux de difficulté progressifs (6 niveaux)
LEVEL_NAMES=("easy" "easy+" "medium" "medium+" "hard" "full")
HARD_PER_GOLD=(2    2      3       4        5      6)
# v8.3 avait réduit easy: 1.0→0.5 pour limiter compute, mais créait un gap ×4 avec le val
# (val toujours à soft=2.0) : modèle ne voyait pas assez de négatifs proches des gold en easy
# → boundary plafonnait à 0.77 sans pouvoir apprendre la discrimination fine requise par val.
# Retour à 1.0 (identique v8.1 qui atteignait 0.927) — gap ×2 au lieu de ×4.
# Compute : ~+30% à niveau easy vs 0.5 (acceptable, idem v8.1 sur 3090).
SOFT_FACTORS=( 1.0  1.25   1.5     2.0      2.0    2.0)

# ── Lambdas NER (têtes principales, labels gold) ─────────────────────────────
# lambda_boundary élevé = priorité absolue : c'est la tête la plus fragile.
# lambda_fine=1.8 identifié empiriquement comme bon compromis fine vs coarse.
L_BOUNDARY=2.5    # Restauré haut (1.0 avait causé -0.70 de F1 boundary)
L_COARSE=1.0
L_FINE=1.8        # Retour valeur v8.0 — 2.2 + focal_fine volait du budget à boundary
# FOCAL_FINE_GAMMA=1.0 : focal loss sur tête fine — down-weight easy (person_name@98%)
# up-weight hard (hint_doctrine@60%). 1.0 au lieu de 1.5 : down-weight moins agressif,
# les classes faciles contribuent encore à la moyenne → convergence plus rapide.
# 1.5 → convergence lente (−8pts à ep14 vs v8.0), 1.0 → meilleur équilibre.
FOCAL_FINE_GAMMA=1.0
# FOCAL_COARSE_GAMMA=0.0 : désactivé — causait régression boundary -5pts via encodeur partagé
# (EVENT/OBJECT bénéficiaient mais boundary payait le prix → net négatif)
FOCAL_COARSE_GAMMA=0.0

# ── Lambdas SVO cibles (têtes secondaires, labels silver Stanza) ─────────────
# Ces lambdas s'appliquent au PLEIN RÉGIME (niveau 5/full).
# v8.3 : réduction SVO vs v8.2 pour équilibre NER/SVO après analyse v8.2.
# v8.2 (L_SVO=0.51, L_SVO_BOUNDARY=0.595) → SVO=0.813 mais NER plafonne à 0.827.
# v8.3 : L_SVO=0.40 (-22%), L_SVO_BOUNDARY=0.45 (-24%) → cible SVO ~0.75-0.78, NER ~0.845+
# Budget SVO total = 1.50 vs budget NER = 5.3 → SVO = 22% du total (vs 27% en v8.2).
L_SVO_BOUNDARY=0.50   # Boundary SVO (silver) — réduit vs v8.2 (0.595) pour ne pas écraser NER
L_SVO=0.50            # Labels SVO (syn labels) — réduit vs v8.2 (0.51) pour équilibre NER/SVO
L_ROLE=0.35           # Rôles SVO — réduit 0.6→0.25 : role fix (ffc3210) a ajouté ~200k spans/ep, gradient trop fort → régression boundary
L_VOICE=0.13        # Retour valeur v8.0 (0.20 trop fort → siphonnait gradient NER boundary)
L_CERTAINTY=0.05       # Certainty active/hypo/etc. (silver) (INCHANGÉ)
L_MORPHO=0.10         # Gender/Number/Person — calibré pour coverage v8.1 (77% vs 43% v8.0 → gradient ×1.8x)
L_VERB_PTR=0.20       # Retour valeur v8.0 (0.5 → plateau boundary à 0.872 confirmé par analyse config)
ROLE_DELAY=${ROLE_DELAY:-12}  # Role démarre 12 epochs après fin warmup NER (v8.5: +4 vs v8.4c=8 car APPOS ×6 → gradient rôle fort)

# Reprise: START_LEVEL=1 START_EPOCH=13 KEEP_CHECKPOINT=1 ./run_adaptive_training.sh
START_LEVEL=${START_LEVEL:-0}    # 0=easy — ramp SVO progressif par niveau (5%→15%→35%→60%→85%→100%)
START_EPOCH=${START_EPOCH:-1}
KEEP_CHECKPOINT=${KEEP_CHECKPOINT:-0}
NER_ONLY_BENCH=${NER_ONLY_BENCH:-0}
# Phase NER-only initiale : stabilise boundary/coarse/fine avant d'introduire les têtes SVO.
# Mettre à 0 pour désactiver (multitask dès le début), ou >0 pour N epochs warmup.
# Empirique : sans warmup (nowarmup), fine_f1 plateau projeté ~0.76 (ratio décélération 0.67).
#             avec warmup 6 (5l4g et v8.0), fine_f1 plateau réel 0.828+ (ratio 0.75).
#             v8.2 (warmup=0 + svoramp25) : SVO=0.813 mais NER plafonne à 0.827 dès ep 27.
#             Analyse v8.2 : la ramp SVO agressive dès ep 1 vole le budget gradient NER
#             pendant sa phase rapide (Δ>0.01/ep jusqu'à ep 12) → NER bloqué à 0.83.
# v8.3 → warmup=12 : couvre toute la phase NER rapide (Δ>0.01/ep), NER atteint ~0.75
#         avant que SVO démarre. Pas 15 (NER déjà ralenti) ni 6 (trop court).
#         SVO ramp RELATIVE warmup : ep 13 → 0%, ep 48 (13+35) → 100% — adouci vs avant (52% à ep13 !)
#         Morpho ramp : ep 21 (13+8) → 0%,  ep 46 (21+25) → 100%
#         Run se termine ep 80 → 32 ep post-ramp SVO, 34 ep post-ramp morpho.
# Note monitoring : pendant le warmup, train/loss < val/loss car λ_SVO=0 en train
#                   mais val/loss inclut SVO. C'est un artefact cosmétique à ignorer.
NER_WARMUP_EPOCHS=${NER_WARMUP_EPOCHS:-0}   # v8.6 fix : retour à 0 comme v8.0/v8.1 (meilleurs scores : 0.918-0.926 boundary)
# nerwarmup=6 causait régression boundary (0.768 vs 0.918 en v8.0). La régression
# "SVO démarre trop tôt" était due au ramp linéaire, maintenant corrigé en svobylevel.

current_level=$START_LEVEL
stagnation_count=0
boundary_stagnation=0
epochs_at_level=0
best_score=-1.0
best_boundary=-1.0
boundary_f1_window=()   # buffer fenêtre glissante pour détection plateau boundary
boundary_rescue_window=()  # buffer plus long (BOUNDARY_RESCUE_WINDOW) pour le NER rescue
ner_rescue_applied=0       # 0=pas encore déclenché, 1=rescue appliqué (once-only)
current_epoch=$START_EPOCH
resume_arg=""
# ── État trigger SVO (métriques-driven, v8.8) ────────────────────────────────
svo_triggered=0          # 0=pas encore déclenché, 1=déclenché
svo_pct=0                # % SVO courant : 0 → 20 → 40 → 60 → 80 → 100
svo_trigger_epoch=0      # epoch à laquelle le trigger a été activé (bnd > SVO_TRIGGER_BND & coarse > SVO_TRIGGER_COARSE)

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

# ── Sources gold — dérivées de GOLD_VERSION (changer la var en tête de fichier) ─────
TRAIN_SILVER="$DATA/train_${GOLD_VERSION}.jsonl"
VAL_SILVER="$DATA/val_${GOLD_VERSION}.jsonl"
TEST_SILVER="$DATA/test_${GOLD_VERSION}.jsonl"

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
DATASET_VERSION="${GOLD_VERSION}-nw0-md8-rd12-svotrig-bnd77c87-v80lam-vptr020-sf10-${TORCH_SHORT}"  # vptr020: verb_ptr retour v8.0 (0.50→0.20), voice retour v8.0 (0.20→0.13) + NER rescue dyn
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
        --max-span-len 8
}

# Build val/test une seule fois
echo "📦 Build val/test datasets..." | tee -a $log_file
python3 build_multitask_dataset.py \
    --input  $VAL_SILVER \
    --output $DATA/val.multitask.jsonl \
    --model-name $MODEL \
    --hard-per-gold 6 \
    --soft-factor 2.0 \
    --max-span-len 8

python3 build_multitask_dataset.py \
    --input  $TEST_SILVER \
    --output $DATA/test.multitask.jsonl \
    --model-name $MODEL \
    --hard-per-gold 6 \
    --soft-factor 2.0 \
    --max-span-len 8

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
            best_boundary=$(python3 -c "import torch; c=torch.load('checkpoint_best_multitask.pt',map_location='cpu'); print(f\"{c.get('best_boundary',-1.0):.4f}\")" 2>/dev/null || echo "-1.0")
            echo "best_score checkpoint: $best_score" | tee -a $log_file
            echo "best_boundary checkpoint: $best_boundary" | tee -a $log_file
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
            # (les compteurs boundary/score du warmup ne reflètent pas le régime multitask)
            stagnation_count=0
            boundary_stagnation=0
            epochs_at_level=0
            best_score=-1.0
            best_boundary=-1.0
        fi

        # Ramp SVO PAR TRIGGER MÉTRIQUES (v8.8) — svo_pct est mis à jour en fin de boucle.
        # 0% jusqu'au trigger (bnd > SVO_TRIGGER_BND & coarse > SVO_TRIGGER_COARSE), puis +20% / SVO_TRIGGER_STEP_EPOCHS.
        ramp_epoch=$((current_epoch - NER_WARMUP_EPOCHS))
        # svo_pct = variable d'état (initialisée à 0, mise à jour après chaque epoch en fin de boucle)
        svo_progress=$(python3 -c "print($svo_pct / 100.0)")
        L_SVO_B_NOW=$(python3 -c "print(f'{$L_SVO_BOUNDARY * $svo_progress:.4f}')")
        L_SVO_NOW=$(python3   -c "print(f'{$L_SVO        * $svo_progress:.4f}')")
        # Role ramp : démarre ROLE_DELAY epochs après fin warmup NER (epoch-based, retard intentionnel)
        # cwp=0 sur role : APPOS/OBLIQUE rares (~1%) boostaient trop → régression boundary via encodeur partagé
        role_ramp_epoch=$((ramp_epoch - ROLE_DELAY))
        role_progress=$(python3 -c "print(min(1.0, max(0.0, $role_ramp_epoch / $SVO_RAMP_EPOCHS)))")
        L_ROLE_NOW=$(python3  -c "print(f'{$L_ROLE * $role_progress:.4f}')")
        L_VOICE_NOW=$(python3 -c "print(f'{$L_VOICE      * $svo_progress:.4f}')")
        L_CERTAINTY_NOW=$(python3 -c "print(f'{$L_CERTAINTY * $svo_progress:.4f}')")
        # Morpho ramp : démarre MORPHO_DELAY epochs après la fin du warmup, epoch-based
        morpho_ramp_epoch=$((ramp_epoch - MORPHO_DELAY))
        morpho_progress=$(python3 -c "print(min(1.0, max(0.0, $morpho_ramp_epoch / $MORPHO_RAMP_EPOCHS)))")
        L_MORPHO_NOW=$(python3 -c "print(f'{$L_MORPHO * $morpho_progress:.4f}')")
        L_VPTR_NOW=$(python3  -c "print(f'{$L_VERB_PTR   * $svo_progress:.4f}')")

        echo "🎛️  Lambdas multitask (SVO trigger: ${svo_pct}% [triggered=${svo_triggered}, bnd>${SVO_TRIGGER_BND} coarse>${SVO_TRIGGER_COARSE}] morpho=${morpho_ramp_epoch}/${MORPHO_RAMP_EPOCHS} role=${role_ramp_epoch}/${SVO_RAMP_EPOCHS})" | tee -a $log_file
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
        --focal-fine-gamma $FOCAL_FINE_GAMMA \
        --focal-coarse-gamma $FOCAL_COARSE_GAMMA \
        --device $DEVICE \
        --layer-lr-decay 0.9 \
        --ema-decay 0.999 \
        --hn-every 1 \
        --class-weight-power 0.5 \
        --ignore-coarse-none \
        --hn-boost-fp 5.0 \
        --hn-boost-fn 2.0 \
        --hn-boost-coarse 2.5 \
        --hn-boost-fine 3.0 \
        --hn-boost-fp-svo $( [ "$NER_ONLY_BENCH" = "1" ] || [ "$in_warmup" = "1" ] && echo "0.0" || echo "3.0" ) \
        --hn-boost-fn-svo $( [ "$NER_ONLY_BENCH" = "1" ] || [ "$in_warmup" = "1" ] && echo "0.0" || echo "2.0" ) \
        --hn-decay 0.85 \
        --hn-max-weight 8.0 \
        --hn-min-weight 0.3 \
        --num-workers $NUM_WORKERS \
        --loss-weighting $LOSS_WEIGHTING \
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
    # Utilisation de la ligne val (commence par "Val   loss=") pour isoler val vs test
    val_log_line=$(grep "Val   loss=" "$epoch_log" | tail -1)
    boundary_f1=$(echo "$val_log_line" | grep -oE "Boundary F1=[0-9.]+" | cut -d= -f2 || echo "?")
    coarse_f1=$(echo "$val_log_line" | grep -oE "Coarse F1=[0-9.]+" | cut -d= -f2 || echo "?")
    fine_f1=$(echo "$val_log_line" | grep -oE "Fine F1=[0-9.]+" | cut -d= -f2 || echo "?")
    svo_f1=$(echo "$val_log_line" | grep -oE "SVO F1=[0-9.]+" | cut -d= -f2 || echo "?")
    voice_f1=$(grep "Voice F1=" $epoch_log | tail -1 | grep -oE "Voice F1=[0-9.]+" | head -1 | cut -d= -f2 || echo "?")
    gender_f1=$(grep "Gender F1=" $epoch_log | tail -1 | grep -oE "Gender F1=[0-9.]+" | cut -d= -f2 || echo "?")
    number_f1=$(grep "Number F1=" $epoch_log | tail -1 | grep -oE "Number F1=[0-9.]+" | cut -d= -f2 || echo "?")
    person_f1=$(grep "Person F1=" $epoch_log | tail -1 | grep -oE "Person F1=[0-9.]+" | cut -d= -f2 || echo "?")
    echo "📊 Epoch $current_epoch — Val Score=$val_score Boundary=$boundary_f1 Coarse=$coarse_f1 Fine=$fine_f1 SVO_F1=$svo_f1 Voice_F1=$voice_f1 Gender_F1=$gender_f1 Number_F1=$number_f1 Person_F1=$person_f1 (best=$best_score)" | tee -a $log_file

    improved=$(python3 -c "print('yes' if float('$val_score') > float('$best_score') + $MIN_DELTA else 'no')")
    boundary_improved=$(python3 -c "print('yes' if '$boundary_f1' != '?' and float('$boundary_f1') > float('$best_boundary') + $MIN_DELTA else 'no')" 2>/dev/null || echo "no")

    if [ "$improved" = "yes" ]; then
        best_score=$val_score
        stagnation_count=0
        echo "✅ Amélioration! best_score=$best_score" | tee -a $log_file
    else
        stagnation_count=$((stagnation_count + 1))
        echo "⏸️  Pas d'amélioration ($stagnation_count/$PATIENCE)" | tee -a $log_file
    fi

    # Mise à jour best_boundary (pour affichage)
    if [ "$boundary_improved" = "yes" ]; then
        best_boundary=$boundary_f1
    fi

    # Fenêtre glissante boundary — détecte le plateau même si micro-améliorations consécutives
    if [ "$boundary_f1" != "?" ]; then
        boundary_f1_window+=("$boundary_f1")
        # Trim au max BOUNDARY_WINDOW éléments
        while [ ${#boundary_f1_window[@]} -gt $BOUNDARY_WINDOW ]; do
            boundary_f1_window=("${boundary_f1_window[@]:1}")
        done
    fi

    # Vérifier le progrès net sur la fenêtre (seulement quand fenêtre pleine)
    if [ ${#boundary_f1_window[@]} -ge $BOUNDARY_WINDOW ] && [ "$boundary_f1" != "?" ]; then
        window_oldest="${boundary_f1_window[0]}"
        boundary_window_progress=$(python3 -c "
try:
    gain = float('$boundary_f1') - float('$window_oldest')
    print('yes' if gain >= $BOUNDARY_WINDOW_DELTA else 'no')
except:
    print('yes')
" 2>/dev/null || echo "yes")
        if [ "$boundary_window_progress" = "no" ]; then
            boundary_stagnation=$((boundary_stagnation + 1))
            echo "⏸️  Boundary plateau sur ${BOUNDARY_WINDOW}ep: $window_oldest → $boundary_f1 (+$(python3 -c "print(f'{float(\"$boundary_f1\")-float(\"$window_oldest\"):.4f}')" 2>/dev/null)) (stagnation=$boundary_stagnation/$PATIENCE)" | tee -a $log_file
        else
            boundary_stagnation=0
            echo "✅ Boundary progresse sur fenêtre ${BOUNDARY_WINDOW}ep: $window_oldest → $boundary_f1 (best=$best_boundary)" | tee -a $log_file
        fi
    else
        echo "✅ Boundary: $boundary_f1 (fenêtre ${#boundary_f1_window[@]}/$BOUNDARY_WINDOW, best=$best_boundary)" | tee -a $log_file
    fi

    # ── NER rescue dynamique : réduit verb_ptr + voice si boundary stagne sous la cible ──
    # Déclenché une seule fois (ner_rescue_applied=0) pour éviter une over-réduction.
    # Fenêtre indépendante (BOUNDARY_RESCUE_WINDOW=10) plus large que le plateau window (5).
    if [ "$ner_rescue_applied" = "0" ] && [ "$in_warmup" != "1" ] && [ "$boundary_f1" != "?" ]; then
        boundary_rescue_window+=("$boundary_f1")
        while [ ${#boundary_rescue_window[@]} -gt $BOUNDARY_RESCUE_WINDOW ]; do
            boundary_rescue_window=("${boundary_rescue_window[@]:1}")
        done
        if [ ${#boundary_rescue_window[@]} -ge $BOUNDARY_RESCUE_WINDOW ]; then
            rescue_oldest="${boundary_rescue_window[0]}"
            rescue_needed=$(python3 -c "
try:
    gain  = float('$boundary_f1') - float('$rescue_oldest')
    below = float('$boundary_f1') < $BOUNDARY_RESCUE_TARGET
    print('yes' if below and gain < $BOUNDARY_RESCUE_DELTA else 'no')
except:
    print('no')
" 2>/dev/null || echo "no")
            if [ "$rescue_needed" = "yes" ]; then
                ner_rescue_applied=1
                L_VERB_PTR=$(python3 -c "print(f'{float(\"$L_VERB_PTR\") * $BOUNDARY_RESCUE_FACTOR:.4f}')")
                L_VOICE=$(python3    -c "print(f'{float(\"$L_VOICE\")    * $BOUNDARY_RESCUE_FACTOR:.4f}')")
                echo "🆘 NER RESCUE ep $current_epoch — boundary stagne à $boundary_f1 (Δ=$rescue_oldest→$boundary_f1) < cible $BOUNDARY_RESCUE_TARGET → L_VERB_PTR=$L_VERB_PTR L_VOICE=$L_VOICE (×$BOUNDARY_RESCUE_FACTOR)" | tee -a $log_file
                boundary_rescue_window=()
            else
                echo "🔭 NER rescue watch ep $current_epoch : bnd fenêtre $rescue_oldest→$boundary_f1 (cible=$BOUNDARY_RESCUE_TARGET, rescue=$([ $ner_rescue_applied -eq 1 ] && echo 'déjà appliqué' || echo 'prêt'))" | tee -a $log_file
            fi
        fi
    fi

    # Pendant le warmup NER : ne pas toucher au niveau ni à la stagnation    # (le score NER-only ne reflète pas encore le régime multitask final)
    if [ "$in_warmup" = "1" ]; then
        resume_arg=""
        [ -f checkpoint_best_multitask.pt ] && resume_arg="--resume checkpoint_best_multitask.pt"
        current_epoch=$((current_epoch + 1))
        continue
    fi

    # ── Trigger SVO basé sur les métriques (v8.8+) ────────────────────────────
    # Démarre à 20% quand boundary > SVO_TRIGGER_BND ET coarse > SVO_TRIGGER_COARSE,
    # puis +20% tous les SVO_TRIGGER_STEP_EPOCHS epochs. Max = 100%.
    # fine exclu : monte trop lentement (ep 20-25), les deux autres suffisent pour
    # confirmer la stabilité NER (bnd=0.77+coarse=0.87 ≈ ep 8-12 sur les bons runs).
    if [ "$NER_ONLY_BENCH" != "1" ]; then
        if [ "$svo_triggered" = "0" ] && [ "$boundary_f1" != "?" ] && [ "$coarse_f1" != "?" ]; then
            all_above=$(python3 -c "
try:
    v = float('$boundary_f1') > $SVO_TRIGGER_BND and float('$coarse_f1') > $SVO_TRIGGER_COARSE
    print('yes' if v else 'no')
except:
    print('no')
" 2>/dev/null || echo "no")
            if [ "$all_above" = "yes" ]; then
                svo_triggered=1
                svo_pct=20
                svo_trigger_epoch=$current_epoch
                echo "🚀 SVO TRIGGER ep $current_epoch — bnd=$boundary_f1 > $SVO_TRIGGER_BND & coarse=$coarse_f1 > $SVO_TRIGGER_COARSE → SVO 20% dès ep suivante" | tee -a $log_file
            else
                echo "⏳ SVO trigger : bnd=$boundary_f1 (>${SVO_TRIGGER_BND}?) coarse=$coarse_f1 (>${SVO_TRIGGER_COARSE}?) — attente seuils" | tee -a $log_file
            fi
        elif [ "$svo_triggered" = "1" ] && [ "$svo_pct" -lt 100 ]; then
            epochs_since_trigger=$((current_epoch - svo_trigger_epoch))
            new_svo_pct=$(python3 -c "print(min(100, 20 + ($epochs_since_trigger // $SVO_TRIGGER_STEP_EPOCHS) * 20))" 2>/dev/null || echo "$svo_pct")
            if [ "$new_svo_pct" -gt "$svo_pct" ]; then
                echo "📈 SVO +20% → ${new_svo_pct}% (ep $current_epoch, +${epochs_since_trigger} depuis trigger ep $svo_trigger_epoch)" | tee -a $log_file
                svo_pct=$new_svo_pct
            fi
        fi
    fi

    epochs_at_level=$((epochs_at_level + 1))
    max_level=$(( ${#LEVEL_NAMES[@]} - 1 ))

    should_advance=0
    if [ $stagnation_count -ge $PATIENCE ]; then
        echo "⏩ PLATEAU score global ($stagnation_count epochs) → advance" | tee -a $log_file
        should_advance=1
    fi
    if [ $boundary_stagnation -ge $PATIENCE ]; then
        echo "⏩ BOUNDARY STAGNE ($boundary_stagnation epochs) → advance hard negatives" | tee -a $log_file
        should_advance=1
    fi
    if [ $epochs_at_level -ge $MAX_EPOCHS_PER_LEVEL ]; then
        echo "⏩ MAX epochs/level ($epochs_at_level/$MAX_EPOCHS_PER_LEVEL) → advance" | tee -a $log_file
        should_advance=1
    fi

    if [ $should_advance -eq 1 ]; then
        if [ $current_level -lt $max_level ]; then
            current_level=$((current_level + 1))
            stagnation_count=0
            boundary_stagnation=0
            epochs_at_level=0
            boundary_f1_window=()
            echo "🚀 ADVANCE to level ${LEVEL_NAMES[$current_level]}" | tee -a $log_file
            rebuild_dataset $current_level
        else
            stagnation_count=0
            boundary_stagnation=0
            epochs_at_level=0
            boundary_f1_window=()
            echo "🔄 Niveau max ${LEVEL_NAMES[$current_level]} — reset compteurs, on continue (ep $current_epoch/$MAX_EPOCHS)" | tee -a $log_file
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

