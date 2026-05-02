#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  setup_runpod.sh — Setup RunPod pour training v5 (labels v5, dataset DVC)
#
#  Usage depuis RunPod (après git clone) :
#    cd pimpmyrag/training/multi-head
#    chmod +x setup_runpod.sh && ./setup_runpod.sh
#
#  Variables d'environnement attendues (RunPod → Secrets) :
#    WANDB_API_KEY           — clé W&B (optionnel, offline si absent)
#    AWS_ACCESS_KEY_ID       — pour DVC remote S3 (si remote = S3)
#    AWS_SECRET_ACCESS_KEY   — idem
#    DVC_REMOTE              — nom du remote DVC à utiliser (défaut: s3remote)
# ═══════════════════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"

echo "📦 Setup RunPod v5 — $(date)"

# ── 1. Dépendances ───────────────────────────────────────────────────────────
echo ""
echo "🐍 Installation des dépendances..."
python3 -m venv venv 2>/dev/null || true
source venv/bin/activate

# PyTorch CUDA (détecte la version CUDA disponible)
CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9.]+" | head -1 || echo "")
if [ -n "$CUDA_VER" ]; then
    CUDA_SHORT=$(echo "$CUDA_VER" | tr -d '.' | cut -c1-3)
    echo "   → CUDA $CUDA_VER détecté — wheel cu${CUDA_SHORT}"
    pip install -q "torch>=2.6.0" --index-url "https://download.pytorch.org/whl/cu${CUDA_SHORT}" \
        || pip install -q "torch>=2.6.0"
else
    echo "   → Pas de GPU détecté — installation CPU"
    pip install -q "torch>=2.6.0"
fi

pip install -q -r requirements.txt
pip install -q wandb dvc dvc-s3

# ── 2. W&B login ─────────────────────────────────────────────────────────────
echo ""
if [ -n "$WANDB_API_KEY" ]; then
    echo "🔑 W&B login..."
    wandb login "$WANDB_API_KEY" --relogin
    echo "✅ W&B connecté"
else
    echo "⚠️  WANDB_API_KEY absent — W&B en mode offline (logs locaux seulement)"
    export WANDB_MODE=offline
fi

# ── 3. DVC pull des datasets v5 ──────────────────────────────────────────────
echo ""
echo "📥 DVC pull datasets v5..."
cd "$REPO_ROOT"

# Cloudflare R2 — injecte les credentials depuis les variables d'env RunPod
# (RunPod → Secrets → DVC_R2_ENDPOINT / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
if [ -n "$DVC_R2_ENDPOINT" ]; then
    dvc remote modify r2remote endpointurl "$DVC_R2_ENDPOINT"
fi
if [ -n "$AWS_ACCESS_KEY_ID" ]; then
    dvc remote modify --local r2remote access_key_id     "$AWS_ACCESS_KEY_ID"
    dvc remote modify --local r2remote secret_access_key "$AWS_SECRET_ACCESS_KEY"
fi

dvc pull training/multi-head/data/train_v5.jsonl \
         training/multi-head/data/val_v5.jsonl \
         training/multi-head/data/test_v5.jsonl

cd training/multi-head

echo "✅ Datasets v5 présents :"
wc -l data/train_v5.jsonl data/val_v5.jsonl data/test_v5.jsonl

# ── 4. Vérification schéma labels v5 ─────────────────────────────────────────
echo ""
echo "🔍 Vérification labels v5 (NUM_FINE=34, NUM_COARSE=10)..."
python3 - <<'PYEOF'
import sys
sys.path.insert(0, '.')
import labels as L
assert L.NUM_FINE == 34, f"NUM_FINE={L.NUM_FINE} attendu 34"
assert len(L.COARSE_LABELS) == 10, f"NUM_COARSE={len(L.COARSE_LABELS)} attendu 10"
assert 'hint_inst_name' in L.FINE2ID, "hint_inst_name manquant"
assert 'hint_document' in L.FINE2ID, "hint_document manquant"
print(f"✅ labels.py v5 OK — NUM_FINE={L.NUM_FINE}  NUM_COARSE={len(L.COARSE_LABELS)}")
PYEOF

# ── 5. Lancement du training ─────────────────────────────────────────────────
echo ""
echo "🚀 Lancement du training adaptatif..."
./run_adaptive_training.sh
TRAINING_EXIT=$?

# ── 6. Upload artefacts vers R2 (DVC) + W&B ──────────────────────────────────
echo ""
echo "📤 Upload artefacts post-training..."

cd "$REPO_ROOT"

# Détermine le run W&B actif (dernier run du projet)
WANDB_RUN_ID=$(python3 - <<'PYEOF' 2>/dev/null || echo "")
import os, wandb
key = os.environ.get("WANDB_API_KEY","")
if not key:
    exit(0)
api = wandb.Api(api_key=key)
runs = list(api.runs("pimpmyrag-ner", order="-created_at"))
r = next((x for x in runs if x.state in ("running","finished")), None)
print(r.id if r else "")
PYEOF

# Artefacts à conserver
CKPT_DIR="training/multi-head"
ARTIFACTS=(
    "$CKPT_DIR/checkpoint_best_multitask.pt"
    "$CKPT_DIR/best_model_multitask.pt"
)

for f in "${ARTIFACTS[@]}"; do
    if [ -f "$f" ]; then
        echo "   → DVC add $f"
        dvc add "$f" 2>/dev/null || true
    fi
done

# Push vers R2
echo "   → dvc push artefacts..."
dvc push "${ARTIFACTS[@]/#/}" 2>&1 | tail -3 || true

# Log comme W&B artifact
if [ -n "$WANDB_API_KEY" ] && [ -n "$WANDB_RUN_ID" ]; then
    python3 - <<PYEOF 2>/dev/null || echo "W&B artifact log skipped"
import os, wandb
key = os.environ.get("WANDB_API_KEY","")
api = wandb.Api(api_key=key)
run = api.run(f"pimpmyrag-ner/$WANDB_RUN_ID")
artifact = wandb.Artifact("pimpmyrag-ner-model", type="model",
    description="checkpoint_best + best_model multitask v5")
for fname in ("training/multi-head/checkpoint_best_multitask.pt",
              "training/multi-head/best_model_multitask.pt"):
    if os.path.exists(fname):
        artifact.add_file(fname)
        print(f"  added {fname}")
run.log_artifact(artifact)
print("W&B artifact logged OK")
PYEOF
fi

echo ""
if [ $TRAINING_EXIT -eq 0 ]; then
    echo "✅ Training + upload termines avec succes"
else
    echo "⚠️  Training s'est termine avec exit=$TRAINING_EXIT — artefacts uploades quand meme"
fi
