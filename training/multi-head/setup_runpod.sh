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

