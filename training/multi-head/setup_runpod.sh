#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  setup_runpod.sh — Setup RunPod pour training v6.1.1 (labels v6.1, dataset DVC)
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

echo "📦 Setup RunPod v6.9 — $(date)"

# ── 1. Dépendances ───────────────────────────────────────────────────────────
echo ""
echo "🐍 Installation des dépendances..."
# --system-site-packages : hérite torch/transformers de l'image de base
# Évite de re-télécharger torch (2 GB) à chaque pod
python3 -m venv venv --system-site-packages 2>/dev/null || true
source venv/bin/activate

# Vérifie si torch >=2.6 est déjà dispo (hérité de l'image)
TORCH_OK=$(python3 -c "import torch; v=tuple(int(x) for x in torch.__version__.split('.')[:2]); print('yes' if v>=(2,6) else 'no')" 2>/dev/null || echo "no")
if [ "$TORCH_OK" = "no" ]; then
    echo "   → torch <2.6 dans l'image, upgrade..."
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9.]+" | head -1 || echo "")
    if [ -n "$CUDA_VER" ]; then
        CUDA_SHORT=$(echo "$CUDA_VER" | tr -d '.' | cut -c1-3)
        pip install -q "torch>=2.6.0" --index-url "https://download.pytorch.org/whl/cu${CUDA_SHORT}" \
            || pip install -q "torch>=2.6.0"
    else
        pip install -q "torch>=2.6.0"
    fi
else
    echo "   ✅ torch $(python3 -c 'import torch; print(torch.__version__)') déjà installé (image de base)"
fi

# Installe seulement requirements.txt (sans torch, déjà présent)
pip install -q -r requirements.txt

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

# ── 3. DVC pull des datasets v6.9 ──────────────────────────────────────────────
echo ""
echo "📥 DVC pull datasets v6.9..."
cd "$REPO_ROOT"

# Cloudflare R2 — injecte les credentials depuis les variables d'env RunPod
if [ -n "$DVC_R2_ENDPOINT" ]; then
    dvc remote modify r2remote endpointurl "$DVC_R2_ENDPOINT"
fi
if [ -n "$AWS_ACCESS_KEY_ID" ]; then
    dvc remote modify --local r2remote access_key_id     "$AWS_ACCESS_KEY_ID"
    dvc remote modify --local r2remote secret_access_key "$AWS_SECRET_ACCESS_KEY"
fi

dvc pull training/multi-head/data/train_v6.9.jsonl \
         training/multi-head/data/val_v6.9.jsonl \
         training/multi-head/data/test_v6.9.jsonl

cd training/multi-head

echo "✅ Datasets v6.9 présents :"
wc -l data/train_v6.9.jsonl data/val_v6.9.jsonl data/test_v6.9.jsonl

# ── 4. Vérification schéma labels v6.9 ─────────────────────────────────────────
echo ""
echo "🔍 Vérification labels v6.9 (NUM_FINE=42, NUM_COARSE=10)..."
python3 - <<'PYEOF'
import sys
sys.path.insert(0, '.')
import labels as L
assert L.NUM_FINE == 42, f"NUM_FINE={L.NUM_FINE} attendu 42"
assert len(L.COARSE_LABELS) == 10, f"NUM_COARSE={len(L.COARSE_LABELS)} attendu 10"
assert 'hint_inst_name'     in L.FINE2ID, "hint_inst_name manquant"
assert 'hint_inst_role'     in L.FINE2ID, "hint_inst_role manquant"
assert 'hint_document'      in L.FINE2ID, "hint_document manquant"
assert 'hint_notion'        in L.FINE2ID, "hint_notion manquant (v6.9)"
assert 'hint_process'       in L.FINE2ID, "hint_process manquant (v6.9)"
assert 'hint_doctrine'      in L.FINE2ID, "hint_doctrine manquant (v6.9)"
print(f"✅ labels.py v6.9 OK — NUM_FINE={L.NUM_FINE}  NUM_COARSE={len(L.COARSE_LABELS)}")
PYEOF

# ── 5. Lancement du training ─────────────────────────────────────────────────
echo ""
echo "🚀 Lancement du training adaptatif..."
./run_adaptive_training.sh
TRAINING_EXIT=$?

# ── 6. Upload artefacts vers R2 (boto3, déjà dans l'image) + W&B ─────────────
# On n'utilise PAS dvc add (nécessite git commit) ni awscli (lourd à installer).
# boto3 est préinstallé dans l'image runpod/pytorch.
echo ""
echo "📤 Upload artefacts post-training (R2 boto3 + W&B artifact)..."

cd "$REPO_ROOT"

CKPT_DIR="training/multi-head"
R2_ENDPOINT="${DVC_R2_ENDPOINT:-https://07027fdcb4c08fe1418a9595986c3ac8.r2.cloudflarestorage.com}"
UPLOAD_TAG=$(date +%Y%m%d-%H%M)
R2_PREFIX="models/setup-${UPLOAD_TAG}"

# Upload R2 via boto3 (pas besoin d'awscli)
if [ -n "$AWS_ACCESS_KEY_ID" ]; then
    python3 - <<PYEOF 2>&1 || echo "⚠ Upload R2 échoué"
import os, boto3
endpoint = os.environ.get("DVC_R2_ENDPOINT", "https://07027fdcb4c08fe1418a9595986c3ac8.r2.cloudflarestorage.com")
s3 = boto3.client("s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)
files = [
    ("$CKPT_DIR/checkpoint_best_multitask.pt", "$R2_PREFIX/checkpoint_best_multitask.pt"),
    ("$CKPT_DIR/best_model_multitask.pt",      "$R2_PREFIX/best_model_multitask.pt"),
]
for local, key in files:
    if os.path.exists(local):
        size_mb = os.path.getsize(local) / 1024**2
        print(f"  → upload {local} ({size_mb:.0f} MB)...")
        s3.upload_file(local, "pimpmyrag-data", key)
        print(f"  ✅ s3://pimpmyrag-data/{key}")
    else:
        print(f"  ⚠ {local} introuvable, ignoré")
print(f"\\n  📋 Récupération locale :")
print(f"     python3 -c \\"import boto3; boto3.client('s3', endpoint_url='{endpoint}', ...).download_file('pimpmyrag-data', '$R2_PREFIX/checkpoint_best_multitask.pt', 'checkpoint_best_multitask.pt')\\"")
PYEOF
else
    echo "   ⚠ AWS_ACCESS_KEY_ID absent — upload R2 ignoré"
fi

# W&B artifact
if [ -n "$WANDB_API_KEY" ]; then
    python3 - <<'PYEOF' 2>&1 | tail -5 || echo "W&B artifact log skipped"
import os, wandb
api = wandb.Api(api_key=os.environ["WANDB_API_KEY"])
runs = list(api.runs("pimpmyrag-ner", order="-created_at"))
run = next((r for r in runs if r.state in ("running","finished")), None)
if run is None:
    print("⚠ Aucun run W&B trouvé — artifact non loggé")
else:
    art = wandb.Artifact("pimpmyrag-ner-model", type="model",
        description="checkpoint_best + best_model multitask v6.9")
    for fname in ("training/multi-head/checkpoint_best_multitask.pt",
                  "training/multi-head/best_model_multitask.pt"):
        if os.path.exists(fname):
            art.add_file(fname)
            print(f"  added {fname}")
    run.log_artifact(art)
    print(f"✅ W&B artifact loggé sur run {run.name}")
PYEOF
else
    echo "   ⚠ WANDB_API_KEY absent — W&B artifact ignoré"
fi

echo ""
if [ $TRAINING_EXIT -eq 0 ]; then
    echo "✅ Training + upload termines avec succes"
else
    echo "⚠️  Training s'est termine avec exit=$TRAINING_EXIT — artefacts uploades quand meme"
fi
