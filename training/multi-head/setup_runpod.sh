#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  setup_runpod.sh — Setup RunPod pour training v8.1 (labels v8.1, dataset DVC)
#
#  Usage depuis RunPod (aprs git clone) :
#    cd pimpmyrag/training/multi-head
#    chmod +x setup_runpod.sh && ./setup_runpod.sh
#
#  Variables d'environnement attendues (RunPod → Secrets) :
#    WANDB_API_KEY           — cl W&B (optionnel, offline si absent)
#    AWS_ACCESS_KEY_ID       — pour DVC remote R2 (Cloudflare)
#    AWS_SECRET_ACCESS_KEY   — idem
#    DVC_R2_ENDPOINT         — URL endpoint R2
# ═══════════════════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"

echo "📦 Setup RunPod v8.1 — $(date)"

# ── 1. Dépendances ───────────────────────────────────────────────────────────
echo ""
echo "🐍 Installation des dépendances..."
# --system-site-packages : hérite torch/transformers de l'image de base
# Évite de re-télécharger torch (2 GB) à chaque pod
python3 -m venv venv --system-site-packages 2>/dev/null || true
source venv/bin/activate

# Vérifie si torch >=2.6 est déjà dispo (hérité de l'image)
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
TORCH_OK=$(python3 -c "import torch; v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); print('yes' if v>=(2,0) else 'no')" 2>/dev/null || echo "no")
if [ "$TORCH_OK" = "no" ]; then
    echo "   → torch absent ou trop ancien, tentative d'upgrade (non bloquant)..."
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9.]+" | head -1 || echo "")
    if [ -n "$CUDA_VER" ]; then
        CUDA_SHORT=$(echo "$CUDA_VER" | tr -d '.' | cut -c1-3)
        pip install -q "torch>=2.0.0" --index-url "https://download.pytorch.org/whl/cu${CUDA_SHORT}" \
            || pip install -q "torch>=2.0.0" \
            || echo "   ⚠️  torch upgrade échoué — on continue avec torch de l'image de base"
    else
        pip install -q "torch>=2.0.0" \
            || echo "   ⚠️  torch upgrade échoué — on continue avec torch de l'image de base"
    fi
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "inconnu")
fi
echo "   ✅ torch ${TORCH_VER} disponible"

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

# ── 3. DVC pull des datasets v8.1 ──────────────────────────────────────────────
echo ""
echo "📥 DVC pull datasets v8.1..."
cd "$REPO_ROOT"

# ── Pré-check credentials R2 ─────────────────────────────────────────────────
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo ""
    echo "⚠️  Credentials R2 manquants (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)"
    echo "   → Sur RunPod : pod template → Environment Variables → ajouter les secrets"
    echo "   → Ou saisir maintenant :"
    echo ""
    if [ -t 0 ]; then
        # Terminal interactif — on peut lire
        read -r -p "   AWS_ACCESS_KEY_ID     : " AWS_ACCESS_KEY_ID
        read -r -s -p "   AWS_SECRET_ACCESS_KEY : " AWS_SECRET_ACCESS_KEY
        echo ""
        read -r -p "   DVC_R2_ENDPOINT (entrée = valeur par défaut) : " _EP
        [ -n "$_EP" ] && DVC_R2_ENDPOINT="$_EP"
        export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY DVC_R2_ENDPOINT
    else
        echo "   ❌ Aucun terminal interactif — impossible de saisir les credentials."
        echo "      Relancer le pod avec les secrets configurés OU exporter les variables"
        echo "      avant d'appeler ce script :"
        echo "        export AWS_ACCESS_KEY_ID=..."
        echo "        export AWS_SECRET_ACCESS_KEY=..."
        echo "        export DVC_R2_ENDPOINT=https://..."
        echo "        ./setup_runpod.sh"
        exit 1
    fi
fi

# Cloudflare R2 — injecte les credentials depuis les variables d'env RunPod
if [ -n "$DVC_R2_ENDPOINT" ]; then
    dvc remote modify r2remote endpointurl "$DVC_R2_ENDPOINT"
fi
dvc remote modify --local r2remote access_key_id     "$AWS_ACCESS_KEY_ID"
dvc remote modify --local r2remote secret_access_key "$AWS_SECRET_ACCESS_KEY"

dvc pull training/multi-head/data/train_v8.1.jsonl \
         training/multi-head/data/val_v8.1.jsonl \
         training/multi-head/data/test_v8.1.jsonl

cd training/multi-head

echo "✅ Datasets v8.1 présents :"
wc -l data/train_v8.1.jsonl data/val_v8.1.jsonl data/test_v8.1.jsonl

# ── 4. Vérification schéma labels v8.1 ─────────────────────────────────────────
echo ""
echo "🔍 Vérification labels v8.1 (NUM_FINE=38, NUM_COARSE=10)..."
python3 - <<'PYEOF'
import sys
sys.path.insert(0, '.')
import labels as L
assert L.NUM_FINE == 38, f"NUM_FINE={L.NUM_FINE} attendu 38"
assert len(L.COARSE_LABELS) == 10, f"NUM_COARSE={len(L.COARSE_LABELS)} attendu 10"
assert 'hint_inst_name'     in L.FINE2ID, "hint_inst_name manquant"
assert 'hint_inst_role'     in L.FINE2ID, "hint_inst_role manquant"
assert 'hint_document'      in L.FINE2ID, "hint_document manquant"
assert 'hint_notion'    in L.FINE2ID, "hint_notion manquant (v8.1)"
assert 'hint_doctrine'  in L.FINE2ID, "hint_doctrine manquant (v8.1)"
assert 'hint_process' not in L.FINE2ID, "hint_process encore present — labels.py pas mis a jour"
assert 'hint_rule'    not in L.FINE2ID, "hint_rule encore present — labels.py pas mis a jour"
assert 'hint_concept'  not in L.FINE2ID, "hint_concept encore present — labels.py pas mis a jour"
assert 'hint_quantity' not in L.FINE2ID, "hint_quantity encore present — labels.py pas mis a jour"
print(f"✅ labels.py v8.1 OK — NUM_FINE={L.NUM_FINE}  NUM_COARSE={len(L.COARSE_LABELS)}")
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
        description="checkpoint_best + best_model multitask v8.1")
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
