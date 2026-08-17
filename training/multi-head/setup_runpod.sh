#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  setup_runpod.sh — Setup RunPod pour training v8.6
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔑  SEUL ENDROIT À CHANGER POUR UPGRADER LE DATASET
GOLD_VERSION="${GOLD_VERSION:-v8.24b_deps}"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "📦 Setup RunPod ${GOLD_VERSION} — $(date)"

# ── 1. Dépendances ───────────────────────────────────────────────────────────
echo ""
echo "🐍 Installation des dépendances..."

# ── venv 100% isolé (PAS de --system-site-packages) ──────────────────────────
# L'image de base runpod/pytorch:2.4 embarque torch 2.4 + cuDNN 8, incompatibles
# avec transformers>=4.53 qui EXIGE torch>=2.6 pour torch.load (CVE-2025-32434).
# Hériter du site-packages système (--system-site-packages) mélange les deux
# installations de torch/cuDNN et fait presque toujours échouer l'upgrade
# (l'ancien torch système reste importable en priorité, ou le test CUDA échoue
# à cause du mismatch cuDNN 8/9 et retombe sur le torch 2.4 cassé).
# → on repart d'un venv totalement isolé et on installe torch 2.6+ proprement.
python3 -m venv venv 2>/dev/null || true
source venv/bin/activate
pip install -q --upgrade pip

echo "⬆️  Installation torch>=2.6.0 (CUDA 12.4, venv isolé)..."
pip install -q "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cu124 \
    || pip install -q "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cu121

TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [ -z "$TORCH_VERSION" ]; then
    echo "   ❌ torch introuvable après installation — abandon."
    exit 1
fi
echo "   torch installé: ${TORCH_VERSION}"

if ! python3 -c "import torch; v=torch.__version__; assert int(v.split('.')[0])>=2 and int(v.split('.')[1])>=6"; then
    echo "   ❌ torch ${TORCH_VERSION} < 2.6 — transformers>=4.53 va planter (CVE-2025-32434). Abandon."
    exit 1
fi

# Vérification CUDA (avertissement seulement — ne JAMAIS redowngrader torch,
# ça garantirait le crash CVE-2025-32434 plus tard dans le training)
if python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" 2>/dev/null; then
    echo "   ✅ torch ${TORCH_VERSION} + CUDA OK"
else
    echo "   ⚠️  CUDA indisponible ou cuDNN incomplet avec torch ${TORCH_VERSION} — on continue quand même"
    echo "      (torch n'est PAS redowngradé : un torch <2.6 ferait planter transformers à coup sûr)"
fi

# Contrainte pour empêcher `pip install -r requirements.txt` de retoucher torch
TORCH_PIN="${TORCH_VERSION%%+*}"
echo "torch==${TORCH_PIN}" > /tmp/torch_constraint.txt
pip install -q -r requirements.txt -c /tmp/torch_constraint.txt

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

# ── 3. DVC pull des datasets ────────────────────────────────────────────────────
echo ""
echo "📥 DVC pull datasets ${GOLD_VERSION}..."
cd "$REPO_ROOT"

# Cloudflare R2 — injecte les credentials depuis les variables d'env RunPod
if [ -n "$DVC_R2_ENDPOINT" ]; then
    dvc remote modify r2remote endpointurl "$DVC_R2_ENDPOINT"
fi
if [ -n "$AWS_ACCESS_KEY_ID" ]; then
    dvc remote modify --local r2remote access_key_id     "$AWS_ACCESS_KEY_ID"
    dvc remote modify --local r2remote secret_access_key "$AWS_SECRET_ACCESS_KEY"
fi

dvc pull training/multi-head/data/train_${GOLD_VERSION}.jsonl \
         training/multi-head/data/val_${GOLD_VERSION}.jsonl \
         training/multi-head/data/test_${GOLD_VERSION}.jsonl

cd training/multi-head

echo "✅ Datasets ${GOLD_VERSION} présents :"
wc -l data/train_${GOLD_VERSION}.jsonl data/val_${GOLD_VERSION}.jsonl data/test_${GOLD_VERSION}.jsonl

# ── 4. Vérification schéma labels v9 ───────────────────────────────────────────
echo ""
echo "🔍 Vérification labels ${GOLD_VERSION} v9 (NUM_FINE=34, NUM_COARSE=9, NUM_GENDER=2)..."
python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, ".")
import labels as L
# v9.0 : taxonomie réduite — fine 40→34, coarse 11→9 (8 positifs + NONE).
assert L.NUM_FINE == 34, f"NUM_FINE={L.NUM_FINE} attendu 34 (v9)"
assert len(L.COARSE_LABELS) == 9, f"NUM_COARSE={len(L.COARSE_LABELS)} attendu 9 (v9 : 8+NONE)"
assert L.NUM_GENDER == 2, f"NUM_GENDER={L.NUM_GENDER} attendu 2"
# labels conservés
assert 'hint_inst_name' in L.FINE2ID, "hint_inst_name manquant"
assert 'hint_inst_role' in L.FINE2ID, "hint_inst_role manquant"
assert 'hint_notion'    in L.FINE2ID, "hint_notion manquant"
assert 'hint_field'     in L.FINE2ID, "hint_field manquant"
assert 'hint_language'  in L.FINE2ID, "hint_language manquant"
# labels fusionnés en v9 → doivent avoir DISPARU
for gone in ('hint_doctrine','hint_state','hint_work_generic','hint_object_name','hint_rate','hint_vegetal'):
    assert gone not in L.FINE2ID, f"{gone} devrait etre fusionne en v9"
# coarse v9 : CONCEPT remplace WORK/ABSTRACT, plus de BIO
for c in ('PER','LOC','ORG','TIME','VALUE','OBJECT','EVENT','CONCEPT','NONE'):
    assert c in L.COARSE2ID, f"coarse {c} manquant"
for c in ('BIO','ABSTRACT','WORK'):
    assert c not in L.COARSE2ID, f"coarse {c} devrait etre supprime en v9"
# attributs transverses
assert L.NUM_ANIMACY == 2 and L.NUM_WORK == 2, "attributs v9 absents"
assert L.derive_attributes('hint_vegetal')['animacy'] == 0, "vegetal doit rester inanimate"
print(f"✅ labels.py v9 OK — NUM_FINE={L.NUM_FINE} NUM_COARSE={len(L.COARSE_LABELS)} + attributs animacy/living/abstract/dynamicity/work")
print(f"   v9 : BIO/ABSTRACT/WORK → attributs ; 6 fusions fine (40→34)")
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
