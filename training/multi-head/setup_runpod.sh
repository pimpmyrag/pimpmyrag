#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  setup_runpod.sh — Setup complet sur RunPod pour relancer le training
#
#  Usage depuis RunPod (après git clone) :
#    cd pimpmyrag/training/multi-head
#    chmod +x setup_runpod.sh && ./setup_runpod.sh
#
#  Ce script :
#    1. Installe les dépendances Python
#    2. Télécharge le modèle Stanza fr
#    3. Régénère les silver SVO (train/val/test) depuis les v3 NER gold
#       → OBLIGATOIRE car build_svo_silver.py a été modifié
#         (ajout verb_char_start/end pour le verb-pointer head)
#    4. Lance le training adaptatif
# ═══════════════════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")"

# ── 1. Dépendances ───────────────────────────────────────────────────────
echo "📦 Installation des dépendances..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>/dev/null \
    || pip install -q torch torchvision torchaudio  # fallback sans CUDA spécifique
pip install -q transformers stanza scikit-learn sentencepiece protobuf accelerate

# ── 2. Stanza fr ─────────────────────────────────────────────────────────
echo "🔤 Téléchargement du modèle Stanza fr..."
python3 - <<'EOF'
import stanza
stanza.download("fr", processors="tokenize,mwt,pos,lemma,depparse", verbose=False)
print("✅ Stanza fr prêt")
EOF

# ── 3. Régénération des silver SVO ───────────────────────────────────────
# Pourquoi régénérer ?
#   build_svo_silver.py a été modifié : chaque argument span stocke maintenant
#   verb_char_start / verb_char_end → requis pour le verb-pointer head.
#   Les anciens fichiers *_svo_silver.jsonl ne contiennent pas ces champs.
#
# Source : data/train_v3.jsonl / val_v3.jsonl / test_v3.jsonl  (dans git, pas LFS)
# Sortie : data/train_svo_silver.jsonl / val_svo_silver.jsonl / test_svo_silver.jsonl

STANZA_BATCH=128  # augmenter sur GPU A100 (256), diminuer si OOM (64)

echo ""
echo "🔄 Régénération silver SVO train (peut prendre 20-40 min sur GPU)..."
python3 data/build_svo_silver.py \
    --data_dir data \
    --split train \
    --suffix_in  _v3 \
    --suffix_out _svo_silver \
    --gpu \
    --batch $STANZA_BATCH \
    --log-every 200

echo ""
echo "🔄 Régénération silver SVO val..."
python3 data/build_svo_silver.py \
    --data_dir data \
    --split val \
    --suffix_in  _v3 \
    --suffix_out _svo_silver \
    --gpu \
    --batch $STANZA_BATCH

echo ""
echo "🔄 Régénération silver SVO test..."
python3 data/build_svo_silver.py \
    --data_dir data \
    --split test \
    --suffix_in  _v3 \
    --suffix_out _svo_silver \
    --gpu \
    --batch $STANZA_BATCH

echo ""
echo "✅ Fichiers silver générés :"
wc -l data/train_svo_silver.jsonl data/val_svo_silver.jsonl data/test_svo_silver.jsonl

# ── 4. Vérification rapide du silver (spot-check verb_char_start) ────────
echo ""
echo "🔍 Spot-check verb_char_start dans le silver..."
python3 - <<'EOF'
import json, sys
ok = bad = 0
with open("data/train_svo_silver.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 500:
            break
        row = json.loads(line)
        for sp in row.get("spans", []):
            if sp["label"] not in {"svo_verb", "neg", "pron_subj", "pron_obj", "pron_dem"}:
                if "verb_char_start" in sp:
                    ok += 1
                else:
                    bad += 1
if bad > 0:
    print(f"⚠️  {bad} spans argument SANS verb_char_start — le silver est ancien !")
    sys.exit(1)
else:
    print(f"✅  {ok} spans argument avec verb_char_start — silver conforme")
EOF

# ── 5. Lancement du training ─────────────────────────────────────────────
echo ""
echo "🚀 Lancement du training adaptatif..."
./run_adaptive_training.sh

