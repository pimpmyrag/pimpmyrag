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

# ── 4. Vérification rapide du silver (coverage verb_char_start) ──────────
# Note : train_v3.jsonl contient déjà des spans SVO de l'ancienne génération
# (sans verb_char_start). C'est normal — ils seront ignorés par le pointer head
# (gov_verb_tok_start=-1). On vérifie juste que des nouveaux spans ont bien
# le champ, et on affiche le taux de couverture.
echo ""
echo "🔍 Coverage verb_char_start dans le silver (500 premiers exemples)..."
python3 - <<'EOF'
import json, sys
from collections import Counter
# Labels SVO argument qui peuvent avoir verb_char_start (extraits dans la boucle verbe)
ARG_LABELS = {"svo_subject", "svo_object", "svo_iobj", "svo_tcomp", "svo_cause"}
ok = bad = 0
with open("data/train_svo_silver.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 500:
            break
        row = json.loads(line)
        for sp in row.get("spans", []):
            if sp["label"] in ARG_LABELS:
                if "verb_char_start" in sp:
                    ok += 1
                else:
                    bad += 1
total = ok + bad
pct = 100.0 * ok / total if total > 0 else 0
print(f"  Spans argument avec verb_char_start : {ok}/{total} ({pct:.1f}%)")
if ok == 0:
    print("⚠️  Aucun span argument avec verb_char_start — vérifier que build_svo_silver.py est bien la nouvelle version !")
    sys.exit(1)
else:
    print(f"✅  Silver conforme — pointer head supervisé sur {pct:.0f}% des spans argument")
    print(f"    (les {bad} spans sans verb_char_start sont des spans hérités de train_v3.jsonl, gov_verb_tok_start=-1)")
EOF

# ── 5. Lancement du training ─────────────────────────────────────────────
echo ""
echo "🚀 Lancement du training adaptatif..."
./run_adaptive_training.sh

