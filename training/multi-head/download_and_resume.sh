#!/bin/bash
set -e
echo "📥 Téléchargement checkpoint R2 depuis run précédent..."
R2_CKPT="models/v8.1-svobylevel-morpho010-nerwarmup0-cwp0-nocoarsenone-t24-deberta-bs48-RTX_3090-0511-0652/checkpoint_best_multitask.pt"

python3 - <<'PYEOF'
import os, boto3, sys
R2_CKPT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("R2_CKPT")
s3 = boto3.client("s3",
    endpoint_url=os.environ["DVC_R2_ENDPOINT"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name="auto")
print(f"  Downloading {R2_CKPT}...")
s3.download_file("pimpmyrag-data", R2_CKPT, "checkpoint_best_multitask.pt")
sz = os.path.getsize("checkpoint_best_multitask.pt") / 1024**3
print(f"  ✅ checkpoint_best_multitask.pt téléchargé ({sz:.1f}GB)")
PYEOF

echo "🚀 Reprise training ep61→80 avec START_LEVEL=5 (SVO 100%)"
export START_LEVEL=5
export START_EPOCH=61
export KEEP_CHECKPOINT=1
export MAX_EPOCHS=80
export R2_CKPT="$R2_CKPT"
./setup_runpod.sh

