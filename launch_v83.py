#!/usr/bin/env python3
"""
Lance un pod RunPod pour training v8.3.
Pattern canonique : git clone main → setup_runpod.sh (gère deps / DVC / training / upload)
"""
import runpod, os, json, time

# Chargement .secrets.env
secrets_path = os.path.join(os.path.dirname(__file__), 'training', 'multi-head', '.secrets.env')
for line in open(secrets_path):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())

runpod.api_key = os.environ['RUNPOD_API_KEY']

# Env vars à injecter dans le pod
env_vars = {k: v for k, v in {
    "WANDB_API_KEY":        os.environ.get("WANDB_API_KEY", ""),
    "AWS_ACCESS_KEY_ID":    os.environ.get("AWS_ACCESS_KEY_ID", ""),
    "AWS_SECRET_ACCESS_KEY":os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    "DVC_R2_ENDPOINT":      os.environ.get("DVC_R2_ENDPOINT", ""),
}.items() if v}

# Kill les pods en cours
print("🔍 Vérification des pods en cours...")
existing = runpod.get_pods()
killed = []
for p in existing:
    if p.get("desiredStatus") in ("RUNNING", "PENDING"):
        print(f"  🛑 Kill {p['id']} ({p.get('name', '?')})...")
        try:
            runpod.terminate_pod(p["id"])
            killed.append(p["id"])
        except Exception as e:
            print(f"     ⚠️  {e}")
if killed:
    print(f"  ✅ {len(killed)} pod(s) terminé(s), attente 5s...")
    time.sleep(5)
else:
    print("  ✅ Aucun pod actif.")

# Commande de démarrage — Checkout 8ef6a03 (contient setup_runpod.sh v8.3 + run_adaptive_training.sh fix)
commit_sha = "8ef6a03"  # Contient TOUT : dataset v8.3 .dvc + setup_runpod.sh v8.3 + ramp fix
start_cmd = (
    "bash -c '"
    "cd /workspace && "
    "git clone https://github.com/pimpmyrag/pimpmyrag.git pimpmyrag 2>&1 | tail -3 && "
    "cd pimpmyrag && git fetch origin && git checkout " + commit_sha + " && "
    "cd training/multi-head && "
    "chmod +x setup_runpod.sh && "
    "./setup_runpod.sh 2>&1 | tee /workspace/training.log"
    "'"
)

# Création du pod
print("\n🚀 Création pod A100 40GB (v8.3 - pronoms + certainty + ramp SVO/morpho)...")
pod = runpod.create_pod(
    name="pimpmyrag-training-v8.3",
    image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    gpu_type_id="NVIDIA A100-SXM4-40GB",   # Supportée torch 2.4->2.6, BS=128, ~3x plus vite
    cloud_type="SECURE",
    gpu_count=1,
    volume_in_gb=50,
    container_disk_in_gb=30,
    docker_args=start_cmd,
    env=env_vars,
    ports="22/tcp",
)

pod_id = pod.get("id", "?")
print(f"\n✅ Pod lancé !")
print(f"   ID     : {pod_id}")
print(f"   Nom    : pimpmyrag-training-v8.3")
print(f"   Dataset: v8.3 (pronoms + certainty)")
print(f"   Ramp   : SVO 35ep (ep13→47), morpho 25ep (ep21→45), warmup NER 12ep")
print(f"   W&B    : https://wandb.ai/pimpmyrag-pimpmyrag/pimpmyrag-ner")
print(f"   Logs   : /workspace/training.log (dans le pod)")
print(f"   Kill   : runpodctl remove pod {pod_id}")
print()
print(json.dumps(pod, indent=2))

