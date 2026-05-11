#!/usr/bin/env python3
"""
Relance h81d4u6n depuis ep60 avec START_LEVEL=5 (SVO 100%) jusqu'à ep80.
Télécharge le checkpoint R2 et reprend le training.
"""
import runpod, os, json, time

secrets_path = 'training/multi-head/.secrets.env'
for line in open(secrets_path):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())

runpod.api_key = os.environ['RUNPOD_API_KEY']

env_vars = {k: v for k, v in {
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
    "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
    "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    "DVC_R2_ENDPOINT": os.environ.get("DVC_R2_ENDPOINT", ""),
}.items() if v}

# Kill pods en cours
print("🔍 Vérification pods...")
for p in runpod.get_pods():
    if p.get("desiredStatus") in ("RUNNING", "PENDING"):
        print(f"  🛑 Kill {p['id']} ({p.get('name', '?')})")
        try:
            runpod.terminate_pod(p["id"])
            time.sleep(3)
        except: pass

# Commande simplifiée : clone + script download_and_resume.sh créé dans le repo
start_cmd = (
    "bash -c '"
    "cd /workspace && "
    "rm -rf pimpmyrag && "
    "git clone https://github.com/pimpmyrag/pimpmyrag.git pimpmyrag 2>&1 | tail -3 && "
    "cd pimpmyrag/training/multi-head && "
    "chmod +x download_and_resume.sh && "
    "./download_and_resume.sh 2>&1 | tee /workspace/training_resume.log"
    "'"
)

print("\n🚀 Création pod (reprise h81d4u6n ep60→80, SVO 100%)...")
pod = runpod.create_pod(
    name="pimpmyrag-v81-resume-svo100",
    image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    gpu_type_id="NVIDIA GeForce RTX 3090",
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
print(f"   Nom    : pimpmyrag-v81-resume-svo100")
print(f"   Commit : HEAD/main (4b2182a)")
print(f"   Reprise: ep61→80, START_LEVEL=5 (SVO 100% + hard=6)")
print(f"   W&B    : https://wandb.ai/pimpmyrag-pimpmyrag/pimpmyrag-ner")
print(f"   Kill   : runpodctl remove pod {pod_id}")
print(json.dumps(pod, indent=2))

