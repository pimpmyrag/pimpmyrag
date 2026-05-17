#!/usr/bin/env python3
"""
Lance un pod RunPod pour training v8.6.
Pattern canonique : git clone main → setup_runpod.sh (gère deps / DVC / training / upload)

Fix v8.6 vs v8.5 :
- Dataset propre : 64 offset errors (vs 8398 en v8.5, 9324 dès v8.2)
- Stanza spans revus par Claude Haiku sur train+val+test
- NER_WARMUP_EPOCHS 6→0 : corrélation claire warmup=0 → meilleur boundary (0.9267 en v8.1)
- GOLD_VERSION centralisé dans setup_runpod.sh + run_adaptive_training.sh
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
    "GOLD_VERSION":         "v8.6",
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

# Commit v8.6 — à mettre à jour après git push
commit_sha = "main"
start_cmd = (
    "bash -c '"
    "cd /workspace && "
    "git clone https://github.com/pimpmyrag/pimpmyrag.git pimpmyrag 2>&1 | tail -3 && "
    "cd pimpmyrag && git fetch origin && git checkout " + commit_sha + " && "
    "cd training/multi-head && "
    "chmod +x setup_runpod.sh && "
    "GOLD_VERSION=v8.6 ./setup_runpod.sh 2>&1 | tee /workspace/training.log"
    "'"
)

# GPU par ordre de préférence prix/perf
GPU_PRIORITY = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A5000",
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA A100-SXM4-40GB",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA L40S",
]

pod = None
for gpu_id in GPU_PRIORITY:
    try:
        print(f"  Tentative sur {gpu_id}...")
        pod = runpod.create_pod(
            name="pimpmyrag-training-v8.6",
            image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            gpu_type_id=gpu_id,
            cloud_type="SECURE",
            gpu_count=1,
            volume_in_gb=50,
            container_disk_in_gb=30,
            docker_args=start_cmd,
            env=env_vars,
            ports="22/tcp",
        )
        print(f"  ✅ Pod créé sur {gpu_id}")
        break
    except Exception as e:
        print(f"  ❌ {gpu_id} non dispo : {e}")

if not pod:
    print("❌ Aucun GPU disponible dans la liste. Réessayez plus tard.")
    exit(1)

pod_id = pod.get("id", "?")
print(f"\n✅ Pod lancé !")
print(f"   ID     : {pod_id}")
print(f"   Nom    : pimpmyrag-training-v8.6")
print(f"   GPU    : RTX 3090 24GB (BS=48×accum=2=96 effectif)")
print(f"   Commit : {commit_sha}")
print(f"   Dataset: v8.6 (64 offset errs, Stanza revus Claude train+val+test)")
print(f"   Fix    : NER_WARMUP 6→0 (corrélation boundary +6pts vs warmup=6)")
print(f"   Config : NER_WARMUP=0 → SVO ramp 20ep → morpho delay 8ep → role delay 12ep")
print(f"   W&B    : https://wandb.ai/pimpmyrag-pimpmyrag/pimpmyrag-ner")
print(f"   Logs   : /workspace/training.log (dans le pod)")
print(f"   Kill   : runpodctl remove pod {pod_id}")
print()
print(json.dumps(pod, indent=2))

