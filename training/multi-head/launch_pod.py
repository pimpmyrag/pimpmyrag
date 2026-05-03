"""Lance un pod RunPod pour le training v6.3 (RTX 5090)."""
import runpod, os, json

# Charge .secrets.env si nécessaire
for line in open('.secrets.env'):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())

runpod.api_key = os.environ['RUNPOD_API_KEY']

# Variables d'environnement à injecter dans le pod
env_vars = {
    "WANDB_API_KEY":        os.environ.get("WANDB_API_KEY", ""),
    "AWS_ACCESS_KEY_ID":    os.environ.get("AWS_ACCESS_KEY_ID", ""),
    "AWS_SECRET_ACCESS_KEY":os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    "DVC_R2_ENDPOINT":      os.environ.get("DVC_R2_ENDPOINT", ""),
    "GITHUB_TOKEN":         os.environ.get("GITHUB_TOKEN", ""),
    "HF_TOKEN":             os.environ.get("HF_TOKEN", ""),
}
# Retire les valeurs vides
env_vars = {k: v for k, v in env_vars.items() if v}

# Commande de démarrage : git clone + setup + training
GITHUB_REPO = "https://github.com/pimpmyrag/pimpmyrag.git"
if env_vars.get("GITHUB_TOKEN"):
    GITHUB_REPO = f"https://{env_vars['GITHUB_TOKEN']}@github.com/pimpmyrag/pimpmyrag.git"

BRANCH = "feature-classif-improvements"

start_cmd = (
    "bash -c '"
    "cd /workspace && "
    f"git clone --branch {BRANCH} {GITHUB_REPO} pimpmyrag 2>&1 | tail -3 && "
    "cd pimpmyrag/training/multi-head && "
    "chmod +x setup_runpod.sh && "
    "./setup_runpod.sh 2>&1 | tee /workspace/training_v63.log"
    "'"
)

pod = runpod.create_pod(
    name="pimpmyrag-training-v6.3",
    image_name="runpod/pytorch:2.6.0-py3.11-cuda12.8.0-devel-ubuntu22.04",
    gpu_type_id="NVIDIA GeForce RTX 5090",
    cloud_type="SECURE",
    gpu_count=1,
    volume_in_gb=50,
    container_disk_in_gb=30,
    docker_args=start_cmd,
    env=env_vars,
    ports="22/tcp",
)

print("Pod lancé !")
print(json.dumps(pod, indent=2))

