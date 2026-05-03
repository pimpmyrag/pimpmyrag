#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
launch_training.py — Lance un pod RunPod, attend la fin du training,
récupère les artefacts (checkpoint + ONNX) et coupe le pod.

Usage :
    python3 launch_training.py                   # GPU auto (A100 SXM si dispo)
    python3 launch_training.py --gpu RTX_4090     # GPU spécifique
    python3 launch_training.py --gpu A100_SXM --spot  # spot (interruptible, -40%)
    python3 launch_training.py --dry-run          # affiche le config sans créer le pod

Prérequis :
    - .secrets.env rempli (RUNPOD_API_KEY, AWS_*, WANDB_API_KEY)
    - dvc push déjà fait (datasets v5 sur R2)
    - pip install runpod
"""
import argparse
import os
import sys
import time
from pathlib import Path
from dotenv import dotenv_values

# ── Chargement des secrets ────────────────────────────────────────────────────
SECRETS_FILE = Path(__file__).parent / ".secrets.env"
if not SECRETS_FILE.exists():
    print(f"❌ {SECRETS_FILE} introuvable — copier .secrets.env.example et remplir")
    sys.exit(1)
secrets = dotenv_values(SECRETS_FILE, encoding='utf-8')

RUNPOD_API_KEY = secrets.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    print("❌ RUNPOD_API_KEY manquant dans .secrets.env")
    sys.exit(1)

import runpod
runpod.api_key = RUNPOD_API_KEY

# ── Config GPU ────────────────────────────────────────────────────────────────
# Prix indicatifs RunPod (mai 2026, on-demand) :
#   RTX_4090        ~$0.74/h   — bon rapport qualité/prix
#   A100_SXM        ~$1.89/h   — plus rapide, batch plus grand
#   RTX_3090        ~$0.44/h   — budget
GPU_PROFILES = {
    "RTX_4090": {
        "gpu_type_id": "NVIDIA GeForce RTX 4090",
        "gpu_count":   1,
        "container_disk_in_gb": 50,
        "volume_in_gb": 20,
    },
    "A100_SXM": {
        "gpu_type_id": "NVIDIA A100 80GB PCIe",
        "gpu_count":   1,
        "container_disk_in_gb": 80,
        "volume_in_gb": 30,
    },
    "RTX_3090": {
        "gpu_type_id": "NVIDIA GeForce RTX 3090",
        "gpu_count":   1,
        "container_disk_in_gb": 50,
        "volume_in_gb": 20,
    },
}
DEFAULT_GPU = "RTX_4090"


def get_env_vars(secrets: dict) -> dict:
    """Construit le dict des variables d env a injecter dans le pod."""
    env = {}
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                "DVC_R2_ENDPOINT", "WANDB_API_KEY"):
        val = secrets.get(key) or os.environ.get(key, "")
        if val:
            env[key] = val
        elif key == "WANDB_API_KEY":
            env["WANDB_MODE"] = "offline"
    return env


def launch_pod(gpu_profile: str, spot: bool, dry_run: bool):
    profile = GPU_PROFILES.get(gpu_profile)
    if not profile:
        print(f"❌ GPU inconnu : {gpu_profile}. Choix : {list(GPU_PROFILES)}")
        sys.exit(1)

    env_vars = get_env_vars(secrets)
    config = {
        "name":             "pimpmyrag-training",
        "image_name":       "runpod/pytorch:1.0.3-cu1290-torch260-ubuntu2204",
        "gpu_type_id":      profile["gpu_type_id"],
        "gpu_count":        profile["gpu_count"],
        "container_disk_in_gb": profile["container_disk_in_gb"],
        "volume_in_gb":     profile["volume_in_gb"],
        "volume_mount_path": "/workspace",
        "env":              env_vars,
        # Commande injectée via docker_args (exécutée au démarrage du container)
        "docker_args": (
            "bash -c 'apt-get update -qq && apt-get install -y -qq git && "
            "git clone https://github.com/pimpmyrag/pimpmyrag.git /workspace/pimpmyrag && "
            "cd /workspace/pimpmyrag/training/multi-head && "
            "chmod +x setup_runpod.sh && ./setup_runpod.sh 2>&1 | tee /workspace/training.log'"
        ),
        "ports": "8888/http",  # Jupyter optionnel si besoin de debug
    }

    print(f"\n🚀 Config pod [{gpu_profile}{'  SPOT' if spot else ''}]")
    print(f"   Image   : {config['image_name']}")
    print(f"   GPU     : {profile['gpu_type_id']} x{profile['gpu_count']}")
    print(f"   Disk    : {profile['container_disk_in_gb']}GB container + {profile['volume_in_gb']}GB volume")
    print(f"   Env vars: {list(env_vars.keys())}")

    if dry_run:
        print("\n  --dry-run : pod NON cree")
        return

    print("\n⏳ Création du pod...")
    try:
        pod = runpod.create_pod(**config)
    except Exception as e:
        print(f"❌ Erreur création pod : {e}")
        sys.exit(1)

    pod_id = pod.get("id") or pod.get("pod", {}).get("id")
    print(f"✅ Pod créé : {pod_id}")
    print(f"   Dashboard : https://www.runpod.io/console/pods/{pod_id}")
    print(f"\n📋 Surveiller les logs :")
    print(f"   runpod pod logs {pod_id} --tail 100\n")

    # ── Attente démarrage ─────────────────────────────────────────────────────
    print("⏳ Attente démarrage du pod (max 5 min)...")
    for i in range(30):
        time.sleep(10)
        try:
            info = runpod.get_pod(pod_id)
            status = info.get("desiredStatus") or info.get("pod", {}).get("desiredStatus", "?")
            print(f"   [{i*10}s] status={status}")
            if status == "RUNNING":
                print("✅ Pod démarré !")
                break
        except Exception:
            pass
    else:
        print("⚠️  Pod pas encore RUNNING après 5 min — vérifier le dashboard")

    print(f"\n💡 Pour arrêter le pod après le training :")
    print(f"   python3 launch_training.py --stop {pod_id}")


def stop_pod(pod_id: str):
    print(f"🛑 Arrêt du pod {pod_id}...")
    try:
        runpod.terminate_pod(pod_id)
        print("✅ Pod terminé")
    except Exception as e:
        print(f"❌ Erreur : {e}")


def list_pods():
    print("📋 Pods actifs :")
    try:
        pods = runpod.get_pods()
        if not pods:
            print("  (aucun)")
            return
        for p in pods:
            pid = p.get("id", "?")
            name = p.get("name", "?")
            status = p.get("desiredStatus", "?")
            gpu = p.get("machine", {}).get("gpuDisplayName", "?")
            print(f"  {pid}  [{status}]  {name}  {gpu}")
    except Exception as e:
        print(f"❌ Erreur : {e}")


def main():
    parser = argparse.ArgumentParser(description="Lance un training RunPod")
    parser.add_argument("--gpu", default=DEFAULT_GPU,
                        choices=list(GPU_PROFILES),
                        help=f"GPU profile (défaut: {DEFAULT_GPU})")
    parser.add_argument("--spot", action="store_true",
                        help="Mode spot/interruptible (environ 40%% moins cher)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche la config sans créer le pod")
    parser.add_argument("--stop", metavar="POD_ID",
                        help="Arrête un pod en cours")
    parser.add_argument("--list", action="store_true",
                        help="Liste les pods actifs")
    args = parser.parse_args()

    if args.list:
        list_pods()
    elif args.stop:
        stop_pod(args.stop)
    else:
        launch_pod(args.gpu, args.spot, args.dry_run)


if __name__ == "__main__":
    main()

