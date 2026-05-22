#!/usr/bin/env python3
"""
Script de lancement RunPod — STABLE, versionné, référence unique.

Usage :
    python3 launch_training.py                        # clone main, GOLD_VERSION courant
    python3 launch_training.py --sha abc1234          # SHA git spécifique
    python3 launch_training.py --gold-version v8.7    # surcharge version dataset
    python3 launch_training.py --gpu "RTX 4090"       # GPU préféré en premier
    python3 launch_training.py --dry-run              # affiche la config sans lancer

Variables d'environnement (depuis training/multi-head/.secrets.env) :
    RUNPOD_API_KEY, WANDB_API_KEY, AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY, DVC_R2_ENDPOINT

Le pod clone le repo, checkout le SHA (ou main), puis lance :
    GOLD_VERSION=<version> ./setup_runpod.sh
Qui lui-même lance run_adaptive_training.sh.
"""
import runpod, os, json, time, argparse

# ── Config par défaut ─────────────────────────────────────────────────────────
DEFAULT_GOLD_VERSION = "v8.18"  # ← seul endroit à mettre à jour entre versions
DEFAULT_BRANCH       = "main"   # branche ou SHA à cloner (surcharge avec --sha)
DEFAULT_GPU_PRIORITY = [
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
REPO_URL   = "https://github.com/pimpmyrag/pimpmyrag.git"
IMAGE_NAME = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
# ──────────────────────────────────────────────────────────────────────────────

def load_secrets():
    secrets_path = os.path.join(os.path.dirname(__file__), 'training', 'multi-head', '.secrets.env')
    for line in open(secrets_path):
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

def build_start_cmd(ref: str, gold_version: str, loss_weighting: str = "fixed") -> str:
    return (
        "bash -c '"
        "cd /workspace && "
        f"git clone {REPO_URL} pimpmyrag 2>&1 | tail -3 && "
        f"cd pimpmyrag && git fetch origin && git checkout {ref} && "
        "cd training/multi-head && "
        "chmod +x setup_runpod.sh && "
        f"GOLD_VERSION={gold_version} LOSS_WEIGHTING={loss_weighting} ./setup_runpod.sh 2>&1 | tee /workspace/training.log"
        "'"
    )

def main():
    parser = argparse.ArgumentParser(description="Lance un pod RunPod pour training pimpmyrag")
    parser.add_argument("--sha",          default=DEFAULT_BRANCH,       help=f"SHA ou branche git (défaut: {DEFAULT_BRANCH})")
    parser.add_argument("--gold-version", default=DEFAULT_GOLD_VERSION, help=f"Version dataset gold (défaut: {DEFAULT_GOLD_VERSION})")
    parser.add_argument("--gpu",          default=None,                 help="GPU à mettre en tête de liste (ex: 'RTX 4090')")
    parser.add_argument("--loss-weighting", default="fixed",            help="Stratégie loss weighting: fixed|uncertainty|gradnorm")
    parser.add_argument("--dry-run",      action="store_true",          help="Affiche la config sans lancer le pod")
    parser.add_argument("--no-kill",      action="store_true",          help="Ne PAS tuer les pods existants (runs parallèles)")
    args = parser.parse_args()

    load_secrets()
    runpod.api_key = os.environ['RUNPOD_API_KEY']

    gold_version = args.gold_version
    ref          = args.sha
    start_cmd    = build_start_cmd(ref, gold_version, args.loss_weighting)

    # GPU priority — permet de mettre un GPU préféré en premier
    gpu_list = list(DEFAULT_GPU_PRIORITY)
    if args.gpu:
        preferred = next((g for g in gpu_list if args.gpu.lower() in g.lower()), None)
        if preferred:
            gpu_list.remove(preferred)
            gpu_list.insert(0, preferred)

    env_vars = {k: v for k, v in {
        "WANDB_API_KEY":        os.environ.get("WANDB_API_KEY", ""),
        "AWS_ACCESS_KEY_ID":    os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY":os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        "DVC_R2_ENDPOINT":      os.environ.get("DVC_R2_ENDPOINT", ""),
        "GOLD_VERSION":         gold_version,
    }.items() if v}

    print(f"🚀 Config training pimpmyrag")
    print(f"   Dataset  : {gold_version}")
    print(f"   Git ref  : {ref}")
    print(f"   GPU list : {gpu_list[0]} (+ {len(gpu_list)-1} fallbacks)")
    print(f"   Image    : {IMAGE_NAME}")

    if args.dry_run:
        print(f"\n🔎 Dry-run — start_cmd :")
        print(f"   {start_cmd}")
        return

    # Kill les pods en cours (sauf si --no-kill)
    print("\n🔍 Vérification des pods en cours...")
    existing = runpod.get_pods()
    killed = []
    if args.no_kill:
        active = [p for p in existing if p.get("desiredStatus") in ("RUNNING", "PENDING")]
        if active:
            print(f"  ℹ️  --no-kill : {len(active)} pod(s) conservé(s) ({', '.join(p['id'] for p in active)})")
        else:
            print("  ✅ Aucun pod actif.")
    else:
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

    # Création du pod
    pod = None
    for gpu_id in gpu_list:
        try:
            print(f"  Tentative sur {gpu_id}...")
            pod = runpod.create_pod(
                name=f"pimpmyrag-training-{gold_version}",
                image_name=IMAGE_NAME,
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
        print("❌ Aucun GPU disponible. Réessayez plus tard.")
        exit(1)

    pod_id = pod.get("id", "?")
    print(f"\n✅ Pod lancé !")
    print(f"   ID      : {pod_id}")
    print(f"   Nom     : pimpmyrag-training-{gold_version}")
    print(f"   Dataset : {gold_version}")
    print(f"   Git ref : {ref}")
    print(f"   W&B     : https://wandb.ai/pimpmyrag-pimpmyrag/pimpmyrag-ner")
    print(f"   Logs    : /workspace/training.log (dans le pod)")
    print(f"   Kill    : runpodctl remove pod {pod_id}")
    print()
    print(json.dumps(pod, indent=2))

if __name__ == "__main__":
    main()

