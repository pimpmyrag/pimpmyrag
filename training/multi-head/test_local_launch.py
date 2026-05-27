#!/usr/bin/env python3
"""
test_local_launch.py — Valide le training en local avant de lancer sur RunPod.

Usage :
    python3 test_local_launch.py                      # utilise GOLD_VERSION courant
    python3 test_local_launch.py --gold-version v8.20 # version explicite
    python3 test_local_launch.py --n 80               # 80 phrases par split (defaut 100)

Ce script :
  1. Prend les N premières phrases de train/val/test_{GOLD_VERSION}.jsonl
  2. Les écrit dans data/train/val/test_mini-local.jsonl
  3. Lance run_training.py --config configs/local-test.json --gold-version mini-local
  4. Vérifie le code de sortie (0 = OK, non-zero = bug à corriger)

Si ce script passe : le code est prêt pour RunPod.
Si ce script échoue : corriger avant de lancer sur RunPod (évite de gaspiller du GPU RunPod).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent


def make_mini(gold_version: str, n: int) -> str:
    """Crée data/train|val|test_mini-local.jsonl depuis les N premières lignes de gold."""
    mini_version = "mini-local"
    for split in ["train", "val", "test"]:
        src = BASE / "data" / f"{split}_{gold_version}.jsonl"
        if not src.exists():
            print(f"❌ Fichier source introuvable : {src}")
            print(f"   Vérifiez que GOLD_VERSION={gold_version!r} est correct et que DVC pull a été fait.")
            sys.exit(1)
        dst = BASE / "data" / f"{split}_{mini_version}.jsonl"
        lines = src.read_text(encoding="utf-8").splitlines(keepends=True)[:n]
        dst.write_text("".join(lines), encoding="utf-8")
        print(f"  ✅ {split}: {len(lines)} phrases → {dst.name}")
    return mini_version


def main():
    parser = argparse.ArgumentParser(description="Teste le training en local sur un mini-dataset.")
    parser.add_argument("--gold-version", default=None,
                        help="Version dataset (défaut : lu dans launch_training.py)")
    parser.add_argument("--n", type=int, default=100,
                        help="Nombre de phrases par split (défaut : 100)")
    args = parser.parse_args()

    # Lire GOLD_VERSION depuis launch_training.py si non fourni
    gold_version = args.gold_version
    if not gold_version:
        launch_py = BASE.parent.parent / "launch_training.py"
        if launch_py.exists():
            for line in launch_py.read_text(encoding="utf-8").splitlines():
                if "DEFAULT_GOLD_VERSION" in line and "=" in line and not line.strip().startswith("#"):
                    raw = line.split("=", 1)[1]
                    raw = raw.split("#")[0]   # retirer commentaire inline
                    gold_version = raw.strip().strip('"').strip("'")
                    break
        if not gold_version:
            gold_version = "v8.20"
        print(f"📌 GOLD_VERSION auto-détecté : {gold_version}")

    print(f"\n🧪 Test local — dataset {gold_version} ({args.n} phrases/split)")
    print("=" * 60)

    # 1. Créer mini dataset
    print("\n📦 Création du mini-dataset...")
    mini_version = make_mini(gold_version, args.n)

    # 2. Lancer le training local
    print(f"\n🚀 Lancement training (2 epochs CPU, config local-test)...")
    cmd = [
        sys.executable, str(BASE / "run_training.py"),
        "--config",       "configs/local-test.json",
        "--gold-version", mini_version,
        "--device",       "cpu",
    ]
    print(f"   Commande : {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(BASE))

    print("\n" + "=" * 60)
    if result.returncode == 0:
        print("✅ TEST LOCAL RÉUSSI — code prêt pour RunPod")
        print("   Lance : python3 launch_training.py --gold-version", gold_version, "--config svo-v819-rc2")
    else:
        print(f"❌ TEST LOCAL ÉCHOUÉ (code {result.returncode}) — NE PAS lancer sur RunPod")
        print("   Corriger l'erreur ci-dessus avant de relancer.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()


