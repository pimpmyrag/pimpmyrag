#!/usr/bin/env python3
"""
run_all.py — Lance les deux benchmarks zero-shot et produit un rapport unifié.

Usage :
    cd training/multi-head
    python benchmark/run_all.py --checkpoint ../../checkpoint_best_multitask.pt
    python benchmark/run_all.py --checkpoint ../../checkpoint_best_multitask.pt --limit 200  # rapide
"""
import argparse
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent
MULTI_HEAD = BENCH_DIR.parent

BENCHMARKS = [
    {
        "name": "MultiNERD-fr (zero-shot)",
        "script": "benchmark_multinerd.py",
        "description": "15 types NER, 3k phrases FR. Recall + F1 PER/LOC/ORG.",
    },
    {
        "name": "Few-NERD (zero-shot cross-lingue FR→EN)",
        "script": "benchmark_fewnerd.py",
        "description": "66 fine types, anglais. Recall fine + coarse sur 500 phrases.",
    },
]

def main():
    ap = argparse.ArgumentParser(description="Lance tous les benchmarks NER zero-shot")
    ap.add_argument("--checkpoint", default=str(MULTI_HEAD.parent.parent / "checkpoint_best_multitask.pt"))
    ap.add_argument("--model-name", default="microsoft/deberta-v3-base")
    ap.add_argument("--limit", type=int, default=0, help="Limiter le nb de phrases (0=tout)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tau-boundary", type=float, default=0.70)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None, choices=["multinerd", "fewnerd"],
                    help="Lancer un seul benchmark")
    args = ap.parse_args()

    print("=" * 70)
    print("  BENCHMARK SUITE — NER Zero-Shot")
    print("=" * 70)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {args.device}")
    print(f"  τ_boundary : {args.tau_boundary}")
    if args.limit:
        print(f"  Limit      : {args.limit} phrases")
    print()

    for bench in BENCHMARKS:
        if args.only and args.only not in bench["script"]:
            continue

        print(f"\n{'#' * 70}")
        print(f"# {bench['name']}")
        print(f"# {bench['description']}")
        print(f"{'#' * 70}\n")

        cmd = [
            sys.executable, str(BENCH_DIR / bench["script"]),
            "--checkpoint", args.checkpoint,
            "--model-name", args.model_name,
            "--batch-size", str(args.batch_size),
            "--tau-boundary", str(args.tau_boundary),
            "--device", args.device,
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]

        result = subprocess.run(cmd, cwd=str(MULTI_HEAD))
        if result.returncode != 0:
            print(f"❌ {bench['name']} a échoué (exit code {result.returncode})")
        else:
            print(f"✅ {bench['name']} terminé")

    print(f"\n{'=' * 70}")
    print("  BENCHMARKS TERMINÉS")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

