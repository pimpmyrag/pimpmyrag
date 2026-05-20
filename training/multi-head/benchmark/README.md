# Benchmark Zero-Shot NER
Evaluation zero-shot sur des benchmarks publics jamais vus pendant l'entrainement.
## Benchmarks
- **MultiNERD-fr** : 15 types, ~3k phrases FR — recall + F1 PER/LOC/ORG
- **Few-NERD** : 66 fine types, EN (cross-lingue) — recall fine/coarse
## Usage
    cd training/multi-head
    python benchmark/run_all.py --checkpoint ../../checkpoint_best_multitask.pt
    python benchmark/run_all.py --limit 200      # rapide
    python benchmark/run_all.py --only multinerd  # un seul
## Deps
    pip install datasets torch transformers
