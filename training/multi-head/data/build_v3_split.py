#!/usr/bin/env python3
"""
build_v3_split.py
==================
Construit les splits train_v3 / val_v3 / test_v3 en fusionnant toutes les sources disponibles,
avec une répartition stratifiée qui garantit une couverture minimale des labels rares
dans val et test.

Sources fusionnées (par ordre de priorité de qualité) :
  - train_v2.jsonl, val_v2.jsonl, test_v2.jsonl      (gold)
  - train_wiki_claude_annotated.jsonl                  (silver Claude + UD Stanza)

Algorithme
──────────
1. Pooler tous les exemples (dédup par id).
2. Étiqueter chaque exemple par l'ensemble de ses labels hint_*.
3. Calculer la fréquence globale de chaque label.
4. Pour chaque label, trier les exemples qui le contiennent par score de rareté
   (somme des inv_freq de ses labels) et allouer un quota dans val et test.
5. Compléter val et test à taille cible avec le reste des exemples
   (tirage stratifié sur la distribution globale).
6. Le reste va dans train.

Usage
─────
  python3 data/build_v3_split.py \\
      --data-dir data/ \\
      --val-size  3000 \\
      --test-size 3000 \\
      --rare-quota 12      # nb min d'exemples par label rare dans val ET test \\
      --rare-threshold 400 # label considéré rare si < N exemples dans le pool \\
      --seed 42
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return examples


def write_jsonl(examples: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Analyse labels
# ─────────────────────────────────────────────────────────────────────────────

def label_set(ex: dict) -> frozenset[str]:
    return frozenset(
        sp["label"] for sp in ex.get("spans", [])
        if sp.get("label", "").startswith("hint_")
    )


def global_label_counts(examples: list[dict]) -> Counter:
    counts: Counter = Counter()
    for ex in examples:
        for lbl in label_set(ex):
            counts[lbl] += 1
    return counts


def rarity_score(ex: dict, inv_freq: dict[str, float]) -> float:
    return sum(inv_freq.get(lbl, 100.0) for lbl in label_set(ex))


# ─────────────────────────────────────────────────────────────────────────────
# Répartition stratifiée
# ─────────────────────────────────────────────────────────────────────────────

def build_splits(
    pool: list[dict],
    val_size: int,
    test_size: int,
    rare_threshold: int,
    rare_quota: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Retourne (train, val, test).
    """
    rng = random.Random(seed)

    label_counts = global_label_counts(pool)
    inv_freq = {
        lbl: 1.0 / (cnt / len(pool)) for lbl, cnt in label_counts.items() if cnt > 0
    }

    # Labels rares
    rare_labels = {lbl for lbl, cnt in label_counts.items() if cnt < rare_threshold}
    print(f"\n  Labels rares (< {rare_threshold} exemples dans le pool) : {len(rare_labels)}")
    for lbl in sorted(rare_labels, key=lambda l: label_counts[l]):
        print(f"    {lbl:<30} : {label_counts[lbl]}")

    # Index : label → liste d'indices triés par score décroissant
    label_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(pool):
        for lbl in label_set(ex):
            label_to_indices[lbl].append(i)

    # Trier par score de rareté décroissant (exemples les plus rares en tête)
    for lbl in label_to_indices:
        label_to_indices[lbl].sort(key=lambda i: rarity_score(pool[i], inv_freq), reverse=True)

    val_indices: set[int] = set()
    test_indices: set[int] = set()

    # 1. Quota pour labels rares dans val et test
    for lbl in rare_labels:
        candidates = label_to_indices.get(lbl, [])
        quota = min(rare_quota, len(candidates) // 3)  # max 1/3 du stock dispo
        added_val = 0
        added_test = 0
        for idx in candidates:
            if added_val < quota and idx not in val_indices and idx not in test_indices:
                val_indices.add(idx)
                added_val += 1
            elif added_test < quota and idx not in val_indices and idx not in test_indices:
                test_indices.add(idx)
                added_test += 1
            if added_val >= quota and added_test >= quota:
                break

    # 2. Compléter val et test à taille cible
    remaining = [i for i in range(len(pool)) if i not in val_indices and i not in test_indices]
    rng.shuffle(remaining)

    need_val = val_size - len(val_indices)
    need_test = test_size - len(test_indices)

    for idx in remaining[:need_val]:
        val_indices.add(idx)
    for idx in remaining[need_val: need_val + need_test]:
        test_indices.add(idx)

    train_indices = [i for i in range(len(pool)) if i not in val_indices and i not in test_indices]
    rng.shuffle(train_indices)

    train = [pool[i] for i in train_indices]
    val   = [pool[i] for i in sorted(val_indices)]
    test  = [pool[i] for i in sorted(test_indices)]

    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Rapport
# ─────────────────────────────────────────────────────────────────────────────

def report(split_name: str, examples: list[dict], ref_counts: Counter) -> None:
    counts = global_label_counts(examples)
    n = len(examples)
    print(f"\n{'─'*65}")
    print(f"  {split_name}  ({n} exemples)")
    print(f"  {'Label':<30} {'n/1k':>7}  {'ref/1k':>7}  {'ratio':>6}")
    print(f"{'─'*65}")
    for lbl in sorted(set(counts) | set(ref_counts), key=lambda l: -ref_counts.get(l, 0)):
        s = counts.get(lbl, 0) / n * 1000 if n else 0
        r = ref_counts.get(lbl, 0)
        ratio = (counts.get(lbl, 0) / r) if r > 0 else float("inf")
        flag = "⚠ " if ratio < 0.5 else "  "
        print(f"  {flag}{lbl:<28} {s:>7.1f}  {r:>7}  {ratio:>6.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build v3 stratified splits")
    parser.add_argument("--data-dir",         default="data")
    parser.add_argument("--val-size",          type=int, default=3000)
    parser.add_argument("--test-size",         type=int, default=3000)
    parser.add_argument("--rare-threshold",    type=int, default=400,
                        help="Label est rare si < N exemples dans le pool global")
    parser.add_argument("--rare-quota",        type=int, default=12,
                        help="Nombre minimum d'exemples par label rare dans val ET test")
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--suffix-out",        default="_v3",
                        help="Suffixe des fichiers de sortie")
    parser.add_argument("--report",            action="store_true",
                        help="Afficher le rapport de distribution après split")
    parser.add_argument("--dry-run",           action="store_true",
                        help="Calculer uniquement, sans écrire les fichiers")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # ── Sources ──────────────────────────────────────────────────────────────
    # On prend les versions *_svo_silver (= gold NER + spans UD/SVO Stanza)
    # plutôt que les *_v2 bruts pour avoir un dataset commun NER+UD.
    sources = [
        ("gold+ud train", data_dir / "train_svo_silver.jsonl"),
        ("gold+ud val",   data_dir / "val_svo_silver.jsonl"),
        ("gold+ud test",  data_dir / "test_svo_silver.jsonl"),
        ("wiki claude",   data_dir / "train_wiki_claude_annotated.jsonl"),
    ]

    pool: list[dict] = []
    seen_ids: set[str] = set()

    for name, path in sources:
        if not path.exists():
            print(f"  ⚠️  {path.name} introuvable, skip.")
            continue
        examples = load_jsonl(path)
        before = len(pool)
        for ex in examples:
            eid = str(ex.get("id", ""))
            if eid and eid in seen_ids:
                continue
            seen_ids.add(eid)
            pool.append(ex)
        added = len(pool) - before
        print(f"  ✅ {path.name:<45} {added:>6} exemples chargés")

    print(f"\n  Pool total : {len(pool)} exemples")

    # ── Références gold ───────────────────────────────────────────────────────
    # Pour le rapport de distribution, on utilise les counts NER du pool entier
    gold_counts = global_label_counts([ex for ex in pool])


    # ── Split ────────────────────────────────────────────────────────────────
    train, val, test = build_splits(
        pool,
        val_size=args.val_size,
        test_size=args.test_size,
        rare_threshold=args.rare_threshold,
        rare_quota=args.rare_quota,
        seed=args.seed,
    )

    print(f"\n  Répartition finale :")
    print(f"    train{args.suffix_out} : {len(train):>6}")
    print(f"    val{args.suffix_out}   : {len(val):>6}")
    print(f"    test{args.suffix_out}  : {len(test):>6}")

    if args.report:
        report(f"val{args.suffix_out}", val, gold_counts)
        report(f"test{args.suffix_out}", test, gold_counts)
        report(f"train{args.suffix_out}", train, gold_counts)

    if not args.dry_run:
        write_jsonl(train, data_dir / f"train{args.suffix_out}.jsonl")
        write_jsonl(val,   data_dir / f"val{args.suffix_out}.jsonl")
        write_jsonl(test,  data_dir / f"test{args.suffix_out}.jsonl")
        print(f"\n  ✅ train{args.suffix_out}.jsonl / val{args.suffix_out}.jsonl / test{args.suffix_out}.jsonl écrits dans {data_dir}/")
    else:
        print("\n  [dry-run] Aucun fichier écrit.")


if __name__ == "__main__":
    main()

