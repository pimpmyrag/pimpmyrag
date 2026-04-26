"""
filter_rare_svo.py
==================
Filtre un fichier SVO silver pour ne conserver que les exemples
les plus "utiles" du point de vue de la rareté des labels NER.

Algorithme
──────────
1. Calcule la fréquence normalisée de chaque label NER dans le gold de référence
   (train_v2.jsonl par défaut).
2. Calcule pour chaque exemple SVO silver un score de rareté :
       score = Σ  (1 / freq_gold_normalisée(label))
   en ne comptant que les spans de type hint_* (NER).
3. Trie les exemples par score décroissant et conserve les top-K%.
4. Écrit le fichier filtré + un rapport de distribution avant/après.

Usage
─────
  python filter_rare_svo.py \\
      --input  train_svo_silver.jsonl \\
      --output train_svo_silver_filtered.jsonl \\
      --gold   train_v2.jsonl \\
      --keep   0.6               # garder 60% (défaut)
      --min-score 0.0            # seuil absolu optionnel
      --report                   # afficher la distribution avant/après
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Calcul des fréquences gold
# ─────────────────────────────────────────────────────────────────────────────

def compute_gold_freqs(gold_path: Path) -> dict[str, float]:
    """
    Retourne un dict {label: spans_per_1000_examples} pour les labels hint_*.
    """
    label_counts: Counter = Counter()
    n = 0
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            n += 1
            for sp in ex.get("spans", []):
                lbl = sp["label"]
                if lbl.startswith("hint_"):
                    label_counts[lbl] += 1
    if n == 0:
        raise ValueError(f"Fichier gold vide : {gold_path}")
    # fréquence normalisée : spans / 1000 exemples
    return {lbl: cnt / n * 1000 for lbl, cnt in label_counts.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_example(ex: dict, inv_freq: dict[str, float]) -> float:
    """
    Score de rareté = Σ inv_freq[label] pour chaque span hint_* de l'exemple.
    Les labels inconnus du gold reçoivent un poids fort (100.0).
    """
    s = 0.0
    for sp in ex.get("spans", []):
        lbl = sp["label"]
        if lbl.startswith("hint_"):
            s += inv_freq.get(lbl, 100.0)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Rapport de distribution
# ─────────────────────────────────────────────────────────────────────────────

def distribution_report(examples: list[dict], gold_freqs: dict[str, float],
                         title: str, n_gold: int) -> None:
    label_counts: Counter = Counter()
    for ex in examples:
        for sp in ex.get("spans", []):
            if sp["label"].startswith("hint_"):
                label_counts[sp["label"]] += 1

    n = len(examples)
    print(f"\n{'─'*62}")
    print(f"  {title}  ({n} exemples)")
    print(f"  {'Label':<30} {'silver/1k':>9}  {'gold/1k':>9}  {'ratio':>7}")
    print(f"{'─'*62}")

    all_labels = sorted(
        set(label_counts) | set(gold_freqs),
        key=lambda l: -gold_freqs.get(l, 0),
    )
    for lbl in all_labels:
        s_freq = label_counts.get(lbl, 0) / n * 1000 if n else 0
        g_freq = gold_freqs.get(lbl, 0)
        ratio  = s_freq / g_freq if g_freq > 0 else float("inf")
        flag   = "⚠️ " if ratio < 0.75 else ("↑  " if ratio > 1.20 else "   ")
        print(f"  {flag}{lbl:<28} {s_freq:>9.1f}  {g_freq:>9.1f}  {ratio:>7.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Filter SVO silver by NER label rarity")
    parser.add_argument("--input",     default="train_svo_silver.jsonl",
                        help="Fichier SVO silver à filtrer")
    parser.add_argument("--output",    default="",
                        help="Fichier de sortie (défaut : <input>_filtered.jsonl)")
    parser.add_argument("--gold",      default="train_v2.jsonl",
                        help="Fichier gold de référence pour les fréquences")
    parser.add_argument("--keep",      type=float, default=0.6,
                        help="Proportion d'exemples à conserver (défaut: 0.6 = 60%%)")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Score minimal pour être conservé (0 = désactivé)")
    parser.add_argument("--report",    action="store_true",
                        help="Afficher le rapport de distribution avant/après")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Afficher les stats sans écrire le fichier")
    args = parser.parse_args()

    input_path  = Path(args.input)
    gold_path   = Path(args.gold)
    output_path = Path(args.output) if args.output else \
                  input_path.with_name(input_path.stem + "_filtered.jsonl")

    # 1. Fréquences gold
    print(f"[FILTER] Calcul des fréquences gold depuis {gold_path.name}…")
    gold_freqs = compute_gold_freqs(gold_path)
    inv_freq   = {lbl: 1000.0 / freq for lbl, freq in gold_freqs.items() if freq > 0}

    n_gold = sum(1 for _ in open(gold_path, encoding="utf-8"))
    print(f"[FILTER] {len(gold_freqs)} labels NER trouvés dans le gold.")

    # 2. Chargement + scoring
    print(f"[FILTER] Scoring de {input_path.name}…")
    examples: list[tuple[float, dict]] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            sc = score_example(ex, inv_freq)
            examples.append((sc, ex))

    n_total = len(examples)
    print(f"[FILTER] {n_total} exemples chargés.")

    # Rapport avant filtrage
    if args.report:
        distribution_report([e for _, e in examples], gold_freqs,
                             f"AVANT filtrage — {input_path.name}", n_gold)

    # 3. Tri + sélection
    examples.sort(key=lambda x: -x[0])  # score décroissant

    # Score stats
    scores = [s for s, _ in examples]
    print(f"\n[FILTER] Score rareté — min: {min(scores):.1f}  "
          f"médiane: {sorted(scores)[n_total//2]:.1f}  "
          f"max: {max(scores):.1f}")

    # Seuil par proportion
    n_keep_prop  = max(1, int(n_total * args.keep))
    # Seuil par score minimal
    n_keep_score = sum(1 for s in scores if s >= args.min_score) if args.min_score > 0 else n_total
    n_keep       = min(n_keep_prop, n_keep_score)

    selected = [ex for _, ex in examples[:n_keep]]
    print(f"[FILTER] Conservation : {n_keep} / {n_total} exemples "
          f"({n_keep/n_total*100:.1f}%)")
    if args.min_score > 0:
        print(f"[FILTER] (seuil min-score={args.min_score:.1f} actif)")

    # Rapport après filtrage
    if args.report:
        distribution_report(selected, gold_freqs,
                             f"APRÈS filtrage — top {args.keep*100:.0f}%", n_gold)

    # 4. Écriture
    if not args.dry_run:
        with open(output_path, "w", encoding="utf-8") as f_out:
            for ex in selected:
                f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"\n[FILTER] ✅ {n_keep} exemples écrits → {output_path}")
    else:
        print("\n[FILTER] dry-run : aucun fichier écrit.")


if __name__ == "__main__":
    main()

