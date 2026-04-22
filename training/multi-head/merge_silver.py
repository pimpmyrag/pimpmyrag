"""
merge_silver.py
===============
Fusionne plusieurs fichiers silver SVO (format jsonl spans) en un seul,
avec pondération par source et déduplication par id.

Le fichier de sortie est passé directement à build_multitask_dataset.py.

Usage :
    python merge_silver.py \\
        --out data/train_svo_silver_merged.jsonl \\
        --sources data/train_svo_silver.jsonl:1.0 \\
                  data/train_obliques_wiki.jsonl:0.6 \\
                  data/train_svo_de.jsonl:0.8

Format de chaque source : chemin:weight
  weight=1.0  → sample_weights inchangés
  weight=0.6  → tous les sample_weights de la source × 0.6
               (utile pour les silver de moindre qualité ou langue secondaire)

Note : le weight est appliqué sur les candidats tokenisés si présents (champ
"candidates"), sinon il est stocké dans un méta-champ "_source_weight" pour
que build_multitask_dataset.py puisse l'appliquer à la construction.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fusionne des fichiers silver SVO")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        metavar="FICHIER:WEIGHT",
        help="Fichiers silver à fusionner, format  chemin:weight  (ex: data/train_svo_silver.jsonl:1.0)"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Fichier de sortie fusionné"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour le shuffle (défaut=42)"
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Désactiver le shuffle final"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        default=True,
        help="Afficher les statistiques de labels par source (défaut=True)"
    )
    args = parser.parse_args()

    # rows_by_id : dict id → row  (fusion de spans si même ID dans plusieurs sources)
    rows_by_id: dict[str, dict] = {}
    rows_order: list[str] = []  # ordre d'insertion (pour le shuffle)
    source_stats: dict[str, dict] = {}

    for spec in args.sources:
        # Séparer chemin et weight (le chemin peut contenir des :// donc on split par le dernier :)
        if ":" not in spec:
            print(f"⚠️  Format invalide '{spec}' — attendu chemin:weight, ignoré.")
            continue

        path_str, weight_str = spec.rsplit(":", 1)
        try:
            weight = float(weight_str)
        except ValueError:
            print(f"⚠️  Weight invalide '{weight_str}' pour {path_str}, défaut=1.0")
            weight = 1.0

        path = Path(path_str)
        if not path.exists():
            print(f"⚠️  {path} introuvable, skip.")
            continue

        n_added = 0
        n_merged = 0
        label_counts: Counter = Counter()

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                uid = str(row.get("id", ""))

                # Compter les labels pour les stats
                for sp in row.get("spans", []):
                    label_counts[sp.get("label", "?")] += 1

                if uid in rows_by_id:
                    # Même ID : fusionner les spans (ajouter ceux absents par clé start/end/label)
                    existing = rows_by_id[uid]
                    existing_keys = {
                        (sp.get("start"), sp.get("end"), sp.get("label"))
                        for sp in existing.get("spans", [])
                    }
                    new_spans = [
                        sp for sp in row.get("spans", [])
                        if (sp.get("start"), sp.get("end"), sp.get("label")) not in existing_keys
                    ]
                    if new_spans:
                        existing.setdefault("spans", []).extend(new_spans)
                    n_merged += 1
                    continue

                # Nouveau ID
                if weight != 1.0:
                    row["_source_weight"] = weight

                # Si les candidats sont déjà pré-construits (rare mais possible),
                # appliquer le weight directement
                if "candidates" in row and weight != 1.0:
                    for c in row["candidates"]:
                        c["sample_weight"] = c.get("sample_weight", 1.0) * weight

                rows_by_id[uid] = row
                rows_order.append(uid)
                n_added += 1

        source_stats[path.name] = {
            "added": n_added,
            "merged": n_merged,
            "weight": weight,
            "labels": dict(label_counts.most_common()),
        }
        print(f"  {path.name:<50} {n_added:>6} nouveaux  {n_merged:>6} fusionnés  weight={weight:.2f}")

    rows = [rows_by_id[uid] for uid in rows_order]

    if not rows:
        print("❌ Aucun exemple chargé — vérifier les chemins des sources.")
        return

    if not args.no_shuffle:
        random.seed(args.seed)
        random.shuffle(rows)
        print(f"\n🔀 Shuffle avec seed={args.seed}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✅ {len(rows)} exemples fusionnés → {out_path}")

    if args.stats:
        print("\n📊 Répartition des labels par source :")
        all_labels: Counter = Counter()
        for src_name, stats in source_stats.items():
            print(f"\n  [{src_name}]  ({stats['added']} exemples, weight={stats['weight']})")
            for label, count in sorted(stats["labels"].items()):
                print(f"    {label:<25} {count:>6}")
                all_labels[label] += count
        print(f"\n  [TOTAL FUSIONNÉ]  ({len(rows)} exemples)")
        for label, count in sorted(all_labels.items()):
            print(f"    {label:<25} {count:>6}")


if __name__ == "__main__":
    main()

