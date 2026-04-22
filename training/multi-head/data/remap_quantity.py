"""
remap_quantity.py
==================
Remapping automatique des spans hint_quantity :
  - contient un nombre + unité physique  →  hint_measure
  - contient un nombre (chiffre ou mot)   →  hint_count
  - pas de nombre détectable              →  supprimé (annotation parasite)

Appliqué sur tous les fichiers *_v2.jsonl (train / val / test) et train_wiki_svo_ner.jsonl.

Usage
─────
  python3 remap_quantity.py [--dry-run] [--data-dir data/]
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Patterns
# ─────────────────────────────────────────────────────────────────────────────

# Unités physiques → hint_measure
PHYSICAL_UNITS = re.compile(
    r'km[²2]?|m[²2³3]|cm|mm|ha'
    r'|kg|mg|tonnes?'
    r'|litres?|ml|cl|dl'
    r'|n\u0153uds?|[Mm]ach'
    r'|°C|°F|kelvin'
    r'|[kKMGT]Hz'
    r'|[kKMGT]W|watts?'
    r'|dB|decibels?'
    r'|volts?|amp\u00e8res?|ohms?'
    r'|bits?|octets?|[KMGTP]o'
    r'|[Mm]\u00e9ga|[Gg]iga|[Tt]\u00e9ra|[Pp]\u00e9ta'
    r'|lieues?|m\u00e8tres?|kilom\u00e8tres?|centim\u00e8tres?|millim\u00e8tres?',
    re.IGNORECASE
)

# Chiffres arabes ou romains simples
HAS_DIGIT = re.compile(r'\d')

# Mots-nombres français
NUMBER_WORDS = re.compile(
    r'\b(un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|'
    r'onze|douze|treize|quatorze|quinze|seize|'
    r'vingt|trente|quarante|cinquante|soixante|'
    r'cent[s]?|mille|million[s]?|milliard[s]?|billion[s]?|'
    r'dizaine[s]?|centaine[s]?|cinquantaine[s]?|'
    r'quelques?|plusieurs|nombreux|nombreuses|'
    r'trentaine[s]?|quarantaine[s]?|'
    r'premier[s]?|première[s]?|dernier[s]?|dernière[s]?|'
    r'demi[s]?|quart[s]?|tiers)\b',
    re.IGNORECASE
)

# Mots parasites sans quantité (drop si span entier matche)
GARBAGE_ONLY = re.compile(
    r'^(quelque|grande?|essentiel[le]?s?|bas[se]?|'
    r'concentration|point|dose|faible|élevé[e]?|'
    r'haut[e]?|peu|beaucoup|très|assez|trop|'
    r'de|du|des|le|la|les|un|une)\s*$',
    re.IGNORECASE
)


def classify_quantity(text: str) -> str | None:
    """
    Retourne 'hint_measure', 'hint_count', ou None (à supprimer).
    """
    t = text.strip()

    # 1. Trop court ou purement parasite
    if len(t) <= 2:
        return None
    if GARBAGE_ONLY.match(t):
        return None

    # 2. Contient une unité physique → hint_measure
    if PHYSICAL_UNITS.search(t):
        return "hint_measure"

    # 3. Contient un chiffre → hint_count
    if HAS_DIGIT.search(t):
        return "hint_count"

    # 4. Contient un mot-nombre → hint_count
    if NUMBER_WORDS.search(t):
        return "hint_count"

    # 5. Pas de quantité détectable → drop
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Traitement d'un fichier
# ─────────────────────────────────────────────────────────────────────────────

def remap_file(input_path: Path, output_path: Path, dry_run: bool = False) -> dict:
    stats = Counter()
    rows = []

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            new_spans = []
            for sp in ex.get("spans", []):
                if sp["label"] != "hint_quantity":
                    new_spans.append(sp)
                    continue
                new_label = classify_quantity(sp["text"])
                stats["total_qty"] += 1
                if new_label is None:
                    stats["dropped"] += 1
                else:
                    stats[f"→ {new_label}"] += 1
                    new_spans.append({**sp, "label": new_label})
            rows.append({**ex, "spans": new_spans})

    if not dry_run:
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Remap hint_quantity → hint_measure | hint_count | drop")
    parser.add_argument("--data-dir", default="data",
                        help="Dossier contenant les fichiers jsonl")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Afficher les stats sans écrire")
    parser.add_argument("--inplace",  action="store_true",
                        help="Écraser les fichiers source (défaut: suffixe _noqty)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    targets = [
        data_dir / "train_v2.jsonl",
        data_dir / "val_v2.jsonl",
        data_dir / "test_v2.jsonl",
        data_dir / "train_wiki_svo_ner.jsonl",
    ]

    for src in targets:
        if not src.exists():
            print(f"  ⚠️  {src.name} introuvable, skip.")
            continue

        if args.inplace:
            dst = src
        else:
            dst = src.with_name(src.stem + "_noqty.jsonl")

        stats = remap_file(src, dst, dry_run=args.dry_run)

        action = "DRY-RUN" if args.dry_run else f"→ {dst.name}"
        print(f"\n{src.name}  [{action}]")
        print(f"  hint_quantity total  : {stats['total_qty']}")
        for k, v in sorted(stats.items()):
            if k != "total_qty":
                print(f"  {k:<25} : {v}")

    if args.dry_run:
        print("\n[dry-run] Aucun fichier modifié.")


if __name__ == "__main__":
    main()

