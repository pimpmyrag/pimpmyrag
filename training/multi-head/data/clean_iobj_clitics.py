"""
clean_iobj_clitics.py
======================
Supprime les spans svo_iobj qui sont des pronoms clitiques français
(lui, y, en, me, te, se, nous, vous, leur, …) des fichiers silver existants.

Ces spans sont redondants avec pron_obj et causent un signal contradictoire
au modèle (même token → deux labels différents).

Usage
─────
  python3 data/clean_iobj_clitics.py [--data-dir data/] [--dry-run]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

# Pronoms clitiques français — doit rester synchronisé avec FR_PERS_PRONOUNS de build_svo_silver.py
CLITICS = {
    "je", "j", "me", "m", "moi",
    "tu", "te", "t", "toi",
    "il", "elle", "le", "la", "lui", "se", "s", "soi",
    "nous", "vous",
    "ils", "elles", "les", "leur", "eux",
    "y", "en",
}


def is_clitic_iobj(sp: dict) -> bool:
    if sp.get("label") != "svo_iobj":
        return False
    txt = sp.get("text", "").strip().lower().rstrip("'")
    return txt in CLITICS or len(sp.get("text", "").strip()) <= 2


def clean_file(src: Path, dst: Path, dry_run: bool = False) -> dict:
    stats = Counter()
    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            new_spans = []
            for sp in ex.get("spans", []):
                if is_clitic_iobj(sp):
                    stats["dropped_iobj_clitic"] += 1
                else:
                    new_spans.append(sp)
            stats["examples"] += 1
            rows.append({**ex, "spans": new_spans})

    if not dry_run:
        with open(dst, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    targets = [
        data_dir / "train_svo_silver.jsonl",
        data_dir / "val_svo_silver.jsonl",
        data_dir / "test_svo_silver.jsonl",
        data_dir / "train_wiki_svo.jsonl",
        data_dir / "train_wiki_svo_ner.jsonl",
        data_dir / "train_wiki_claude_annotated.jsonl",
    ]

    for src in targets:
        if not src.exists():
            print(f"  ⚠️  {src.name} introuvable, skip.")
            continue
        stats = clean_file(src, src, dry_run=args.dry_run)
        action = "DRY-RUN" if args.dry_run else "nettoyé"
        print(f"{src.name}  [{action}]  {stats['examples']} ex  "
              f"→ {stats['dropped_iobj_clitic']} svo_iobj clitiques supprimés")

    if args.dry_run:
        print("\n[dry-run] Aucun fichier modifié.")


if __name__ == "__main__":
    main()

