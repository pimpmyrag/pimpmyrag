#!/usr/bin/env python3
"""
Vérifie les doublons entre wikinews_rare_candidates.jsonl et le dataset existant (train/val/test).
Compare par texte normalisé (lowercase, strip, espaces multiples → simple).
"""
import argparse
import json
import re
from collections import defaultdict


def normalize(text: str) -> str:
    """Normalisation pour comparaison de doublons."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_texts(path: str) -> set:
    """Charge les textes normalisés d'un JSONL."""
    texts = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            texts.add(normalize(obj["text"]))
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/wikinews_rare_candidates.jsonl")
    parser.add_argument("--dataset", nargs="+", default=[
        "data/train.jsonl",
        "data/val.jsonl",
        "data/test.jsonl",
    ], help="Fichiers du dataset existant à comparer")
    parser.add_argument("--output", default=None, help="Si fourni, écrit les candidates dédoublonnées")
    args = parser.parse_args()

    # 1. Charger les textes du dataset existant
    existing = set()
    for path in args.dataset:
        try:
            texts = load_texts(path)
            print(f"📂 {path}: {len(texts)} phrases")
            existing |= texts
        except FileNotFoundError:
            print(f"⚠️  {path} non trouvé, ignoré")

    print(f"\n📊 Total phrases existantes (dédoublonnées): {len(existing)}")

    # 2. Vérifier les candidates
    n_total = 0
    n_dupes = 0
    n_clean = 0
    dupes = []
    clean = []

    # Vérifier aussi les doublons internes aux candidates
    seen_candidates = set()
    n_internal_dupes = 0

    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            n_total += 1
            norm = normalize(obj["text"])

            # Doublon interne
            if norm in seen_candidates:
                n_internal_dupes += 1
                continue
            seen_candidates.add(norm)

            # Doublon avec dataset existant
            if norm in existing:
                n_dupes += 1
                dupes.append(obj)
            else:
                n_clean += 1
                clean.append(obj)

    print(f"\n{'='*60}")
    print(f"📝 Candidates analysées:     {n_total}")
    print(f"🔁 Doublons internes:        {n_internal_dupes}")
    print(f"❌ Doublons avec dataset:     {n_dupes}")
    print(f"✅ Phrases uniques:          {n_clean}")
    print(f"{'='*60}")

    if n_dupes > 0:
        print(f"\n🔍 Exemples de doublons (max 10):")
        for d in dupes[:10]:
            print(f"  [{d['id']}] {d['text'][:100]}...")

    # 3. Écrire le fichier nettoyé si demandé
    if args.output:
        with open(args.output, "w", encoding="utf-8") as out:
            for obj in clean:
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"\n💾 {n_clean} phrases uniques → {args.output}")


if __name__ == "__main__":
    main()

