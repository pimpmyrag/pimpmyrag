#!/usr/bin/env python3
"""
Post-correction rule-based des spans hint_concept évidents restant après Haiku.
Applique des règles lexicales pour reclasser les cas que Haiku a été trop conservateur
à garder comme hint_concept.

Usage:
  python3 patch_concept_fallback.py --input data/val_v6.8.jsonl --output data/val_v6.8p.jsonl
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

# ─── Règles lexicales ─────────────────────────────────────────────────────────
# Chaque règle : (set de tokens normalisés, nouveau label)
# Appliquée si le texte normalisé du span matche exactement un token OU contient un des tokens.

# hint_work_generic : production culturelle générique sans titre (film seul, œuvre seule…)
WORK_GENERIC_EXACT = {
    "film", "films", "œuvre", "oeuvre", "œuvres", "oeuvres",
    "lettres", "livre", "livres", "chanson", "chansons",
    "album", "albums", "série", "séries", "émission", "émissions",
    "morceaux musicaux", "classiques",
}

# hint_notion : abstraction pure, valeur, méta-concept philosophique
NOTION_EXACT = {
    "notion", "notions", "idée", "idées", "sens",
    "vision", "existence", "influence", "mémoire",
    "signification", "conscience", "nature", "essence",
    "intention", "particularité", "essentiel",
    "besoin", "vérité", "réalité", "pensée",
    "liberté", "justice", "dignité", "équité",
    "valeur", "principe", "concept",
}

# hint_process : processus continu / dynamique de transformation
PROCESS_EXACT = {
    "cycle", "cycles", "rythme", "algorithmes", "algorithme",
    "transmission", "diffusion", "évolution", "convergence",
    "sampling",
}

# hint_field : domaine / secteur d'activité
FIELD_EXACT = {
    "chimie", "physique", "biologie", "informatique",
    "mathématiques", "mathématique", "algèbre",
    "médecine", "électronique", "astronomie",
    "géologie", "linguistique", "économie", "sociologie",
}

# hint_rule : règle, procédure, norme opérationnelle
RULE_EXACT = {
    "instruction", "instructions", "conseils", "conseil",
    "protocole", "procédure", "procédures",
}


def normalize(text: str) -> str:
    return text.strip().lower()


def classify_fallback(text: str) -> str | None:
    t = normalize(text)
    if t in WORK_GENERIC_EXACT:
        return "hint_work_generic"
    if t in NOTION_EXACT:
        return "hint_notion"
    if t in PROCESS_EXACT:
        return "hint_process"
    if t in FIELD_EXACT:
        return "hint_field"
    if t in RULE_EXACT:
        return "hint_rule"
    return None  # garder hint_concept


def patch_file(input_path: str, output_path: str):
    stats = Counter()
    n_items = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            n_items += 1
            new_spans = []
            for sp in item.get("spans", []):
                if sp.get("label") == "hint_concept":
                    new_label = classify_fallback(sp.get("text", ""))
                    if new_label:
                        new_spans.append({**sp, "label": new_label})
                        stats[f"hint_concept → {new_label}"] += 1
                    else:
                        new_spans.append(sp)
                        stats["hint_concept kept"] += 1
                else:
                    new_spans.append(sp)
            item["spans"] = new_spans
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ {n_items} phrases → {output_path}")
    print("\n📊 Corrections rule-based :")
    for k, v in sorted(stats.items()):
        print(f"  {k:<40} {v:>5}")
    total_corrected = sum(v for k, v in stats.items() if "→" in k)
    total_kept = stats.get("hint_concept kept", 0)
    print(f"\n  Corrigés : {total_corrected}  |  Conservés hint_concept : {total_kept}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    patch_file(args.input, args.output)


if __name__ == "__main__":
    main()

