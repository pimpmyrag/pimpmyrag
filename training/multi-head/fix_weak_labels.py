#!/usr/bin/env python3
"""
fix_weak_labels.py
==================
Nettoie les spans problématiques dans les labels faibles :
  - hint_object_generic → hint_document  : spans contenant des mots de document
  - hint_state           → DROP          : adjectifs seuls (fragments inutilisables)
  - hint_field           → DROP          : fragments trop courts sans nom de domaine
  - hint_notion          → DROP          : fragments purement adjectivaux ou trop courts
  - hint_doctrine        → DROP          : spans d'un seul mot trop générique

Usage:
    python3 fix_weak_labels.py data/train_v8.1.jsonl -o data/train_v8.1.jsonl
    python3 fix_weak_labels.py data/train_v8.1.jsonl -o /tmp/check.jsonl --dry-run
"""

import json
import re
import argparse
from collections import Counter

# ── hint_object_generic → hint_document ──────────────────────────────────────
DOCUMENT_WORDS = re.compile(
    r'\b(donn[eé]es?|liste|listes|fichier|fichiers|rapport|rapports|'
    r'bulletin|bulletins|note|notes|document|documents|acte|actes|'
    r'formulaire|formulaires|d[eé]cret|d[eé]crets|arr[eê]t[eé]|arr[eê]t[eé]s|'
    r'proc[eè]s-verbal|ordonnance|circulaire|m[eé]morandum|'
    r'registre|registres|archives?|dossier|dossiers|'
    r'contrat|contrats|accord|accords|trait[eé]|convention)\b',
    re.IGNORECASE
)

# ── hint_state → DROP : adjectifs seuls ou fragments adjectivaux ──────────────
# Un span hint_state d'1-2 mots purement adjectival sans nom est un fragment
STATE_ADJ_ONLY = re.compile(
    r'^(totale?|collective?|sanitaires?|majeures?|grave|graves|'
    r'critiqu[eé]|profondes?|s[eé]v[eè]re|s[eé]v[eè]res|'
    r'alarmante?|pr[eé]occupante?|instable|instables|'
    r'volatile|fragile|fragiles|tendu|tendue|tendues?|'
    r'acute?|chronique|chroniques|endémique|end[eé]miques?|'
    r'structurelle?|conjoncturelle?|résiduelle?|transitoire)s?$',
    re.IGNORECASE
)

# ── hint_field → DROP : fragments trop courts sans nom de domaine ──────────
# Uniquement les adjectifs de taille SANS le nom du secteur
FIELD_SIZE_ADJ = re.compile(
    r'^(petites?|moyennes?|grandes?|PME|TPE)$',
    re.IGNORECASE
)

# ── hint_notion → DROP : fragments adjectivaux seuls ─────────────────────────
NOTION_ADJ_ONLY = re.compile(
    r'^(majeures?|mineures?|importantes?|fondamentales?|essentielles?|'
    r'principales?|secondaires?|multiples?|diverses?|'
    r'positives?|n[eé]gatives?|fausses?|vraies?|'
    r'nouvelles?|anciennes?|modernes?|contemporaines?|'
    r'complexes?|simples?|claires?|obscures?)$',
    re.IGNORECASE
)

# ── hint_doctrine → DROP : mots seuls trop génériques ────────────────────────
DOCTRINE_TOO_VAGUE = re.compile(
    r'^(approche|approches|ligne|lignes|m[eé]thode|m[eé]thodes|'
    r'voie|voies|strat[eé]gie|strat[eé]gies|'
    r'tendance|tendances|logique|logiques|'
    r'vision|visions|position|positions|'
    r'lecture|lectures|interpr[eé]tation|interpr[eé]tations?|'
    r'paradigme?s?|mod[eè]le?s?|cadre|cadres|'
    r'doctrine|option|options)$',
    re.IGNORECASE
)

HAS_DIGIT = re.compile(r'\d')


def reclassify(span: dict, phrase_text: str):
    label = span.get("label", "")
    text = span.get("text", "").strip()
    words = text.split()

    if label == "hint_object_generic":
        if DOCUMENT_WORDS.search(text):
            return "hint_document", f"doc_word in '{text}'"

    elif label == "hint_state":
        # DROP si : 1 seul mot adjectival, OU adjectif+adjectif court sans nom
        if len(words) <= 2 and not HAS_DIGIT.search(text):
            if STATE_ADJ_ONLY.match(text):
                return "DROP", f"adj_fragment '{text}'"
        # DROP si fragment de 1 mot purement adjectival non nominalisé
        if len(words) == 1 and not HAS_DIGIT.search(text) and text[0].islower():
            if re.match(r'.+(al|el|if|ive|ique|aire|oire|iste|ant|ent|eux|euse)s?$', text, re.I):
                return "DROP", f"single_adj '{text}'"

    elif label == "hint_field":
        if len(words) <= 1 and FIELD_SIZE_ADJ.match(text):
            return "DROP", f"field_size_adj '{text}'"
        # Fragment comme "petites et moyennes" sans "entreprises"
        if re.match(r'^(petites?|grandes?|moyennes?)\s+(et\s+)?(moyennes?|grandes?|petites?)?$', text, re.I):
            return "DROP", f"field_size_only '{text}'"

    elif label == "hint_notion":
        if len(words) == 1 and not HAS_DIGIT.search(text) and NOTION_ADJ_ONLY.match(text):
            return "DROP", f"notion_adj_only '{text}'"

    elif label == "hint_doctrine":
        if len(words) == 1 and DOCTRINE_TOO_VAGUE.match(text):
            return "DROP", f"doctrine_too_vague '{text}'"

    return label, ""


def fix_file(input_path, output_path, dry_run=False):
    with open(input_path) as f:
        phrases = [json.loads(l) for l in f]

    stats = Counter()
    log = []

    for p in phrases:
        kept = []
        for span in p.get("spans", []):
            new_label, reason = reclassify(span, p.get("text", ""))
            if new_label == "DROP":
                stats["dropped"] += 1
                log.append(("DROP", span["label"], span.get("text",""), reason))
                continue
            if new_label != span["label"]:
                old = span["label"]
                span["label"] = new_label
                stats[f"{old}→{new_label}"] += 1
                log.append(("REMAP", old, span.get("text",""), f"→{new_label} [{reason}]"))
            kept.append(span)
        p["spans"] = kept

    print(f"\n=== Résumé ===")
    for k, v in sorted(stats.items()):
        print(f"  {k:<40} {v:>5}")
    print(f"  {'TOTAL':<40} {sum(stats.values()):>5}")

    print(f"\n=== Échantillon ===")
    by_type = {}
    for op, old, txt, reason in log:
        by_type.setdefault(f"{op}:{old}", []).append((txt, reason))
    for key, items in sorted(by_type.items()):
        print(f"\n  [{key}] ({len(items)})")
        for txt, reason in items[:6]:
            print(f"    \"{txt[:45]}\"  →  {reason}")

    if dry_run:
        print("\n[DRY RUN]")
        return

    with open(output_path, "w") as f:
        for p in phrases:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nSauvegardé → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    fix_file(args.input, args.output, args.dry_run)


if __name__ == "__main__":
    main()

