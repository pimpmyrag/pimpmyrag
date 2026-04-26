#!/usr/bin/env python3
"""
Reannotation automatique du dataset existant pour les nouveaux labels ABSTRACT.

Stratégie :
  - hint_law      : remap sûr pour "accord de...", "traité de...", "loi ...", etc.
  - hint_disease  : remap sûr pour "COVID-19", "Ebola", "grippe", "peste", etc.
  - hint_concept  : très peu de cas, on ne touche pas automatiquement
  - hint_language : TROP AMBIGU pour l'auto — "français" est souvent hint_norp légitimement
  - hint_work_of_art : impossible à détecter par mots-clés

Deux modes :
  --dry-run  : affiche les changements sans modifier
  --apply    : écrit les fichiers modifiés

Usage:
  python remap_to_abstract.py --dry-run
  python remap_to_abstract.py --apply
"""
from __future__ import annotations

import json
import re
import sys
import argparse
from pathlib import Path
from collections import Counter


# ─────────────────────────────────────────────────────────────
# Règles de remap : (condition, nouveau_label)
# condition = fonction(span_text, label, sentence) -> bool
# ─────────────────────────────────────────────────────────────

def _lower(text):
    return text.lower().strip()


def make_law_rule():
    """Remap -> hint_law si le span EST un texte juridique nommé."""
    # Patterns : "traité de X", "loi Y", "accord de X", etc.
    # On ne remap PAS les cas ambigus (ex: "accord" seul = hint_event_nominal ok)
    patterns = [
        r"^(?:le |la |les |l'|l')?(?:traité|trait) (?:de |d'|du |sur )",
        r"^(?:le |la |les |l'|l')?(?:édit|edit) (?:de |d'|du )",
        r"^(?:le |la |les |l'|l')?loi (?:sur |de |du |Hadopi|Taubira|Veil|Évin|Evin|Toubon|Gayssot)",
        r"^(?:le |la |les |l'|l')?(?:décret|decret) (?:de |d'|du |sur |n°)",
        r"^(?:le |la |les |l'|l')?constitution(?:$| )",
        r"^(?:le |la |les |l'|l')?convention (?:de |d'|du |sur |européenne)",
        r"^(?:le |la |les |l'|l')?(?:accords?|accord) (?:de |d'|du |sur )",
        r"^(?:le |la |les |l'|l')?pacte (?:de |d'|du )",
        r"^(?:le |la |les |l'|l')?charte (?:de |d'|du |des )",
        r"^(?:le |la |les |l'|l')?protocole (?:de |d'|du |sur )",
        r"^(?:le |la |les |l'|l')?directive ",
        r"^(?:le |la |les |l'|l')?ordonnance (?:de |d'|du )",
        r"^(?:le |la |les |l'|l')?concordat",
        r"^(?:le |la |les |l'|l')?règlement (?:de |d'|du |sur |général)",
        r"^(?:le |la |les |l'|l')?armistice",
        r"^(?:le |la |les |l'|l')?(?:déclaration|declaration) (?:de |d'|du |des |universelle)",
        r"^(?:le |la |les |l'|l')?Code (?:civil|pénal|du travail|de commerce)",
        r"^Magna Carta",
        r"^Bill of Rights",
        r"^Habeas Corpus",
        # Patterns nommés spécifiques
        r"(?:Loi sur la réduction|loi sur la réduction)",
        r"^nouvelle constitution$",
        r"^accord de paix",
        r"^accord(?:s)? de (?:Camp David|Grenelle|Schengen|Matignon|Minsk|Oslo|Munich)",
    ]
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    def rule(span_text, label, sentence):
        txt = span_text
        for pat in compiled:
            if pat.search(txt):
                return True
        return False

    return rule


def make_disease_rule():
    """Remap -> hint_disease si le span EST une maladie/pathologie."""
    exact_diseases = {
        'covid-19', 'covid', 'sars-cov-2', 'ebola', 'sras', 'sars',
        'vih', 'sida', 'mpox', 'zika', 'h1n1', 'h5n1',
    }
    disease_patterns = [
        r"^(?:le |la |les |l'|l')?grippe(?: .+)?$",
        r"^(?:le |la |les |l'|l')?peste(?: .+)?$",
        r"^(?:le |la |les |l'|l')?choléra$",
        r"^(?:le |la |les |l'|l')?cancer(?: .+)?$",
        r"^(?:le |la |les |l'|l')?diabète(?: .+)?$",
        r"^(?:le |la |les |l'|l')?tuberculose$",
        r"^(?:le |la |les |l'|l')?variole$",
        r"^(?:le |la |les |l'|l')?paludisme$",
        r"^(?:le |la |les |l'|l')?rougeole$",
        r"^(?:le |la |les |l'|l')?dengue$",
        r"^(?:le |la |les |l'|l')?maladie (?:de |d'|du )",
        r"^(?:le |la |les |l'|l')?virus (?:Ebola|Zika|de la |du )",
        r"^(?:le |la |les |l'|l')?syndrome (?:de |d'|du )",
        r"^(?:le |la |les |l'|l')?fièvre (?:jaune|typhoïde|hémorragique|de )",
        r"^(?:le |la |les |l'|l')?sclérose",
        r"^(?:le |la |les |l'|l')?hépatite",
        r"^(?:le |la |les |l'|l')?méningite",
        r"^(?:le |la |les |l'|l')?coqueluche$",
        r"^(?:le |la |les |l'|l')?polio(?:myélite)?$",
    ]
    compiled = [re.compile(p, re.IGNORECASE) for p in disease_patterns]

    def rule(span_text, label, sentence):
        txt_low = span_text.lower().strip()
        if txt_low in exact_diseases:
            return True
        for pat in compiled:
            if pat.search(span_text):
                return True
        return False

    return rule


RULES = [
    ("hint_law", make_law_rule()),
    ("hint_disease", make_disease_rule()),
]


def process_file(path: str, dry_run: bool = True):
    rows = []
    changes = []

    with open(path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            for sp in row.get('spans', []):
                old_label = sp['label']
                # Don't remap if already one of the new labels
                if old_label.startswith('hint_law') or old_label.startswith('hint_disease') or \
                   old_label.startswith('hint_concept') or old_label.startswith('hint_language') or \
                   old_label.startswith('hint_work_of_art'):
                    continue

                for new_label, rule_fn in RULES:
                    if rule_fn(sp['text'], old_label, row['text']):
                        changes.append({
                            'id': row['id'],
                            'text': sp['text'][:80],
                            'old': old_label,
                            'new': new_label,
                            'sentence': row['text'][:120],
                        })
                        if not dry_run:
                            sp['label'] = new_label
                        break

            rows.append(row)

    if not dry_run:
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    return changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', default=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()

    dry_run = not args.apply

    base = Path(__file__).parent
    files = ['train.jsonl', 'val.jsonl', 'test.jsonl']

    all_changes = []
    for fname in files:
        path = base / fname
        if not path.exists():
            print(f"⚠️  {path} not found, skipping")
            continue
        changes = process_file(str(path), dry_run=dry_run)
        all_changes.extend(changes)
        print(f"\n{'📋' if dry_run else '✅'} {fname}: {len(changes)} remaps")

    # Summary
    print(f"\n{'='*70}")
    print(f"  TOTAL: {len(all_changes)} remaps {'(DRY RUN)' if dry_run else '(APPLIED)'}")
    print(f"{'='*70}")

    by_new = Counter(c['new'] for c in all_changes)
    for new_label, count in by_new.most_common():
        print(f"\n  → {new_label}: {count}")
        by_old = Counter(c['old'] for c in all_changes if c['new'] == new_label)
        for old_label, cnt in by_old.most_common():
            print(f"      from {old_label:30s} {cnt:4d}")

    print(f"\n  Exemples de remaps:")
    for c in all_changes[:30]:
        print(f"    {c['old']:25s} → {c['new']:20s} | \"{c['text']}\"")

    if dry_run:
        print(f"\n💡 Pour appliquer: python {__file__} --apply")


if __name__ == '__main__':
    main()

