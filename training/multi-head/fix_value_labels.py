#!/usr/bin/env python3
"""
fix_value_labels.py
===================
Nettoie les confusions entre hint_measure / hint_rate / hint_count / hint_percentage
dans un fichier .jsonl.

Règles (larges volontairement) :
  1. hint_measure → hint_rate    : contient une unité de vitesse/taux (km/h, m/s, %, °/s…)
  2. hint_measure → hint_count   : contient un nom dénombrable OU est une quantité approx.
  3. hint_measure → DROP         : span abstrait sans chiffre ni unité physique reconnaissable
  4. hint_rate    → hint_measure : vitesse physique pure déjà bien placée (km/h gardé dans rate)
  5. hint_percentage → hint_rate : si contexte de taux/variation (taux, croissance, inflation…)

Usage:
    python3 fix_value_labels.py data/train_v8.1.jsonl -o data/train_v8.1_fixed.jsonl
    python3 fix_value_labels.py data/train_v8.1.jsonl -o data/train_v8.1_fixed.jsonl --dry-run
"""

import json
import re
import sys
import argparse
from collections import Counter

# ── Indicateurs pour hint_measure → hint_rate ────────────────────────────────
# Unités de vitesse / fréquence / débit / taux physique
SPEED_UNITS = re.compile(
    r'\b(km/h|m/s|mph|mi/h|km/s|km·h|kn\b|nœud|noeuds?|knot|mach|bauds?|'
    r'tr/min|rpm|Hz|kHz|MHz|GHz|l/s|m³/s|m3/s|L/min)\b',
    re.IGNORECASE
)

# ── Indicateurs pour hint_measure → hint_count ────────────────────────────────
# Noms dénombrables (personnes, objets, entités)
COUNTABLE_NOUNS = re.compile(
    r'\b(personne|personnes|habitant|habitants|individu|individus|'
    r'victime|victimes|bless[eé]|bless[eé]s|mort|morts|décédé|décédés|'
    r'tu[eé]|tu[eé]s|soldat|soldats|militaire|militaires|'
    r'manifestant|manifestants|gréviste|grévistes|syndicaliste|syndicalistes|'
    r'patient|patients|malade|malades|infirmier|infirmiers|médecin|médecins|'
    r'[eé]l[eè]ve|[eé]l[eè]ves|[eé]tudiant|[eé]tudiants|enfant|enfants|'
    r'homme|hommes|femme|femmes|adulte|adultes|jeune|jeunes|mineur|mineurs|'
    r'citoyen|citoyens|r[eé]fugi[eé]|r[eé]fugi[eé]s|d[eé]tenu|d[eé]tenus|'
    r'employ[eé]|employ[eé]s|salari[eé]|salari[eé]s|ouvrier|ouvriers|'
    r'travailleur|travailleurs|agriculteur|agriculteurs|p[eê]cheur|p[eê]cheurs|'
    r'voiture|voitures|v[eé]hicule|v[eé]hicules|camion|camions|bus|cars?|'
    r'navire|navires|bateau|bateaux|avion|avions|a[eé]ronef|drone|drones|'
    r'logement|logements|appartement|appartements|maison|maisons|b[aâ]timent|b[aâ]timents|'
    r'entreprise|entreprises|soci[eé]t[eé]|soci[eé]t[eé]s|[eé]tablissement|[eé]tablissements|'
    r'pays|r[eé]gion|r[eé]gions|ville|villes|commune|communes|d[eé]partement|'
    r'millier|milliers|million|millions|milliard|milliards|billion|billions|'
    r'centaine|centaines|dizaine|dizaines|douzaine|douzaines|'
    r'vingtaine|trentaine|quarantaine|cinquantaine|soixantaine|'
    r'point|points|but|buts|essai|essais|set|sets|match|matches|'
    r'cas|incident|incidents|[eé]v[eè]nement|[eé]v[eè]nements|'
    r'fois|occurrence|occurrences|exemple|exemples|'
    r'mort|morts|bless[eé]|bless[eé]s|disparu|disparus)\b',
    re.IGNORECASE
)

# Quantités approximatives sans unité physique
APPROX_QTY = re.compile(
    r'\b(plusieurs|quelques|de nombreux|de nombreuses|'
    r'une vingtaine|une dizaine|une douzaine|une trentaine|'
    r'une quarantaine|une cinquantaine|une centaine|'
    r'des milliers|des centaines|des dizaines|des millions|des milliards|'
    r'plus d[eu]|moins d[eu]|environ|près de|autour de|'
    r'au moins|au plus|à peine)\b',
    re.IGNORECASE
)

# ── Indicateurs pour hint_measure → DROP ─────────────────────────────────────
# Spans purement abstraits : pas de chiffre, pas d'unité physique
HAS_DIGIT = re.compile(r'\d')
PHYSICAL_UNITS = re.compile(
    r'\b(kg|g|mg|t\b|tonne|tonnes|km|m\b|cm|mm|nm|'
    r'km²|hectare|ha\b|m²|m³|'
    r'°C|°F|°K|kelvin|celsius|'
    r'an|ans|ann[eé]e|ann[eé]es|mois|jour|jours|semaine|semaines|heure|heures|minute|minutes|seconde|secondes|'
    r'watt|watts|kW|MW|GW|volt|ampère|joule|cal|kcal|'
    r'litre|litres|cl|ml|dl|'
    r'bit|octet|Mo|Go|To|Ko)\b',
    re.IGNORECASE
)

# Nombres écrits en lettres (→ hint_count, pas DROP)
NUMBER_WORDS = re.compile(
    r'\b(z[eé]ro|un\b|une\b|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|'
    r'onze|douze|treize|quatorze|quinze|seize|dix-sept|dix-huit|dix-neuf|'
    r'vingt|trente|quarante|cinquante|soixante|soixante-dix|quatre-vingts?|'
    r'cent|cents|mille|plusieurs|quelques|nombreux|nombreuses)\b',
    re.IGNORECASE
)

# Mots abstraits clairement hors mesure
ABSTRACT_ONLY = re.compile(
    r'^(dommage|dommages|pr[eé]judice|pr[eé]judices|perte|pertes|'
    r'co[uû]t|co[uû]ts|risque|risques|niveau|niveaux|d[eé]ficit|exc[eé]dent|'
    r'impact|impacts|effet|effets|cons[eé]quence|cons[eé]quences|'
    r'moyen\b|moyenne\b|am[eé]lioration|d[eé]gradation)\s*(mat[eé]riel|moral|humain|financier|[eé]conomique)?$',
    re.IGNORECASE
)

# ── Indicateurs pour hint_percentage → hint_rate ─────────────────────────────
# Contexte de taux dans la phrase entière
RATE_CONTEXT_IN_PHRASE = re.compile(
    r'\b(taux|croissance|inflation|ch[oô]mage|emploi|fertili|natalit|'
    r'mortalit|variation|[eé]volution|progression|hausse|baisse|'
    r'rendement|retour|r[eé]mun[eé]ration|int[eé]r[eê]t|dividende|'
    r'performance|efficacit|productivit|rencherissement|r[eé]enchérissement)\b',
    re.IGNORECASE
)

# ── Logique principale ────────────────────────────────────────────────────────

def reclassify_span(span: dict, phrase_text: str) -> tuple[str, str]:
    """
    Retourne (new_label, reason) ou (original_label, '') si pas de changement.
    reason == 'DROP' → supprimer le span.
    """
    label = span.get("label", "")
    text = span.get("text", "").strip()
    text_low = text.lower()

    if label == "hint_measure":
        # Règle 1 : → hint_rate si unité de vitesse/fréquence
        if SPEED_UNITS.search(text):
            return "hint_rate", f"speed_unit in '{text}'"

        # Règle 2 : → hint_count si nom dénombrable
        if COUNTABLE_NOUNS.search(text):
            return "hint_count", f"countable_noun in '{text}'"

        # Règle 2b : → hint_count si quantité approximative
        if APPROX_QTY.search(text):
            return "hint_count", f"approx_qty in '{text}'"

        # Règle 3 : DROP si abstrait (pas de chiffre ni nombre-en-lettres ni unité physique)
        if not HAS_DIGIT.search(text) and not PHYSICAL_UNITS.search(text):
            # Nb en lettres → hint_count plutôt que DROP
            if NUMBER_WORDS.search(text) or COUNTABLE_NOUNS.search(text):
                return "hint_count", f"number_word/countable in '{text}'"
            if ABSTRACT_ONLY.match(text):
                return "DROP", f"abstract_no_unit '{text}'"
            # Large : tout span hint_measure sans chiffre ni unité physique connue
            if len(text.split()) <= 4:  # court et vague
                return "DROP", f"no_digit_no_unit short '{text}'"

    elif label == "hint_percentage":
        # Règle 5 : → hint_rate si contexte de taux dans la phrase
        if RATE_CONTEXT_IN_PHRASE.search(phrase_text):
            if re.search(r'%|taux|croissance|inflation|ch[oô]mage', text_low):
                return "hint_rate", f"rate_context in phrase for '{text}'"

    elif label == "hint_rate":
        # Règle 6 : → hint_measure si unité de fréquence physique (CPU, signal)
        if re.search(r'\b(GHz|MHz|kHz|Hz)\b', text, re.IGNORECASE):
            return "hint_measure", f"freq_unit in '{text}'"

        # Règle 7 : → hint_count si "X fois" SANS "par" (fréquence dénombrable)
        if re.search(r'\bfois\b', text_low) and not re.search(r'\bpar\b', text_low):
            return "hint_count", f"'fois' sans 'par' → count in '{text}'"

        # Règle 8 : → hint_percentage si fraction (2/3, 1/4…) sans contexte de taux
        if re.search(r'^\d+/\d+$', text.strip()):
            return "hint_percentage", f"fraction '{text}'"

        # Règle 9 : → hint_notion si acronyme macro-éco connu sans valeur numérique
        MACRO_ACRONYMS = re.compile(
            r'^(PIB|PNB|PPA|IDH|IPC|IPP|PMI|ISM|CAC|CAF|FAB|TVA|RSA|SMIG|SMIC)$',
            re.IGNORECASE
        )
        if MACRO_ACRONYMS.match(text.strip()) and not HAS_DIGIT.search(text):
            return "hint_notion", f"macro_acronym '{text}'"

        # Règle 10 : DROP si mot vague sans valeur numérique ni unité
        DROP_RATE_WORDS = re.compile(
            r'^(niveau|niveaux|seuil|palier|indice|valeur|degr[eé]|rang)$',
            re.IGNORECASE
        )
        if DROP_RATE_WORDS.match(text.strip()) and not HAS_DIGIT.search(text):
            return "DROP", f"vague_rate_word '{text}'"

    return label, ""


def fix_file(input_path: str, output_path: str, dry_run: bool = False):
    with open(input_path) as f:
        phrases = [json.loads(l) for l in f]

    stats = Counter()
    changes_log = []

    for p in phrases:
        to_keep = []
        for span in p.get("spans", []):
            new_label, reason = reclassify_span(span, p.get("text", ""))

            if new_label == "DROP":
                stats["dropped"] += 1
                changes_log.append(("DROP", span.get("label"), span.get("text", ""), reason))
                continue

            if new_label != span.get("label"):
                old = span["label"]
                span["label"] = new_label
                stats[f"{old}→{new_label}"] += 1
                changes_log.append(("REMAP", old, span.get("text", ""), f"→{new_label}  [{reason}]"))

            to_keep.append(span)

        p["spans"] = to_keep

    print(f"\n=== Résumé des corrections ===")
    for key, cnt in sorted(stats.items()):
        print(f"  {key:<35} {cnt:>5}")
    print(f"  {'TOTAL spans modifiés':<35} {sum(stats.values()):>5}")

    print(f"\n=== Échantillon des changements ===")
    by_type = {}
    for op, old_lb, text, reason in changes_log:
        key = f"{op}:{old_lb}"
        by_type.setdefault(key, []).append((text, reason))

    for key, items in sorted(by_type.items()):
        print(f"\n  [{key}]  ({len(items)} total)")
        for text, reason in items[:5]:
            print(f"    \"{text[:45]}\"  →  {reason}")

    if dry_run:
        print("\n[DRY RUN] Aucune écriture.")
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

