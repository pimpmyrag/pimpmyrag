#!/usr/bin/env python3
"""
Diagnostic qualité des candidates : accents, markup, doublons de spans.
+ nettoyage → fichier prêt pour Mistral.
"""
import json
import re
import unicodedata

INPUT = "data/wikinews_rare_candidates_clean.jsonl"
OUTPUT = "data/wikinews_ready_for_mistral.jsonl"
REJECTED = "data/wikinews_rejected.jsonl"

MARKUP_PATTERNS = ['__NOTOC__', '__TOC__', 'colspan', 'rowspan', '{|', '|}', '[[', ']]', '{{', '}}', '<ref', '</ref', '<br', '<!--']

def has_accent_issues(text):
    """Détecte si le texte a des accents manquants (mots français courants sans accents)."""
    # Mots français qui devraient avoir des accents
    suspect_words = [
        r'\bprsid', r'\blect', r'\bdfai', r'\bscur', r'\bdcid',
        r'\bsupri', r'\brpubli', r'\bgnral', r'\bfvri', r'\bdcem',
        r'\bfranai', r'\bcono', r'\bscurit', r'\brcent',
        r'\bminist', r'\bdmont', r'\bncessai', r'\bspcial',
        r'\bdclar', r'\bamnag', r'\bintress', r'\borganis',
    ]
    count = sum(1 for p in suspect_words if re.search(p, text, re.IGNORECASE))
    return count >= 2  # au moins 2 mots suspects

def has_markup(text):
    return any(x in text for x in MARKUP_PATTERNS)

def ratio_ascii_letters(text):
    """Ratio de lettres sans accents parmi toutes les lettres."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    accented = sum(1 for c in letters if unicodedata.combining(unicodedata.normalize('NFD', c)[-1]) if len(unicodedata.normalize('NFD', c)) > 1)
    # Plus simple : compter les lettres avec diacritiques
    accented = sum(1 for c in letters if ord(c) > 127)
    return accented / len(letters)

def is_good_french(text):
    """Vérifie que le texte contient au moins quelques lettres accentuées françaises typiques."""
    fr_accented = set('àâäéèêëïîôùûüçœæÀÂÄÉÈÊËÏÎÔÙÛÜÇŒÆ')
    has_any = any(c in fr_accented for c in text)
    # Si le texte fait > 80 chars et contient aucun accent, c'est suspect
    if len(text) > 80 and not has_any:
        return False
    return True

def main():
    with open(INPUT) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    print(f"Total: {len(lines)} phrases\n")

    # Diagnostic
    n_markup = 0
    n_no_accent = 0
    n_accent_issue = 0
    n_too_many_preds = 0
    n_ok = 0

    good = []
    rejected = []

    for obj in lines:
        text = obj["text"]
        problems = []

        if has_markup(text):
            problems.append("markup")
            n_markup += 1

        if not is_good_french(text) and has_accent_issues(text):
            problems.append("accents_manquants")
            n_accent_issue += 1
        elif not is_good_french(text):
            problems.append("pas_accents")
            n_no_accent += 1

        # Trop de prédictions (spam du modèle) — seulement si c'est vraiment excessif
        if len(obj.get("predictions", [])) > 25:
            problems.append("trop_de_preds")
            n_too_many_preds += 1

        if problems:
            obj["_problems"] = problems
            rejected.append(obj)
        else:
            n_ok += 1
            good.append(obj)

    print(f"✅ Phrases OK:              {n_ok}")
    print(f"❌ Markup résiduel:          {n_markup}")
    print(f"❌ Accents manquants:        {n_accent_issue}")
    print(f"⚠️  Pas d'accents (court?):  {n_no_accent}")
    print(f"⚠️  Trop de prédictions:     {n_too_many_preds}")
    print(f"❌ Total rejetées:           {len(rejected)}")

    # Exemples de rejetées
    if rejected:
        print(f"\n🔍 Exemples rejetées (max 10):")
        for r in rejected[:10]:
            print(f"  [{r['id']}] {r['_problems']} → {r['text'][:100]}")

    # Écrire
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for obj in good:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(REJECTED, "w", encoding="utf-8") as out:
        for obj in rejected:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"\n💾 {len(good)} phrases propres → {OUTPUT}")
    print(f"💾 {len(rejected)} phrases rejetées → {REJECTED}")

    # Stats labels dans les phrases gardées
    from collections import Counter
    label_counts = Counter()
    for obj in good:
        for p in obj.get("predictions", []):
            fl = p.get("fine", "")
            label_counts[fl] += 1

    print(f"\n📊 Labels dans les phrases propres:")
    for label, count in label_counts.most_common():
        print(f"  {label:<25} {count:>6}")


if __name__ == "__main__":
    main()

