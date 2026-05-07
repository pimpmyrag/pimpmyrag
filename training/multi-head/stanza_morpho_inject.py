#!/usr/bin/env python3
"""
stanza_morpho_inject.py  — v2  (avec dep parse + gender_guesser)
================================================================
Injecte gender/number/person dans les spans "nus" d'un fichier .jsonl.

Stratégie par type de span :
  - Noms communs (NOUN)   : feats Stanza directement sur le token tête
  - Noms propres (PROPN)  : 3 niveaux
      1. NOUN tête de dépendance avec gender (titre/rôle apposé)
      2. Participe passé accordé (nsubj:pass → Part+Gender)
      3. gender_guesser sur le prénom (premier token du span)

Usage:
    python3 stanza_morpho_inject.py data/train.jsonl -o data/train_v8.1.jsonl
"""

import json
import sys
import argparse
from collections import Counter

import stanza
import gender_guesser.detector as _gg_module

# ── Constantes ────────────────────────────────────────────────────────────────

GENDER_MAP = {"Masc": "M", "Fem": "F", "Neut": "N"}
NUMBER_MAP  = {"Sing": "SG", "Plur": "PL"}
PERSON_MAP  = {"1": "1", "2": "2", "3": "3"}

# Labels dont les tokens sont des noms propres → logique spécifique
PROPN_LABELS = {"hint_person_name"}

# Labels pour lesquels on ne cherche pas de gender
SKIP_LABELS = {"verb_trigger"}

# gender_guesser: résultats fiables
GG_TO_DATASET = {
    "male": "M", "mostly_male": "M",
    "female": "F", "mostly_female": "F",
}

_gg_detector = _gg_module.Detector(case_sensitive=False)


# ── Utilitaires ───────────────────────────────────────────────────────────────

def parse_feats(feats_str: str) -> dict:
    if not feats_str or feats_str == "_":
        return {}
    return {k: v for k, v in (p.split("=", 1) for p in feats_str.split("|") if "=" in p)}


def words_in_span(all_words, span_start: int, span_end: int):
    """Tokens Stanza dont la plage caractère chevauche [span_start, span_end)."""
    return [
        w for w in all_words
        if w.start_char < span_end and w.end_char > span_start
    ]


# ── Stratégie noms communs ────────────────────────────────────────────────────

def noun_morpho(tokens) -> dict:
    """Premier NOUN/PROPN avec Gender dans le span."""
    candidates = [t for t in tokens if t.feats and "Gender" in t.feats]
    if not candidates:
        return {}
    for t in candidates:
        if t.upos in ("NOUN", "PROPN"):
            return parse_feats(t.feats)
    return parse_feats(candidates[0].feats)


# ── Stratégie noms propres ────────────────────────────────────────────────────

def propn_morpho_from_context(span_tokens, all_words_by_id: dict) -> dict:
    """
    Niveau 1 & 2 : inférence via graphe de dépendances Stanza.

    Niveau 1 — titre/rôle apposé (deprel strict) :
      PROPN → head=NOUN avec Gender ET deprel ∈ {appos, nsubj, nsubj:pass}
      ex: "le président [Macron]" (deprel=appos) → président.Gender=Masc → M

    Niveau 2 — participe passé accordé :
      PROPN → head=VERB (VerbForm=Part + Gender) ET deprel ∈ {nsubj, nsubj:pass}
      ex: "[Simone Veil] a été élue" → élue.Gender=Fem → F

    On n'accepte PAS n'importe quel head NOUN pour éviter les faux positifs.
    """
    VALID_DEPRELS = {"appos", "nsubj", "nsubj:pass"}

    for tok in span_tokens:
        if tok.upos != "PROPN":
            continue
        head_id = tok.head
        if head_id == 0:
            continue
        head = all_words_by_id.get(head_id)
        if head is None:
            continue

        deprel = tok.deprel or ""
        # Ignorer les relations qui ne correspondent pas à un titre/accord
        if deprel not in VALID_DEPRELS and not any(d in deprel for d in ("nsubj", "appos")):
            continue

        feats = parse_feats(head.feats or "")

        # Niveau 1 : tête NOUN avec gender (rôle/titre apposé)
        if head.upos == "NOUN" and "Gender" in feats:
            # On ne retourne QUE le gender (pas le number : la personne est toujours SG)
            return {"Gender": feats["Gender"]}

        # Niveau 2 : participe passé accordé (nsubj ou nsubj:pass)
        if (head.upos in ("VERB", "AUX")
                and feats.get("VerbForm") == "Part"
                and "Gender" in feats):
            return {"Gender": feats["Gender"]}

    return {}


def propn_morpho_from_name(span_text: str) -> dict:
    """
    Niveau 3 : gender_guesser sur le prénom (premier mot).
    Retourne {"Gender": "Masc"|"Fem"} si confiant, {} sinon.
    """
    first_name = span_text.split()[0].strip(".,;:-'\"")
    result = _gg_detector.get_gender(first_name, "france")
    if result in GG_TO_DATASET:
        g = "Masc" if GG_TO_DATASET[result] == "M" else "Fem"
        return {"Gender": g}
    # Sans contrainte de pays pour les prénoms étrangers
    result = _gg_detector.get_gender(first_name)
    if result in ("male", "female"):
        g = "Masc" if result == "male" else "Fem"
        return {"Gender": g}
    return {}


# ── Injection principale ──────────────────────────────────────────────────────

def inject_morpho(phrase: dict, all_words: list) -> Counter:
    """
    Injecte gender/number/person dans les spans nus d'une phrase.
    Retourne des stats Counter.
    """
    # Index id→word (MWT ont id=tuple, on les exclut)
    all_words_by_id = {w.id: w for w in all_words if isinstance(w.id, int)}
    stats = Counter()

    for span in phrase.get("spans", []):
        label = span.get("label", "")
        if label in SKIP_LABELS:
            continue
        # Pour les labels PROPN, on force le re-passage même si déjà annoté
        # (les anciennes annotations Stanza peuvent être fausses sur les noms propres)
        if label not in PROPN_LABELS:
            if span.get("gender") or span.get("_score") or span.get("svo_role"):
                continue

        span_tokens = words_in_span(all_words, span["start"], span["end"])
        if not span_tokens:
            continue

        label = span.get("label", "")
        morpho = {}

        if label in PROPN_LABELS:
            # Niveau 1 & 2 : contexte syntaxique
            morpho = propn_morpho_from_context(span_tokens, all_words_by_id)
            if morpho:
                stats["injected_dep"] += 1
            else:
                # Niveau 3 : gender_guesser
                morpho = propn_morpho_from_name(span.get("text", ""))
                if morpho:
                    stats["injected_gg"] += 1
                else:
                    stats["failed_propn"] += 1
        else:
            morpho = noun_morpho(span_tokens)
            if morpho:
                stats["injected_noun"] += 1
            else:
                stats["failed_noun"] += 1

        if not morpho:
            continue

        if "Gender" in morpho:
            span["gender"] = GENDER_MAP.get(morpho["Gender"], morpho["Gender"])
        # Pour les noms propres de personnes : toujours SG, pas de number hérité du contexte
        if label in PROPN_LABELS:
            span["number"] = "SG"
        elif "Number" in morpho:
            span["number"] = NUMBER_MAP.get(morpho["Number"], morpho["Number"])
        if "Person" in morpho and morpho.get("Person") in PERSON_MAP:
            span["person"] = PERSON_MAP[morpho["Person"]]

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Injecte les annotations morpho Stanza dans les spans nus"
    )
    parser.add_argument("input", help="Fichier jsonl d'entrée")
    parser.add_argument("-o", "--output", required=True, help="Fichier jsonl de sortie")
    args = parser.parse_args()

    print(f"Chargement {args.input}...")
    with open(args.input) as f:
        phrases = [json.loads(l) for l in f]
    print(f"  {len(phrases):,} phrases chargées")

    to_process_idx = []
    for i, p in enumerate(phrases):
        needs_processing = False
        for span in p.get("spans", []):
            label = span.get("label", "")
            if label in SKIP_LABELS:
                continue
            # Re-passer sur tous les PROPN_LABELS (même déjà annotés)
            if label in PROPN_LABELS:
                needs_processing = True
                break
            # Pour les autres : seulement les spans nus
            if not span.get("gender") and not span.get("_score") and not span.get("svo_role"):
                needs_processing = True
                break
        if needs_processing:
            to_process_idx.append(i)

    print(f"  {len(to_process_idx):,} phrases avec spans nus à annoter")
    if not to_process_idx:
        print("Rien à faire.")
        with open(args.output, "w") as f:
            for p in phrases:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        return

    print("Initialisation Stanza fr (tokenize, mwt, pos, lemma, depparse)...")
    nlp = stanza.Pipeline("fr", processors="tokenize,mwt,pos,lemma,depparse", verbose=False)
    print("  Stanza prêt.\n")

    total_stats = Counter()
    n = len(to_process_idx)

    for i, idx in enumerate(to_process_idx):
        phrase = phrases[idx]
        doc = nlp(phrase["text"])
        # Filtrer les tokens MWT sans offset
        all_words = [
            w for sent in doc.sentences for w in sent.words
            if isinstance(w.id, int) and w.start_char is not None
        ]
        stats = inject_morpho(phrase, all_words)
        total_stats.update(stats)

        if (i + 1) % 200 == 0 or (i + 1) == n:
            total_inj = (total_stats["injected_noun"]
                         + total_stats["injected_dep"]
                         + total_stats["injected_gg"])
            print(
                f"  [{i+1:>5}/{n}  {100*(i+1)/n:.0f}%]  injectés={total_inj}  "
                f"(noun={total_stats['injected_noun']} "
                f"dep={total_stats['injected_dep']} "
                f"gg={total_stats['injected_gg']} "
                f"failed={total_stats['failed_propn']+total_stats['failed_noun']})",
                end="\r",
            )

    total_inj = (total_stats["injected_noun"]
                 + total_stats["injected_dep"]
                 + total_stats["injected_gg"])
    print(f"\n\n=== Résultat ===")
    print(f"  Noms communs  (NOUN direct)    : {total_stats['injected_noun']:>6}")
    print(f"  Noms propres  (dep parse)      : {total_stats['injected_dep']:>6}")
    print(f"  Noms propres  (gender_guesser) : {total_stats['injected_gg']:>6}")
    print(f"  Non résolus   (GPE, chiffres…) : {total_stats['failed_propn']+total_stats['failed_noun']:>6}")
    print(f"  ──────────────────────────────────────")
    print(f"  Total injectés                 : {total_inj:>6}")

    print(f"\nSauvegarde → {args.output}")
    with open(args.output, "w") as f:
        for p in phrases:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("Terminé.")


if __name__ == "__main__":
    main()

