#!/usr/bin/env python3
"""
annotate_nominal_parents.py
===========================
Annote chaque span NER enfant avec :
  - nominal_parent_start : char offset du span parent (contenant OU frère,
                           relié via l'arbre de dépendances Stanza)
  - nominal_relation     : APPOS / NMOD / POSS / AMOD / COMPOUND / SOURCE / LOC / TIME / MISC
  - semantic_role        : rôle dérivé de la relation nominale
                           (écrase le rôle du mapper verbal, souvent incorrect pour les spans imbriqués)

Deux mécanismes de rattachement, dans cet ordre :
  1. Containment (historique) : le plus petit span qui CONTIENT physiquement l'enfant.
  2. Dépendances Stanza (frères) : si aucun containment trouvé, on suit le token-tête
     de l'enfant jusqu'à son gouverneur syntaxique ; si ce gouverneur appartient à un
     AUTRE span NER (même non-imbriqué, ex: "révolte" + "du 17 mai" = 2 spans frères),
     on les relie quand même. Ça couvre la majorité des cas manqués par le containment
     seul (ex: "réseau ferroviaire" / "Saint-Étienne", "révolte" / "17 mai").

Cache Stanza : --cache-output écrit l'arbre de dépendances brut (id/text/head/deprel/
offsets par mot, groupé par phrase Stanza) pour réutilisation ultérieure sans reparser
(--cache-input). Précieux car un run complet sur train prend ~4h30.

Pour les spans sans parent (arguments directs du verbe), semantic_role est laissé inchangé.

Usage:
    python3 annotate_nominal_parents.py data/train_v8.22_semrole.jsonl -o data/train_v8.23.jsonl \\
        --cache-output data/train_v8.22_semrole_stanza_cache.jsonl
    # Réutiliser le cache (aucun reparsing Stanza) :
    python3 annotate_nominal_parents.py data/train_v8.22_semrole.jsonl -o data/train_v8.24.jsonl \\
        --cache-input data/train_v8.22_semrole_stanza_cache.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import stanza


# ── Stanza deprel → NOMINAL_RELATION ─────────────────────────────────────────
DEPREL_TO_NOMINAL_REL = {
    "nmod":          "NMOD",
    "nmod:poss":     "POSS",
    "nmod:subj":     "NMOD",
    "amod":          "AMOD",
    "appos":         "APPOS",
    "flat":          "COMPOUND",
    "flat:name":     "COMPOUND",
    "flat:foreign":  "COMPOUND",
    "compound":      "COMPOUND",
    "det":           "AMOD",
    "nsubj":         "SOURCE",   # sujet nominalisé d'un EVENT_NOMINAL Communication
    "nsubj:pass":    "NMOD",
    "obl":           "MISC",
    "obl:mod":       "MISC",
    "obl:agent":     "MISC",
    "acl":           "NMOD",
    "acl:relcl":     "NMOD",
    "advmod":        "MISC",
    "conj":          "NMOD",
    "cc":            "AMOD",
}

# Affinage obl/obl:mod → TIME ou LOC selon le hint NER
_NER_TIME = {"hint_time_date", "hint_time_clock", "hint_time_duration"}
_NER_LOC  = {"hint_gpe", "hint_fac_name", "hint_loc_generic", "hint_infra"}

# NOMINAL_RELATION → semantic_role string  (None = SKIP = OBLIQUE_UNRESOLVED)
NOMINAL_REL_TO_SEMANTIC = {
    "SOURCE":   "SOURCE",
    "LOC":      "LOCATION",
    "TIME":     "TEMPORAL",
    "APPOS":    "IDENTITY",
    "POSS":     "OWNER",
    # NMOD affiné par hint de l'enfant (cf. get_nmod_semantic_role) : un
    # complément du nom ("le budget de la santé" -> "santé") a presque toujours
    # un rôle thématique réel (DOMAIN/LOCATION/PART_OF/TEMPORAL) plutôt que rien.
    "AMOD":     None,       # adjectif qualificatif → non supervisé
    "COMPOUND": None,       # nom propre multi-tokens → non supervisé
    "MISC":     None,       # fallback → non supervisé
}

# Hints utilisés pour affiner NMOD (alignés sur build_multitask_dataset.py)
_NER_DOMAIN_HINTS = {"hint_field", "hint_doctrine", "hint_notion", "hint_language"}
_NER_ORG_HINTS    = {"hint_org_name", "hint_inst_name", "hint_inst_role"}

# Labels à ne jamais utiliser comme parent nominal (domaine verbal, pas nominal)
_SKIP_PARENT_LABELS = {"verb_trigger", "pron_subj", "pron_obj"}


def get_nmod_semantic_role(hint: str, parent: dict | None = None, child: dict | None = None) -> str:
    """Rôle sémantique d'un complément du nom (NMOD) selon le hint de l'enfant.
    Ex: "la santé" dans "le budget de la santé" -> DOMAIN plutôt que NONE.

    Cas ORG (hint_org_name/hint_inst_name/hint_inst_role) : PART_OF suppose deux
    entités DISTINCTES en relation d'appartenance (ex: "la filiale de Google").
    Mais très souvent, parent et enfant ne sont que deux découpages du MÊME
    mention à des granularités différentes (ex: "ministère des Transports" /
    "Transports" tous deux hint_inst_name) — dans ce cas "Transports" n'est pas
    "une partie du" ministère, c'est son domaine de compétence (DOMAIN).
    Heuristique : même label parent/enfant, ou enfant qui se termine exactement
    là où finit le parent (suffixe d'un nom composé) → quasi toujours la même
    entité redécoupée → DOMAIN plutôt que PART_OF.
    """
    if hint in _NER_TIME:         return "TEMPORAL"
    if hint in _NER_LOC:          return "LOCATION"
    if hint in _NER_DOMAIN_HINTS: return "DOMAIN"
    if hint in _NER_ORG_HINTS:
        same_label = parent is not None and parent.get("label") == hint
        same_end   = (
            parent is not None and child is not None
            and parent.get("end") == child.get("end")
        )
        if same_label or same_end:
            return "DOMAIN"
        return "PART_OF"
    return "DOMAIN"   # complément du nom générique : rôle thématique par défaut

# Préfixes des labels NER
_NER_PREFIXES = ("hint_",)


# ── Cache Stanza (parse brut, réutilisable) ──────────────────────────────────

class SimpleWord:
    """Substitut léger d'un stanza.Word, reconstruit depuis le cache JSONL."""
    __slots__ = ("id", "text", "head", "deprel", "start_char", "end_char")

    def __init__(self, id, text, head, deprel, start_char, end_char):
        self.id = id
        self.text = text
        self.head = head
        self.deprel = deprel
        self.start_char = start_char
        self.end_char = end_char


def doc_to_cache(doc) -> list[list[dict]]:
    """stanza.Document -> liste de phrases Stanza, chacune une liste de mots
    sérialisables (id/text/head/deprel/offsets). Format stable, indépendant
    de la version de stanza installée."""
    return [
        [
            {
                "id": w.id, "text": w.text, "head": w.head, "deprel": w.deprel,
                "start_char": w.start_char, "end_char": w.end_char,
            }
            for w in sent.words
        ]
        for sent in doc.sentences
    ]


def cache_to_stanza_sentences(cached) -> list[list[SimpleWord]]:
    return [[SimpleWord(**wd) for wd in sent] for sent in cached]


def doc_to_stanza_sentences(doc) -> list[list[SimpleWord]]:
    return cache_to_stanza_sentences(doc_to_cache(doc))


# ── Utilitaires Stanza ────────────────────────────────────────────────────────

def words_in_span(all_words, span_start: int, span_end: int):
    """Tokens Stanza dont la plage char chevauche [span_start, span_end)."""
    return [
        w for w in all_words
        if w.start_char is not None and w.end_char is not None
        and w.start_char < span_end and w.end_char > span_start
    ]


def find_span_head_token(span_words):
    """
    Token-tête du span = celui dont le HEAD est en dehors du span.
    En cas d'ambiguïté, prend le token dont le HEAD id est le plus petit
    (plus proche de la racine).
    """
    if not span_words:
        return None
    span_ids = {w.id for w in span_words}
    candidates = [w for w in span_words if w.head == 0 or w.head not in span_ids]
    if not candidates:
        return span_words[0]
    return min(candidates, key=lambda w: w.head if w.head > 0 else float("inf"))


def _relation_from_deprel(deprel: str, hint: str) -> str:
    deprel_base = (deprel or "").split(":")[0]
    rel = DEPREL_TO_NOMINAL_REL.get(deprel)
    if rel is None:
        rel = DEPREL_TO_NOMINAL_REL.get(deprel_base)
    if rel is None:
        rel = "MISC"
    # Affiner MISC/NMOD → TIME / LOC selon le hint NER (ex: "par jour", "à Lyon")
    if rel in ("MISC", "NMOD") and deprel_base in ("obl", "advmod", "nmod"):
        if hint in _NER_TIME:
            rel = "TIME"
        elif hint in _NER_LOC:
            rel = "LOC"
    return rel


def get_nominal_relation(child_words, hint: str) -> str:
    """
    Détermine la NOMINAL_RELATION du child en lisant le deprel
    du token-tête du child vers son gouverneur.
    """
    if not child_words:
        return "MISC"
    head_tok = find_span_head_token(child_words)
    if head_tok is None:
        return "MISC"
    return _relation_from_deprel(head_tok.deprel or "", hint)


# ── Logique principale ────────────────────────────────────────────────────────

def find_parent_span(child, ner_spans):
    """
    Retourne le plus petit span annoté DISTINCT qui contient strictement child.
    None si child est un argument de premier niveau (ou pas de containment).
    """
    cs, ce = child["start"], child["end"]
    best = None
    best_size = float("inf")
    for other in ner_spans:
        if other is child:
            continue
        ps, pe = other["start"], other["end"]
        # other contient child (et n'est pas identique)
        if ps <= cs and ce <= pe and (ps, pe) != (cs, ce):
            size = pe - ps
            if size < best_size:
                best_size = size
                best = other
    return best


def find_span_for_char_range(start: int, end: int, spans, min_overlap: float = 0.6):
    """Span NER qui recouvre le mieux [start, end) (pas forcément containment strict).
    Sert à résoudre le gouverneur syntaxique d'un token vers SON span, y compris
    quand ce span est un frère (pas un parent au sens containment)."""
    best = None
    best_score = min_overlap
    for sp in spans:
        s, e = sp["start"], sp["end"]
        inter = max(0, min(end, e) - max(start, s))
        denom = min(end - start, e - s) or 1
        score = inter / denom
        if score > best_score:
            best_score = score
            best = sp
    return best


def find_sibling_parent(span, ner_spans, stanza_sentences):
    """
    Fallback quand aucun containment n'a été trouvé : suit le token-tête du
    span jusqu'à son gouverneur syntaxique Stanza, et regarde si ce gouverneur
    appartient à un AUTRE span NER (frère, pas imbriqué). Ex: "révolte" + "du
    17 mai" sont deux spans distincts non-imbriqués — le containment seul ne
    peut jamais les relier, mais l'arbre de dépendances si.

    Retourne (parent_span, relation) ou (None, None).
    """
    cs, ce = span["start"], span["end"]
    for sent_words in stanza_sentences:
        overlapping = [
            w for w in sent_words
            if w.start_char is not None and w.end_char is not None
            and w.start_char < ce and w.end_char > cs
        ]
        if not overlapping:
            continue

        head_tok = find_span_head_token(overlapping)
        if head_tok is None or head_tok.head == 0:
            return None, None

        gov = next((w for w in sent_words if w.id == head_tok.head), None)
        if gov is None or gov.start_char is None:
            return None, None

        gov_span = find_span_for_char_range(gov.start_char, gov.end_char, ner_spans)
        if gov_span is None or gov_span is span:
            return None, None
        if gov_span.get("label") in _SKIP_PARENT_LABELS:
            return None, None  # gouverneur = verbe/pronom → domaine verbal (SVO), pas nominal

        rel = _relation_from_deprel(head_tok.deprel or "", span.get("label", ""))
        return gov_span, rel

    return None, None


def inject_nominal_parents(sentence_data, stanza_sentences):
    """
    Annote nominal_parent_start / nominal_relation / semantic_role sur les spans enfants.
    `stanza_sentences` : liste de phrases Stanza (list[SimpleWord]), fraîches ou depuis cache.
    Retourne (new_sentence_data, n_annotated, rel_counts, n_via_sibling).
    """
    all_words = [w for sent in stanza_sentences for w in sent]

    spans     = sentence_data.get("spans", [])
    ner_spans = [s for s in spans if s.get("label", "").startswith("hint_")]

    n_annotated  = 0
    n_via_sibling = 0
    rel_counts   = Counter()
    new_spans    = []

    for span in spans:
        label  = span.get("label", "")
        is_ner = label.startswith("hint_")

        if not is_ner:
            new_spans.append(span)
            continue

        # Déjà annoté → laisser tel quel
        if "nominal_parent_start" in span:
            new_spans.append(span)
            continue

        parent = find_parent_span(span, ner_spans)
        via_sibling = False

        if parent is not None:
            child_words  = words_in_span(all_words, span["start"], span["end"])
            parent_words = words_in_span(all_words, parent["start"], parent["end"])

            # ── Vérifier que child n'est pas la TÊTE du parent NP ────────────
            # Si le head-token de child pointe vers l'extérieur du parent,
            # alors child EST la tête du NP (ex: "cancer" dans "cancer colorectal").
            # Dans ce cas on n'annote PAS ce span via containment (on retente
            # quand même le fallback sibling ci-dessous).
            child_head = find_span_head_token(child_words)
            head_outside_parent = (
                child_head is not None
                and (
                    child_head.head == 0
                    or child_head.head not in {w.id for w in parent_words}
                )
            )
            if head_outside_parent:
                parent = None
            else:
                rel = get_nominal_relation(child_words, label)

        if parent is None:
            # ── Fallback : rattachement par dépendances (frères, pas de containment) ──
            parent, rel = find_sibling_parent(span, ner_spans, stanza_sentences)
            via_sibling = parent is not None

        if parent is None:
            # Argument de premier niveau : pas de parent nominal → inchangé
            new_spans.append(span)
            continue

        # Semantic role dérivé de la relation nominale
        if rel == "NMOD":
            semantic_role = get_nmod_semantic_role(label, parent=parent, child=span)
        else:
            sr_str = NOMINAL_REL_TO_SEMANTIC.get(rel)  # None → SKIP
            semantic_role = sr_str if sr_str is not None else "OBLIQUE_UNRESOLVED"

        new_span = dict(span)
        new_span["nominal_parent_start"] = parent["start"]
        new_span["nominal_relation"]     = rel
        new_span["semantic_role"]        = semantic_role
        if via_sibling:
            new_span["nominal_parent_source"] = "stanza_sibling"

        new_spans.append(new_span)
        n_annotated += 1
        if via_sibling:
            n_via_sibling += 1
        rel_counts[rel] += 1

    result         = dict(sentence_data)
    result["spans"] = new_spans
    return result, n_annotated, rel_counts, n_via_sibling


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Annote nominal_parent_start / nominal_relation / semantic_role via Stanza"
    )
    parser.add_argument("input",  help="JSONL source (ex: data/train_v8.22_semrole.jsonl)")
    parser.add_argument("-o", "--output", required=True, help="JSONL de sortie")
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--max-sentences", type=int, default=None, help="Debug : limiter à N phrases")
    parser.add_argument("--cache-input",  default=None, help="Cache Stanza JSONL déjà généré (skip reparsing)")
    parser.add_argument("--cache-output", default=None, help="Écrit le cache Stanza JSONL (id -> parse brut)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    print(f"📂 Chargement {input_path}...")
    sentences = []
    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_sentences and i >= args.max_sentences:
                break
            sentences.append(json.loads(line))
    print(f"   {len(sentences):,} phrases\n")

    # ── Cache Stanza (par id de phrase) ──────────────────────────────────────
    cache_by_id: dict[str, list] = {}
    if args.cache_input:
        cache_path = Path(args.cache_input)
        if cache_path.exists():
            print(f"📦 Chargement cache Stanza {cache_path}...")
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    cache_by_id[rec["id"]] = rec["parse"]
            print(f"   {len(cache_by_id):,} phrases en cache\n")
        else:
            print(f"⚠️  Cache introuvable ({cache_path}), sera reconstruit intégralement.\n")

    n_missing_cache = sum(1 for s in sentences if s.get("id") not in cache_by_id)
    nlp = None
    if n_missing_cache > 0:
        print(f"🔧 Chargement pipeline Stanza (fr) — {n_missing_cache:,} phrases à parser...")
        nlp = stanza.Pipeline(
            "fr",
            processors="tokenize,mwt,pos,lemma,depparse",
            tokenize_pretokenized=False,
            verbose=False,
            use_gpu=False,
        )
        print("✅ Pipeline Stanza prêt\n")
    else:
        print("✅ Toutes les phrases sont déjà en cache, aucun reparsing Stanza nécessaire\n")

    n_ner_total      = sum(1 for s in sentences for sp in s.get("spans", [])
                           if sp.get("label", "").startswith("hint_"))
    n_already_parent = sum(1 for s in sentences for sp in s.get("spans", [])
                           if "nominal_parent_start" in sp)
    print(f"📊 Spans NER total          : {n_ner_total:,}")
    print(f"   Déjà avec parent annoté  : {n_already_parent:,}")
    print(f"   À analyser               : {n_ner_total - n_already_parent:,}\n")

    total_annotated  = 0
    total_via_sibling = 0
    total_rel        = Counter()

    cache_out_f = open(args.cache_output, "w", encoding="utf-8") if args.cache_output else None

    print(f"🚀 Traitement par batchs de {args.batch_size}...")
    with open(output_path, "w", encoding="utf-8") as f_out:
        for batch_start in range(0, len(sentences), args.batch_size):
            batch = sentences[batch_start : batch_start + args.batch_size]

            for sent_data in batch:
                sid = sent_data.get("id")
                cached = cache_by_id.get(sid)
                if cached is not None:
                    stanza_sentences = cache_to_stanza_sentences(cached)
                else:
                    doc = nlp(sent_data.get("text", ""))
                    cached = doc_to_cache(doc)
                    stanza_sentences = cache_to_stanza_sentences(cached)
                    cache_by_id[sid] = cached  # dispo pour un run ultérieur dans le même process

                if cache_out_f is not None:
                    cache_out_f.write(json.dumps({"id": sid, "parse": cached}, ensure_ascii=False) + "\n")

                result, n, rels, n_sib = inject_nominal_parents(sent_data, stanza_sentences)
                total_annotated  += n
                total_via_sibling += n_sib
                total_rel        += rels
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

            done = min(batch_start + args.batch_size, len(sentences))
            pct  = done / len(sentences) * 100
            print(f"  [{done:>6,}/{len(sentences):,}] {pct:5.1f}%  "
                  f"+{total_annotated:,} parents annotés ({total_via_sibling:,} via frères)", end="\r")

    if cache_out_f is not None:
        cache_out_f.close()

    print(f"\n\n{'='*60}")
    print(f"✅ RÉSULTAT")
    print(f"{'='*60}")
    print(f"  Sortie              : {output_path}")
    print(f"  Parents injectés    : {total_annotated:,}")
    print(f"    dont via frères   : {total_via_sibling:,} (rattachement par dépendances, pas containment)")
    print(f"  Taux de couverture  : {total_annotated / max(1, n_ner_total - n_already_parent) * 100:.1f}%")
    print(f"\n  Distribution NOMINAL_RELATION :")
    for rel, cnt in total_rel.most_common():
        sr = NOMINAL_REL_TO_SEMANTIC.get(rel)
        sr_label = sr if sr is not None else "SKIP"
        print(f"    {rel:12s}  {cnt:6,}  → semantic_role={sr_label}")
    print(f"{'='*60}")
    if args.cache_output:
        print(f"\n💾 Cache Stanza écrit : {args.cache_output}")
    print(f"\n💡 Prochaine étape :")
    print(f"   Répéter sur val + test, puis :")
    print(f"   dvc add data/train_v8.24.jsonl ... && dvc push ...")
    print(f"   Mettre à jour DEFAULT_GOLD_VERSION dans launch_training.py")


if __name__ == "__main__":
    main()

