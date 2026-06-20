#!/usr/bin/env python3
"""
stanza_inject_nominal_parents.py — v1.1 conservative (v8.22+)
============================================================

Étape 2 du pipeline nominal_parent_pointer :
Utilise Stanza (dep parse UD) pour proposer des relations nominales,
mais uniquement si elles passent des garde-fous sémantiques stricts.

Input  : *_rules.jsonl
Output : *_stanza.jsonl

Optionnel :
-cache-output *_stanza_cache.jsonl  → snapshot persistant des sorties Stanza
                                      pour réutilisation ultérieure sans recalcul.

Principes :
- Rules > Stanza
- Stanza propose, les règles valident
- Haiku/review si conflit, source/citation, OCR suspect, parent ambigu
- Ne jamais attacher verb_trigger/pron_* via nominal_parent
- Ne jamais accepter APPOS/NMOD absurdes entre MEASURE et ORG/EVENT/etc.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

OVERLAP_MIN = 0.60

SKIP_LABELS = {"verb_trigger", "pron_subj", "pron_obj"}

MEASURE_LABELS = {
    "hint_measure",
    "hint_rate",
    "hint_money",
    "hint_value",
}

EVENT_NOMINAL_LABELS = {
    "hint_event_nominal",
    "hint_event_named",
}

COMMUNICATION_EVENT_WORDS = {
    "déclaration", "déclarations",
    "annonce", "annonces",
    "rapport", "rapports",
    "communiqué", "communiqués",
    "discours", "allocution", "propos",
    "interview", "entretien",
}

# Relations UD de base. Certaines sont ensuite spécialisées avec les marqueurs case.
UD_TO_NOMINAL_BASE = {
    "appos":        "APPOS",
    "nmod":         "NMOD",
    "nmod:poss":    "POSS",
    "nmod:de":      "NMOD",
    "nmod:arg":     "NMOD",
    "amod":         "AMOD",
    "compound":     "COMPOUND",
    "flat:name":    "COMPOUND",
    "flat:foreign": "COMPOUND",
    "flat":         "COMPOUND",
    # advmod n'est PAS nominal en général. On le garde seulement si validation stricte.
    "advmod":       "AMOD",
    # obl n'est pas nominal en général, mais peut exprimer un lien sur un EVENT_NOMINAL.
    "obl":          "MISC",
    "obl:de":       "NMOD",
}

CASE_SOURCE = {"de", "du", "des", "d'", "d’"}
CASE_LOC = {"à", "au", "aux", "dans", "en", "sur", "chez", "vers"}
CASE_MEDIUM = {"dans", "sur"}


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_stanza():
    import stanza
    print("⏳ Chargement Stanza fr...", flush=True)
    nlp = stanza.Pipeline(
        lang="fr",
        processors="tokenize,pos,lemma,depparse",
        tokenize_no_ssplit=True,
        verbose=False,
    )
    print("✅ Stanza prêt", flush=True)
    return nlp


# ─────────────────────────────────────────────────────────────────────────────
# Span helpers
# ─────────────────────────────────────────────────────────────────────────────

def span_key(sp: dict) -> tuple[int, int]:
    return sp["start"], sp["end"]


def span_len(sp: dict) -> int:
    return max(1, sp["end"] - sp["start"])


def span_text(sp: dict) -> str:
    return sp.get("text", "")


def is_same_span(a: dict, b: dict) -> bool:
    return a["start"] == b["start"] and a["end"] == b["end"]


def contains_span(parent: dict, child: dict) -> bool:
    return (
            parent["start"] <= child["start"]
            and parent["end"] >= child["end"]
            and not is_same_span(parent, child)
    )


def overlaps_span(a: dict, b: dict) -> bool:
    return min(a["end"], b["end"]) - max(a["start"], b["start"]) > 0


def gap_between(a: dict, b: dict) -> int:
    if overlaps_span(a, b):
        return 0
    if a["end"] <= b["start"]:
        return b["start"] - a["end"]
    return a["start"] - b["end"]


def label_of(sp: dict) -> str:
    return sp.get("label") or sp.get("type") or ""


def is_verb_or_pron(sp: dict) -> bool:
    return label_of(sp) in SKIP_LABELS


def is_measure(sp: dict) -> bool:
    return label_of(sp) in MEASURE_LABELS or coarse_of(sp) == "VALUE"


def coarse_of(sp: dict) -> str | None:
    if sp.get("coarse"):
        return sp["coarse"]

    label = label_of(sp)

    if label.startswith("hint_person") or label in {"hint_norp", "hint_group_role"}:
        return "PER"
    if label in {"hint_org_name", "hint_inst_name"}:
        return "ORG"
    if label in {"hint_fac_name", "hint_gpe", "hint_loc_generic", "hint_infra"}:
        return "LOC"
    if label.startswith("hint_time"):
        return "TIME"
    if label in {"hint_measure", "hint_rate", "hint_money", "hint_value"}:
        return "VALUE"
    if label.startswith("hint_event"):
        return "EVENT"
    if label in {"hint_document", "hint_work_generic", "hint_law"}:
        return "WORK"
    if label in {"hint_notion", "hint_field", "hint_state"}:
        return "ABSTRACT"
    if label in {"hint_object_name", "hint_vehicle", "hint_tool", "hint_substance"}:
        return "OBJECT"

    return None


def is_nominal_candidate(sp: dict) -> bool:
    if is_verb_or_pron(sp):
        return False
    if sp.get("start") is None or sp.get("end") is None:
        return False
    if sp["end"] <= sp["start"]:
        return False
    return True


def char_overlap(tok_start: int, tok_end: int, span_start: int, span_end: int) -> float:
    inter_start = max(tok_start, span_start)
    inter_end = min(tok_end, span_end)
    inter = max(0, inter_end - inter_start)
    tok_len = max(1, tok_end - tok_start)
    span_len_ = max(1, span_end - span_start)
    return inter / min(tok_len, span_len_)


def find_span_for_token(token_start: int, token_end: int, spans: list[dict]) -> dict | None:
    """
    Trouve le meilleur span pour un token.
    Préfère :
    - fort overlap
    - span plus long si overlap identique
    - score éventuel plus élevé
    """
    best = None
    best_tuple = (OVERLAP_MIN, -1, -1.0)

    for sp in spans:
        if not is_nominal_candidate(sp):
            continue

        score = char_overlap(token_start, token_end, sp["start"], sp["end"])
        if score < OVERLAP_MIN:
            continue

        ner_score = float(sp.get("score") or sp.get("confidence") or 0.0)
        candidate_tuple = (score, span_len(sp), ner_score)

        if candidate_tuple > best_tuple:
            best_tuple = candidate_tuple
            best = sp

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Source block / OCR detection
# ─────────────────────────────────────────────────────────────────────────────

QUOTE_RE = re.compile(r'["«“](.+?)["»”]')
CONSULT_RE = re.compile(
    r"\bconsult[ée]?\s+le\s+\d{1,2}/\d{1,2}/\d{2,4}",
    flags=re.IGNORECASE,
)

OCR_GLUE_RE = re.compile(
    r"(heure|heures|h|mSv/h|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)(Le|La|Les|L'|L’)\b"
)


def has_suspicious_segmentation(text: str) -> bool:
    """
    Détecte des collages OCR/segmentation typiques :
    - heureLe Midi Libre
    - marsLe ...
    """
    return OCR_GLUE_RE.search(text) is not None


def detect_source_blocks(text: str, spans: list[dict]) -> list[dict]:
    """
    Détecte grossièrement les blocs bibliographiques/citationnels.
    Exemple :
      Le Midi Libre, "très forte radioactivité...", consulté le 27/03/2011
    """
    blocks = []

    for m in QUOTE_RE.finditer(text):
        quote_start, quote_end = m.span()

        # Publication candidate avant la citation.
        previous_sources = [
            sp for sp in spans
            if is_nominal_candidate(sp)
               and sp["end"] <= quote_start
               and quote_start - sp["end"] <= 80
               and coarse_of(sp) in {"ORG", "WORK"}
        ]
        publication = max(previous_sources, key=lambda sp: (sp["end"], span_len(sp)), default=None)

        block_start = publication["start"] if publication else quote_start
        block_end = quote_end

        # Étendre avec "consulté le DATE" si présent après citation.
        after = text[quote_end: quote_end + 120]
        cm = CONSULT_RE.search(after)
        if cm:
            block_end = quote_end + cm.end()

        blocks.append({
            "start": block_start,
            "end": block_end,
            "quote_start": quote_start,
            "quote_end": quote_end,
            "publication_text": publication.get("text") if publication else None,
            "publication_start": publication.get("start") if publication else None,
            "publication_end": publication.get("end") if publication else None,
            "source": "heuristic",
        })

    return blocks


def block_index_for_span(sp: dict, blocks: list[dict]) -> int | None:
    center = (sp["start"] + sp["end"]) // 2
    for i, b in enumerate(blocks):
        if b["start"] <= center < b["end"]:
            return i
    return None


def crosses_source_block(child: dict, parent: dict, blocks: list[dict]) -> bool:
    c_block = block_index_for_span(child, blocks)
    p_block = block_index_for_span(parent, blocks)
    return c_block != p_block and (c_block is not None or p_block is not None)


# ─────────────────────────────────────────────────────────────────────────────
# Relation specialisation and validation
# ─────────────────────────────────────────────────────────────────────────────

def normalize_case_marker(s: str) -> str:
    return s.lower().replace("’", "'").strip()


def relation_from_ud(
        deprel: str,
        case_markers: list[str],
        child_span: dict,
        parent_span: dict,
) -> str | None:
    base = UD_TO_NOMINAL_BASE.get(deprel)
    if base is None:
        return None

    cases = {normalize_case_marker(c) for c in case_markers}

    child_coarse = coarse_of(child_span)
    parent_coarse = coarse_of(parent_span)
    child_text = span_text(child_span).lower()
    parent_text = span_text(parent_span).lower()

    # advmod ne doit pas devenir AMOD sauf cas très strict :
    # child strictement inclus dans parent, typiquement adjectif span dans NP entier.
    if deprel == "advmod":
        if contains_span(parent_span, child_span):
            return "AMOD"
        return None

    # amod : normalement child adjectival -> parent nominal.
    if deprel == "amod":
        return "AMOD"

    # appos direct.
    if deprel == "appos":
        return "APPOS"

    # Possession.
    if deprel == "nmod:poss":
        return "POSS"

    # NMOD/OBL avec préposition.
    if deprel.startswith("nmod") or deprel.startswith("obl"):
        if cases & CASE_SOURCE:
            if parent_coarse in {"EVENT", "WORK"} or label_of(parent_span) in EVENT_NOMINAL_LABELS:
                if child_coarse in {"PER", "ORG", "WORK"}:
                    return "SOURCE"
            return "NMOD"

        if cases & CASE_MEDIUM:
            # "dans les colonnes du journal X", "sur Twitter", etc.
            if (
                    "journal" in child_text
                    or "colonnes" in child_text
                    or child_coarse in {"ORG", "WORK"}
                    or parent_coarse in {"EVENT", "WORK"}
            ):
                return "MEDIUM"

        if cases & CASE_LOC:
            if child_coarse == "LOC" or label_of(child_span) in {"hint_fac_name", "hint_gpe", "hint_infra"}:
                return "LOC"
            return "NMOD"

        return base

    return base


def valid_appos(parent: dict, child: dict) -> bool:
    """
    APPOS est très contraint.
    On évite toutes les appositions absurdes type MEASURE↔ORG/EVENT.
    """
    if is_measure(parent) or is_measure(child):
        return False

    p = coarse_of(parent)
    c = coarse_of(child)

    if p is None or c is None:
        return False

    # Même grande famille, souvent acceptable.
    if p == c and p in {"PER", "ORG", "LOC", "WORK", "OBJECT"}:
        return True

    # ROLE/PER déjà couvert par PER/PER via coarse.
    # WORK générique + ORG peut être publication ("journal Les Échos").
    if p == "WORK" and c in {"ORG", "WORK"}:
        return True
    if c == "WORK" and p in {"ORG", "WORK"}:
        return True

    return False


def valid_amod(parent: dict, child: dict) -> bool:
    """
    AMOD attendu : child inclus dans parent ou très proche.
    Exemple : solides -> fondations solides.
    Rejette Midi Libre -> Midi.
    """
    if is_measure(parent) or is_measure(child):
        return False

    # Le cas le plus propre : l'adjectif/qualificatif est inclus dans le span parent.
    if contains_span(parent, child):
        return True

    # Sinon proximité faible et pas d'entité propre longue vers fragment.
    if contains_span(child, parent):
        return False

    return gap_between(parent, child) <= 3


def validate_edge(
        child: dict,
        parent: dict,
        relation: str,
        source_blocks: list[dict],
) -> tuple[bool, str | None]:
    """
    Garde-fous sémantiques.
    Retourne (valid, reason_if_invalid).
    """
    if not is_nominal_candidate(child):
        return False, "child_not_nominal"
    if not is_nominal_candidate(parent):
        return False, "parent_not_nominal"

    if is_same_span(child, parent):
        return False, "same_span"

    if is_verb_or_pron(child) or is_verb_or_pron(parent):
        return False, "verb_or_pron_endpoint"

    if crosses_source_block(child, parent, source_blocks):
        return False, "crosses_source_block"

    # Éviter les edges entre fragments nested qui relèvent plutôt de NER tree.
    if contains_span(child, parent) and relation in {"APPOS", "AMOD", "NMOD", "COMPOUND"}:
        return False, "child_contains_parent_fragment"

    # Les mesures ne doivent pas être parents nominaux génériques.
    if is_measure(parent) and relation not in {"UNIT", "QUALIFIER", "TIME", "COMPARE_TO"}:
        return False, "measure_as_nominal_parent"

    # APPOS stricte.
    if relation == "APPOS" and not valid_appos(parent, child):
        return False, "invalid_appos_compatibility"

    # AMOD stricte.
    if relation == "AMOD" and not valid_amod(parent, child):
        return False, "invalid_amod_shape"

    # COMPOUND/FLAT : éviter d'utiliser ça entre entités longues incompatibles.
    if relation == "COMPOUND":
        if is_measure(parent) or is_measure(child):
            return False, "invalid_compound_measure"
        if gap_between(parent, child) > 8 and not overlaps_span(parent, child):
            return False, "compound_too_far"

    # NMOD/SOURCE/MEDIUM/LOC/TIME : distance raisonnable, sauf si containment.
    if relation in {"NMOD", "SOURCE", "MEDIUM", "LOC", "TIME", "POSS", "MISC"}:
        if gap_between(parent, child) > 120 and not overlaps_span(parent, child):
            return False, "nominal_edge_too_far"

    return True, None


# ─────────────────────────────────────────────────────────────────────────────
# Stanza extraction
# ─────────────────────────────────────────────────────────────────────────────

def stanza_edges_for_sentence(
        doc_sent,
        spans: list[dict],
        text: str,
        source_blocks: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Extrait des arêtes nominales depuis une phrase Stanza.
    Retourne (valid_edges, rejected_edges).
    """
    edges = []
    rejected = []

    tokens = list(doc_sent.tokens)
    words = list(doc_sent.words)

    word_id_to_token = {}
    word_id_to_word = {}
    for tok in tokens:
        for w in tok.words:
            word_id_to_token[w.id] = tok
            word_id_to_word[w.id] = w

    # Index des case markers de chaque mot head.
    case_markers_by_head: dict[int, list[str]] = defaultdict(list)
    for w in words:
        if (w.deprel or "") == "case" and w.head:
            case_markers_by_head[w.head].append((w.text or "").lower())

    for word in words:
        deprel = word.deprel or ""
        if deprel not in UD_TO_NOMINAL_BASE:
            continue

        child_tok = word_id_to_token.get(word.id)
        if child_tok is None:
            continue

        head_id = word.head
        if head_id == 0:
            continue

        head_tok = word_id_to_token.get(head_id)
        if head_tok is None:
            continue

        child_span = find_span_for_token(child_tok.start_char, child_tok.end_char, spans)
        parent_span = find_span_for_token(head_tok.start_char, head_tok.end_char, spans)

        if child_span is None or parent_span is None:
            continue

        case_markers = case_markers_by_head.get(word.id, [])
        relation = relation_from_ud(deprel, case_markers, child_span, parent_span)
        if relation is None:
            continue

        valid, reason = validate_edge(child_span, parent_span, relation, source_blocks)

        edge = {
            "child_start": child_span["start"],
            "child_end": child_span["end"],
            "child_text": child_span["text"],
            "parent_start": parent_span["start"],
            "parent_end": parent_span["end"],
            "parent_text": parent_span["text"],
            "relation": relation,
            "confidence": 0.82,
            "source": "stanza",
            "ud_deprel": deprel,
            "case_markers": case_markers,
        }

        if valid:
            edges.append(edge)
        else:
            bad = dict(edge)
            bad["rejected_reason"] = reason
            rejected.append(bad)

    return edges, rejected


# ─────────────────────────────────────────────────────────────────────────────
# Merge and needs_haiku
# ─────────────────────────────────────────────────────────────────────────────

def edge_agrees(a: dict, b: dict) -> bool:
    return (
            a.get("parent_start") == b.get("parent_start")
            and a.get("parent_end") == b.get("parent_end")
            and a.get("relation") == b.get("relation")
    )


def merge_edges(rule_edges: list[dict], stanza_edges: list[dict]) -> list[dict]:
    """
    Fusion rule + stanza par enfant.
    Accord = même parent + même relation.
    Parent différent ou relation différente = conflict.
    """
    by_child: dict[tuple[int, int], list[dict]] = defaultdict(list)

    for e in rule_edges + stanza_edges:
        if e.get("child_start") is None or e.get("child_end") is None:
            continue
        by_child[(e["child_start"], e["child_end"])].append(e)

    merged = []

    for _, es in by_child.items():
        rule_es = [e for e in es if e.get("source") == "rule"]
        stanza_es = [e for e in es if e.get("source") == "stanza"]

        if rule_es and stanza_es:
            best_rule = max(rule_es, key=lambda e: e.get("confidence", 0.0))
            best_stanza = max(stanza_es, key=lambda e: e.get("confidence", 0.0))

            if edge_agrees(best_rule, best_stanza):
                boosted = dict(best_rule)
                boosted["confidence"] = min(1.0, float(best_rule.get("confidence", 0.0)) + 0.05)
                boosted["confirmed_by_stanza"] = True
                merged.append(boosted)
            else:
                # On garde la rule comme edge candidate, mais on marque conflit.
                conflict = dict(best_rule)
                conflict["source"] = "conflict"
                conflict["confidence"] = max(
                    float(best_rule.get("confidence", 0.0)),
                    float(best_stanza.get("confidence", 0.0)),
                ) * 0.85
                conflict["stanza_parent_start"] = best_stanza.get("parent_start")
                conflict["stanza_parent_end"] = best_stanza.get("parent_end")
                conflict["stanza_parent_text"] = best_stanza.get("parent_text")
                conflict["stanza_relation"] = best_stanza.get("relation")
                merged.append(conflict)

        elif rule_es:
            merged.append(max(rule_es, key=lambda e: e.get("confidence", 0.0)))

        elif stanza_es:
            merged.append(max(stanza_es, key=lambda e: e.get("confidence", 0.0)))

    return merged


def needs_haiku(
        row: dict,
        edges: list[dict],
        rejected_edges: list[dict],
        source_blocks: list[dict],
        suspicious_segmentation: bool,
) -> bool:
    spans = row.get("spans", [])
    edge_children = {(e["child_start"], e["child_end"]) for e in edges}

    # Conflit rule/stanza.
    if any(e.get("source") == "conflict" for e in edges):
        return True

    # Segmentation/OCR suspecte.
    if suspicious_segmentation:
        return True

    # Source/citation bibliographique : souvent nécessite arbitrage ou au moins contrôle.
    if source_blocks:
        return True

    # Rejets critiques.
    critical_reject_reasons = {
        "verb_or_pron_endpoint",
        "measure_as_nominal_parent",
        "invalid_appos_compatibility",
        "crosses_source_block",
        "child_contains_parent_fragment",
    }
    if any(e.get("rejected_reason") in critical_reject_reasons for e in rejected_edges):
        return True

    # APPOS sans parent.
    for sp in spans:
        if sp.get("svo_role") == "APPOS" or sp.get("role") == "APPOS":
            if (sp["start"], sp["end"]) not in edge_children:
                return True

    # EVENT_NOMINAL/Document communication sans SOURCE résolu.
    for sp in spans:
        txt = sp.get("text", "").lower()
        if label_of(sp) in EVENT_NOMINAL_LABELS or label_of(sp) == "hint_document":
            if txt in COMMUNICATION_EVENT_WORDS:
                has_source_child = any(
                    e.get("parent_start") == sp["start"]
                    and e.get("parent_end") == sp["end"]
                    and e.get("relation") in {"SOURCE", "NMOD"}
                    for e in edges
                )
                if not has_source_child:
                    return True

    # Pronoms intraphrase ambigus.
    prons = [s for s in spans if label_of(s) in {"pron_subj", "pron_obj"}]
    for pron in prons:
        candidates = [
            s for s in spans
            if is_nominal_candidate(s)
               and s["end"] <= pron["start"]
               and (
                       s.get("gender") == pron.get("gender")
                       or s.get("gender") is None
                       or pron.get("gender") is None
               )
               and (
                       s.get("number") == pron.get("number")
                       or s.get("number") is None
                       or pron.get("number") is None
               )
        ]
        if len(candidates) > 1:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Row processing
# ─────────────────────────────────────────────────────────────────────────────

def process_row_with_stanza(row: dict, nlp) -> dict:
    text = row["text"]
    spans = row.get("spans", [])
    rule_edges = row.get("candidate_edges", [])

    source_blocks = detect_source_blocks(text, spans)
    suspicious_segmentation = has_suspicious_segmentation(text)

    stanza_edges: list[dict] = []
    rejected_edges: list[dict] = []

    # Si segmentation très suspecte, ne pas faire confiance à Stanza.
    # On garde les rules existantes et on envoie vers Haiku/review.
    if not suspicious_segmentation:
        doc = nlp(text)
        for sent in doc.sentences:
            valid, rejected = stanza_edges_for_sentence(sent, spans, text, source_blocks)
            stanza_edges.extend(valid)
            rejected_edges.extend(rejected)
    else:
        rejected_edges.append({
            "source": "precheck",
            "rejected_reason": "suspicious_segmentation",
            "text_excerpt": text[:220],
        })

    merged = merge_edges(rule_edges, stanza_edges)

    # Index par child. On n'injecte pas les conflicts dans les spans.
    edge_by_child: dict[tuple[int, int], dict] = {}
    for e in merged:
        key = (e["child_start"], e["child_end"])
        if e.get("source") == "conflict":
            continue

        prev = edge_by_child.get(key)
        if prev is None or float(e.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
            edge_by_child[key] = e

    annotated_spans = []
    for sp in spans:
        sp_copy = dict(sp)
        key = (sp["start"], sp["end"])
        e = edge_by_child.get(key)

        if e is not None:
            sp_copy["nominal_parent_start"] = e["parent_start"]
            sp_copy["nominal_parent_end"] = e["parent_end"]
            sp_copy["nominal_parent_text"] = e["parent_text"]
            sp_copy["nominal_relation"] = e["relation"]
            sp_copy["nominal_parent_confidence"] = e["confidence"]
            sp_copy["nominal_parent_source"] = e["source"]

            if e.get("confirmed_by_stanza"):
                sp_copy["nominal_parent_confirmed_by_stanza"] = True

        annotated_spans.append(sp_copy)

    result = dict(row)
    result["spans"] = annotated_spans
    result["candidate_edges"] = merged
    result["rejected_edges"] = rejected_edges
    result["source_blocks"] = source_blocks
    result["suspicious_segmentation"] = suspicious_segmentation
    result["needs_haiku"] = needs_haiku(
        row=row,
        edges=merged,
        rejected_edges=rejected_edges,
        source_blocks=source_blocks,
        suspicious_segmentation=suspicious_segmentation,
    )

    return result


def snapshot_for_cache(result: dict) -> dict:
    """Retourne une copie persistable de la sortie Stanza pour usage ultérieur."""
    cached = dict(result)
    cached.pop("needs_haiku", None)
    return cached


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stanza nominal parent injection v1.1 conservative")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-output", default=None,
                        help="JSONL cache persistant des sorties Stanza")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip les phrases déjà traitées dans le checkpoint courant")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ckpt_path = Path(args.output + ".ckpt")
    done_ids: set[str] = set()

    if args.skip_existing and ckpt_path.exists():
        done_ids = {l.strip() for l in ckpt_path.read_text().splitlines() if l.strip()}
        print(f"♻️  Reprise : {len(done_ids):,} phrases déjà traitées", flush=True)

    nlp = load_stanza()

    rows_buffer = []
    n_total = 0
    n_haiku = 0
    n_stanza_edges = 0
    n_rejected = 0
    n_source_blocks = 0
    n_suspicious = 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache_output) if args.cache_output else None
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as fout, \
            open(str(ckpt_path), "a", encoding="utf-8") as fckpt, \
            open(str(cache_path), "w", encoding="utf-8") if cache_path is not None else open("/dev/null", "w", encoding="utf-8") as fcache:

        def flush_buffer():
            nonlocal n_haiku, n_stanza_edges, n_rejected, n_source_blocks, n_suspicious

            for row in rows_buffer:
                # Attention : dans cette version, skip-existing évite seulement de retraiter
                # dans une même reprise checkpoint, mais n'écrit pas l'ancien résultat.
                # Usage recommandé : relancer sans écraser l'output partiel, ou gérer append externe.
                if row["id"] in done_ids:
                    continue

                result = process_row_with_stanza(row, nlp)

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                if cache_path is not None:
                    fcache.write(json.dumps(snapshot_for_cache(result), ensure_ascii=False) + "\n")
                fckpt.write(result["id"] + "\n")
                fckpt.flush()

                if result.get("needs_haiku"):
                    n_haiku += 1

                n_stanza_edges += sum(
                    1 for e in result.get("candidate_edges", [])
                    if e.get("source") in {"stanza", "conflict"}
                )
                n_rejected += len(result.get("rejected_edges", []))
                n_source_blocks += len(result.get("source_blocks", []))
                if result.get("suspicious_segmentation"):
                    n_suspicious += 1

        for row in load_jsonl(args.input):
            n_total += 1
            rows_buffer.append(row)

            if len(rows_buffer) >= args.batch_size:
                flush_buffer()
                rows_buffer.clear()
                print(
                    f"  {n_total:,} phrases — "
                    f"{n_haiku:,} → Haiku — "
                    f"{n_stanza_edges:,} arêtes Stanza/conflict — "
                    f"{n_rejected:,} rejetées — "
                    f"{n_source_blocks:,} source_blocks — "
                    f"{n_suspicious:,} segmentation suspecte",
                    flush=True,
                )

        if rows_buffer:
            flush_buffer()

    pct_haiku = n_haiku / max(1, n_total) * 100
    print(f"\n✅ Terminé : {n_total:,} phrases")
    print(f"   Arêtes Stanza/conflict : {n_stanza_edges:,}")
    print(f"   Arêtes rejetées        : {n_rejected:,}")
    print(f"   Source blocks          : {n_source_blocks:,}")
    print(f"   Segmentation suspecte  : {n_suspicious:,}")
    print(f"   Phrases → Haiku        : {n_haiku:,} ({pct_haiku:.1f}%)")
    print(f"   Output                 : {args.output}")

    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"   Checkpoint supprimé    : {ckpt_path}")


if __name__ == "__main__":
    main()