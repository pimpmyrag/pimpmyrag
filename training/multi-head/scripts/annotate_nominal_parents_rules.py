#!/usr/bin/env python3
"""
annotate_nominal_parents_rules.py — v1.1 conservative (v8.22+)
==============================================================

Étape 1 du pipeline nominal_parent_pointer.

Applique uniquement des règles haute précision pour proposer des arêtes
nominales child -> parent.

Important :
- Les règles ne sont que des candidates.
- Chaque arête passe par validate_edge().
- Aucun verb_trigger / pronom ne peut recevoir ou porter nominal_parent.
- Les champs nominal_parent_* existants sont nettoyés puis reconstruits.
- Les cas OCR/citation/source sont marqués needs_haiku / exclude nominal training.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

SKIP_NOMINAL_LABELS = {
    "verb_trigger",
    "pron_subj",
    "pron_obj",
}

NOMINAL_PARENT_FIELDS = {
    "nominal_parent_start",
    "nominal_parent_end",
    "nominal_parent_text",
    "nominal_relation",
    "nominal_parent_confidence",
    "nominal_parent_source",
    "nominal_parent_confirmed_by_stanza",
}

ROLE_LABELS = {
    "hint_person_role",
    "hint_group_role",
    "hint_inst_role",
}

PROPER_LABELS = {
    "hint_person_name",
    "hint_org_name",
    "hint_inst_name",
    "hint_event_named",
    "hint_gpe",
    "hint_object_name",
}

ORG_LABELS = {
    "hint_org_name",
    "hint_inst_name",
}

DOC_LABELS = {
    "hint_document",
    "hint_work_of_art",
    "hint_law",
    "hint_work_generic",
}

WORK_OR_MEDIA_LABELS = {
    "hint_document",
    "hint_work_generic",
    "hint_work_of_art",
    "hint_law",
    "hint_org_name",
    "hint_inst_name",
}

COMM_EVENT_WORDS = {
    "déclarations",
    "déclaration",
    "annonce",
    "annonces",
    "rapport",
    "rapports",
    "communiqué",
    "communiqués",
    "lettre",
    "lettres",
    "note",
    "notes",
    "avis",
    "discours",
    "allocution",
    "conférence",
    "interview",
    "entretien",
    "message",
    "décret",
    "arrêté",
    "propos",
}

COMM_EVENT_LABELS = {
    "hint_event_nominal",
    "hint_document",
}

NMOD_PREPS = {
    "de",
    "du",
    "des",
    "d'",
    "d’",
}

POSS_DETS = {
    "son",
    "sa",
    "ses",
    "leur",
    "leurs",
    "mon",
    "ma",
    "mes",
    "ton",
    "ta",
    "tes",
    "notre",
    "votre",
    "nos",
    "vos",
}

MEASURE_LABELS = {
    "hint_measure",
    "hint_rate",
    "hint_money",
    "hint_value",
}

TIME_LABELS = {
    "hint_time_date",
    "hint_time_duration",
    "hint_time_clock",
}

LOC_LABELS = {
    "hint_fac_name",
    "hint_gpe",
    "hint_loc_generic",
    "hint_infra",
}

SOURCE_MARKERS = {
    "selon",
    "d'après",
    "d’apres",
    "d’après",
    "suivant",
}

OCR_GLUE_RE = re.compile(
    r"(heure|heures|h|mSv/h|Sv/h|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)(Le|La|Les|L'|L’)\b"
)

QUOTE_RE = re.compile(r'["«“].+?["»”]')
CONSULT_RE = re.compile(
    r"\bconsult[ée]?\s+le\s+\d{1,2}/\d{1,2}/\d{2,4}",
    flags=re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers spans
# ─────────────────────────────────────────────────────────────────────────────

def label_of(sp: dict) -> str:
    return sp.get("label") or ""


def span_key(sp: dict) -> tuple[int, int]:
    return sp["start"], sp["end"]


def span_len(sp: dict) -> int:
    return max(1, sp["end"] - sp["start"])


def is_same_span(a: dict, b: dict) -> bool:
    return a["start"] == b["start"] and a["end"] == b["end"]


def overlaps(a: dict, b: dict) -> bool:
    return min(a["end"], b["end"]) - max(a["start"], b["start"]) > 0


def contains(parent: dict, child: dict) -> bool:
    return (
            parent["start"] <= child["start"]
            and parent["end"] >= child["end"]
            and not is_same_span(parent, child)
    )


def gap_between(a: dict, b: dict) -> int:
    if overlaps(a, b):
        return 0
    if a["end"] <= b["start"]:
        return b["start"] - a["end"]
    return a["start"] - b["end"]


def strip_nominal_fields(sp: dict) -> dict:
    clean = dict(sp)
    for k in NOMINAL_PARENT_FIELDS:
        clean.pop(k, None)
    return clean


def is_nominal_span(sp: dict) -> bool:
    if label_of(sp) in SKIP_NOMINAL_LABELS:
        return False
    if sp.get("start") is None or sp.get("end") is None:
        return False
    if sp["end"] <= sp["start"]:
        return False
    return True


def coarse_of(sp: dict) -> str | None:
    if sp.get("coarse"):
        return sp["coarse"]

    label = label_of(sp)

    if label.startswith("hint_person") or label in {"hint_norp", "hint_group_role"}:
        return "PER"
    if label in {"hint_org_name", "hint_inst_name", "hint_inst_role"}:
        return "ORG"
    if label in LOC_LABELS:
        return "LOC"
    if label in TIME_LABELS:
        return "TIME"
    if label in MEASURE_LABELS:
        return "VALUE"
    if label.startswith("hint_event"):
        return "EVENT"
    if label in DOC_LABELS:
        return "WORK"
    if label in {"hint_notion", "hint_field", "hint_state"}:
        return "ABSTRACT"
    if label in {"hint_object_name", "hint_vehicle", "hint_tool", "hint_substance"}:
        return "OBJECT"

    return None


def is_measure(sp: dict) -> bool:
    return label_of(sp) in MEASURE_LABELS or coarse_of(sp) == "VALUE"


def is_time(sp: dict) -> bool:
    return label_of(sp) in TIME_LABELS or coarse_of(sp) == "TIME"


def is_source_like_text(text: str) -> bool:
    low = text.lower()
    return (
            "consulté le" in low
            or "consultée le" in low
            or '"' in text
            or "«" in text
            or "»" in text
    )


# ─────────────────────────────────────────────────────────────────────────────
# Texte / segmentation / blocs source
# ─────────────────────────────────────────────────────────────────────────────

def text_between(s1: dict, s2: dict, text: str) -> str:
    if s1["end"] <= s2["start"]:
        return text[s1["end"]:s2["start"]].strip().lower()
    return ""


def raw_between(s1: dict, s2: dict, text: str) -> str:
    if s1["end"] <= s2["start"]:
        return text[s1["end"]:s2["start"]]
    return ""


def spans_adjacent(s1: dict, s2: dict, text: str, max_gap: int = 4) -> bool:
    if s1["end"] > s2["start"]:
        return False
    between = text[s1["end"]:s2["start"]]
    if len(between) > max_gap:
        return False
    return bool(re.match(r'^[\s\-–—\'«»"“”]*$', between))


def has_suspicious_segmentation(text: str) -> bool:
    return OCR_GLUE_RE.search(text) is not None


def find_suspicious_boundary(text: str) -> int | None:
    m = OCR_GLUE_RE.search(text)
    if not m:
        return None
    return m.start(2)


def edge_crosses_boundary(child: dict, parent: dict, boundary: int | None) -> bool:
    if boundary is None:
        return False

    c = (child["start"] + child["end"]) // 2
    p = (parent["start"] + parent["end"]) // 2
    return (c < boundary <= p) or (p < boundary <= c)


def detect_source_blocks(text: str, spans: list[dict]) -> list[dict]:
    blocks = []

    for quote in QUOTE_RE.finditer(text):
        q_start, q_end = quote.span()

        previous_sources = [
            sp for sp in spans
            if is_nominal_span(sp)
               and sp["end"] <= q_start
               and q_start - sp["end"] <= 90
               and coarse_of(sp) in {"ORG", "WORK"}
        ]

        publication = max(
            previous_sources,
            key=lambda sp: (sp["end"], span_len(sp)),
            default=None,
        )

        block_start = publication["start"] if publication else q_start
        block_end = q_end

        after = text[q_end:q_end + 140]
        cm = CONSULT_RE.search(after)
        if cm:
            block_end = q_end + cm.end()

        blocks.append({
            "start": block_start,
            "end": block_end,
            "quote_start": q_start,
            "quote_end": q_end,
            "publication_text": publication.get("text") if publication else None,
            "publication_start": publication.get("start") if publication else None,
            "publication_end": publication.get("end") if publication else None,
            "source": "heuristic",
        })

    return blocks


def block_index_for_span(sp: dict, blocks: list[dict]) -> int | None:
    center = (sp["start"] + sp["end"]) // 2
    for i, block in enumerate(blocks):
        if block["start"] <= center < block["end"]:
            return i
    return None


def touches_source_block(sp: dict, blocks: list[dict]) -> bool:
    return block_index_for_span(sp, blocks) is not None


def crosses_source_block(child: dict, parent: dict, blocks: list[dict]) -> bool:
    c = block_index_for_span(child, blocks)
    p = block_index_for_span(parent, blocks)
    return c != p and (c is not None or p is not None)


# ─────────────────────────────────────────────────────────────────────────────
# Validation edges
# ─────────────────────────────────────────────────────────────────────────────

def valid_appos(parent: dict, child: dict) -> bool:
    if is_measure(parent) or is_measure(child):
        return False

    p = coarse_of(parent)
    c = coarse_of(child)

    if p is None or c is None:
        return False

    if p == c and p in {"PER", "ORG", "LOC", "WORK", "OBJECT"}:
        return True

    # journal Les Échos : WORK -> ORG/WORK
    if p == "WORK" and c in {"ORG", "WORK"}:
        return True

    # groupe LVMH : ORG generic/name éventuel
    if p == "ORG" and c == "ORG":
        return True

    return False


def valid_amod(parent: dict, child: dict) -> bool:
    if is_measure(parent) or is_measure(child):
        return False

    # Cas propre : adjectif/qualifier inclus dans le span parent.
    if contains(parent, child):
        return True

    # Refuser l’inverse type "Midi Libre" -> "Midi".
    if contains(child, parent):
        return False

    # Sinon seulement très proche.
    return gap_between(parent, child) <= 3


def validate_edge(
        child: dict,
        parent: dict,
        relation: str,
        text: str,
        source_blocks: list[dict],
        suspicious_boundary: int | None,
) -> tuple[bool, str | None]:
    if relation is None:
        return False, "missing_relation"

    if not is_nominal_span(child):
        return False, "child_not_nominal"

    if not is_nominal_span(parent):
        return False, "parent_not_nominal"

    if is_same_span(child, parent):
        return False, "same_span"

    if label_of(child) in SKIP_NOMINAL_LABELS or label_of(parent) in SKIP_NOMINAL_LABELS:
        return False, "verb_or_pron_endpoint"

    if edge_crosses_boundary(child, parent, suspicious_boundary):
        return False, "crosses_suspicious_boundary"

    if crosses_source_block(child, parent, source_blocks):
        return False, "crosses_source_block"

    # Très conservateur : ne pas produire de nominal training dans un bloc citation/source.
    if touches_source_block(child, source_blocks) or touches_source_block(parent, source_blocks):
        return False, "inside_source_block"

    # Mesure comme parent nominal = quasi toujours faux.
    if is_measure(parent) and relation not in {"UNIT", "QUALIFIER", "TIME", "COMPARE_TO"}:
        return False, "measure_as_nominal_parent"

    # Mesure comme enfant NMOD/APPOS/AMOD/COMPOUND = souvent faux pour ce pipeline.
    if is_measure(child) and relation in {"APPOS", "NMOD", "AMOD", "COMPOUND", "POSS"}:
        return False, "measure_as_nominal_child"

    # TIME comme APPOS/NMOD child : trop bruité.
    if is_time(child) and relation in {"APPOS", "NMOD", "AMOD", "COMPOUND"}:
        return False, "time_as_nominal_child"

    # Fragment inverse : enfant qui contient parent.
    if contains(child, parent) and relation in {"APPOS", "NMOD", "AMOD", "COMPOUND"}:
        return False, "child_contains_parent_fragment"

    if relation == "APPOS" and not valid_appos(parent, child):
        return False, "invalid_appos_compatibility"

    if relation == "AMOD" and not valid_amod(parent, child):
        return False, "invalid_amod_shape"

    if relation in {"NMOD", "SOURCE", "POSS", "LOC", "MEDIUM", "MISC"}:
        if gap_between(parent, child) > 120 and not overlaps(parent, child):
            return False, "edge_too_far"

    # Éviter de faire passer des edges à travers ponctuation lourde pour les relations locales.
    if relation in {"APPOS", "NMOD", "AMOD", "COMPOUND"}:
        zone = raw_between(parent, child, text)
        if any(p in zone for p in [".", ";", ":", "\n"]):
            return False, "hard_punctuation_between"

    return True, None


def make_edge(child: dict, parent: dict, relation: str, confidence: float, source: str = "rule") -> dict:
    return {
        "child_start": child["start"],
        "child_end": child["end"],
        "child_text": child["text"],
        "parent_start": parent["start"],
        "parent_end": parent["end"],
        "parent_text": parent["text"],
        "relation": relation,
        "confidence": confidence,
        "source": source,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rules
# ─────────────────────────────────────────────────────────────────────────────

def starts_with_poss(span: dict, text: str, window: int = 8) -> str | None:
    pre = text[max(0, span["start"] - window):span["start"]].strip().lower()
    for det in POSS_DETS:
        if re.search(rf"\b{re.escape(det)}\s*$", pre):
            return det
    return None


def is_communication_event(sp: dict) -> bool:
    return (
            label_of(sp) in COMM_EVENT_LABELS
            and sp.get("text", "").lower() in COMM_EVENT_WORDS
    )


def apply_rules(row: dict) -> tuple[list[dict], list[dict], dict]:
    text = row["text"]
    spans = row.get("spans", [])
    spans_sorted = sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"])))

    source_blocks = detect_source_blocks(text, spans)
    suspicious_segmentation = has_suspicious_segmentation(text)
    suspicious_boundary = find_suspicious_boundary(text)

    edges: list[dict] = []
    rejected_edges: list[dict] = []

    seen_child: set[tuple[int, int]] = set()

    def try_add(child: dict, parent: dict, relation: str, confidence: float, source: str = "rule"):
        e = make_edge(child, parent, relation, confidence, source=source)
        valid, reason = validate_edge(
            child=child,
            parent=parent,
            relation=relation,
            text=text,
            source_blocks=source_blocks,
            suspicious_boundary=suspicious_boundary,
        )

        if not valid:
            bad = dict(e)
            bad["rejected_reason"] = reason
            rejected_edges.append(bad)
            return

        key = (e["child_start"], e["child_end"])
        if key in seen_child:
            # Une seule edge par child dans cette étape rule.
            # Les arbitrages plus complexes iront à Stanza/Haiku.
            return

        seen_child.add(key)
        edges.append(e)

    # ── Règle 1 : APPOS adjacent ROLE + PROPER
    # PDG Bernard Arnault -> Bernard Arnault APPOS PDG
    for i, s1 in enumerate(spans_sorted):
        for j in range(i + 1, len(spans_sorted)):
            s2 = spans_sorted[j]
            if s2["start"] > s1["end"] + 8:
                break

            if label_of(s1) in ROLE_LABELS and label_of(s2) in PROPER_LABELS:
                if spans_adjacent(s1, s2, text, max_gap=5):
                    try_add(s2, s1, "APPOS", 0.95)

            elif label_of(s1) in PROPER_LABELS and label_of(s2) in ROLE_LABELS:
                if spans_adjacent(s1, s2, text, max_gap=5):
                    try_add(s2, s1, "APPOS", 0.88)

    # ── Règle 2 : APPOS adjacent DOC/WORK/ORG generic + proper
    # journal Les Échos, groupe LVMH, société SpaceX
    generic_heads = DOC_LABELS | {"hint_org_generic", "hint_company_generic", "hint_object_generic"}
    proper_children = ORG_LABELS | PROPER_LABELS

    for i, s1 in enumerate(spans_sorted):
        for j in range(i + 1, len(spans_sorted)):
            s2 = spans_sorted[j]
            if s2["start"] > s1["end"] + 8:
                break

            if label_of(s1) in generic_heads and label_of(s2) in proper_children:
                if spans_adjacent(s1, s2, text, max_gap=5):
                    try_add(s2, s1, "APPOS", 0.90)

    # ── Règle 3 : NMOD avec de/du/des/d'
    # salle de turbines du réacteur 3 -> réacteur 3 NMOD salle de turbines
    for i, s1 in enumerate(spans_sorted):
        if not is_nominal_span(s1):
            continue

        for j in range(i + 1, len(spans_sorted)):
            s2 = spans_sorted[j]

            if s2["start"] > s1["end"] + 35:
                break

            if not is_nominal_span(s2):
                continue

            between = text_between(s1, s2, text)
            zone = raw_between(s1, s2, text)

            if not between:
                continue

            if between in NMOD_PREPS or any(between.startswith(p + " ") for p in NMOD_PREPS):
                if "," in zone or ";" in zone or "." in zone or ":" in zone:
                    continue

                relation = "NMOD"

                # Si parent est un event/document de communication : source.
                if is_communication_event(s1) and coarse_of(s2) in {"PER", "ORG", "WORK"}:
                    relation = "SOURCE"

                try_add(s2, s1, relation, 0.78)

    # ── Règle 4 : AMOD / QUALIFIER inclus dans span parent
    # fondations solides -> solides AMOD fondations solides
    for child in spans_sorted:
        if not is_nominal_span(child):
            continue
        if label_of(child) in PROPER_LABELS:
            continue

        for parent in spans_sorted:
            if child is parent:
                continue
            if not is_nominal_span(parent):
                continue

            if contains(parent, child) and child["start"] > parent["start"]:
                gender_ok = (
                        child.get("gender") == parent.get("gender")
                        or child.get("gender") is None
                        or parent.get("gender") is None
                )
                number_ok = (
                        child.get("number") == parent.get("number")
                        or child.get("number") is None
                        or parent.get("number") is None
                )

                if gender_ok and number_ok:
                    try_add(child, parent, "AMOD", 0.88)

    # ── Règle 5 : POSS très conservateur
    # On NE relie pas automatiquement "son PDG" au span le plus proche.
    # On crée POSS seulement si un possesseur ORG/PER clair existe avant, sans event nominal plus proche.
    for child in spans_sorted:
        if not is_nominal_span(child):
            continue

        det = starts_with_poss(child, text)
        if not det:
            continue

        # Seulement les rôles/personnes/institutions.
        if label_of(child) not in ROLE_LABELS and coarse_of(child) not in {"PER", "ORG"}:
            continue

        previous_candidates = [
            s for s in spans_sorted
            if is_nominal_span(s)
               and s["end"] <= child["start"]
               and child["start"] - s["end"] <= 100
               and coarse_of(s) in {"ORG", "PER"}
        ]

        if not previous_candidates:
            continue

        parent = max(previous_candidates, key=lambda s: (s["end"], span_len(s)))
        try_add(child, parent, "POSS", 0.70)

    # ── Règle 6 : SOURCE explicite EVENT_NOMINAL + de + PER/ORG
    for event in spans_sorted:
        if not is_communication_event(event):
            continue

        for child in spans_sorted:
            if child is event:
                continue
            if child["start"] < event["end"]:
                continue
            if child["start"] > event["end"] + 30:
                break
            if not is_nominal_span(child):
                continue

            between = text_between(event, child, text)
            if between in NMOD_PREPS or any(between.startswith(p + " ") for p in NMOD_PREPS):
                if coarse_of(child) in {"PER", "ORG", "WORK"}:
                    try_add(child, event, "SOURCE", 0.85)

    meta = {
        "source_blocks": source_blocks,
        "suspicious_segmentation": suspicious_segmentation,
        "suspicious_boundary": suspicious_boundary,
    }

    return edges, rejected_edges, meta


# ─────────────────────────────────────────────────────────────────────────────
# Injection / needs_haiku
# ─────────────────────────────────────────────────────────────────────────────

def edges_to_span_annotations(spans: list[dict], edges: list[dict]) -> list[dict]:
    edge_by_child: dict[tuple[int, int], dict] = {}

    for e in edges:
        key = (e["child_start"], e["child_end"])
        prev = edge_by_child.get(key)
        if prev is None or e["confidence"] > prev["confidence"]:
            edge_by_child[key] = e

    annotated = []

    for sp in spans:
        sp_copy = strip_nominal_fields(sp)
        key = (sp["start"], sp["end"])
        e = edge_by_child.get(key)

        if e is not None:
            sp_copy["nominal_parent_start"] = e["parent_start"]
            sp_copy["nominal_parent_end"] = e["parent_end"]
            sp_copy["nominal_parent_text"] = e["parent_text"]
            sp_copy["nominal_relation"] = e["relation"]
            sp_copy["nominal_parent_confidence"] = e["confidence"]
            sp_copy["nominal_parent_source"] = e["source"]

        annotated.append(sp_copy)

    return annotated


def needs_haiku(row: dict, edges: list[dict], rejected_edges: list[dict], meta: dict) -> bool:
    spans = row.get("spans", [])
    edge_children = {(e["child_start"], e["child_end"]) for e in edges}

    if meta.get("suspicious_segmentation"):
        return True

    if meta.get("source_blocks"):
        return True

    critical_rejections = {
        "verb_or_pron_endpoint",
        "parent_not_nominal",
        "child_not_nominal",
        "measure_as_nominal_parent",
        "measure_as_nominal_child",
        "invalid_appos_compatibility",
        "inside_source_block",
        "crosses_source_block",
        "crosses_suspicious_boundary",
        "child_contains_parent_fragment",
    }

    if any(e.get("rejected_reason") in critical_rejections for e in rejected_edges):
        return True

    for sp in spans:
        if sp.get("svo_role") == "APPOS" or sp.get("role") == "APPOS":
            if (sp["start"], sp["end"]) not in edge_children:
                return True

    return False


def process_row(row: dict) -> dict:
    edges, rejected_edges, meta = apply_rules(row)
    annotated_spans = edges_to_span_annotations(row.get("spans", []), edges)

    result = dict(row)
    result["spans"] = annotated_spans
    result["candidate_edges"] = edges
    result["rejected_edges"] = rejected_edges
    result["source_blocks"] = meta["source_blocks"]
    result["suspicious_segmentation"] = meta["suspicious_segmentation"]
    result["needs_haiku"] = needs_haiku(row, edges, rejected_edges, meta)

    # Ne pas entraîner nominal_parent_pointer sur ces cas, mais garder pour NER/SVO.
    result["exclude_from_nominal_parent_training"] = bool(
        result["suspicious_segmentation"]
        or result["source_blocks"]
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Annotation nominal parents — rules v1.1 conservative")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    n_total = 0
    n_annotated = 0
    n_edges = 0
    n_rejected = 0
    n_haiku = 0
    n_excluded = 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as fout:
        for row in load_jsonl(args.input):
            n_total += 1

            if args.skip_existing and row.get("candidate_edges"):
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            result = process_row(row)

            edges = result.get("candidate_edges", [])
            rejected = result.get("rejected_edges", [])

            if edges:
                n_annotated += 1
                n_edges += len(edges)

            n_rejected += len(rejected)

            if result.get("needs_haiku"):
                n_haiku += 1

            if result.get("exclude_from_nominal_parent_training"):
                n_excluded += 1

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            if n_total % 1000 == 0:
                print(
                    f"  {n_total:,} phrases — "
                    f"{n_annotated:,} annotées — "
                    f"{n_edges:,} edges — "
                    f"{n_rejected:,} rejetées — "
                    f"{n_haiku:,} Haiku — "
                    f"{n_excluded:,} excluded_nominal",
                    flush=True,
                )

    avg_edges = n_edges / max(1, n_annotated)

    print(f"\n✅ Terminé : {n_total:,} phrases")
    print(f"   Phrases avec arêtes rule-based : {n_annotated:,} ({n_annotated / max(1, n_total) * 100:.1f}%)")
    print(f"   Total arêtes candidates        : {n_edges:,} (avg {avg_edges:.1f}/phrase annotée)")
    print(f"   Arêtes rejetées                : {n_rejected:,}")
    print(f"   Phrases → Haiku                : {n_haiku:,} ({n_haiku / max(1, n_total) * 100:.1f}%)")
    print(f"   Exclues nominal training       : {n_excluded:,} ({n_excluded / max(1, n_total) * 100:.1f}%)")
    print(f"   Output                         : {args.output}")


if __name__ == "__main__":
    main()