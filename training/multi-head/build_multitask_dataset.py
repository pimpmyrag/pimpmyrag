# build_multitask_dataset.py
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set

from transformers import AutoTokenizer

from labels import (
    FINE2ID, FINE_NONE_ID,
    fine_label_to_coarse_id,
    COARSE_NONE_ID,
    SYN2ID, SYN_NONE_ID, ALL_SYN_LABELS,
    ROLE2ID, ROLE_NONE_ID,
    ROLE_FINE_TO_COARSE_ID, ROLE_COARSE_NONE_ID, ROLE_COARSE_OTHER_ID,
    NER_TIME_LABELS, NER_LOC_LABELS,
    VOICE2ID, VOICE_NONE_ID,
    CERTAINTY2ID, CERTAINTY_NONE_ID,
    GENDER2ID, GENDER_NONE_ID,
    NUMBER2ID, NUMBER_NONE_ID,
    PERSON2ID, PERSON_NONE_ID,
    # VerbFam
    VERB_FAMILY2ID, VERB_FAMILY_NONE_ID,
    VERB_FAMILY_FINE2ID, VERB_FAMILY_FINE_NONE_ID,
    VERB_POLARITY2ID, VERB_POLARITY_NONE_ID,
    VERB_ASPECT2ID, VERB_ASPECT_NONE_ID,
    VERB_SOURCE2ID, VERB_SOURCE_NONE_ID,
    # Nominal parent pointer (v8.22)
    NOMINAL_RELATION2ID, NOMINAL_RELATION_NONE_ID,
    # Semantic role (v8.22+)
    SEMANTIC_ROLE2ID, SEMANTIC_ROLE_NONE_ID, SEMANTIC_ROLE_SKIP_ID,
    # compat aliases
    SVO2ID, SVO_NONE_ID, ALL_SVO_LABELS,
)
# Attributs transverses v9 (dérivés du fine label — 0 annotation)
from labels_v9 import (
    derive_attributes, to_v9_fine,
    ANIMACY_NONE_ID, LIVING_NONE_ID, ABSTRACT_NONE_ID, DYNAMICITY_NONE_ID, WORK_NONE_ID,
)

# ─── Mapper sémantique f(svo_role, hint, verb_family, voice) → semantic_role_id ──
# Dérivé de semantic_role_mapper_v1.py Phase 2.
_NER_TIME       = {"hint_time_date","hint_time_clock","hint_time_duration"}
_NER_LOC        = {"hint_gpe","hint_fac_name","hint_loc_generic","hint_infra"}
_NER_MEASURE    = {"hint_measure","hint_percentage","hint_count","hint_money","hint_rate"}
_NER_DOMAIN     = {"hint_field","hint_doctrine","hint_notion","hint_language"}
_NER_WORK       = {"hint_work_of_art","hint_law","hint_document","hint_work_generic"}
_NER_EVENT      = {"hint_event_nominal","hint_event_named","hint_state"}
_NER_INSTRUMENT = {"hint_tool","hint_weapon","hint_vehicle"}
_NER_MATERIAL   = {"hint_substance","hint_food"}
_NER_OBJECT_GEN = {"hint_object_generic","hint_object_name"}
_NER_BIO        = {"hint_animal","hint_vegetal","hint_disease"}
_NER_ORG        = {"hint_org_name","hint_inst_name","hint_inst_role"}
_NER_HUMAN      = {"hint_person_name","hint_person_role","hint_norp","hint_group_role",
                   "hint_inst_name","hint_inst_role","hint_org_name"}
_CC = {"Communication","Cognition"}
_CA = {"Causality","State_Change","Conflict"}
# Extension causale pour SUBJECT : au-delà des event_nominal/named/state, une loi,
# une infrastructure ou une maladie peut aussi être la CAUSE grammaticale d'un
# verbe causatif ("la loi a mis fin à...", "l'échangeur a provoqué...",
# "l'ostéoporose fragilise..."). Sans cette extension, ces sujets inanimés
# retombaient par défaut sur AGENT malgré un verb_family clairement causal.
_NER_CAUSAL_SUBJ = _NER_EVENT | {"hint_law", "hint_infra", "hint_disease", "hint_doctrine"}

def _map_semantic_role_id(svo_role: str | None, hint: str | None,
                          verb_family: str | None, voice: str | None) -> int:
    """Retourne le SEMANTIC_ROLE label_id pour un span NER."""
    if not svo_role or svo_role == "NONE":
        return SEMANTIC_ROLE_NONE_ID
    h, vf, vc = hint or "", verb_family or "", voice or "active"
    TYPED = {
        "OBLIQUE_AGENT": "AGENT", "OBLIQUE_CAUSE": "CAUSE",
        "OBLIQUE_ADVERSARY": "ADVERSARY", "OBLIQUE_BENEFICIARY": "BENEFICIARY",
        "OBLIQUE_COMITATIVE": "COMITATIVE", "OBLIQUE_DOMAIN": "DOMAIN",
        "OBLIQUE_SOURCE": "SOURCE",
    }
    if svo_role in TYPED:
        return SEMANTIC_ROLE2ID[TYPED[svo_role]]
    if svo_role == "APPOS":
        return SEMANTIC_ROLE2ID["IDENTITY"]
    if svo_role == "SUBJECT":
        if vc == "passive": return SEMANTIC_ROLE2ID["PATIENT"]
        if h in _NER_TIME: return SEMANTIC_ROLE2ID["TEMPORAL"]
        if h in _NER_CAUSAL_SUBJ and vf in _CA: return SEMANTIC_ROLE2ID["CAUSE"]
        return SEMANTIC_ROLE2ID["AGENT"]
    if svo_role == "OBJECT":
        if h in _NER_TIME:    return SEMANTIC_ROLE2ID["TEMPORAL"]
        if h in _NER_MEASURE: return SEMANTIC_ROLE2ID["MEASURE"]
        if h in _NER_LOC:     return SEMANTIC_ROLE2ID["LOCATION"]
        if vf in _CC:
            if h in _NER_EVENT | _NER_WORK | _NER_DOMAIN: return SEMANTIC_ROLE2ID["CONTENT"]
            if h in _NER_HUMAN: return SEMANTIC_ROLE2ID["PATIENT"]
            return SEMANTIC_ROLE2ID["CONTENT"]
        return SEMANTIC_ROLE2ID["PATIENT"]
    if svo_role == "OBLIQUE":
        if h in _NER_TIME:    return SEMANTIC_ROLE2ID["TEMPORAL"]
        if h in _NER_LOC:     return SEMANTIC_ROLE2ID["LOCATION"]
        if h in _NER_MEASURE: return SEMANTIC_ROLE2ID["MEASURE"]
        if h in _NER_DOMAIN:  return SEMANTIC_ROLE2ID["DOMAIN"]
        if h in _NER_WORK:    return SEMANTIC_ROLE2ID["CONTENT"] if vf in _CC else SEMANTIC_ROLE2ID["SOURCE"]
        if h in _NER_EVENT:
            if vf == "Temporal": return SEMANTIC_ROLE2ID["TEMPORAL"]
            if vf in _CA | {"Conflict"}: return SEMANTIC_ROLE2ID["CAUSE"]
            if vf in _CC: return SEMANTIC_ROLE2ID["CONTENT"]
            return SEMANTIC_ROLE2ID["CAUSE"]
        if h in _NER_INSTRUMENT:
            if vf in {"Possession"}: return SEMANTIC_ROLE2ID["PATIENT"]
            if vf in _CC: return SEMANTIC_ROLE2ID["DOMAIN"]
            return SEMANTIC_ROLE2ID["INSTRUMENT"]
        if h in _NER_MATERIAL:
            return SEMANTIC_ROLE2ID["DOMAIN"] if vf in _CC else SEMANTIC_ROLE2ID["PATIENT"]
        if h in _NER_OBJECT_GEN:
            return SEMANTIC_ROLE2ID["DOMAIN"] if vf in _CC else SEMANTIC_ROLE2ID["PATIENT"]
        if h in _NER_BIO:
            return SEMANTIC_ROLE2ID["DOMAIN"] if vf in _CC else SEMANTIC_ROLE2ID["PATIENT"]
        if h == "hint_norp":
            return SEMANTIC_ROLE2ID["DOMAIN"] if vf in {"Relation","OTHER"} else \
                   SEMANTIC_ROLE2ID["BENEFICIARY"] if vf == "Social" else SEMANTIC_ROLE2ID["PATIENT"]
        if h in _NER_ORG:
            if vf == "Relation": return SEMANTIC_ROLE2ID["PART_OF"]
            if vf == "Social":   return SEMANTIC_ROLE2ID["BENEFICIARY"]
            if h == "hint_inst_name" and vf in _CA | {"State_Change"}: return SEMANTIC_ROLE2ID["LOCATION"]
            if vf in _CC: return SEMANTIC_ROLE2ID["BENEFICIARY"]
            return SEMANTIC_ROLE2ID["PATIENT"]
        if h in _NER_HUMAN:
            if vf == "Social":     return SEMANTIC_ROLE2ID["BENEFICIARY"]
            if vf == "Possession": return SEMANTIC_ROLE2ID["OWNER"]
            if vf in _CC:          return SEMANTIC_ROLE2ID["BENEFICIARY"]
            if vf == "Relation":   return SEMANTIC_ROLE2ID["PART_OF"]
            if vf in _CA | {"Perception","State_Change","Conflict","OTHER"}:
                return SEMANTIC_ROLE2ID["PATIENT"]
            return SEMANTIC_ROLE_SKIP_ID   # OBLIQUE_UNRESOLVED → non supervisé
        return SEMANTIC_ROLE_SKIP_ID       # tout le reste → non supervisé
    return SEMANTIC_ROLE_NONE_ID

# ─── Fallback nominal : spans sans gouverneur verbal, régis par une relation ──
# nominale (nominal_parent_start / nominal_relation, v8.22). Exemple : dans
# "le budget de la santé", "santé" n'a pas de svo_role/gov_verb (pas de verbe),
# seulement une relation NMOD vers "budget" → sans ce fallback il retombait
# systématiquement à SEMANTIC_ROLE_NONE_ID au lieu d'un rôle exploitable.
_NOMINAL_RELATION_TO_SEMANTIC = {
    "APPOS":  "IDENTITY",
    "POSS":   "OWNER",
    "SOURCE": "SOURCE",
    "MEDIUM": "SOURCE",
    "LOC":    "LOCATION",
    "TIME":   "TEMPORAL",
}

def _map_semantic_role_from_nominal(nominal_relation: str | None, hint: str | None,
                                     parent_label: str | None = None,
                                     parent_end: int | None = None,
                                     child_end: int | None = None) -> int:
    """Rôle sémantique dérivé de la relation nominale parent→enfant (pas de verbe).
    Ne s'applique qu'en fallback, quand le mapper SVO principal n'a rien trouvé
    (span sans gov_verb_start, donc pas d'AGENT/PATIENT/etc. dérivable d'un verbe).
    """
    if not nominal_relation:
        return SEMANTIC_ROLE_NONE_ID
    if nominal_relation in _NOMINAL_RELATION_TO_SEMANTIC:
        return SEMANTIC_ROLE2ID[_NOMINAL_RELATION_TO_SEMANTIC[nominal_relation]]
    if nominal_relation == "NMOD":
        h = hint or ""
        if h in _NER_TIME:   return SEMANTIC_ROLE2ID["TEMPORAL"]
        if h in _NER_LOC:    return SEMANTIC_ROLE2ID["LOCATION"]
        if h in _NER_DOMAIN: return SEMANTIC_ROLE2ID["DOMAIN"]
        if h in _NER_ORG:
            # PART_OF suppose deux entités DISTINCTES (ex: "la filiale de Google").
            # Mais si parent/enfant partagent le même label, ou si l'enfant se
            # termine exactement là où finit le parent (suffixe d'un nom composé,
            # ex: "ministère des Transports" / "Transports"), il s'agit très
            # probablement de la MÊME entité redécoupée en granularités
            # différentes → DOMAIN (son domaine de compétence) plutôt que PART_OF.
            same_label = parent_label is not None and parent_label == h
            same_end   = (
                parent_end is not None and child_end is not None
                and parent_end == child_end
            )
            if same_label or same_end:
                return SEMANTIC_ROLE2ID["DOMAIN"]
            return SEMANTIC_ROLE2ID["PART_OF"]
        return SEMANTIC_ROLE2ID["DOMAIN"]   # complément du nom générique
    # AMOD (adjectif), COMPOUND (même entité éclatée), MISC : pas de rôle
    # sémantique propre à superviser ici → laisse SKIP (non supervisé).
    return SEMANTIC_ROLE_SKIP_ID

# ─── Override par indices de surface : PURPOSE / MEMBER_OF / OWNER ────────────
# Le mapper principal ne voit ni la préposition ni le connecteur qui précède le
# span. Or ces trois rôles sont quasi exclusivement portés par un marqueur lexical
# de surface ("afin de", "membre de", "propriété de"...). Sans cette couche ils
# restaient sous-dérivés (OWNER ~0.1% du gold, PURPOSE/MEMBER_OF = 0). On inspecte
# les ~40 caractères précédant le span, gaté par le type NER de la cible.
_PURPOSE_MARKERS = (
    "afin de ", "afin d'", "dans le but de ", "dans le but d'",
    "en vue de ", "en vue d'", "dans l'objectif de ", "dans l'optique de ",
    "destiné à ", "destinée à ", "destinés à ", "destinées à ",
    "visant à ", "de manière à ", "pour objectif de ",
    "à des fins de ", "à des fins d'", "en prévision de ", "en prévision d'",
    "en préparation de ", "en préparation d'", "dans une optique de ",
    "dans une perspective de ", "en quête de ", "en quête d'",
    "à la recherche de ", "à la recherche d'",
)
_MEMBER_MARKERS = (
    "membre de ", "membres de ", "membre du ", "membres du ",
    "membre des ", "membres des ", "fait partie de ", "font partie de ",
    "faisait partie de ", "faisaient partie de ", "au sein de ", "au sein du ",
    "au sein des ", "adhère à ", "adhèrent à ", "adhérent de ", "affilié à ",
    "affiliée à ", "affiliés à ", "rattaché à ", "rattachée à ",
)
_OWNER_MARKERS = (
    "propriété de ", "propriété du ", "propriété des ",
    "détenu par ", "détenue par ", "détenus par ", "détenues par ",
    "possédé par ", "possédée par ", "possédés par ",
    "appartient à ", "appartiennent à ", "appartenant à ",
    "aux mains de ", "aux mains du ", "aux mains des ", "dans les mains de ",
)
# Cibles PURPOSE : buts abstraits (jamais une personne/org → celles-ci = BENEFICIARY).
_NER_PURPOSE_TARGETS = {
    "hint_event_nominal", "hint_notion", "hint_state", "hint_field", "hint_doctrine",
}
# Cibles MEMBER_OF : l'ensemble d'appartenance est une org/collectif/lieu.
_NER_MEMBER_TARGETS = _NER_ORG | {"hint_gpe", "hint_group_role", "hint_norp"}

# Déterminants possibles entre le marqueur et le span (adjacence tolérante à l'article).
_DETS = ("", "l'", "le ", "la ", "les ", "un ", "une ", "d'",
         "son ", "sa ", "ses ", "leur ", "leurs ", "ce ", "cet ", "cette ", "ces ")


def _ends_with_marker(prefix: str, markers: tuple[str, ...]) -> bool:
    """True si le préfixe se termine par un marqueur suivi d'un déterminant optionnel,
    c.-à-d. que le span suit IMMÉDIATEMENT le marqueur (évite qu'un marqueur lointain
    gouvernant un autre nom ne contamine le span courant)."""
    for m in markers:
        for d in _DETS:
            if prefix.endswith(m + d):
                return True
    return False


def _surface_override_role(sentence_text: str | None, span: dict) -> int | None:
    """Détecte PURPOSE / MEMBER_OF / OWNER via le préfixe lexical précédant le span.
    Retourne un semantic_role_id, ou None si aucun marqueur fort ne s'applique.
    Priorité forte : quand un marqueur match, il prime sur le mapper de base.

    - MEMBER_OF / OWNER : le marqueur gouverne le nom qui suit → matching par
      ADJACENCE (span juste après le marqueur, modulo un déterminant).
    - PURPOSE : idem adjacence → ne capte que les buts NOMINAUX directs
      ("en vue de la réforme"), pas l'objet d'un infinitif ("afin de RÉDUIRE X",
      où X garde son rôle local patient/adversaire).
    """
    if not sentence_text:
        return None
    start = span.get("start")
    if start is None:
        return None
    hint = span.get("label") or ""
    prefix = sentence_text[max(0, start - 40):start].lower().replace("\u2019", "'")

    if hint in _NER_MEMBER_TARGETS and _ends_with_marker(prefix, _MEMBER_MARKERS):
        return SEMANTIC_ROLE2ID["MEMBER_OF"]
    if hint in _NER_HUMAN and _ends_with_marker(prefix, _OWNER_MARKERS):
        return SEMANTIC_ROLE2ID["OWNER"]
    if hint in _NER_PURPOSE_TARGETS and _ends_with_marker(prefix, _PURPOSE_MARKERS):
        return SEMANTIC_ROLE2ID["PURPOSE"]
    return None

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

def tokenize_with_offsets(tokenizer, text: str):
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    return enc["input_ids"], enc["offset_mapping"]


def _canon_label(s: str) -> str:
    """Normalise un label pour matching robuste (casse, tirets, espaces, préfixes)."""
    s = (s or "").strip()
    if not s:
        return ""
    s = s.strip("[]")
    s = re.sub(r"^verb[_\-\s]*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-zA-Z0-9]+", "", s)
    return s.lower()


def _build_canon_map(d: dict[str, int]) -> dict[str, int]:
    return {_canon_label(k): v for k, v in d.items()}


_VFAM_CANON_MAP = _build_canon_map(VERB_FAMILY2ID)
_VFINE_CANON_MAP = _build_canon_map(VERB_FAMILY_FINE2ID)
_VPOL_CANON_MAP = _build_canon_map(VERB_POLARITY2ID)
_VASP_CANON_MAP = _build_canon_map(VERB_ASPECT2ID)
_VSRC_CANON_MAP = _build_canon_map(VERB_SOURCE2ID)
_NOMINAL_RELATION_CANON_MAP = _build_canon_map(NOMINAL_RELATION2ID)


def _parse_verb_labels(sp: dict) -> tuple[int, int, int, int, int]:
    """Parse les labels verbaux en supportant les formats legacy + actuels.

    Legacy support:
    - verb_family = "Communication_Demande" (coarse+fine combinés)
    - verb_family = "Verb_Communication_Annonce"
    """
    vfam_raw = sp.get("verb_family")
    vfine_raw = sp.get("verb_family_fine")
    vpol_raw = sp.get("verb_polarity")
    vasp_raw = sp.get("verb_aspect")
    vsrc_raw = sp.get("verb_source")

    vfam_id = _VFAM_CANON_MAP.get(_canon_label(vfam_raw), VERB_FAMILY_NONE_ID)
    vfine_id = _VFINE_CANON_MAP.get(_canon_label(vfine_raw), VERB_FAMILY_FINE_NONE_ID)

    # Compat legacy: champ unique "Family_Fine" dans verb_family.
    if vfam_id == VERB_FAMILY_NONE_ID and isinstance(vfam_raw, str) and vfam_raw.strip():
        parts = [p for p in re.split(r"[_\-/\s]+", vfam_raw.strip("[]")) if p]
        if parts and parts[0].lower() == "verb":
            parts = parts[1:]
        if len(parts) >= 2:
            fam_part = parts[0]
            fine_part = "".join(parts[1:])
            vfam_id = _VFAM_CANON_MAP.get(_canon_label(fam_part), vfam_id)
            if vfine_id == VERB_FAMILY_FINE_NONE_ID:
                vfine_id = _VFINE_CANON_MAP.get(_canon_label(fine_part), vfine_id)

    vpol_id = _VPOL_CANON_MAP.get(_canon_label(vpol_raw), VERB_POLARITY_NONE_ID)
    vasp_id = _VASP_CANON_MAP.get(_canon_label(vasp_raw), VERB_ASPECT_NONE_ID)
    vsrc_id = _VSRC_CANON_MAP.get(_canon_label(vsrc_raw), VERB_SOURCE_NONE_ID)
    return vfam_id, vfine_id, vpol_id, vasp_id, vsrc_id

def char_span_to_token_span(offsets: List[Tuple[int, int]], start: int, end: int):
    """
    Convertit un span char -> span token couvrant la zone [start, end)
    même si les frontières char ne tombent pas exactement sur des offsets token.
    """
    tok_start = None
    tok_end = None

    for i, (s, e) in enumerate(offsets):
        if e <= start:
            continue
        if s >= end:
            break
        if tok_start is None:
            tok_start = i
        tok_end = i

    if tok_start is None or tok_end is None:
        return None
    return tok_start, tok_end

def token_span_to_char_span(offsets: List[Tuple[int, int]], tok_start: int, tok_end: int):
    return offsets[tok_start][0], offsets[tok_end][1]

def spans_overlap(a_start, a_end, b_start, b_end):
    return not (a_end <= b_start or b_end <= a_start)

def token_span_iou(a, b):
    a0, a1 = a
    b0, b1 = b
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return inter / union if union > 0 else 0.0

def build_gold_candidates(row, tokenizer):
    text = row["text"]
    input_ids, offsets = tokenize_with_offsets(tokenizer, text)

    gold_candidates = []
    gold_token_spans = []
    gold_char_spans = set()

    # Index verb_trigger.start -> voice — permet de résoudre gov_verb_voice pour
    # le mapper sémantique (cf. _map_semantic_role_id, règle SUBJECT+passive→PATIENT).
    # Sans cet index, gov_verb_voice restait toujours None et cette règle ne se
    # déclenchait jamais (bug constaté empiriquement sur des sujets passifs mappés AGENT).
    verb_voice_by_start = {
        s["start"]: s.get("voice")
        for s in row.get("spans", [])
        if s.get("label") == "verb_trigger"
    }

    # Index start -> span, pour résoudre le label/end du parent nominal
    # (cf. _map_semantic_role_from_nominal, distinction PART_OF vs DOMAIN).
    span_by_start: dict[int, dict] = {}
    for s in row.get("spans", []):
        span_by_start.setdefault(s["start"], s)

    for sp in row.get("spans", []):
        _orig_label = sp["label"]
        # v9 : fusionne object_name→object_generic, rate→measure,
        # work_generic→work_of_art, state→event_nominal, doctrine→notion,
        # vegetal→animal. Les attributs sont dérivés du label d'ORIGINE.
        label = to_v9_fine(_orig_label)
        start = sp["start"]
        end   = sp["end"]

        tok_span = char_span_to_token_span(offsets, start, end)
        if tok_span is None:
            continue
        tok_start, tok_end = tok_span

        # ── Spans syntaxiques v4 : verb_trigger, pron_subj, pron_obj ────────
        if label in ALL_SYN_LABELS:
            syn_id = SYN2ID[label]

            # Voice + certainty sur verb_trigger
            voice_id     = VOICE_NONE_ID
            certainty_id = CERTAINTY_NONE_ID
            # VerbFam (verb_trigger uniquement)
            vfam_id      = VERB_FAMILY_NONE_ID
            vfam_fine_id = VERB_FAMILY_FINE_NONE_ID
            vpol_id      = VERB_POLARITY_NONE_ID
            vasp_id      = VERB_ASPECT_NONE_ID
            vsrc_id      = VERB_SOURCE_NONE_ID
            if label == "verb_trigger":
                voice_str = sp.get("voice", "")
                voice_id  = VOICE2ID.get(voice_str, VOICE_NONE_ID)
                cert_str  = sp.get("certainty", "")
                certainty_id = CERTAINTY2ID.get(cert_str, CERTAINTY_NONE_ID)
                vfam_id, vfam_fine_id, vpol_id, vasp_id, vsrc_id = _parse_verb_labels(sp)

            # Rôle SVO du pronom + gov_verb_tok_start
            role_id          = ROLE2ID.get(sp.get("svo_role", "NONE"), ROLE_NONE_ID)
            role_coarse_id   = ROLE_FINE_TO_COARSE_ID.get(role_id, ROLE_COARSE_NONE_ID)
            gov_verb_tok_start = -1
            gvs = sp.get("gov_verb_start")
            if gvs is not None:
                v_tok = char_span_to_token_span(offsets, gvs, gvs + 1)
                if v_tok is not None:
                    gov_verb_tok_start = v_tok[0]

            # Conversion du champ person (peut être int ou str dans différentes versions du dataset)
            person_raw = sp.get("person")
            person_id = PERSON2ID.get(str(person_raw) if person_raw is not None else "", PERSON_NONE_ID)

            # Nominal parent pointer — syn spans n'ont pas de parent nominal
            syn_nominal_parent_tok_start = -1
            syn_nominal_relation_label_id = NOMINAL_RELATION_NONE_ID

            cand = {
                "char_start":          start,
                "char_end":            end,
                "tok_start":           tok_start,
                "tok_end":             tok_end,
                "boundary_label":      0,            # pas un span NER
                "svo_boundary_label":  1,            # span syntaxique
                "coarse_label_id":     COARSE_NONE_ID,
                "fine_label_id":       FINE_NONE_ID,
                "syn_label_id":        syn_id,
                "role_label_id":       role_id,
                "role_coarse_label_id": role_coarse_id,
                "semantic_role_label_id": SEMANTIC_ROLE_SKIP_ID,  # syn spans = non-oblique
                "voice_label_id":      voice_id,
                "certainty_label_id":  certainty_id,
                "gender_label_id":     GENDER2ID.get(sp.get("gender"), GENDER_NONE_ID),
                "number_label_id":     NUMBER2ID.get(sp.get("number"), NUMBER_NONE_ID),
                "person_label_id":     person_id,
                "gov_verb_tok_start":  gov_verb_tok_start,
                "mod_of_tok_start":    -1,
                # VerbFam
                "verb_family_label_id":      vfam_id,
                "verb_family_fine_label_id": vfam_fine_id,
                "verb_polarity_label_id":    vpol_id,
                "verb_aspect_label_id":      vasp_id,
                "verb_source_label_id":      vsrc_id,
                # Nominal parent pointer (v8.22) — non supervisé pour les spans syntaxiques
                "nominal_parent_tok_start":  syn_nominal_parent_tok_start,
                "nominal_relation_label_id": syn_nominal_relation_label_id,
                "neg_type":            "syn_gold",
                "sample_weight":       1.0,
                "text":                sp.get("text", text[start:end]),
            }
            gold_candidates.append(cand)
            gold_token_spans.append((tok_start, tok_end))
            gold_char_spans.add((start, end))
            continue

        # ── Spans NER classiques (hint_*) — portent maintenant svo_role ─────
        if label not in FINE2ID:
            # Ignorer les labels inconnus (anciens labels Stanza si mélange de données)
            continue

        fine_id   = FINE2ID[label]
        coarse_id = fine_label_to_coarse_id(label)

        # Rôle SVO annoté sur ce span NER
        role_id = ROLE2ID.get(sp.get("svo_role", "NONE"), ROLE_NONE_ID)
        # NER gold sans svo_role → OTHER (entité hors argument annoté, pas sentinel)
        # ROLE_FINE_TO_COARSE_ID mappe NONE→OTHER pour les spans NER gold
        role_coarse_id = ROLE_FINE_TO_COARSE_ID.get(role_id, ROLE_COARSE_NONE_ID)

        # Rôle sémantique fin (Phase 2 mapper, remplace role_oblique)
        svo_role_str = sp.get("svo_role", "NONE") or "NONE"
        # Pour dériver le semantic_role, on a besoin du verb_family du verbe gouverneur.
        # Il sera disponible si le dataset contient des verb_trigger annotés avec verb_family.
        # Sinon → SEMANTIC_ROLE_SKIP_ID (non supervisé, résolu au preprocessing)
        gov_verb_family = sp.get("gov_verb_family")   # champ ajouté par build_v822_semrole.py
        gvs_lookup = sp.get("gov_verb_start")
        # Résolu via l'index verb_voice_by_start construit en tête de fonction
        # (avant ce fix : toujours None → règle SUBJECT+passive→PATIENT jamais déclenchée)
        gov_verb_voice = verb_voice_by_start.get(gvs_lookup) if gvs_lookup is not None else None
        semantic_role_id = _map_semantic_role_id(svo_role_str, label, gov_verb_family, gov_verb_voice)
        # Fallback/override nominal : un span emboîté dans un span parent nominal
        # annoté (nominal_parent_start/nominal_relation, v8.22) hérite souvent d'un
        # svo_role de son parent (SUBJECT/OBJECT...) qui ne reflète PAS son propre
        # rôle sémantique — ex: "santé" dans "le budget de la santé" (NMOD de
        # "budget", pas de verbe propre). La relation nominale PRIME donc sur le
        # rôle verbal hérité dès qu'elle est annotée (cohérent avec
        # annotate_nominal_parents.py, qui applique la même priorité en amont).
        nom_rel_str = sp.get("nominal_relation")
        if nom_rel_str:
            nps_lookup = sp.get("nominal_parent_start")
            parent_sp = span_by_start.get(nps_lookup) if nps_lookup is not None else None
            semantic_role_id = _map_semantic_role_from_nominal(
                nom_rel_str, label,
                parent_label=parent_sp.get("label") if parent_sp else None,
                parent_end=parent_sp.get("end") if parent_sp else None,
                child_end=sp.get("end"),
            )
        # Override par indices de surface (PURPOSE / MEMBER_OF / OWNER) : prime sur
        # le mapper de base/nominal quand un marqueur lexical fort précède le span.
        _surf = _surface_override_role(text, sp)
        if _surf is not None:
            semantic_role_id = _surf
        # Si le dataset porte déjà semantic_role (v8.22+), l'utiliser directement
        if "semantic_role" in sp:
            sr_str = sp["semantic_role"]
            if sr_str == "OBLIQUE_UNRESOLVED":
                semantic_role_id = SEMANTIC_ROLE_SKIP_ID
            elif sr_str in SEMANTIC_ROLE2ID:
                semantic_role_id = SEMANTIC_ROLE2ID[sr_str]
            else:
                semantic_role_id = SEMANTIC_ROLE_NONE_ID

        # Pointeur vers le verbe gouverneur
        gov_verb_tok_start = -1
        gvs = sp.get("gov_verb_start")
        if gvs is not None:
            v_tok = char_span_to_token_span(offsets, gvs, gvs + 1)
            if v_tok is not None:
                gov_verb_tok_start = v_tok[0]

        # Pointeur vers le span dont ce span est modificateur nominal
        mod_of_tok_start = -1
        mos = sp.get("mod_of_start")
        if mos is not None:
            m_tok = char_span_to_token_span(offsets, mos, mos + 1)
            if m_tok is not None:
                mod_of_tok_start = m_tok[0]

        # ── Nominal parent pointer (v8.22) ────────────────────────────────
        # nominal_parent_start : char offset du span parent nominal annoté
        # nominal_relation     : relation APPOS/NMOD/POSS/AMOD/COMPOUND/SOURCE/MEDIUM/LOC/TIME/MISC
        # -1 = NO_PARENT (racine ou non supervisé)
        nominal_parent_tok_start = -1
        nps = sp.get("nominal_parent_start")
        if nps is not None:
            np_tok = char_span_to_token_span(offsets, nps, nps + 1)
            if np_tok is not None:
                nominal_parent_tok_start = np_tok[0]

        nominal_relation_str = sp.get("nominal_relation", "")
        nominal_relation_label_id = NOMINAL_RELATION2ID.get(
            nominal_relation_str, NOMINAL_RELATION_NONE_ID
        )
        # Si pas de parent annoté → sentinel (non supervisé pour le pointer)
        if nominal_parent_tok_start < 0:
            nominal_relation_label_id = NOMINAL_RELATION_NONE_ID

        # Attributs transverses v9 dérivés du fine label d'ORIGINE (span NER positif)
        _attrs_v9 = derive_attributes(_orig_label)

        cand = {
            "char_start":          start,
            "char_end":            end,
            "tok_start":           tok_start,
            "tok_end":             tok_end,
            "boundary_label":      1,            # span NER
            "svo_boundary_label":  0,
            "coarse_label_id":     coarse_id,
            "fine_label_id":       fine_id,
            "syn_label_id":        SYN_NONE_ID,
            "role_label_id":       role_id,
            "role_coarse_label_id": role_coarse_id,
            "semantic_role_label_id": semantic_role_id,
            "voice_label_id":      VOICE_NONE_ID,
            "certainty_label_id":  CERTAINTY_NONE_ID,
            "gender_label_id":     GENDER2ID.get(sp.get("gender"), GENDER_NONE_ID),
            "number_label_id":     NUMBER2ID.get(sp.get("number"), NUMBER_NONE_ID),
            "person_label_id":     PERSON_NONE_ID,
            "gov_verb_tok_start":  gov_verb_tok_start,
            "mod_of_tok_start":    mod_of_tok_start,
            # Nominal parent pointer (v8.22)
            "nominal_parent_tok_start":  nominal_parent_tok_start,
            "nominal_relation_label_id": nominal_relation_label_id,
            # VerbFam : toujours NONE pour les spans NER (ce sont des entités, pas des verbes)
            "verb_family_label_id":      VERB_FAMILY_NONE_ID,
            "verb_family_fine_label_id": VERB_FAMILY_FINE_NONE_ID,
            "verb_polarity_label_id":    VERB_POLARITY_NONE_ID,
            "verb_aspect_label_id":      VERB_ASPECT_NONE_ID,
            "verb_source_label_id":      VERB_SOURCE_NONE_ID,
            # Attributs transverses v9 (dérivés du fine label d'origine)
            "animacy_label_id":    _attrs_v9["animacy"],
            "living_label_id":     _attrs_v9["living"],
            "abstract_label_id":   _attrs_v9["abstract"],
            "dynamicity_label_id": _attrs_v9["dynamicity"],
            "work_label_id":       _attrs_v9["work"],
            "neg_type":            "gold",
            "sample_weight":       1.0,
            "text":                sp.get("text", text[start:end]),
        }
        gold_candidates.append(cand)
        gold_token_spans.append((tok_start, tok_end))
        gold_char_spans.add((start, end))

    # ── Déduplication des rôles SVO sur spans NER overlappants ─────────────
    # Pour les spans NER (neg_type="gold") avec un rôle SVO, on garde uniquement
    # le rôle du span le PLUS LONG dans chaque groupe qui se chevauchent.
    # Les spans plus courts perdent leur rôle (→ OTHER) mais gardent leurs labels NER.
    # Élimine : (1) redondances "Agence" + "Agence internationale" même rôle,
    #            (2) contradictions "Millau"=OBLIQUE dans "viaduc de Millau"=SUBJECT.
    ner_role_indices = [
        i for i, c in enumerate(gold_candidates)
        if c.get("neg_type") == "gold"
        and c.get("role_coarse_label_id", ROLE_COARSE_NONE_ID) not in (ROLE_COARSE_NONE_ID, ROLE_COARSE_OTHER_ID)
    ]
    # Trier par longueur de span décroissante (le plus long est prioritaire)
    ner_role_indices.sort(
        key=lambda i: gold_candidates[i]["tok_end"] - gold_candidates[i]["tok_start"],
        reverse=True
    )
    claimed = []  # (tok_start, tok_end) des spans qui ont gardé leur rôle
    for idx in ner_role_indices:
        c = gold_candidates[idx]
        ts, te = c["tok_start"], c["tok_end"]
        overlaps = any(not (te < ks or ts > ke) for ks, ke in claimed)
        if overlaps:
            # Ce span perd son rôle SVO → devient OTHER (entité NER sans rôle annoté)
            gold_candidates[idx]["role_label_id"]        = ROLE_NONE_ID
            gold_candidates[idx]["role_coarse_label_id"] = ROLE_COARSE_OTHER_ID
            gold_candidates[idx]["semantic_role_label_id"] = SEMANTIC_ROLE_SKIP_ID
        else:
            claimed.append((ts, te))
    # ────────────────────────────────────────────────────────────────────────

    return text, input_ids, offsets, gold_candidates, gold_token_spans, gold_char_spans

def _make_negative(neg_type: str, char_start: int, char_end: int,
                   tok_start: int, tok_end: int, weight: float = 1.0) -> dict:
    """Crée un candidat négatif avec tous les champs v4 à NONE/sentinel."""
    return {
        "char_start":          char_start,
        "char_end":            char_end,
        "tok_start":           tok_start,
        "tok_end":             tok_end,
        "boundary_label":      0,
        "svo_boundary_label":  0,
        "coarse_label_id":     COARSE_NONE_ID,
        "fine_label_id":       FINE_NONE_ID,
        "syn_label_id":        SYN_NONE_ID,
        "role_label_id":       ROLE_NONE_ID,
        "role_coarse_label_id": ROLE_COARSE_NONE_ID,
        "semantic_role_label_id": SEMANTIC_ROLE_SKIP_ID,
        "voice_label_id":      VOICE_NONE_ID,
        "certainty_label_id":  CERTAINTY_NONE_ID,
        "gender_label_id":     GENDER_NONE_ID,
        "number_label_id":     NUMBER_NONE_ID,
        "person_label_id":     PERSON_NONE_ID,
        "gov_verb_tok_start":  -1,
        "mod_of_tok_start":    -1,
        # Nominal parent pointer (v8.22) — sentinel pour les négatifs
        "nominal_parent_tok_start":  -1,
        "nominal_relation_label_id": NOMINAL_RELATION_NONE_ID,
        # VerbFam : NONE pour les spans négatifs
        "verb_family_label_id":      VERB_FAMILY_NONE_ID,
        "verb_family_fine_label_id": VERB_FAMILY_FINE_NONE_ID,
        "verb_polarity_label_id":    VERB_POLARITY_NONE_ID,
        "verb_aspect_label_id":      VERB_ASPECT_NONE_ID,
        "verb_source_label_id":      VERB_SOURCE_NONE_ID,
        "neg_type":            neg_type,
        "sample_weight":       weight,
        "text":                None,
    }


def generate_hard_negatives(offsets, gold_candidates, gold_char_spans, max_per_gold=6):
    """
    Hard negatives = variantes de frontières autour des spans golds.
    """
    n_tokens = len(offsets)
    out = []
    seen = set()

    for gc in gold_candidates:
        l = gc["tok_start"]
        r = gc["tok_end"]

        proposals = []
        for dl in [-2, -1, 0, 1, 2]:
            for dr in [-2, -1, 0, 1, 2]:
                nl = l + dl
                nr = r + dr
                if nl < 0 or nr < 0 or nl >= n_tokens or nr >= n_tokens or nl > nr:
                    continue
                if nl == l and nr == r:
                    continue

                cstart, cend = token_span_to_char_span(offsets, nl, nr)
                if (cstart, cend) in gold_char_spans:
                    continue

                # On veut plutôt des spans "proches" d'un gold
                iou = token_span_iou((l, r), (nl, nr))
                if iou <= 0.0:
                    continue

                proposals.append((nl, nr, cstart, cend, iou))

        # garder les plus proches
        proposals.sort(key=lambda x: x[-1], reverse=True)
        kept = 0
        for nl, nr, cstart, cend, iou in proposals:
            key = (cstart, cend)
            if key in seen:
                continue
            seen.add(key)
            out.append(_make_negative("hard_neg", cstart, cend, nl, nr))
            kept += 1
            if kept >= max_per_gold:
                break

    return out


def generate_svo_hard_negatives(offsets, gold_candidates, gold_char_spans, max_per_gold=4):
    """
    Hard negatives pour svo_boundary : variantes de frontières autour des spans SYNTAXIQUES golds.
    
    Contrairement aux hard negatives NER qui ciblent boundary_label,
    ceux-ci ciblent svo_boundary_label.
    
    Pour chaque span syntaxique gold (verb_trigger, pron_subj, pron_obj),
    génère des variantes proches (IoU > 0) qui devraient être prédites comme NON-syntaxiques.
    """
    n_tokens = len(offsets)
    out = []
    seen = set()

    # Filtrer les gold_candidates qui sont des spans syntaxiques (svo_boundary_label == 1)
    syn_golds = [gc for gc in gold_candidates if gc.get("svo_boundary_label", 0) == 1]

    for gc in syn_golds:
        l = gc["tok_start"]
        r = gc["tok_end"]

        proposals = []
        for dl in [-2, -1, 0, 1, 2]:
            for dr in [-2, -1, 0, 1, 2]:
                nl = l + dl
                nr = r + dr
                if nl < 0 or nr < 0 or nl >= n_tokens or nr >= n_tokens or nl > nr:
                    continue
                if nl == l and nr == r:
                    continue

                cstart, cend = token_span_to_char_span(offsets, nl, nr)
                if (cstart, cend) in gold_char_spans:
                    continue

                # On veut des spans "proches" d'un gold SYNTAXIQUE
                iou = token_span_iou((l, r), (nl, nr))
                if iou <= 0.0:
                    continue

                proposals.append((nl, nr, cstart, cend, iou))

        # garder les plus proches
        proposals.sort(key=lambda x: x[-1], reverse=True)
        kept = 0
        for nl, nr, cstart, cend, iou in proposals:
            key = (cstart, cend)
            if key in seen:
                continue
            seen.add(key)
            # Créer un candidat négatif pour svo_boundary
            out.append({
                "char_start": cstart,
                "char_end": cend,
                "tok_start": nl,
                "tok_end": nr,
                "boundary_label": 0,           # Pas un span NER
                "svo_boundary_label": 0,      # Négatif pour svo_boundary (c'est le but !)
                "coarse_label_id": COARSE_NONE_ID,
                "fine_label_id": FINE_NONE_ID,
                "syn_label_id": SYN_NONE_ID,
                "role_label_id": ROLE_NONE_ID,
                "role_coarse_label_id": ROLE_COARSE_NONE_ID,
                "semantic_role_label_id": SEMANTIC_ROLE_SKIP_ID,
                "voice_label_id": VOICE_NONE_ID,
                "certainty_label_id": CERTAINTY_NONE_ID,
                "gender_label_id": GENDER_NONE_ID,
                "number_label_id": NUMBER_NONE_ID,
                "person_label_id": PERSON_NONE_ID,
                "gov_verb_tok_start": -1,
                "mod_of_tok_start": -1,
                "verb_family_label_id": VERB_FAMILY_NONE_ID,
                "verb_family_fine_label_id": VERB_FAMILY_FINE_NONE_ID,
                "verb_polarity_label_id": VERB_POLARITY_NONE_ID,
                "verb_aspect_label_id": VERB_ASPECT_NONE_ID,
                "verb_source_label_id": VERB_SOURCE_NONE_ID,
                "neg_type": "svo_hard_neg",
                "sample_weight": 1.0,
                "text": None,
            })
            kept += 1
            if kept >= max_per_gold:
                break

    return out


def generate_soft_negatives(offsets, gold_token_spans, gold_char_spans, num_soft=20, max_span_len=8, seed=13):
    """
    Soft negatives = spans aléatoires non-overlap.
    """
    rnd = random.Random(seed)
    n_tokens = len(offsets)
    out = []
    seen = set()

    attempts = 0
    max_attempts = num_soft * 50

    while len(out) < num_soft and attempts < max_attempts:
        attempts += 1
        if n_tokens == 0:
            break

        l = rnd.randint(0, n_tokens - 1)
        span_len = rnd.randint(1, max_span_len)
        r = min(n_tokens - 1, l + span_len - 1)

        cstart, cend = token_span_to_char_span(offsets, l, r)
        key = (cstart, cend)
        if key in seen or key in gold_char_spans:
            continue

        # pas de recouvrement avec les golds
        overlap = False
        for gl, gr in gold_token_spans:
            if token_span_iou((l, r), (gl, gr)) > 0.0:
                overlap = True
                break
        if overlap:
            continue

        seen.add(key)
        out.append(_make_negative("soft_neg", cstart, cend, l, r, weight=0.35))

    return out


def generate_englobant_negatives(offsets, gold_candidates, gold_char_spans, max_per_gold=3, max_span_len=12):
    """
    Englobant negatives = spans larges qui CONTIENNENT un gold mais débordent
    suffisamment pour ne plus être une entité valide.
    Enseigne au boundary head que 'Simon Bolivar est considéré comme le Libérateur'
    n'est PAS une entité même si ça contient 'Simon Bolivar'.
    """
    n_tokens = len(offsets)
    out = []
    seen = set()

    for gc in gold_candidates:
        l = gc["tok_start"]
        r = gc["tok_end"]
        gold_len = r - l + 1

        proposals = []
        # Étendre à gauche et/ou à droite de 2 à 6 tokens au total
        for expand_left in range(0, 5):
            for expand_right in range(0, 5):
                total_expand = expand_left + expand_right
                if total_expand < 2:
                    # Au moins 2 tokens d'expansion pour être un vrai englobant
                    continue
                nl = l - expand_left
                nr = r + expand_right
                if nl < 0 or nr >= n_tokens or nr - nl + 1 > max_span_len:
                    continue

                cstart, cend = token_span_to_char_span(offsets, nl, nr)
                if (cstart, cend) in gold_char_spans:
                    continue

                new_len = nr - nl + 1
                # Plus l'expansion est grande, plus c'est un bon négatif
                expansion_ratio = new_len / max(1, gold_len)
                proposals.append((nl, nr, cstart, cend, expansion_ratio))

        # Trier par expansion décroissante (les plus larges d'abord = les plus informatifs)
        proposals.sort(key=lambda x: x[-1], reverse=True)
        kept = 0
        for nl, nr, cstart, cend, _ in proposals:
            key = (cstart, cend)
            if key in seen:
                continue
            seen.add(key)
            out.append(_make_negative("englobant_neg", cstart, cend, nl, nr, weight=1.5))
            kept += 1
            if kept >= max_per_gold:
                break

    return out


def generate_multi_entity_negatives(gold_candidates, gold_char_spans, max_negatives=5):
    """
    Multi-entity negatives = spans qui englobent 2+ entités adjacentes.
    Enseigne que 'Winston Churchill et Franklin D. Roosevelt' n'est PAS un seul span.
    """
    out = []
    seen = set()

    # Trier les golds par position
    sorted_golds = sorted(gold_candidates, key=lambda g: g["tok_start"])

    for i in range(len(sorted_golds) - 1):
        g1 = sorted_golds[i]
        g2 = sorted_golds[i + 1]

        # Vérifier qu'ils sont proches (gap <= 5 tokens)
        gap = g2["tok_start"] - g1["tok_end"]
        if gap > 5 or gap < 0:
            continue

        nl = g1["tok_start"]
        nr = g2["tok_end"]
        cstart = g1["char_start"]
        cend = g2["char_end"]

        key = (cstart, cend)
        if key in seen or key in gold_char_spans:
            continue
        seen.add(key)
        seen.add(key)
        out.append(_make_negative("multi_entity_neg", cstart, cend, nl, nr, weight=2.0))

        if len(out) >= max_negatives:
            break

    return out


def make_multitask_row(row, tokenizer, hard_per_gold=6, soft_factor=2.0, max_span_len=8, seed=13, svo_hard_per_gold=4):
    text, input_ids, offsets, gold_candidates, gold_token_spans, gold_char_spans = build_gold_candidates(row, tokenizer)

    num_soft = max(1, int(len(gold_candidates) * soft_factor))
    hard_negs = generate_hard_negatives(
        offsets,
        gold_candidates,
        gold_char_spans,
        max_per_gold=hard_per_gold
    )
    # Hard negatives pour svo_boundary : variantes autour des spans syntaxiques
    svo_hard_negs = generate_svo_hard_negatives(
        offsets,
        gold_candidates,
        gold_char_spans,
        max_per_gold=svo_hard_per_gold
    )
    soft_negs = generate_soft_negatives(
        offsets,
        gold_token_spans,
        gold_char_spans,
        num_soft=num_soft,
        max_span_len=max_span_len,
        seed=seed
    )
    englobant_negs = generate_englobant_negatives(
        offsets,
        gold_candidates,
        gold_char_spans,
        max_per_gold=3,
        max_span_len=max_span_len,
    )
    multi_ent_negs = generate_multi_entity_negatives(
        gold_candidates,
        gold_char_spans,
        max_negatives=5,
    )

    candidates = gold_candidates + hard_negs + svo_hard_negs + soft_negs + englobant_negs + multi_ent_negs

    # renseigner le texte du span si absent
    for c in candidates:
        if c["text"] is None:
            c["text"] = text[c["char_start"]:c["char_end"]]

    # Appliquer le _source_weight de merge_silver.py si présent
    source_weight = row.get("_source_weight", 1.0)
    if source_weight != 1.0:
        for c in candidates:
            c["sample_weight"] = c.get("sample_weight", 1.0) * source_weight

    return {
        "id": row["id"],
        "text": text,
        "candidates": candidates,
        "meta": {
            "num_gold": len(gold_candidates),
            "num_hard_neg": len(hard_negs),
            "num_svo_hard_neg": len(svo_hard_negs),
            "num_soft_neg": len(soft_negs),
            "num_englobant_neg": len(englobant_negs),
            "num_multi_entity_neg": len(multi_ent_negs),
            "num_tokens": len(input_ids),
        }
    }

def export_head_views(rows, out_prefix: str):
    """
    Exporte des vues séparées si tu veux inspecter / debugger chaque tête.
    """
    boundary_rows = []
    coarse_rows = []
    fine_rows = []

    for row in rows:
        boundary_rows.append({
            "id": row["id"],
            "text": row["text"],
            "spans": [
                {
                    "start": c["char_start"],
                    "end": c["char_end"],
                    "text": c["text"],
                    "label": c["boundary_label"],
                    "neg_type": c["neg_type"],
                    "sample_weight": c["sample_weight"],
                }
                for c in row["candidates"]
            ]
        })
        coarse_rows.append({
            "id": row["id"],
            "text": row["text"],
            "spans": [
                {
                    "start": c["char_start"],
                    "end": c["char_end"],
                    "text": c["text"],
                    "label_id": c["coarse_label_id"],
                    "neg_type": c["neg_type"],
                    "sample_weight": c["sample_weight"],
                }
                for c in row["candidates"]
            ]
        })
        fine_rows.append({
            "id": row["id"],
            "text": row["text"],
            "spans": [
                {
                    "start": c["char_start"],
                    "end": c["char_end"],
                    "text": c["text"],
                    "label_id": c["fine_label_id"],
                    "neg_type": c["neg_type"],
                    "sample_weight": c["sample_weight"],
                }
                for c in row["candidates"]
            ]
        })

    write_jsonl(out_prefix + ".boundary.jsonl", boundary_rows)
    write_jsonl(out_prefix + ".coarse.jsonl", coarse_rows)
    write_jsonl(out_prefix + ".fine.jsonl", fine_rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="dataset JSONL source (positifs uniquement)")
    parser.add_argument("--output", required=True, help="dataset JSONL multitask de sortie")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--tokenizer-path", default=None, help="si tu veux utiliser ton tokenizer local fast")
    parser.add_argument("--hard-per-gold", type=int, default=6)
    parser.add_argument("--svo-hard-per-gold", type=int, default=4, help="nb hard negatives SVO par span syntaxique gold")
    parser.add_argument("--soft-factor", type=float, default=2.0, help="nb soft neg = soft_factor * nb gold")
    parser.add_argument("--max-span-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--export-head-views-prefix", default=None, help="si défini, exporte aussi boundary/coarse/fine séparés")
    args = parser.parse_args()

    random.seed(args.seed)

    tokenizer_source = args.tokenizer_path or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    rows = []
    for row in load_jsonl(args.input):
        mt_row = make_multitask_row(
            row,
            tokenizer=tokenizer,
            hard_per_gold=args.hard_per_gold,
            svo_hard_per_gold=args.svo_hard_per_gold,
            soft_factor=args.soft_factor,
            max_span_len=args.max_span_len,
            seed=args.seed
        )
        rows.append(mt_row)

    write_jsonl(args.output, rows)
    print(f"✅ dataset multitask écrit dans {args.output} ({len(rows)} lignes)")

    if args.export_head_views_prefix:
        export_head_views(rows, args.export_head_views_prefix)
        print(f"✅ vues séparées écrites avec préfixe {args.export_head_views_prefix}")

if __name__ == "__main__":
    main()
