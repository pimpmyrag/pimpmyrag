# labels_v9.py
# ─────────────────────────────────────────────────────────────
# TAXONOMIE v9.0 — PROPOSITION (non câblée)
# ─────────────────────────────────────────────────────────────
# Refonte : le coarse ne porte plus que l'axe ONTOLOGIQUE (qu'est-ce que
# c'est ?). Les axes PROPRIÉTÉ (vivant, abstrait, dynamique, œuvre) passent
# en ATTRIBUTS binaires, tous DÉRIVÉS du label d'origine → zéro ré-annotation
# et zéro perte d'information (state/doctrine restent récupérables via les
# attributs même après fusion des fines).
#
# Changements vs labels.py (v8.x) :
#   COARSE 11 → 8 (+NONE) : suppression de BIO et ABSTRACT, fusion de WORK
#     - OBJECT  absorbe animal, vegetal          (ex-BIO physique)
#     - EVENT   absorbe disease                  (condition)
#     - CONCEPT absorbe ex-ABSTRACT + ex-WORK
#   FINE 40 → 34 (option B — on garde field ET language) :
#     - hint_state       → hint_event_nominal (info via dynamicity)
#     - hint_doctrine    → hint_notion         (info via abstractness)
#     - hint_work_generic→ hint_work_of_art    (info via attribut work)
#     - hint_object_name → hint_object_generic (distinction nommé/générique bruitée)
#     - hint_rate        → hint_measure        (sous-type rare de mesure)
#     - hint_vegetal     → hint_animal         (bucket vivant non-humain ; animacy
#                                               distingue animal/plante via attribut)
#   ATTRIBUTS (5 têtes binaires, dérivées) :
#     animacy, living, abstractness, dynamicity, work
#
# Les blocs INCHANGÉS (SYN, ROLE, ROLE_COARSE, ROLE_OBLIQUE, VOICE, CERTAINTY,
# MORPHO, VERB_*, NOMINAL_RELATION, SEMANTIC_ROLE) restent définis dans
# labels.py : ils seront hérités tels quels lors de la promotion en labels.py.
from __future__ import annotations
import torch

# ─────────────────────────────────────────────────────────────
# FINE LABELS v9 (38 — POSITIVE ONLY)
# ─────────────────────────────────────────────────────────────
FINE_LABELS = [
    "hint_person_name",      # 0
    "hint_person_role",      # 1
    "hint_norp",             # 2
    "hint_group_role",       # 3
    "hint_org_name",         # 4
    "hint_inst_name",        # 5
    "hint_gpe",              # 6
    "hint_fac_name",         # 7
    "hint_loc_generic",      # 8
    "hint_weapon",           # 9
    "hint_vehicle",          # 10
    "hint_substance",        # 11
    "hint_food",             # 12
    "hint_infra",            # 13
    "hint_tool",             # 14
    "hint_object_generic",   # 15  ← absorbe hint_object_name
    "hint_event_nominal",    # 16  ← absorbe hint_state (dynamicity=stative)
    "hint_event_named",      # 17
    "hint_time_date",        # 18
    "hint_time_clock",       # 19
    "hint_time_duration",    # 20
    "hint_measure",          # 21  ← absorbe hint_rate
    "hint_percentage",       # 22
    "hint_count",            # 23
    "hint_money",            # 24
    "hint_work_of_art",      # 25  ← absorbe hint_work_generic (attribut work=1)
    "hint_law",              # 26
    "hint_document",         # 27
    "hint_disease",          # 28  coarse EVENT (condition), dynamicity=stative
    "hint_language",         # 29
    "hint_inst_role",        # 30
    "hint_animal",           # 31  vivant non-humain (absorbe hint_vegetal ;
                             #      animacy distingue animal[animate]/plante[inanimate])
    "hint_notion",           # 32  ← absorbe hint_doctrine
    "hint_field",            # 33  domaine / secteur (conservé pour le rôle DOMAIN)
]

FINE2ID = {x: i for i, x in enumerate(FINE_LABELS)}
ID2FINE = {i: x for x, i in FINE2ID.items()}
NUM_FINE = len(FINE_LABELS)          # = 34
FINE_NONE_ID = NUM_FINE               # = 34, hors range [0..33]

# ─────────────────────────────────────────────────────────────
# MIGRATION des anciens fines (v8.x) → v9
# Utilisé au build : on lit le label d'origine, on le mappe en fine v9,
# et on dérive les attributs À PARTIR DE L'ORIGINAL (pas de perte).
# ─────────────────────────────────────────────────────────────
LEGACY_TO_V9_FINE = {
    "hint_state":        "hint_event_nominal",
    "hint_doctrine":     "hint_notion",
    "hint_work_generic": "hint_work_of_art",
    "hint_object_name":  "hint_object_generic",
    "hint_rate":         "hint_measure",
    "hint_vegetal":      "hint_animal",
    # tous les autres = identité
}

def to_v9_fine(label: str) -> str:
    """Mappe un fine label (v8.x ou v9) vers le fine v9 canonique."""
    return LEGACY_TO_V9_FINE.get(label, label)

# ─────────────────────────────────────────────────────────────
# COARSE LABELS v9 (8 + NONE)
# ─────────────────────────────────────────────────────────────
COARSE_LABELS = [
    "PER",       # 0
    "LOC",       # 1
    "ORG",       # 2
    "TIME",      # 3
    "VALUE",     # 4
    "OBJECT",    # 5  artefacts physiques + vivant physique (animal, vegetal)
    "EVENT",     # 6  événements + états/conditions (disease)
    "CONCEPT",   # 7  informationnel/abstrait (ex-ABSTRACT + ex-WORK)
    "NONE",      # 8
]
COARSE2ID = {x: i for i, x in enumerate(COARSE_LABELS)}
ID2COARSE = {i: x for x, i in COARSE2ID.items()}
COARSE_NONE_ID = COARSE2ID["NONE"]

# ─────────────────────────────────────────────────────────────
# COARSE → FINE MAPPING v9
# ─────────────────────────────────────────────────────────────
COARSE_TO_FINE = {
    COARSE2ID["PER"]: [
        FINE2ID["hint_person_name"],
        FINE2ID["hint_person_role"],
        FINE2ID["hint_norp"],
        FINE2ID["hint_group_role"],
    ],
    COARSE2ID["LOC"]: [
        FINE2ID["hint_gpe"],
        FINE2ID["hint_fac_name"],
        FINE2ID["hint_loc_generic"],
        FINE2ID["hint_infra"],
    ],
    COARSE2ID["ORG"]: [
        FINE2ID["hint_org_name"],
        FINE2ID["hint_inst_name"],
        FINE2ID["hint_inst_role"],
    ],
    COARSE2ID["TIME"]: [
        FINE2ID["hint_time_date"],
        FINE2ID["hint_time_clock"],
        FINE2ID["hint_time_duration"],
    ],
    COARSE2ID["VALUE"]: [
        FINE2ID["hint_measure"],
        FINE2ID["hint_percentage"],
        FINE2ID["hint_count"],
        FINE2ID["hint_money"],
    ],
    COARSE2ID["OBJECT"]: [
        FINE2ID["hint_weapon"],
        FINE2ID["hint_vehicle"],
        FINE2ID["hint_substance"],
        FINE2ID["hint_food"],
        FINE2ID["hint_tool"],
        FINE2ID["hint_object_generic"],
        FINE2ID["hint_animal"],
    ],
    COARSE2ID["EVENT"]: [
        FINE2ID["hint_event_nominal"],
        FINE2ID["hint_event_named"],
        FINE2ID["hint_disease"],
    ],
    COARSE2ID["CONCEPT"]: [
        FINE2ID["hint_work_of_art"],
        FINE2ID["hint_law"],
        FINE2ID["hint_document"],
        FINE2ID["hint_language"],
        FINE2ID["hint_notion"],
        FINE2ID["hint_field"],
    ],
}

# FINE → COARSE reverse mapping
_FINE_TO_COARSE = {}
for _c, _fines in COARSE_TO_FINE.items():
    for _f in _fines:
        _FINE_TO_COARSE[_f] = _c
        _FINE_TO_COARSE[FINE_LABELS[_f]] = _c

def fine_label_to_coarse_id(fine_label: str | int) -> int:
    return _FINE_TO_COARSE.get(fine_label, COARSE_NONE_ID)

def build_coarse_to_fine_mask() -> torch.Tensor:
    mask = torch.zeros(len(COARSE_LABELS), NUM_FINE, dtype=torch.bool)
    for c, fines in COARSE_TO_FINE.items():
        for f in fines:
            mask[c, f] = True
    return mask

# ─────────────────────────────────────────────────────────────
# ATTRIBUTS v9 — 5 têtes binaires
# Chaque attribut a un sentinel NONE (= 2) pour les spans où il n'est
# pas supervisé (span négatif, ou non pertinent p.ex. dynamicity hors EVENT).
# Les tables ci-dessous sont exprimées sur les fines v9 ; la dérivation
# depuis un label LEGACY passe d'abord par to_v9_fine() PUIS applique les
# surcharges d'origine (state/doctrine) pour ne rien perdre.
# ─────────────────────────────────────────────────────────────

# --- animacy : {inanimate=0, animate=1} ---
ANIMACY_LABELS  = ["inanimate", "animate"]
ANIMACY2ID      = {x: i for i, x in enumerate(ANIMACY_LABELS)}
NUM_ANIMACY     = len(ANIMACY_LABELS)
ANIMACY_NONE_ID = NUM_ANIMACY
# Sur label d'ORIGINE (vegetal exclu → une plante n'est pas animate)
_ANIMATE_ORIG = {
    "hint_person_name", "hint_person_role", "hint_norp",
    "hint_group_role", "hint_animal",
}

# --- living : {non_living=0, living=1}  (biologique : + vegetal) ---
LIVING_LABELS  = ["non_living", "living"]
LIVING2ID      = {x: i for i, x in enumerate(LIVING_LABELS)}
NUM_LIVING     = len(LIVING_LABELS)
LIVING_NONE_ID = NUM_LIVING
_LIVING_ORIG = _ANIMATE_ORIG | {"hint_vegetal"}

# --- abstractness : {concrete=0, abstract=1} ---
ABSTRACT_LABELS  = ["concrete", "abstract"]
ABSTRACT2ID      = {x: i for i, x in enumerate(ABSTRACT_LABELS)}
NUM_ABSTRACT     = len(ABSTRACT_LABELS)
ABSTRACT_NONE_ID = NUM_ABSTRACT
# Tout CONCEPT + disease sont abstraits ; les states (fusionnés dans
# event_nominal) le sont aussi → surcharge via label d'origine.
_ABSTRACT_FINES = {
    "hint_work_of_art", "hint_law", "hint_document", "hint_language",
    "hint_notion", "hint_field", "hint_disease",
}

# --- dynamicity : {stative=0, dynamic=1}  (supervisé sur EVENT uniquement) ---
DYNAMICITY_LABELS  = ["stative", "dynamic"]
DYNAMICITY2ID      = {x: i for i, x in enumerate(DYNAMICITY_LABELS)}
NUM_DYNAMICITY     = len(DYNAMICITY_LABELS)
DYNAMICITY_NONE_ID = NUM_DYNAMICITY
# EVENT coarse = {event_nominal, event_named, disease}. Par défaut dynamic ;
# stative pour les ex-state et disease (conditions). L'ex-state se détecte
# via le label d'origine hint_state.
_STATIVE_FINES = {"hint_disease"}   # + hint_state via override d'origine

# --- work : {non_work=0, work=1}  (production intellectuelle/culturelle) ---
WORK_LABELS  = ["non_work", "work"]
WORK2ID      = {x: i for i, x in enumerate(WORK_LABELS)}
NUM_WORK     = len(WORK_LABELS)
WORK_NONE_ID = NUM_WORK
_WORK_FINES = {"hint_work_of_art", "hint_law", "hint_document"}


def derive_attributes(legacy_label: str) -> dict[str, int]:
    """
    Dérive les 5 attributs v9 à partir du label d'ORIGINE (v8.x ou v9).
    Passe par le label d'origine pour ne pas perdre l'info fusionnée
    (hint_state → dynamicity=stative ; hint_doctrine reste abstract=1).
    Retourne un dict d'IDs d'attributs (jamais NONE ici : pour un span
    positif tous les attributs sont définis ; le NONE_ID est réservé aux
    spans négatifs, géré au build).
    """
    v9 = to_v9_fine(legacy_label)
    is_event = fine_label_to_coarse_id(v9) == COARSE2ID["EVENT"]
    stative = (legacy_label == "hint_state") or (v9 in _STATIVE_FINES)
    # abstractness : par fine v9, MAIS surcharge d'origine (un ex-state
    # fusionné dans event_nominal reste une condition abstraite).
    is_abstract = (v9 in _ABSTRACT_FINES) or (legacy_label == "hint_state")
    return {
        # animacy/living : intrinsèques → basés sur le label d'ORIGINE
        # (une plante fusionnée dans hint_animal reste inanimate + living).
        "animacy":    ANIMACY2ID["animate"] if legacy_label in _ANIMATE_ORIG else ANIMACY2ID["inanimate"],
        "living":     LIVING2ID["living"] if legacy_label in _LIVING_ORIG else LIVING2ID["non_living"],
        "abstract":   ABSTRACT2ID["abstract"] if is_abstract else ABSTRACT2ID["concrete"],
        # dynamicity : significatif uniquement sur EVENT, sinon NONE (non supervisé)
        "dynamicity": (DYNAMICITY2ID["stative"] if stative else DYNAMICITY2ID["dynamic"]) if is_event else DYNAMICITY_NONE_ID,
        "work":       WORK2ID["work"] if v9 in _WORK_FINES else WORK2ID["non_work"],
    }


# ─────────────────────────────────────────────────────────────
# Auto-test taxonomie (python3 labels_v9.py)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Cohérence coarse↔fine : tout fine mappé exactement une fois
    mapped = [f for fines in COARSE_TO_FINE.values() for f in fines]
    assert len(mapped) == NUM_FINE == 34, f"fine mapping {len(mapped)} != {NUM_FINE}"
    assert len(set(mapped)) == NUM_FINE, "fine mappé en double"
    assert len(COARSE_LABELS) == 9, "coarse != 8+NONE"

    # 2. Dérivations attendues
    def chk(label, **exp):
        a = derive_attributes(label)
        for k, v in exp.items():
            got = a[k]
            name = {
                "animacy": ANIMACY_LABELS, "living": LIVING_LABELS,
                "abstract": ABSTRACT_LABELS, "work": WORK_LABELS,
                "dynamicity": DYNAMICITY_LABELS + ["NONE"],
            }[k][got]
            assert name == v, f"{label}.{k} = {name} != {v}"

    chk("hint_person_name", animacy="animate", living="living", abstract="concrete", work="non_work")
    chk("hint_animal",      animacy="animate", living="living", abstract="concrete")
    chk("hint_org_name",    animacy="inanimate", living="non_living", abstract="concrete")
    chk("hint_notion",      abstract="abstract", work="non_work")
    chk("hint_law",         abstract="abstract", work="work")
    chk("hint_work_of_art", abstract="abstract", work="work")
    chk("hint_field",       abstract="abstract")
    chk("hint_language",    abstract="abstract")
    chk("hint_disease",     abstract="abstract", dynamicity="stative")
    chk("hint_event_named", dynamicity="dynamic")
    chk("hint_event_nominal", dynamicity="dynamic")
    # legacy merges
    chk("hint_state",    dynamicity="stative", abstract="abstract")
    assert to_v9_fine("hint_state") == "hint_event_nominal"
    chk("hint_doctrine", abstract="abstract", work="non_work")
    assert to_v9_fine("hint_doctrine") == "hint_notion"
    chk("hint_work_generic", abstract="abstract", work="work")
    assert to_v9_fine("hint_work_generic") == "hint_work_of_art"
    chk("hint_object_name", animacy="inanimate", abstract="concrete", work="non_work")
    assert to_v9_fine("hint_object_name") == "hint_object_generic"
    chk("hint_rate", abstract="concrete")
    assert to_v9_fine("hint_rate") == "hint_measure"
    # vegetal fusionné dans animal MAIS reste inanimate + living (via label d'origine)
    chk("hint_vegetal", animacy="inanimate", living="living", abstract="concrete")
    assert to_v9_fine("hint_vegetal") == "hint_animal"

    print(f"✅ labels_v9 OK — NUM_FINE={NUM_FINE} NUM_COARSE={len(COARSE_LABELS)} "
          f"| attributs: animacy/living/abstract/dynamicity/work")

