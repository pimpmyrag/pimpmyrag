# labels.py
from __future__ import annotations
import torch

# ─────────────────────────────────────────────────────────────
# FINE LABELS (POSITIVE ONLY)
# ─────────────────────────────────────────────────────────────

FINE_LABELS = [
    "hint_person_name",      # 0
    "hint_person_role",      # 1
    "hint_norp",             # 2
    "hint_group_role",       # 3
    "hint_org_name",         # 4
    "hint_gpe",              # 5
    "hint_fac_name",         # 6
    "hint_loc_generic",      # 7
    "hint_weapon",           # 8
    "hint_vehicle",          # 9
    "hint_substance",        # 10
    "hint_food",             # 11
    "hint_infra",            # 12
    "hint_tool",             # 13
    "hint_object_generic",   # 14
    "hint_object_name",      # 15
    "hint_event_nominal",    # 16
    "hint_event_named",      # 17
    "hint_time_date",        # 18
    "hint_time_clock",       # 19
    "hint_time_duration",    # 20
    "hint_quantity",         # 21
    "hint_measure",          # 22
    "hint_percentage",       # 23
    "hint_count",            # 24
    "hint_money",            # 25
    "hint_rate",             # 26
    "hint_law",              # 27
    "hint_work_of_art",      # 28
    "hint_concept",          # 29
    "hint_disease",          # 30
    "hint_language",         # 31
]

FINE2ID = {x: i for i, x in enumerate(FINE_LABELS)}
ID2FINE = {i: x for x, i in FINE2ID.items()}

NUM_FINE = len(FINE_LABELS)

# Sentinel pour les spans négatifs (pas un vrai label fine)
FINE_NONE_ID = NUM_FINE  # = 32, hors range [0..31]

# ─────────────────────────────────────────────────────────────
# COARSE LABELS
# ─────────────────────────────────────────────────────────────

COARSE_LABELS = [
    "PER",      # 0
    "LOC",      # 1
    "ORG",      # 2
    "TIME",     # 3
    "EVENT",    # 4
    "OBJECT",   # 5
    "VALUE",    # 6
    "ABSTRACT", # 7
    "NONE",     # 8
]

COARSE2ID = {x: i for i, x in enumerate(COARSE_LABELS)}
ID2COARSE = {i: x for x, i in COARSE2ID.items()}

COARSE_NONE_ID = COARSE2ID["NONE"]

# ─────────────────────────────────────────────────────────────
# COARSE → FINE MAPPING
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
    ],
    COARSE2ID["TIME"]: [
        FINE2ID["hint_time_date"],
        FINE2ID["hint_time_clock"],
        FINE2ID["hint_time_duration"],
    ],
    COARSE2ID["EVENT"]: [
        FINE2ID["hint_event_nominal"],
        FINE2ID["hint_event_named"],
    ],
    COARSE2ID["OBJECT"]: [
        FINE2ID["hint_weapon"],
        FINE2ID["hint_vehicle"],
        FINE2ID["hint_substance"],
        FINE2ID["hint_food"],
        FINE2ID["hint_tool"],
        FINE2ID["hint_object_generic"],
        FINE2ID["hint_object_name"],
    ],
    COARSE2ID["VALUE"]: [
        FINE2ID["hint_quantity"],
        FINE2ID["hint_measure"],
        FINE2ID["hint_percentage"],
        FINE2ID["hint_count"],
        FINE2ID["hint_money"],
        FINE2ID["hint_rate"],
    ],
    COARSE2ID["ABSTRACT"]: [
        FINE2ID["hint_law"],
        FINE2ID["hint_work_of_art"],
        FINE2ID["hint_concept"],
        FINE2ID["hint_disease"],
        FINE2ID["hint_language"],
    ],
}

# ─────────────────────────────────────────────────────────────
# FINE → COARSE reverse mapping
# ─────────────────────────────────────────────────────────────

_FINE_TO_COARSE = {}
for _c, _fines in COARSE_TO_FINE.items():
    for _f in _fines:
        _FINE_TO_COARSE[_f] = _c
        _FINE_TO_COARSE[FINE_LABELS[_f]] = _c


def fine_label_to_coarse_id(fine_label: str | int) -> int:
    """Renvoie le coarse_id pour un fine label (str ou int)."""
    if isinstance(fine_label, int):
        return _FINE_TO_COARSE.get(fine_label, COARSE_NONE_ID)
    return _FINE_TO_COARSE.get(fine_label, COARSE_NONE_ID)

def build_coarse_to_fine_mask() -> torch.Tensor:
    """
    Matrice [num_coarse, num_fine] binaire :
    mask[c, f] = 1 si f est autorisé pour le coarse c
    """
    mask = torch.zeros(len(COARSE_LABELS), NUM_FINE, dtype=torch.bool)
    for c, fines in COARSE_TO_FINE.items():
        for f in fines:
            mask[c, f] = True
    return mask


# ─────────────────────────────────────────────────────────────
# SVO LABELS  (tête syntaxique / silver Stanza)
# ─────────────────────────────────────────────────────────────

SVO_LABELS = [
    "svo_verb",      # 0  verbe principal (+ aux)
    "svo_subject",   # 1  sujet grammatical (NP)
    "svo_object",    # 2  objet direct
    "svo_iobj",      # 3  objet indirect / oblique prép.
    "svo_tcomp",     # 4  CC de temps
    "svo_lcomp",     # 5  CC de lieu
    "svo_cause",     # 6  proposition / GN causal
    "attr",          # 7  attribut du sujet (copule)
    "nom_event",     # 8  NOUN déverbal avec argument ("l'arrestation de X")
    "ent_appos",     # 9  apposition NE → rôle/titre
    "pron_subj",     # 10 pronom sujet
    "pron_obj",      # 11 pronom objet
    "pron_dem",      # 12 pronom démonstratif ("celui-ci", "ça")
    "neg",           # 13 marqueur de négation
]

SVO2ID   = {x: i for i, x in enumerate(SVO_LABELS)}
ID2SVO   = {i: x for x, i in SVO2ID.items()}
NUM_SVO  = len(SVO_LABELS)
# Sentinel pour les spans sans rôle SVO (NER purs, négatifs)
SVO_NONE_ID = NUM_SVO   # = 6, hors range [0..5]

# ─────────────────────────────────────────────────────────────
# VOICE LABELS  (pour la tête voice, prédite sur les svo_verb)
# ─────────────────────────────────────────────────────────────

VOICE_LABELS = ["ACTIVE", "PASSIVE"]
VOICE2ID     = {x: i for i, x in enumerate(VOICE_LABELS)}
ID2VOICE     = {i: x for x, i in VOICE2ID.items()}
NUM_VOICE    = len(VOICE_LABELS)
VOICE_NONE_ID = NUM_VOICE   # sentinel

# Ensemble des labels silver SVO (pour le routage dans build_multitask_dataset)
ALL_SVO_LABELS: set[str] = set(SVO_LABELS)

# ─────────────────────────────────────────────────────────────
# MORPHO LABELS  (gender + number, pour la coréf future)
# Supervisés sur tous les spans SVO actifs (svo_label != SVO_NONE_ID)
# ─────────────────────────────────────────────────────────────

GENDER_LABELS  = ["Masc", "Fem", "NONE"]   # 0, 1, 2
GENDER2ID      = {x: i for i, x in enumerate(GENDER_LABELS)}
ID2GENDER      = {i: x for x, i in GENDER2ID.items()}
NUM_GENDER     = len(GENDER_LABELS)
GENDER_NONE_ID = GENDER2ID["NONE"]         # = 2

NUMBER_LABELS  = ["Sing", "Plur", "NONE"]  # 0, 1, 2
NUMBER2ID      = {x: i for i, x in enumerate(NUMBER_LABELS)}
ID2NUMBER      = {i: x for x, i in NUMBER2ID.items()}
NUM_NUMBER     = len(NUMBER_LABELS)
NUMBER_NONE_ID = NUMBER2ID["NONE"]         # = 2

# PERSON : supervisé sur les pronoms (pron_subj / pron_obj / pron_dem)
PERSON_LABELS  = ["1", "2", "3", "NONE"]   # 0, 1, 2, 3
PERSON2ID      = {x: i for i, x in enumerate(PERSON_LABELS)}
ID2PERSON      = {i: x for x, i in PERSON2ID.items()}
NUM_PERSON     = len(PERSON_LABELS)
PERSON_NONE_ID = PERSON2ID["NONE"]         # = 3

