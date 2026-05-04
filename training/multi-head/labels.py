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
    "hint_inst_name",        # 5  institution NOMMÉE (sigle ou nom propre qualifié : "ONU", "OTAN", "Commission européenne", "Sénat américain")
    "hint_gpe",              # 6
    "hint_fac_name",         # 7
    "hint_loc_generic",      # 8
    "hint_weapon",           # 9
    "hint_vehicle",          # 10
    "hint_substance",        # 11
    "hint_food",             # 12
    "hint_infra",            # 13
    "hint_tool",             # 14
    "hint_object_generic",   # 15
    "hint_object_name",      # 16
    "hint_event_nominal",    # 17
    "hint_event_named",      # 18
    "hint_time_date",        # 19
    "hint_time_clock",       # 20
    "hint_time_duration",    # 21
    "hint_quantity",         # 22
    "hint_measure",          # 23
    "hint_percentage",       # 24
    "hint_count",            # 25
    "hint_money",            # 26
    "hint_rate",             # 27
    "hint_work_of_art",      # 28
    "hint_law",              # 29
    "hint_document",         # 30  rapport, lettre, communiqué, données, contrat…
    "hint_disease",          # 31
    "hint_language",         # 32
    "hint_inst_role",        # 33  institution GÉNÉRIQUE sans qualificatif (gouvernement, police, armée, parlement, tribunal…)
    # ▼ v6.9 : hint_concept éclaté en 7 sous-types + fallback (ancien hint_concept_named fusionné dans hint_concept puis redistribué)
    "hint_rule",             # 34  règle, procédure, norme, protocole opérationnel
    "hint_doctrine",         # 35  doctrine, idéologie, courant de pensée, théorie (nommée ou non)
    "hint_state",            # 36  état, condition, situation abstraite (pauvreté, crise, guerre, paix…)
    "hint_notion",           # 37  notion, concept abstrait pur, valeur, principe philosophique
    "hint_work_generic",     # 38  production culturelle générique sans titre (film, livre, presse, médias…)
    "hint_field",            # 39  domaine / secteur d'activité (santé, éducation, agriculture, finance…)
    "hint_process",          # 40  processus socio-économique continu (réforme, privatisation, mondialisation…)
    "hint_concept",          # 41  ← fallback : concept abstrait inclassable dans les 7 sous-types ci-dessus
]

FINE2ID = {x: i for i, x in enumerate(FINE_LABELS)}
ID2FINE = {i: x for x, i in FINE2ID.items()}

NUM_FINE = len(FINE_LABELS)

# Sentinel pour les spans négatifs (pas un vrai label fine)
FINE_NONE_ID = NUM_FINE  # = 42, hors range [0..41]

# ─────────────────────────────────────────────────────────────
# COARSE LABELS
# ─────────────────────────────────────────────────────────────

COARSE_LABELS = [
    "PER",      # 0
    "LOC",      # 1
    "ORG",      # 2
    "TIME",     # 3
    "EVENT",    # 4
    "OBJECT",   # 5  artefacts physiques
    "VALUE",    # 6
    "WORK",     # 7  ← nouveau : productions intellectuelles/culturelles (oeuvre, loi)
    "ABSTRACT", # 8
    "NONE",     # 9
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
        FINE2ID["hint_inst_name"],   # institution publique NOMMÉE (sigle, nom propre qualifié)
        FINE2ID["hint_inst_role"],   # institution générique (gouvernement, police, armée…)
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
    COARSE2ID["WORK"]: [
        FINE2ID["hint_work_of_art"],
        FINE2ID["hint_law"],
        FINE2ID["hint_document"],       # rapport, lettre, communiqué, données, contrat…
        FINE2ID["hint_work_generic"],   # v6.9 : production culturelle générique sans titre (film, presse, cinéma…)
    ],
    COARSE2ID["ABSTRACT"]: [
        FINE2ID["hint_disease"],
        FINE2ID["hint_language"],
        # v6.9 : hint_concept redistributé en 6 sous-types + fallback
        FINE2ID["hint_rule"],
        FINE2ID["hint_doctrine"],
        FINE2ID["hint_state"],
        FINE2ID["hint_notion"],
        FINE2ID["hint_field"],
        FINE2ID["hint_process"],
        FINE2ID["hint_concept"],  # fallback résiduel
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
# SYNTACTIC SPAN LABELS  (v4 — annotation Claude gold)
# Remplace les 14 labels SVO Stanza silver.
# verb_trigger et pronoms sont des spans syntaxiques distincts
# des spans NER.
# ─────────────────────────────────────────────────────────────

SYN_LABELS = [
    "verb_trigger",  # 0  verbe d'action gouverneur (+ auxiliaire)
    "pron_subj",     # 1  pronom sujet
    "pron_obj",      # 2  pronom objet clitique
]
SYN2ID      = {x: i for i, x in enumerate(SYN_LABELS)}
ID2SYN      = {i: x for x, i in SYN2ID.items()}
NUM_SYN     = len(SYN_LABELS)
SYN_NONE_ID = NUM_SYN   # sentinel (span sans rôle syntaxique)

# Compat alias (utilisé dans build_multitask_dataset + train)
ALL_SYN_LABELS: set[str] = set(SYN_LABELS)

# ─────────────────────────────────────────────────────────────
# ROLE LABELS  (rôle SVO prédit sur chaque span NER et pronom)
# ─────────────────────────────────────────────────────────────

ROLE_LABELS = [
    "SUBJECT",        # 0  sujet grammatical
    "OBJECT",         # 1  objet direct
    "OBLIQUE",        # 2  complément circonstanciel (lieu/temps/manière)
    "OBLIQUE_AGENT",  # 3  agent sémantique en passif ("par la police")
    "OBLIQUE_CAUSE",  # 4  complément causal ("suite à…")
    "APPOS",          # 5  apposition
    "NONE",           # 6  pas d'argument verbal direct
]
ROLE2ID      = {x: i for i, x in enumerate(ROLE_LABELS)}
ID2ROLE      = {i: x for x, i in ROLE2ID.items()}
NUM_ROLE     = len(ROLE_LABELS)
ROLE_NONE_ID = ROLE2ID["NONE"]   # = 6

# ─────────────────────────────────────────────────────────────
# VOICE LABELS  (prédit sur les verb_trigger)
# ─────────────────────────────────────────────────────────────

VOICE_LABELS  = ["active", "passive"]
VOICE2ID      = {x: i for i, x in enumerate(VOICE_LABELS)}
ID2VOICE      = {i: x for x, i in VOICE2ID.items()}
NUM_VOICE     = len(VOICE_LABELS)
VOICE_NONE_ID = NUM_VOICE   # sentinel

# ─────────────────────────────────────────────────────────────
# CERTAINTY LABELS  (prédit sur les verb_trigger)
# ─────────────────────────────────────────────────────────────

CERTAINTY_LABELS  = ["certain", "modal", "denied"]
CERTAINTY2ID      = {x: i for i, x in enumerate(CERTAINTY_LABELS)}
ID2CERTAINTY      = {i: x for x, i in CERTAINTY2ID.items()}
NUM_CERTAINTY     = len(CERTAINTY_LABELS)
CERTAINTY_NONE_ID = NUM_CERTAINTY   # sentinel

# ─────────────────────────────────────────────────────────────
# MORPHO LABELS  (gender + number + person)
# gender/number : sur spans NER PER/ORG/EVENT + pronoms
# person        : sur pronoms uniquement
# ─────────────────────────────────────────────────────────────

GENDER_LABELS  = ["M", "F", "N"]    # 0 Masc, 1 Fem, 2 Neutre
GENDER2ID      = {x: i for i, x in enumerate(GENDER_LABELS)}
# Compat anciens labels Stanza
GENDER2ID["Masc"] = GENDER2ID["M"]
GENDER2ID["Fem"]  = GENDER2ID["F"]
ID2GENDER      = {0: "M", 1: "F", 2: "N"}
NUM_GENDER     = 3
GENDER_NONE_ID = NUM_GENDER   # sentinel = 3

NUMBER_LABELS  = ["SG", "PL"]
NUMBER2ID      = {x: i for i, x in enumerate(NUMBER_LABELS)}
# Compat anciens labels Stanza
NUMBER2ID["Sing"] = NUMBER2ID["SG"]
NUMBER2ID["Plur"] = NUMBER2ID["PL"]
ID2NUMBER      = {0: "SG", 1: "PL"}
NUM_NUMBER     = 2
NUMBER_NONE_ID = NUM_NUMBER   # sentinel = 2

PERSON_LABELS  = ["1", "2", "3"]
PERSON2ID      = {x: i for i, x in enumerate(PERSON_LABELS)}
ID2PERSON      = {i: x for x, i in PERSON2ID.items()}
NUM_PERSON     = len(PERSON_LABELS)
PERSON_NONE_ID = NUM_PERSON   # sentinel = 3

# ─────────────────────────────────────────────────────────────
# Compat aliases (pour ne pas casser les imports existants)
# ─────────────────────────────────────────────────────────────
SVO_LABELS   = SYN_LABELS    # alias
SVO2ID       = SYN2ID
ID2SVO       = ID2SYN
NUM_SVO      = NUM_SYN
SVO_NONE_ID  = SYN_NONE_ID
ALL_SVO_LABELS = ALL_SYN_LABELS

