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
    "hint_inst_name",        # 5  institution NOMMÉE (sigle ou nom propre qualifié)
    "hint_gpe",              # 6
    "hint_fac_name",         # 7
    "hint_loc_generic",      # 8
    "hint_weapon",           # 9
    "hint_vehicle",          # 10
    "hint_substance",        # 11
    "hint_food",             # 12
    "hint_infra",            # 13
    "hint_tool",             # 14
    "hint_object_generic",   # 15  ← v9 : absorbe hint_object_name
    "hint_event_nominal",    # 16  ← v9 : absorbe hint_state (attribut dynamicity=stative)
    "hint_event_named",      # 17
    "hint_time_date",        # 18
    "hint_time_clock",       # 19
    "hint_time_duration",    # 20
    "hint_measure",          # 21  ← v9 : absorbe hint_rate
    "hint_percentage",       # 22
    "hint_count",            # 23
    "hint_money",            # 24
    "hint_work_of_art",      # 25  ← v9 : absorbe hint_work_generic (attribut work=1)
    "hint_law",              # 26
    "hint_document",         # 27  rapport, lettre, communiqué, données, contrat…
    "hint_disease",          # 28  coarse EVENT (condition), dynamicity=stative
    "hint_language",         # 29
    "hint_inst_role",        # 30  institution GÉNÉRIQUE (gouvernement, police, armée…)
    "hint_animal",           # 31  vivant non-humain (v9 : absorbe hint_vegetal ;
                             #      animacy distingue animal[animate]/plante[inanimate])
    "hint_notion",           # 32  ← v9 : absorbe hint_doctrine
    "hint_field",            # 33  domaine / secteur d'activité (santé, éducation…)
    # ── v9.0 : taxonomie réduite 40→34. Fusions : object_name→object_generic,
    #    rate→measure, work_generic→work_of_art, state→event_nominal,
    #    doctrine→notion, vegetal→animal. BIO/ABSTRACT/WORK deviennent des
    #    ATTRIBUTS transverses (animacy/living/abstract/dynamicity/work).
]

FINE2ID = {x: i for i, x in enumerate(FINE_LABELS)}
ID2FINE = {i: x for x, i in FINE2ID.items()}

NUM_FINE = len(FINE_LABELS)

# Sentinel pour les spans négatifs (pas un vrai label fine)
FINE_NONE_ID = NUM_FINE  # = 34, hors range [0..33]

# ─────────────────────────────────────────────────────────────
# GROUPES FINE LABELS : CONCRETE vs ABSTRACT
# CONCRETE : entités nommées prototypiques (NER classique) — faciles, fréquentes
# ABSTRACT : labels génériques / sémantiques — plus difficiles, moins distinctifs
# ─────────────────────────────────────────────────────────────
FINE_CONCRETE_LABELS = [
    "hint_person_name",    # nom propre de personne
    "hint_org_name",       # nom d'organisation
    "hint_inst_name",      # institution nommée (sigle, nom propre qualifié)
    "hint_gpe",            # entité géopolitique
    "hint_fac_name",       # facility nommée
    "hint_event_named",    # événement nommé
    "hint_time_date",      # date
    "hint_time_clock",     # heure
    "hint_time_duration",  # durée
    "hint_measure",        # mesure
    "hint_percentage",     # pourcentage
    "hint_count",          # compte numérique
    "hint_money",          # montant monétaire
    "hint_work_of_art",    # œuvre nommée
    "hint_law",            # loi / texte officiel nommé
]

FINE_ABSTRACT_LABELS = [
    "hint_person_role",    # rôle (fonctionnel, non-nom propre)
    "hint_norp",           # nationalité / religion / politique
    "hint_group_role",     # groupe générique (l opposition, les civils…)
    "hint_loc_generic",    # lieu générique (non nommé)
    "hint_infra",          # infrastructure générique
    "hint_weapon",         # arme générique
    "hint_vehicle",        # véhicule générique
    "hint_substance",      # substance (pétrole, gaz…)
    "hint_food",           # nourriture
    "hint_tool",           # outil
    "hint_object_generic", # objet générique
    "hint_event_nominal",  # événement nominalé / non nommé
    "hint_document",       # document générique (rapport, lettre…)
    "hint_disease",        # maladie
    "hint_language",       # langue
    "hint_inst_role",      # institution générique (gouvernement, armée…)
    "hint_notion",         # notion / concept abstrait
    "hint_field",          # domaine d'activité
]

FINE_CONCRETE_IDS: frozenset[int] = frozenset(FINE2ID[l] for l in FINE_CONCRETE_LABELS if l in FINE2ID)
FINE_ABSTRACT_IDS: frozenset[int] = frozenset(FINE2ID[l] for l in FINE_ABSTRACT_LABELS if l in FINE2ID)

# ─────────────────────────────────────────────────────────────
# COARSE LABELS
# ─────────────────────────────────────────────────────────────

COARSE_LABELS = [
    "PER",       # 0
    "LOC",       # 1
    "ORG",       # 2
    "TIME",      # 3
    "VALUE",     # 4
    "OBJECT",    # 5  artefacts physiques + vivant physique (animal/plante)
    "EVENT",     # 6  événements + états/conditions (disease)
    "CONCEPT",   # 7  informationnel/abstrait (v9 : ex-ABSTRACT + ex-WORK)
    "NONE",      # 8
    # ── v9.0 : BIO et ABSTRACT supprimés (→ attributs), WORK fusionné dans CONCEPT.
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
# v9.0 — MIGRATION anciens fines → v9  +  ATTRIBUTS TRANSVERSES
# ─────────────────────────────────────────────────────────────
# 6 fusions v8.x → v9 (les attributs portent l'info fusionnée, sans perte).
LEGACY_TO_V9_FINE = {
    "hint_state":        "hint_event_nominal",
    "hint_doctrine":     "hint_notion",
    "hint_work_generic": "hint_work_of_art",
    "hint_object_name":  "hint_object_generic",
    "hint_rate":         "hint_measure",
    "hint_vegetal":      "hint_animal",
}

def to_v9_fine(label: str) -> str:
    """Mappe un fine label (v8.x ou v9) vers le fine v9 canonique."""
    return LEGACY_TO_V9_FINE.get(label, label)

# --- 5 attributs binaires (sentinel NONE = nombre de classes) -----------------
ANIMACY_LABELS    = ["inanimate", "animate"]
ANIMACY2ID        = {x: i for i, x in enumerate(ANIMACY_LABELS)}
NUM_ANIMACY       = len(ANIMACY_LABELS)
ANIMACY_NONE_ID   = NUM_ANIMACY
_ANIMATE_ORIG     = {"hint_person_name", "hint_person_role", "hint_norp",
                     "hint_group_role", "hint_animal"}

LIVING_LABELS     = ["non_living", "living"]
LIVING2ID         = {x: i for i, x in enumerate(LIVING_LABELS)}
NUM_LIVING        = len(LIVING_LABELS)
LIVING_NONE_ID    = NUM_LIVING
_LIVING_ORIG      = _ANIMATE_ORIG | {"hint_vegetal"}

ABSTRACT_LABELS   = ["concrete", "abstract"]
ABSTRACT2ID       = {x: i for i, x in enumerate(ABSTRACT_LABELS)}
NUM_ABSTRACT      = len(ABSTRACT_LABELS)
ABSTRACT_NONE_ID  = NUM_ABSTRACT
_ABSTRACT_FINES   = {"hint_work_of_art", "hint_law", "hint_document", "hint_language",
                     "hint_notion", "hint_field", "hint_disease"}

DYNAMICITY_LABELS   = ["stative", "dynamic"]
DYNAMICITY2ID       = {x: i for i, x in enumerate(DYNAMICITY_LABELS)}
NUM_DYNAMICITY      = len(DYNAMICITY_LABELS)
DYNAMICITY_NONE_ID  = NUM_DYNAMICITY
_STATIVE_FINES      = {"hint_disease"}   # + hint_state via override d'origine

WORK_LABELS       = ["non_work", "work"]
WORK2ID           = {x: i for i, x in enumerate(WORK_LABELS)}
NUM_WORK          = len(WORK_LABELS)
WORK_NONE_ID      = NUM_WORK
_WORK_FINES       = {"hint_work_of_art", "hint_law", "hint_document"}


def derive_attributes(legacy_label: str) -> dict[str, int]:
    """
    Dérive les 5 attributs v9 depuis le label d'ORIGINE (v8.x ou v9), sans perte.
    animacy/living : intrinsèques → basés sur le label d'ORIGINE (une plante
    fusionnée dans hint_animal reste inanimate + living). abstract/work : label v9.
    dynamicity : sur EVENT uniquement (ex-state → stative), sinon NONE.
    """
    v9 = to_v9_fine(legacy_label)
    is_event = fine_label_to_coarse_id(v9) == COARSE2ID["EVENT"]
    stative = (legacy_label == "hint_state") or (v9 in _STATIVE_FINES)
    is_abstract = (v9 in _ABSTRACT_FINES) or (legacy_label == "hint_state")
    return {
        "animacy":    ANIMACY2ID["animate"] if legacy_label in _ANIMATE_ORIG else ANIMACY2ID["inanimate"],
        "living":     LIVING2ID["living"] if legacy_label in _LIVING_ORIG else LIVING2ID["non_living"],
        "abstract":   ABSTRACT2ID["abstract"] if is_abstract else ABSTRACT2ID["concrete"],
        "dynamicity": (DYNAMICITY2ID["stative"] if stative else DYNAMICITY2ID["dynamic"]) if is_event else DYNAMICITY_NONE_ID,
        "work":       WORK2ID["work"] if v9 in _WORK_FINES else WORK2ID["non_work"],
    }


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
    "SUBJECT",              # 0  sujet grammatical
    "OBJECT",               # 1  objet direct
    "OBLIQUE",              # 2  complément circonstanciel générique (lieu/temps/manière)
    "OBLIQUE_AGENT",        # 3  agent sémantique en passif ("par la police")
    "OBLIQUE_CAUSE",        # 4  complément causal ("suite à…", "grâce à…")
    "APPOS",                # 5  apposition
    "NONE",                 # 6  pas d'argument verbal direct
    "OBLIQUE_ADVERSARY",    # 7  adversaire ("contre la France", "face à l'opposition")
    "OBLIQUE_BENEFICIARY",  # 8  bénéficiaire ("pour les victimes", "en faveur de…")
    "OBLIQUE_COMITATIVE",   # 9  comitatif ("avec ses alliés", "aux côtés de…")
    "OBLIQUE_DOMAIN",       # 10 domaine / thème ("sur l'éducation", "concernant…")
    "OBLIQUE_SOURCE",       # 11 source épistémique ("selon Reuters", "d'après…")
]
ROLE2ID      = {x: i for i, x in enumerate(ROLE_LABELS)}
ID2ROLE      = {i: x for x, i in ROLE2ID.items()}
NUM_ROLE     = len(ROLE_LABELS)
ROLE_NONE_ID = ROLE2ID["NONE"]   # = 6

# ─────────────────────────────────────────────────────────────
# ROLE COARSE  (analogue au coarse NER — force la discrimination
# SUBJ/OBJ/OBLIQ avant d'apprendre les sous-types fins)
# ─────────────────────────────────────────────────────────────
ROLE_COARSE_LABELS = [
    "SUBJ",   # 0  ← SUBJECT
    "OBJ",    # 1  ← OBJECT
    "OBLIQ",  # 2  ← OBLIQUE + tous les OBLIQUE_* collapsés
    "APPOS",  # 3  ← APPOS (relation appositionnelle explicite)
    "OTHER",  # 4  ← spans NER gold sans svo_role annoté (cascade SVO→NER)
]
ROLE_COARSE2ID      = {x: i for i, x in enumerate(ROLE_COARSE_LABELS)}
ID2ROLE_COARSE      = {i: x for x, i in ROLE_COARSE2ID.items()}
NUM_ROLE_COARSE     = len(ROLE_COARSE_LABELS)       # = 5
ROLE_COARSE_NONE_ID = NUM_ROLE_COARSE  # = 5  (sentinel : spans négatifs, verb_trigger)
ROLE_COARSE_OTHER_ID = ROLE_COARSE2ID["OTHER"]  # = 4  (NER gold sans svo_role — poids réduit dans loss)

# Mapping fine_role_id → coarse_role_id  (construit à partir de ROLE_LABELS)
_RC = ROLE_COARSE2ID
ROLE_FINE_TO_COARSE_ID: dict[int, int] = {
    ROLE2ID["SUBJECT"]:            _RC["SUBJ"],
    ROLE2ID["OBJECT"]:             _RC["OBJ"],
    ROLE2ID["OBLIQUE"]:            _RC["OBLIQ"],
    ROLE2ID["OBLIQUE_AGENT"]:      _RC["OBLIQ"],
    ROLE2ID["OBLIQUE_CAUSE"]:      _RC["OBLIQ"],
    ROLE2ID["OBLIQUE_ADVERSARY"]:  _RC["OBLIQ"],
    ROLE2ID["OBLIQUE_BENEFICIARY"]:_RC["OBLIQ"],
    ROLE2ID["OBLIQUE_COMITATIVE"]: _RC["OBLIQ"],
    ROLE2ID["OBLIQUE_DOMAIN"]:     _RC["OBLIQ"],
    ROLE2ID["OBLIQUE_SOURCE"]:     _RC["OBLIQ"],
    ROLE2ID["APPOS"]:              _RC["APPOS"],
    ROLE2ID["NONE"]:               _RC["OTHER"],  # NER gold sans svo_role → OTHER (cascade, poids minimal dans loss)
}

# ─────────────────────────────────────────────────────────────
# ROLE OBLIQUE FINE  (tête fine conditionnée sur OBLIQ coarse)
# Supervise uniquement les spans où role_coarse = OBLIQ
# ─────────────────────────────────────────────────────────────
ROLE_OBLIQUE_LABELS = [
    "OBLIQUE",              # 0  oblique générique (sous-type non spécifié)
    "OBLIQUE_AGENT",        # 1  agent de passif ("par la France", "de la part de…")
    "OBLIQUE_CAUSE",        # 2  cause ("en raison de", "suite à…")
    "OBLIQUE_ADVERSARY",    # 3  adversaire ("contre", "face à…")
    "OBLIQUE_BENEFICIARY",  # 4  bénéficiaire ("pour", "en faveur de…")
    "OBLIQUE_COMITATIVE",   # 5  comitatif ("avec", "aux côtés de…")
    "OBLIQUE_DOMAIN",       # 6  domaine / thème ("sur", "concernant…")
    "OBLIQUE_SOURCE",       # 7  source épistémique ("selon", "d'après…")
    "OBLIQUE_TIME",         # 8  NEW — inféré de hint_time_* en position oblique
    "OBLIQUE_LOC",          # 9  NEW — inféré de hint_loc_*/hint_gpe/hint_fac_name en oblique
]
ROLE_OBLIQUE2ID      = {x: i for i, x in enumerate(ROLE_OBLIQUE_LABELS)}
ID2ROLE_OBLIQUE      = {i: x for x, i in ROLE_OBLIQUE2ID.items()}
NUM_ROLE_OBLIQUE     = len(ROLE_OBLIQUE_LABELS)
ROLE_OBLIQUE_NONE_ID = NUM_ROLE_OBLIQUE  # = 10 (sentinel : non-oblique ou non supervisé)

# Mapping role fine → oblique fine id (pour les spans OBLIQUE_*)
ROLE_TO_OBLIQUE_ID: dict[int, int] = {
    ROLE2ID["OBLIQUE"]:            ROLE_OBLIQUE2ID["OBLIQUE"],
    ROLE2ID["OBLIQUE_AGENT"]:      ROLE_OBLIQUE2ID["OBLIQUE_AGENT"],
    ROLE2ID["OBLIQUE_CAUSE"]:      ROLE_OBLIQUE2ID["OBLIQUE_CAUSE"],
    ROLE2ID["OBLIQUE_ADVERSARY"]:  ROLE_OBLIQUE2ID["OBLIQUE_ADVERSARY"],
    ROLE2ID["OBLIQUE_BENEFICIARY"]:ROLE_OBLIQUE2ID["OBLIQUE_BENEFICIARY"],
    ROLE2ID["OBLIQUE_COMITATIVE"]: ROLE_OBLIQUE2ID["OBLIQUE_COMITATIVE"],
    ROLE2ID["OBLIQUE_DOMAIN"]:     ROLE_OBLIQUE2ID["OBLIQUE_DOMAIN"],
    ROLE2ID["OBLIQUE_SOURCE"]:     ROLE_OBLIQUE2ID["OBLIQUE_SOURCE"],
    # OBLIQUE_TIME et OBLIQUE_LOC inférés dans build_multitask_dataset.py via NER label
}

# Labels NER qui permettent d'inférer le type oblique
NER_TIME_LABELS = {"hint_time_date", "hint_time_clock", "hint_time_duration"}
NER_LOC_LABELS  = {"hint_loc_generic", "hint_gpe", "hint_fac_name", "hint_infra"}

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

GENDER_LABELS  = ["M", "F"]         # 0 Masc, 1 Fem  (N supprimé : <0.1% du dataset, impossble à apprendre)
GENDER2ID      = {x: i for i, x in enumerate(GENDER_LABELS)}
# Compat anciens labels Stanza
GENDER2ID["Masc"] = GENDER2ID["M"]
GENDER2ID["Fem"]  = GENDER2ID["F"]
# N=Neutre → NONE (exclu de la loss et du scorer)
GENDER2ID["N"]    = len(GENDER_LABELS)   # sera mappé à NONE_ID
ID2GENDER      = {0: "M", 1: "F"}
NUM_GENDER     = 2
GENDER_NONE_ID = NUM_GENDER   # sentinel = 2

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
# ROLE COARSE DÉRIVÉE DEPUIS ROLE_HEAD (12 labels)
# Groupes d'IDs pour agrégation logsumexp : 0=SUBJ 1=OBJ 2=OBLIQ 3=APPOS
# Ordre aligné sur ROLE_COARSE_LABELS[:4]
# ─────────────────────────────────────────────────────────────
ROLE_DERIVED_SUBJ_IDS   = [ROLE2ID["SUBJECT"]]
ROLE_DERIVED_OBJ_IDS    = [ROLE2ID["OBJECT"]]
ROLE_DERIVED_OBLIQ_IDS  = [
    ROLE2ID["OBLIQUE"],
    ROLE2ID["OBLIQUE_AGENT"],
    ROLE2ID["OBLIQUE_CAUSE"],
    ROLE2ID["OBLIQUE_ADVERSARY"],
    ROLE2ID["OBLIQUE_BENEFICIARY"],
    ROLE2ID["OBLIQUE_COMITATIVE"],
    ROLE2ID["OBLIQUE_DOMAIN"],
    ROLE2ID["OBLIQUE_SOURCE"],
]
ROLE_DERIVED_APPOS_IDS  = [ROLE2ID["APPOS"]]

# ─────────────────────────────────────────────────────────────
# Compat aliases (pour ne pas casser les imports existants)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# COARSE DÉRIVÉE DEPUIS ROLE_HEAD (12 labels)
# Groupes d'IDs pour agrégation logsumexp : 0=SUBJ 1=OBJ 2=OBLIQ 3=APPOS
# Ordre aligné sur ROLE_COARSE_LABELS[:4]
# ─────────────────────────────────────────────────────────────
ROLE_DERIVED_SUBJ_IDS   = [ROLE2ID["SUBJECT"]]
ROLE_DERIVED_OBJ_IDS    = [ROLE2ID["OBJECT"]]
ROLE_DERIVED_OBLIQ_IDS  = [
    ROLE2ID["OBLIQUE"],
    ROLE2ID["OBLIQUE_AGENT"],
    ROLE2ID["OBLIQUE_CAUSE"],
    ROLE2ID["OBLIQUE_ADVERSARY"],
    ROLE2ID["OBLIQUE_BENEFICIARY"],
    ROLE2ID["OBLIQUE_COMITATIVE"],
    ROLE2ID["OBLIQUE_DOMAIN"],
    ROLE2ID["OBLIQUE_SOURCE"],
]
ROLE_DERIVED_APPOS_IDS  = [ROLE2ID["APPOS"]]
SVO_LABELS   = SYN_LABELS    # alias
SVO2ID       = SYN2ID
ID2SVO       = ID2SYN
NUM_SVO      = NUM_SYN
SVO_NONE_ID  = SYN_NONE_ID
ALL_SVO_LABELS = ALL_SYN_LABELS

# ─────────────────────────────────────────────────────────────
# VERBFAM LABELS  (prédit sur verb_trigger uniquement)
# ─────────────────────────────────────────────────────────────

VERB_FAMILY_LABELS = [
    "Causality",     # 0
    "Cognition",     # 1
    "Communication", # 2
    "Conflict",      # 3
    "Movement",      # 4
    "OTHER",         # 5
    "Perception",    # 6
    "Possession",    # 7
    "Relation",      # 8
    "Social",        # 9
    "State_Change",  # 10
    "Temporal",      # 11
]
VERB_FAMILY2ID      = {x: i for i, x in enumerate(VERB_FAMILY_LABELS)}
ID2VERB_FAMILY      = {i: x for x, i in VERB_FAMILY2ID.items()}
NUM_VERB_FAMILY     = len(VERB_FAMILY_LABELS)
VERB_FAMILY_NONE_ID = NUM_VERB_FAMILY   # sentinel = 12

VERB_FAMILY_FINE_LABELS = [
    "Achat",        # 0
    "Annonce",      # 1
    "Appartenance", # 2
    "Cognitive",    # 3
    "Combat",       # 4
    "Concerne",     # 5
    "Contenu",      # 6
    "Creation",     # 7
    "Croyance",     # 8
    "Debut",        # 9
    "Decision",     # 10
    "Demande",      # 11
    "Deplacement",  # 12
    "Destruction",  # 13
    "Don",          # 14
    "Duree",        # 15
    "Ecrit",        # 16
    "Election",     # 17
    "Fin",          # 18
    "Intention",    # 19
    "Jugement",     # 20
    "Legislation",  # 21
    "Lien",         # 22
    "Negatif",      # 23
    "Negociation",  # 24
    "Nomination",   # 25
    "OTHER",        # 26
    "Opposition",   # 27
    "Permission",   # 28
    "Positif",      # 29
    "Reponse",      # 30
    "Savoir",       # 31
    "Sensorielle",  # 32
    "Transformation",# 33
    "Transport",    # 34
    "Vente",        # 35
    "Visuelle",     # 36
    "Voyage",       # 37
]
VERB_FAMILY_FINE2ID      = {x: i for i, x in enumerate(VERB_FAMILY_FINE_LABELS)}
ID2VERB_FAMILY_FINE      = {i: x for x, i in VERB_FAMILY_FINE2ID.items()}
NUM_VERB_FAMILY_FINE     = len(VERB_FAMILY_FINE_LABELS)
VERB_FAMILY_FINE_NONE_ID = NUM_VERB_FAMILY_FINE   # sentinel = 38

# Mapping family coarse → fine  (pour le soft mask)
VERB_FAMILY_TO_FINE: dict[int, list[int]] = {
    VERB_FAMILY2ID["Causality"]:     [VERB_FAMILY_FINE2ID[x] for x in ["Negatif", "Positif", "Transformation", "Destruction", "Creation"]],
    VERB_FAMILY2ID["Cognition"]:     [VERB_FAMILY_FINE2ID[x] for x in ["Cognitive", "Croyance", "Decision", "Intention", "Savoir", "Jugement"]],
    VERB_FAMILY2ID["Communication"]: [VERB_FAMILY_FINE2ID[x] for x in ["Annonce", "Demande", "Ecrit", "Reponse", "Contenu", "Negociation"]],
    VERB_FAMILY2ID["Conflict"]:      [VERB_FAMILY_FINE2ID[x] for x in ["Combat", "Opposition", "Negatif"]],
    VERB_FAMILY2ID["Movement"]:      [VERB_FAMILY_FINE2ID[x] for x in ["Deplacement", "Transport", "Voyage"]],
    VERB_FAMILY2ID["OTHER"]:         [VERB_FAMILY_FINE2ID[x] for x in ["OTHER", "Concerne", "Lien"]],
    VERB_FAMILY2ID["Perception"]:    [VERB_FAMILY_FINE2ID[x] for x in ["Sensorielle", "Visuelle", "Cognitive"]],
    VERB_FAMILY2ID["Possession"]:    [VERB_FAMILY_FINE2ID[x] for x in ["Achat", "Appartenance", "Don", "Vente"]],
    VERB_FAMILY2ID["Relation"]:      [VERB_FAMILY_FINE2ID[x] for x in ["Appartenance", "Lien", "Concerne"]],
    VERB_FAMILY2ID["Social"]:        [VERB_FAMILY_FINE2ID[x] for x in ["Election", "Legislation", "Nomination", "Negociation", "Permission"]],
    VERB_FAMILY2ID["State_Change"]:  [VERB_FAMILY_FINE2ID[x] for x in ["Debut", "Fin", "Transformation", "Nomination"]],
    VERB_FAMILY2ID["Temporal"]:      [VERB_FAMILY_FINE2ID[x] for x in ["Debut", "Duree", "Fin"]],
}

def build_verb_family_fine_mask() -> "torch.Tensor":
    import torch
    mask = torch.zeros(NUM_VERB_FAMILY, NUM_VERB_FAMILY_FINE, dtype=torch.bool)
    for fam_id, fine_ids in VERB_FAMILY_TO_FINE.items():
        for f in fine_ids:
            mask[fam_id, f] = True
    return mask

VERB_FAMILY_FINE_MASK = build_verb_family_fine_mask()

# ─────────────────────────────────────────────────────────────
# VERB POLARITY  (prédit sur verb_trigger)
# ─────────────────────────────────────────────────────────────
VERB_POLARITY_LABELS  = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
VERB_POLARITY2ID      = {x: i for i, x in enumerate(VERB_POLARITY_LABELS)}
ID2VERB_POLARITY      = {i: x for x, i in VERB_POLARITY2ID.items()}
NUM_VERB_POLARITY     = len(VERB_POLARITY_LABELS)
VERB_POLARITY_NONE_ID = NUM_VERB_POLARITY   # sentinel = 3

# ─────────────────────────────────────────────────────────────
# VERB ASPECT  (prédit sur verb_trigger)
# ─────────────────────────────────────────────────────────────
VERB_ASPECT_LABELS  = ["DURATIF", "PONCTUEL"]
VERB_ASPECT2ID      = {x: i for i, x in enumerate(VERB_ASPECT_LABELS)}
ID2VERB_ASPECT      = {i: x for x, i in VERB_ASPECT2ID.items()}
NUM_VERB_ASPECT     = len(VERB_ASPECT_LABELS)
VERB_ASPECT_NONE_ID = NUM_VERB_ASPECT   # sentinel = 2

# ─────────────────────────────────────────────────────────────
# VERB SOURCE  (prédit sur verb_trigger)
# ─────────────────────────────────────────────────────────────
VERB_SOURCE_LABELS  = ["DIRECT", "HYPOTHETICAL", "REPORTED"]
VERB_SOURCE2ID      = {x: i for i, x in enumerate(VERB_SOURCE_LABELS)}
ID2VERB_SOURCE      = {i: x for x, i in VERB_SOURCE2ID.items()}
NUM_VERB_SOURCE     = len(VERB_SOURCE_LABELS)
VERB_SOURCE_NONE_ID = NUM_VERB_SOURCE   # sentinel = 3


# ─────────────────────────────────────────────────────────────
# NOMINAL RELATION LABELS  (relation nominale parent→enfant)
# Prédit sur chaque span nominal enfant (NER boundary=1)
# quand un parent nominal est annoté dans la phrase.
# ─────────────────────────────────────────────────────────────

NOMINAL_RELATION_LABELS = [
    "APPOS",     # 0  apposition rôle/titre + nom propre : "PDG Bernard Arnault"
    "NMOD",      # 1  complément du nom : "action du groupe LVMH"
    "POSS",      # 2  possessif : "son PDG", "sa filiale"
    "AMOD",      # 3  adjectif qualificatif : "fondations solides"
    "COMPOUND",  # 4  nom propre multi-tokens / flat : "Union européenne"
    "SOURCE",    # 5  complément source d'EVENT_NOMINAL Communication : "déclarations du PDG"
    "MEDIUM",    # 6  support de publication : "dans le journal Les Échos"
    "LOC",       # 7  complément nominal locatif
    "TIME",      # 8  complément nominal temporel
    "MISC",      # 9  fallback conservateur
]
NOMINAL_RELATION2ID      = {x: i for i, x in enumerate(NOMINAL_RELATION_LABELS)}
ID2NOMINAL_RELATION      = {i: x for x, i in NOMINAL_RELATION2ID.items()}
NUM_NOMINAL_RELATION     = len(NOMINAL_RELATION_LABELS)   # = 10
NOMINAL_RELATION_NONE_ID = NUM_NOMINAL_RELATION           # = 10 (sentinel NO_PARENT / non-supervisé)


# ─────────────────────────────────────────────────────────────
# SEMANTIC ROLE LABELS  (rôle sémantique fin — Phase 3)
# Prédit sur chaque span NER + pronoms, conditionné sur svo_role.
# Dérivé du mapper f(svo_role, hint_*, verb_family, voice).
# ─────────────────────────────────────────────────────────────
SEMANTIC_ROLE_LABELS = [
    "AGENT",        # 0   initiateur de l'action (sujet actif)
    "PATIENT",      # 1   entité affectée / objet de l'action
    "CONTENT",      # 2   contenu propositionnel (ce qui est dit / pensé)
    "SOURCE",       # 3   source épistémique ("selon X", "d'après X")
    "LOCATION",     # 4   lieu de l'action / destination
    "TEMPORAL",     # 5   ancrage temporel
    "CAUSE",        # 6   cause / déclencheur de l'événement
    "PURPOSE",      # 7   but / intention ("pour X", "afin de X")
    "MEASURE",      # 8   quantité / valeur numérique
    "BENEFICIARY",  # 9   bénéficiaire ("pour X", "en faveur de X")
    "COMITATIVE",   # 10  co-participant ("avec X", "aux côtés de X")
    "ADVERSARY",    # 11  opposant ("contre X", "face à X")
    "DOMAIN",       # 12  domaine / thème ("sur X", "en matière de X")
    "INSTRUMENT",   # 13  moyen / outil ("avec X", "via X", "à l'aide de X")
    "PART_OF",      # 14  inclusion / appartenance ("appartient à X")
    "MEMBER_OF",    # 15  membre d'un ensemble ("fait partie de X")
    "OWNER",        # 16  possesseur ("détenu par X", "propriété de X")
    "IDENTITY",     # 17  apposition identitaire (X = Y)
    "NONE",         # 18  pas de rôle sémantique (span sans gouverneur verbal)
]
SEMANTIC_ROLE2ID      = {x: i for i, x in enumerate(SEMANTIC_ROLE_LABELS)}
ID2SEMANTIC_ROLE      = {i: x for x, i in SEMANTIC_ROLE2ID.items()}
NUM_SEMANTIC_ROLE     = len(SEMANTIC_ROLE_LABELS)   # = 19
SEMANTIC_ROLE_NONE_ID = SEMANTIC_ROLE2ID["NONE"]    # = 18

# Sentinel pour les spans non supervisés (OBLIQUE_UNRESOLVED ou spans syntaxiques)
SEMANTIC_ROLE_SKIP_ID = NUM_SEMANTIC_ROLE           # = 19  (hors range actif)

# Sous-groupes utiles pour le scoring et le masquage de loss
SEMANTIC_ROLE_CORE_IDS = [
    SEMANTIC_ROLE2ID["AGENT"],
    SEMANTIC_ROLE2ID["PATIENT"],
    SEMANTIC_ROLE2ID["CONTENT"],
    SEMANTIC_ROLE2ID["CAUSE"],
    SEMANTIC_ROLE2ID["BENEFICIARY"],
    SEMANTIC_ROLE2ID["LOCATION"],
    SEMANTIC_ROLE2ID["TEMPORAL"],
]


