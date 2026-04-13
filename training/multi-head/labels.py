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
]

FINE2ID = {x: i for i, x in enumerate(FINE_LABELS)}
ID2FINE = {i: x for x, i in FINE2ID.items()}

NUM_FINE = len(FINE_LABELS)

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
    "NONE",     # 6
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
        FINE2ID["hint_quantity"],
    ],
}

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