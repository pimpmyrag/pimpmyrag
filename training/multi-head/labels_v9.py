# labels_v9.py
# ─────────────────────────────────────────────────────────────
# SHIM de compatibilité — la taxonomie v9 (34 fine / 8 coarse + 5 attributs)
# est désormais DÉFINIE dans labels.py (source unique). Ce module ré-exporte
# tout pour ne pas casser les imports existants `from labels_v9 import ...`.
# ─────────────────────────────────────────────────────────────
from labels import *  # noqa: F401,F403
from labels import (  # ré-exports explicites (attributs v9)
    to_v9_fine, LEGACY_TO_V9_FINE, derive_attributes,
    ANIMACY_LABELS, ANIMACY2ID, NUM_ANIMACY, ANIMACY_NONE_ID,
    LIVING_LABELS, LIVING2ID, NUM_LIVING, LIVING_NONE_ID,
    ABSTRACT_LABELS, ABSTRACT2ID, NUM_ABSTRACT, ABSTRACT_NONE_ID,
    DYNAMICITY_LABELS, DYNAMICITY2ID, NUM_DYNAMICITY, DYNAMICITY_NONE_ID,
    WORK_LABELS, WORK2ID, NUM_WORK, WORK_NONE_ID,
)

if __name__ == "__main__":
    import labels as L
    assert L.NUM_FINE == 34 and len(L.COARSE_LABELS) == 9
    print(f"✅ labels_v9 shim OK — NUM_FINE={L.NUM_FINE} NUM_COARSE={len(L.COARSE_LABELS)}")
