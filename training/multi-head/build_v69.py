"""
Version 6.9 : Relabélisation fine hint_concept → 7 sous-types + fallback

Provenance :
  v6.7  : fusionhint_concept_named → hint_concept (via remap_concept_named_to_concept.py)
  v6.8  : relabélisation Claude Haiku Batch API sur hint_concept → 7 sous-types
  v6.8p : corrections rule-based post-Haiku
  v6.9  : version finale pour training

Nouveaux labels ABSTRACT :
  - hint_rule         : règle, procédure, norme, protocole
  - hint_doctrine     : doctrine, idéologie, courant de pensée, théorie (nommée ou non)
  - hint_state        : état, condition, situation abstraite
  - hint_notion       : notion, concept abstrait pur, valeur
  - hint_work_generic : production culturelle générique sans titre
  - hint_field        : domaine / secteur d'activité
  - hint_process      : processus socio-économique continu
  - hint_concept      : fallback résiduel (10% des spans)

total ABSTRACT : 14 167 spans (vs 12 379 en v6.7)
"""

import json
from pathlib import Path
from collections import Counter

DATA = Path("data")
VERSION = "v6.9"

CONCEPT_LABELS = {
    "hint_rule", "hint_doctrine", "hint_state", "hint_notion",
    "hint_work_generic", "hint_field", "hint_process", "hint_concept"
}

print(__doc__)

for split in ["train", "val", "test"]:
    path = DATA / f"{split}_{VERSION}.jsonl"
    if not path.exists():
        print(f"⚠️  {path} absent")
        continue

    concept_counts = Counter()
    total_spans = 0
    n_items = 0

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            n_items += 1
            for sp in item.get("spans", []):
                total_spans += 1
                lbl = sp.get("label", "")
                if lbl in CONCEPT_LABELS:
                    concept_counts[lbl] += 1

    total_concept = sum(concept_counts.values())
    print(f"\n{'='*60}")
    print(f"  {split.upper():5} {VERSION}  —  {n_items} phrases | {total_concept} spans ABSTRACT")
    print(f"{'='*60}")
    for lbl, cnt in concept_counts.most_common():
        pct = cnt / max(total_concept, 1) * 100
        bar = "█" * int(pct / 3)
        print(f"  {lbl:<28} {cnt:>5}  {pct:5.1f}%  {bar}")

print(f"\n✅ Version {VERSION} prête pour training")

