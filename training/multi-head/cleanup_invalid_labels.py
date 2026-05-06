#!/usr/bin/env python3
"""
Supprime des splits train/val/test tous les spans dont le label
n'est pas dans FINE_LABELS ni SYN_LABELS (= labels inconnus / supprimés).
"""
import json, shutil
from collections import Counter
from pathlib import Path

from labels import FINE_LABELS, SYN_LABELS

DATA_DIR = Path("data")
VALID_LABELS = set(FINE_LABELS) | set(SYN_LABELS)

# Labels à remapper vers un équivalent valide
# (artefacts de versions antérieures ou labels renommés)
REMAP = {
    "hint_concept":       "hint_notion",      # v7.0 fallback : hint_concept → hint_notion
    "hint_quantity":      "hint_measure",     # hint_quantity supprimé → measure (fallback générique VALUE)
    "hint_actor":         "hint_inst_role",   # autorités/gouvernements → institution générique
    "hint_org_role":      "hint_inst_role",   # commission/service/cabinet → institution générique
    "hint_org_component": "hint_inst_role",   # comité/inspection → composante fonctionnelle générique
}
# Labels à supprimer (hint_concept : trop hétérogène, pas de fallback safe v7.0)
DROP = {
    "hint_classification",
    "hint_cause",
    "hint_document_type",
    "hint_issue",
    "hint_location",
    "hint_org_generic",    # syndicat/entreprise/presse : pas d'équivalent dans la taxo (×25)
}

def load_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Labels valides : {len(VALID_LABELS)}")
print()

total_dropped = Counter()

for split in ["train", "val", "test"]:
    path = DATA_DIR / f"{split}.jsonl"
    rows = load_jsonl(path)

    # Backup
    bak = DATA_DIR / f"{split}.jsonl.pre_cleanup"
    if not bak.exists():
        shutil.copy2(path, bak)

    n_spans_before = sum(len(r.get("spans", [])) for r in rows)
    dropped_here = Counter()
    remapped_here = Counter()

    remapped_here = Counter()
    for row in rows:
        clean_spans = []
        for sp in row.get("spans", []):
            lbl = sp.get("label", "")
            if lbl in REMAP:
                sp = dict(sp)
                sp["label"] = REMAP[lbl]
                remapped_here[f"{lbl} → {REMAP[lbl]}"] += 1
                clean_spans.append(sp)
            elif lbl in DROP:
                dropped_here[lbl] += 1
            elif lbl in VALID_LABELS:
                clean_spans.append(sp)
            else:
                dropped_here[lbl] += 1  # inconnu non mappé
        row["spans"] = clean_spans

    n_spans_after = sum(len(r.get("spans", [])) for r in rows)
    write_jsonl(path, rows)

    print(f"  {split:<6}  {n_spans_before:>7} → {n_spans_after:>7} spans")
    for key, cnt in sorted(remapped_here.items(), key=lambda x: -x[1]):
        print(f"           REMAP  {key}  x{cnt}")
    for lbl, cnt in sorted(dropped_here.items(), key=lambda x: -x[1]):
        print(f"           DROP   {lbl:<30} x{cnt}")
    total_dropped.update(dropped_here)

print()
print(f"{'='*60}")
print(f"TOTAL SPANS SUPPRIMÉS  : {sum(total_dropped.values())}")
print(f"{'='*60}")
for lbl, cnt in sorted(total_dropped.items(), key=lambda x: -x[1]):
    print(f"  DROP  {lbl:<35} {cnt:>6}")

