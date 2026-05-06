#!/usr/bin/env python3
"""
Merge final de toutes les sources pipeline avant envoi en review Claude.

Sources :
  - data/pipeline_generated_filtered.jsonl       (r1 : filtré leakages)
  - data/pipeline_generated_r3_targeted.jsonl    (r3 : labels fragiles)
  - data/mistral_targeted_generations_r2.jsonl   (r2/r4 dont hint_rate)
    → ce fichier contient les raw Mistral (pas encore preannotés par DeBERTa)
    → on filtre pour ne garder que les lignes avec _confirmed_label OU on les
      inclut si elles ont des spans (provenant du pipeline)

Output :
  - data/pipeline_generated_merged_final.jsonl

Usage :  python3 finalize_for_review.py
"""
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data")

SOURCES = [
    DATA_DIR / "pipeline_generated_filtered.jsonl",
    DATA_DIR / "pipeline_generated_r3_targeted.jsonl",
]

# r2 raw : contient hint_rate mais ce sont des raw Mistral (sans spans DeBERTa)
# On cherche d'abord un fichier annoté spécifique
EXTRA_SOURCES = [
    DATA_DIR / "pipeline_generated_r4_hint_rate.jsonl",  # si pipeline a tourné en r4
]

OUT = DATA_DIR / "pipeline_generated_merged_final.jsonl"

print("=" * 70)
print("MERGE FINAL POUR REVIEW")
print("=" * 70)

rows = []
seen: set[str] = set()  # clés de déduplication (text normalisé)
stats_per_source: dict[str, dict] = {}

all_sources = SOURCES + [s for s in EXTRA_SOURCES if s.exists()]

for src in all_sources:
    if not src.exists():
        print(f"  skip {src.name} (absent)")
        continue
    n_before = len(rows)
    n_dup = 0
    n_json_error = 0   # lignes non parsables
    n_no_label = 0     # lignes sans _confirmed_label ni spans
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                n_json_error += 1
                continue
            # Garder seulement les lignes avec _confirmed_label ou spans
            if not row.get("_confirmed_label") and not row.get("spans"):
                n_no_label += 1
                continue
            key = row.get("text", "").lower().strip()
            if not key:
                n_no_label += 1
                continue
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            rows.append(row)
    added = len(rows) - n_before
    stats_per_source[src.name] = {
        "added": added,
        "dup": n_dup,
        "json_error": n_json_error,
        "no_label": n_no_label,
    }
    print(
        f"  {src.name:<50}  +{added:4d}"
        f"  ({n_dup} dup, {n_json_error} json_err, {n_no_label} no_label)"
    )

print(f"\n  Total : {len(rows)} phrases uniques")

# Récapitulatif global des skips
total_dup = sum(s["dup"] for s in stats_per_source.values())
total_json = sum(s["json_error"] for s in stats_per_source.values())
total_no_label = sum(s["no_label"] for s in stats_per_source.values())
print(f"  Skips  : {total_dup} dup  |  {total_json} json_err  |  {total_no_label} no_label")

# Audit rapide
confirmed = Counter(row.get("_confirmed_label", "") for row in rows)
span_labels = Counter()
svo_roles = Counter()
for row in rows:
    for sp in row.get("spans", []):
        lbl = sp.get("label", "")
        if lbl:
            span_labels[lbl] += 1
        role = sp.get("svo_role", "")
        if role and role != "NONE":
            svo_roles[role] += 1

print(f"\n{'=' * 70}")
print("LABELS CONFIRMES")
print("=" * 70)
for lbl, cnt in sorted(confirmed.items(), key=lambda x: -x[1]):
    if lbl:
        print(f"  {lbl:<35} {cnt:5d}")

print(f"\n{'=' * 70}")
print("TOP SPANS DETECTES")
print("=" * 70)
for lbl, cnt in span_labels.most_common(20):
    print(f"  {lbl:<35} {cnt:6d}")

print(f"\nRoles SVO :")
for role, cnt in sorted(svo_roles.items(), key=lambda x: -x[1]):
    print(f"  {role:<20} {cnt:6d}")

# Ecriture
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"\n-> {OUT}  ({len(rows)} phrases)")
print(f"\nEtape suivante :")
print(f"  python3 prepare_review_inputs.py")
