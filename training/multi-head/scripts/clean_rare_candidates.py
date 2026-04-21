#!/usr/bin/env python3
import json
from collections import Counter

items = [json.loads(l) for l in open("data/rare_candidates.jsonl") if l.strip()]
print(f"Avant nettoyage: {len(items)} phrases")

# 1. Filtrer textes trop longs/courts
items = [i for i in items if 40 <= len(i["text"]) <= 600]
print(f"Après filtre longueur (40-600): {len(items)}")

# 2. Filtrer phrases avec trop de predictions
items = [i for i in items if len(i.get("predictions", [])) <= 25]
print(f"Après filtre predictions (<=25): {len(items)}")

# 3. Capper les labels surreprésentés à 1000 chacun
TARGET = 1000
RARE_LABELS_ORDER = [
    "hint_concept", "hint_count", "hint_disease", "hint_fac_name",
    "hint_food", "hint_infra", "hint_language", "hint_law",
    "hint_measure", "hint_money", "hint_object_name", "hint_percentage",
    "hint_rate", "hint_substance", "hint_time_clock", "hint_tool",
    "hint_weapon", "hint_work_of_art"
]

label_counts = Counter()
kept = []
items_sorted = sorted(items, key=lambda x: len(x["text"]))
for item in items_sorted:
    rare = item.get("rare_labels", [])
    useful = [l for l in rare if label_counts[l] < TARGET]
    if useful:
        kept.append(item)
        for l in rare:
            label_counts[l] += 1

print(f"Après cap à {TARGET}/label: {len(kept)} phrases")

print("\nDistribution finale:")
final_counts = Counter()
for item in kept:
    for lbl in item.get("rare_labels", []):
        final_counts[lbl] += 1
for lbl in RARE_LABELS_ORDER:
    cnt = final_counts.get(lbl, 0)
    bar = "█" * min(cnt // 20, 50)
    status = "✅" if cnt >= TARGET else "⚠️ "
    print(f"  {status} {lbl:<25} {cnt:>5}  {bar}")

with open("data/rare_candidates_clean.jsonl", "w") as f:
    for item in kept:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"\n✅ data/rare_candidates_clean.jsonl ({len(kept)} phrases)")

# Estimation coût Claude (batch = 50% tarif normal)
# claude-3-5-sonnet: $3/M input tokens, $15/M output tokens
# estimation grossière: ~200 tokens input/phrase + 100 output
tokens_in = len(kept) * 200
tokens_out = len(kept) * 100
cost = (tokens_in / 1e6 * 3 + tokens_out / 1e6 * 15) * 0.5
print(f"\n💰 Estimation coût batch Claude-3.5-sonnet: ~${cost:.2f} USD")

