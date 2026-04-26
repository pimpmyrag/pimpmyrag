#!/usr/bin/env python3
"""
Merge les annotations Claude validées avec le dataset existant,
déduplique, re-split 80/10/10, backup les originaux.
"""
import json, os, re, shutil, random, sys
from collections import Counter
from pathlib import Path

CLAUDE = sys.argv[1] if len(sys.argv) > 1 else "data/wikinews_claude_annotated_valid.jsonl"
DATA_DIR = "data"
SEED = 42

def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())

def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# 1. Backup
for name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
    src = f"{DATA_DIR}/{name}"
    bak = f"{DATA_DIR}/{name}.bak"
    if os.path.exists(src) and not os.path.exists(bak):
        shutil.copy2(src, bak)
        print(f"📦 Backup {src} → {bak}")

# 2. Load existing
existing = []
for name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
    path = f"{DATA_DIR}/{name}"
    if os.path.exists(path):
        items = load_jsonl(path)
        print(f"📂 {path}: {len(items)} phrases")
        existing.extend(items)

print(f"\n📊 Total existant: {len(existing)}")

# 3. Load Claude
claude_items = load_jsonl(CLAUDE)
print(f"📂 Claude: {len(claude_items)} phrases")

# 4. Deduplicate
seen = set()
all_items = []
n_dupes = 0

for obj in existing:
    norm = normalize(obj["text"])
    if norm not in seen:
        seen.add(norm)
        all_items.append(obj)

for obj in claude_items:
    norm = normalize(obj["text"])
    if norm not in seen:
        seen.add(norm)
        # Ensure format matches (remove extra fields)
        clean = {
            "id": obj.get("id", ""),
            "text": obj["text"],
            "spans": obj.get("spans", []),
        }
        all_items.append(clean)
    else:
        n_dupes += 1

print(f"\n📊 Après dédup: {len(all_items)} phrases ({n_dupes} doublons supprimés)")

# 5. Shuffle and split
random.seed(SEED)
random.shuffle(all_items)

n = len(all_items)
n_val = n_test = max(int(n * 0.1), 1)
n_train = n - n_val - n_test

train = all_items[:n_train]
val = all_items[n_train:n_train + n_val]
test = all_items[n_train + n_val:]

# 6. Write
write_jsonl(f"{DATA_DIR}/train.jsonl", train)
write_jsonl(f"{DATA_DIR}/val.jsonl", val)
write_jsonl(f"{DATA_DIR}/test.jsonl", test)

# 7. Stats
for name, split in [("train", train), ("val", val), ("test", test)]:
    labels = Counter()
    for obj in split:
        for s in obj.get("spans", []):
            labels[s.get("label", "")] += 1
    print(f"\n📊 {name}: {len(split)} phrases, {sum(labels.values())} spans")
    for label, count in labels.most_common():
        print(f"  {label:<25} {count:>6}")

print(f"\n✅ Done! train={len(train)} / val={len(val)} / test={len(test)}")

