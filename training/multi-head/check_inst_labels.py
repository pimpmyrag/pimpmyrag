"""
Audit des hint_inst_name generiques (commencant par minuscule)
dans les fichiers de training/validation.
"""
import json, os

DATA_DIR = os.path.dirname(__file__) + "/data"

files = [
    "train_v4_claude.jsonl",
    "train_v5.jsonl",
    "test_v4_inst_fixed.jsonl",
]

for fname in files:
    path = os.path.join(DATA_DIR, fname)
    bad = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                for sp in ex.get("spans", []):
                    if sp["label"] == "hint_inst_name":
                        t = sp["text"].strip()
                        if t and t[0].islower():
                            bad.append((ex["id"], t))
        print(f"\n{fname}: {len(bad)} cas suspects (hint_inst_name generique)")
        for eid, txt in bad[:20]:
            print(f"  [{eid}] \"{txt}\"")
        if len(bad) > 20:
            print(f"  ... et {len(bad)-20} autres")
    except FileNotFoundError:
        print(f"{fname}: fichier introuvable")

