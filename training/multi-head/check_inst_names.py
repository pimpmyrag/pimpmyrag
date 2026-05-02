"""
Affiche toutes les valeurs distinctes de hint_inst_name dans les fichiers de data,
avec leur frequence et un flag si elles n'ont pas de majuscule (candidats hint_group_role).
"""
import json, collections

DATA_DIR = "data"
FILES = ["train_v6.jsonl", "val_v6.jsonl", "test_v6.jsonl"]

for fname in FILES:
    path = f"{DATA_DIR}/{fname}"
    print(f"\n{'='*60}")
    print(f"  {fname}")
    print(f"{'='*60}")
    inst_spans = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                for sp in ex.get("spans", []):
                    if sp["label"] == "hint_inst_name":
                        inst_spans.append(sp["text"].strip())
    except FileNotFoundError:
        print("  INTROUVABLE")
        continue

    counter = collections.Counter(inst_spans)
    no_upper = [(t, c) for t, c in counter.items() if not any(ch.isupper() for ch in t)]
    with_upper = [(t, c) for t, c in counter.items() if any(ch.isupper() for ch in t)]

    print(f"  Total spans hint_inst_name : {len(inst_spans)}")
    print(f"  Valeurs distinctes         : {len(counter)}")
    print(f"  Dont SANS majuscule        : {len(no_upper)}  (candidats → hint_group_role)")
    print(f"  Dont AVEC majuscule        : {len(with_upper)}  (conservent hint_inst_name)")

    if no_upper:
        print(f"\n  ⚠️  SANS majuscule (seraient convertis en hint_group_role) :")
        for text, count in sorted(no_upper, key=lambda x: -x[1]):
            print(f"      [{count:3d}x]  {text!r}")

    print(f"\n  ✅ AVEC majuscule (hint_inst_name OK) :")
    for text, count in sorted(with_upper, key=lambda x: -x[1]):
        print(f"      [{count:3d}x]  {text!r}")

print("\nTermine.")

