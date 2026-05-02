"""
Corrige les hint_inst_name generiques (commencant par minuscule)
→ relab en hint_group_role dans les fichiers de data.

Logique : si le span hint_inst_name commence par une lettre minuscule,
c'est un nom commun generique → hint_group_role.
Les vrais noms propres (majuscule initiale ou sigle) restent hint_inst_name.
"""
import json, os, shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

FILES_TO_FIX = [
    "train_v5.jsonl",
    "test_v4_inst_fixed.jsonl",
]

for fname in FILES_TO_FIX:
    path = os.path.join(DATA_DIR, fname)
    backup = path + ".bak"
    if not os.path.exists(path):
        print(f"{fname}: introuvable, skip")
        continue

    shutil.copy2(path, backup)
    print(f"{fname}: backup -> {backup}")

    fixed_count = 0
    out_lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                out_lines.append(line)
                continue
            ex = json.loads(line)
            for sp in ex.get("spans", []):
                if sp["label"] == "hint_inst_name":
                    t = sp["text"].strip()
                    if t and t[0].islower():
                        sp["label"] = "hint_group_role"
                        fixed_count += 1
            out_lines.append(json.dumps(ex, ensure_ascii=False))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"  → {fixed_count} spans corriges (hint_inst_name → hint_group_role)")

print("\nTermine.")

