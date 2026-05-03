"""
Corrige les hint_inst_name generiques → hint_group_role dans les fichiers de data.

Regle :
  - hint_inst_name RESTE si le span contient au moins une majuscule
    ex: "police de Grenoble", "parquet de Paris" → reste hint_inst_name
  - hint_inst_name → hint_group_role si le span ne contient AUCUNE majuscule
    ex: "gouvernement", "police", "ministere", "armee reguliere" → hint_group_role
"""
import json, os, shutil


def is_generic_inst(text: str) -> bool:
    """True si le texte ne contient aucune majuscule → institution generique non nommee."""
    return not any(c.isupper() for c in text)


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

    # Restaure le backup si present (pour re-appliquer proprement)
    if os.path.exists(backup):
        shutil.copy2(backup, path)
        print(f"{fname}: restaure depuis backup")
    else:
        shutil.copy2(path, backup)
        print(f"{fname}: backup cree -> {backup}")

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
                    if t and is_generic_inst(t):
                        sp["label"] = "hint_group_role"
                        fixed_count += 1
            out_lines.append(json.dumps(ex, ensure_ascii=False))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"  → {fixed_count} spans corriges (hint_inst_name → hint_group_role)")

print("\nTermine.")
