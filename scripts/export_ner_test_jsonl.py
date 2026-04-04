#!/usr/bin/env python3
"""
Extrait la liste TESTS de ner_candidates_test.py vers un fichier JSONL.
Utilise ast.literal_eval — aucun code HTTP n'est exécuté.

Usage :
  python scripts/export_ner_test_jsonl.py
  python scripts/export_ner_test_jsonl.py --out path/to/output.jsonl
"""
import ast
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--out", default=None,
    help="Chemin de sortie (défaut: radar-nli-toolkit/src/test/resources/ner_candidates_tests.jsonl)"
)
args = parser.parse_args()

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPTS_DIR)

src_path = os.path.join(_SCRIPTS_DIR, "ner_candidates_test.py")
out_path = args.out or os.path.join(
    _REPO_ROOT, "radar-nli-toolkit", "src", "test", "resources",
    "ner_candidates_tests.jsonl"
)

# ── Extraction de la liste TESTS via ast (pas d'exécution HTTP) ──────────────
with open(src_path, "r", encoding="utf-8") as f:
    src = f.read()

tree = ast.parse(src, filename=src_path)

tests_node = None
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "TESTS":
                tests_node = node.value
                break
    if tests_node:
        break

if tests_node is None:
    raise RuntimeError("Impossible de trouver la variable TESTS dans ner_candidates_test.py")

tests_value: list = ast.literal_eval(tests_node)

# ── Conversion en JSONL ───────────────────────────────────────────────────────
os.makedirs(os.path.dirname(out_path), exist_ok=True)

count = 0
with open(out_path, "w", encoding="utf-8") as f:
    for i, entry in enumerate(tests_value, 1):
        text, expected_list, category, note = entry

        expected = []
        for type_spec, span_text in expected_list:
            if ":" in type_spec:
                typ, hint = type_spec.split(":", 1)
            else:
                typ, hint = type_spec, None
            expected.append({
                "type":  typ,
                "text":  span_text,
                "hint":  hint,
            })

        record = {
            "id":       i,
            "text":     text,
            "expected": expected,
            "category": category,
            "note":     note,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

print(f"✅ {count} cas de test exportés → {out_path}")

