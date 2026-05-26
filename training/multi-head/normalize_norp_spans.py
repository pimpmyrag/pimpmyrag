"""
normalize_norp_spans.py — Normalise les patterns hint_norp adjacent à un span rôle.

Pattern B (incohérent) :
   [hint_group_role "autorités"][5-14]  [hint_norp "allemandes"][15-24]

→ Pattern A (cohérent, cible) :
   [hint_group_role "autorités allemandes"][5-24]  ⊃  [hint_norp "allemandes"][15-24]

Règles de sécurité :
  - Le texte entre parent et norp doit être uniquement des espaces (≤ 2 chars)
  - L'extension ne doit pas chevaucher d'autres spans existants
  - Labels étendables : hint_group_role, hint_person_role, hint_inst_role,
                        hint_org_name, hint_person_name, hint_org_role,
                        hint_loc_generic, hint_vehicle, hint_object_generic
  - Côtés gérés : norp_after (parent ← norp étend END) et norp_before (norp → parent étend START)
  - Itération stable : on recalcule après chaque extension (gère les NORPs multiples)
"""
import json, re
from pathlib import Path
from collections import Counter

DATA = Path("/Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head/data")
SVO_LABELS   = {"verb_trigger", "pron_subj", "pron_obj"}
# Labels dont on peut étendre les frontières
EXTENSIBLE   = {
    "hint_group_role", "hint_person_role", "hint_inst_role", "hint_org_role",
    "hint_org_name", "hint_person_name",
    "hint_loc_generic", "hint_vehicle", "hint_object_generic",
}
MAX_BETWEEN  = 2   # max chars entre parent et norp (typiquement 1 espace)

import sys as _sys
# Par défaut opère sur les originaux ; passer --filled pour les _filled
_MODE = "filled" if "--filled" in _sys.argv else "orig"
FILES = (
    ["train_v8.18_filled.jsonl", "val_v8.18_filled.jsonl", "test_v8.18_filled.jsonl"]
    if _MODE == "filled" else
    ["train_v8.18.jsonl", "val_v8.18.jsonl", "test_v8.18.jsonl"]
)

def is_pure_whitespace(s: str) -> bool:
    return bool(re.fullmatch(r'\s*', s))

def find_collision(new_start, new_end, spans, exclude_ids):
    """Retourne True si [new_start, new_end] chevauche un span existant non exclu."""
    for i, sp in enumerate(spans):
        if i in exclude_ids:
            continue
        s, e = sp["start"], sp["end"]
        if s < new_end and e > new_start:  # chevauchement
            return True
    return False

def process_row(row: dict, text: str, stats: Counter, examples: list) -> int:
    """
    Traite une phrase. Retourne le nb d'extensions effectuées.
    Modifie row in-place.
    """
    spans = row.get("spans", [])
    n_extended = 0

    # Itération stable : on repasse tant qu'on trouve des extensions
    for _iteration in range(10):  # max 10 passes (chaînes longues)
        changed = False
        ner = [s for s in spans if s.get("label") not in SVO_LABELS]

        for i, parent in enumerate(ner):
            if parent.get("label") not in EXTENSIBLE:
                continue
            ps, pe = parent["start"], parent["end"]

            for j, norp in enumerate(ner):
                if norp is parent or norp.get("label") != "hint_norp":
                    continue
                ns, ne = norp["start"], norp["end"]

                # Déjà imbriqué ?
                if ps <= ns and pe >= ne:
                    continue

                # Calcule côté et texte entre
                if pe <= ns:
                    between = text[pe:ns]
                    side = "norp_after"
                    new_start, new_end = ps, ne
                elif ne <= ps:
                    between = text[ne:ps]
                    side = "norp_before"
                    new_start, new_end = ns, pe
                else:
                    continue  # chevauchement partiel → skip

                if len(between) > MAX_BETWEEN:
                    continue
                if not is_pure_whitespace(between):
                    continue

                # Vérif collision — on ignore parent et norp eux-mêmes
                parent_idx = spans.index(parent)
                norp_idx   = spans.index(norp)
                if find_collision(new_start, new_end, spans, {parent_idx, norp_idx}):
                    continue

                # Applique l'extension
                old_text = text[ps:pe]
                new_text = text[new_start:new_end]

                parent["start"] = new_start
                parent["end"]   = new_end
                parent["text"]  = new_text

                n_extended += 1
                changed = True
                stats[f"extended_{side}"] += 1
                stats[f"extended_parent_{parent['label']}"] += 1

                if len(examples) < 12:
                    examples.append({
                        "old": f"[{parent['label']} \"{old_text}\"] + [hint_norp \"{text[ns:ne]}\"]",
                        "new": f"[{parent['label']} \"{new_text}\"] ⊃ [hint_norp \"{text[ns:ne]}\"]",
                        "ctx": text[max(0, new_start-35):min(len(text), new_end+35)],
                    })
                break  # repart depuis le début (spans modifiés)

            if changed:
                break  # recommence l'itération externe

        if not changed:
            break

    return n_extended


def run(dry_run=False):
    stats    = Counter()
    examples = []

    for fname in FILES:
        orig_path = DATA / fname
        if not orig_path.exists():
            print(f"⚠️  {fname} manquant"); continue

        data = []
        with open(orig_path) as f:
            for line in f:
                data.append(json.loads(line))

        n_rows_changed = 0
        n_spans_extended = 0

        for row in data:
            text = row.get("text", "")
            n = process_row(row, text, stats, examples)
            if n:
                n_rows_changed += 1
                n_spans_extended += n

        print(f"\n  {fname}")
        print(f"    Phrases modifiées  : {n_rows_changed:,}")
        print(f"    Spans étendus      : {n_spans_extended:,}")

        if not dry_run:
            suffix = "_filled" if _MODE == "filled" else ""
            base   = fname.replace("_filled.jsonl", "").replace(".jsonl", "")
            out_path = DATA / f"{base}_normed{suffix}.jsonl"
            with open(out_path, "w") as f:
                for row in data:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"    → {out_path.name}")

    print(f"\n{'━'*60}")
    print(f"  Stats globales :")
    for k, v in sorted(stats.items(), key=lambda x: -x[1])[:20]:
        print(f"    {k:<45} {v:>5}")

    print(f"\n  Exemples d'extensions :")
    for ex in examples:
        print(f"\n    AVANT : {ex['old']}")
        print(f"    APRÈS : {ex['new']}")
        print(f"    ctx   : ...{ex['ctx']}...")

if __name__ == "__main__":
    dry = "--dry-run" in _sys.argv
    print(f"{'DRY-RUN' if dry else 'LIVE'} — mode={'filled' if _MODE=='filled' else 'orig'} — normalisation hint_norp adjacents")
    run(dry_run=dry)

