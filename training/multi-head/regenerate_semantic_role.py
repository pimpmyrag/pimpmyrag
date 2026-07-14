#!/usr/bin/env python3
"""
regenerate_semantic_role.py — Recalcule le champ "semantic_role" de chaque span
NER (hint_*) d'un dataset JSONL en utilisant le mapper heuristique corrigé de
build_multitask_dataset.py (_map_semantic_role_id + _map_semantic_role_from_nominal).

Pourquoi ce script :
  Le champ "semantic_role" était jusqu'ici calculé une fois (script externe
  build_v822_semrole.py, absent du repo) et mis en cache dans les JSONL sources
  (train_v8.22_semrole.jsonl, etc.). build_multitask_dataset.py fait TOUJOURS
  primer ce cache sur un recalcul (cf. `if "semantic_role" in sp: ... l'utiliser
  directement`) — donc corriger le mapper seul est sans effet tant que ce cache
  n'est pas régénéré. Ce script régénère explicitement ce cache, AVANT le build
  multitask, pour :
    1. Profiter des fixes du mapper (propagation voice, extension causale,
       fallback nominal_relation).
    2. Rendre le résultat inspectable/éditable directement dans les JSONL
       sources (utile pour une review humaine ou LLM ultérieure).

Usage:
  python3 regenerate_semantic_role.py \
      --input  data/train_v8.22_semrole.jsonl \
      --output data/train_v8.23_semrole_fixed.jsonl \
      --report data/train_v8.23_semrole_diff_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_multitask_dataset import (
    _map_semantic_role_id,
    _map_semantic_role_from_nominal,
    ALL_SYN_LABELS,
)
from labels import (
    ID2SEMANTIC_ROLE, SEMANTIC_ROLE2ID,
    SEMANTIC_ROLE_SKIP_ID, SEMANTIC_ROLE_NONE_ID,
)

# String utilisée dans les JSONL sources pour le sentinel SKIP (cohérent avec
# le format déjà en place — cf. build_multitask_dataset.py: sr_str == "OBLIQUE_UNRESOLVED").
_SKIP_STR = "OBLIQUE_UNRESOLVED"
_NONE_STR = "NONE"


def _id_to_str(role_id: int) -> str:
    if role_id == SEMANTIC_ROLE_SKIP_ID:
        return _SKIP_STR
    if role_id == SEMANTIC_ROLE_NONE_ID:
        return _NONE_STR
    return ID2SEMANTIC_ROLE[role_id]


def recompute_row(row: dict, stats: Counter, diffs: list) -> dict:
    spans = row.get("spans", [])

    # Index verb_trigger.start -> voice (même logique que build_multitask_dataset.py)
    verb_voice_by_start = {
        s["start"]: s.get("voice")
        for s in spans
        if s.get("label") == "verb_trigger"
    }

    # Index start -> span, pour résoudre le label/end du parent nominal
    # (cf. _map_semantic_role_from_nominal, distinction PART_OF vs DOMAIN).
    span_by_start: dict[int, dict] = {}
    for s in spans:
        span_by_start.setdefault(s["start"], s)

    for sp in spans:
        label = sp.get("label")
        if label in ALL_SYN_LABELS:
            continue  # verb_trigger/pron_* : semantic_role non calculé (cf. dataset builder)

        svo_role_str = sp.get("svo_role", "NONE") or "NONE"
        gov_verb_family = sp.get("gov_verb_family")
        gvs = sp.get("gov_verb_start")
        gov_verb_voice = verb_voice_by_start.get(gvs) if gvs is not None else None

        new_id = _map_semantic_role_id(svo_role_str, label, gov_verb_family, gov_verb_voice)
        # La relation nominale prime dès qu'elle est annotée (cohérent avec
        # build_multitask_dataset.py et annotate_nominal_parents.py) : un span
        # emboîté hérite souvent d'un svo_role de son parent qui ne reflète pas
        # son propre rôle sémantique.
        nom_rel = sp.get("nominal_relation")
        if nom_rel:
            nps = sp.get("nominal_parent_start")
            parent_sp = span_by_start.get(nps) if nps is not None else None
            new_id = _map_semantic_role_from_nominal(
                nom_rel, label,
                parent_label=parent_sp.get("label") if parent_sp else None,
                parent_end=parent_sp.get("end") if parent_sp else None,
                child_end=sp.get("end"),
            )

        new_str = _id_to_str(new_id)
        old_str = sp.get("semantic_role")

        stats["total_spans"] += 1
        if old_str is not None:
            stats["had_cached_value"] += 1
            if old_str != new_str:
                stats["changed"] += 1
                diffs.append({
                    "id": row.get("id"),
                    "text": sp.get("text"),
                    "hint": label,
                    "svo_role": svo_role_str,
                    "gov_verb_family": gov_verb_family,
                    "old_semantic_role": old_str,
                    "new_semantic_role": new_str,
                })
            else:
                stats["unchanged"] += 1
        else:
            stats["newly_assigned"] += 1
            if new_str not in (_NONE_STR,):
                stats["newly_assigned_non_none"] += 1

        sp["semantic_role"] = new_str

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default=None, help="JSON listant les spans dont la valeur a changé")
    ap.add_argument("--dry-run", action="store_true", help="n'écrit pas --output, affiche seulement les stats")
    args = ap.parse_args()

    stats = Counter()
    diffs = []
    out_rows = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out_rows.append(recompute_row(row, stats, diffs))

    print(f"Phrases traitées      : {len(out_rows)}")
    print(f"Spans totaux (NER)    : {stats['total_spans']}")
    print(f"  avec cache existant : {stats['had_cached_value']}")
    print(f"    → changés         : {stats['changed']}")
    print(f"    → inchangés       : {stats['unchanged']}")
    print(f"  sans cache (nouveau): {stats['newly_assigned']}")
    print(f"    dont non-NONE     : {stats['newly_assigned_non_none']}")

    if not args.dry_run:
        with open(args.output, "w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"✅ Écrit : {args.output}")

        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(diffs, f, ensure_ascii=False, indent=2)
            print(f"✅ Rapport de diff ({len(diffs)} changements) : {args.report}")
    else:
        print("⚠️  --dry-run : rien écrit sur disque")


if __name__ == "__main__":
    main()

