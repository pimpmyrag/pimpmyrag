#!/usr/bin/env python3
import json, re
from pathlib import Path

INFRA_TO_LOC = {
    "poste", "base", "bureaux", "bureau",
    "batiment", "btiment", "batiments", "btiments",
    "terminal", "centrale", "infrastructure",
    "structures", "installation", "installations",
    "constructions", "construction", "escalier", "escaliers",
    "frontiere", "frontire", "frontieres", "frontires",
}

INFRA_TO_OBJ = {
    "barricade", "barricades", "la chemine", "chemine",
    "la cheminee", "cheminee",
    "clture", "clture de scurit",
    "barrire", "barrires", "barrires de scurit",
    "cloture", "barriere", "barrieres",
}

TOOL_TO_OBJ = {
    "electrons", "lectrons", "protons", "neutrons", "photons",
    "atomes", "molecules", "molcules", "ions", "isotopes",
    "logithque", "logiteque", "portail",
    "controles", "contrle", "mainframes",
    "deliberation", "dlibration",
    "technologie", "technologies", "algorithme", "algorithmes",
    "application", "applications",
    "piano", "guitare", "violon", "trompette", "saxophone",
    "baguettes",
}

TOOL_TO_WEAPON = {
    "traquenards", "traquenard", "pieges", "pige", "piege", "piges",
    "trappes", "trappe",
}

def normalize(t): return t.strip().lower()

def relabel_infra(text):
    t = text.strip()
    n = normalize(t)
    if n in {x.lower() for x in INFRA_TO_OBJ}:
        return "hint_object_generic"
    if n in {x.lower() for x in INFRA_TO_LOC}:
        return "hint_loc_generic"
    return None

def relabel_tool(text):
    t = text.strip()
    n = normalize(t)
    if n in {x.lower() for x in TOOL_TO_WEAPON}:
        return "hint_weapon"
    if n in {x.lower() for x in TOOL_TO_OBJ}:
        return "hint_object_generic"
    return None

def preview(path):
    by_infra, by_tool = {}, {}
    for line in Path(path).open():
        row = json.loads(line)
        for sp in row.get("spans", []):
            t = sp.get("text", "")
            lbl = sp.get("label", "")
            if lbl == "hint_infra":
                nl = relabel_infra(t) or "hint_infra (garde)"
                by_infra.setdefault(nl, []).append(t)
            elif lbl == "hint_tool":
                nl = relabel_tool(t) or "hint_tool (garde)"
                by_tool.setdefault(nl, []).append(t)
    for name, by in [("hint_infra", by_infra), ("hint_tool", by_tool)]:
        print(f"\n{chr(35)*60}\n  {name}\n{chr(35)*60}")
        for nl, ex in sorted(by.items()):
            print(f"  -> {nl} ({len(ex)} cas)")
            for e in ex[:10]: print(f'      [{e}]')
            if len(ex) > 10: print(f'      ...+{len(ex)-10}')

def process(inp, out):
    stats = {}
    rows = []
    for line in Path(inp).open():
        row = json.loads(line)
        spans = []
        for sp in row.get("spans", []):
            lbl = sp.get("label", "")
            t = sp.get("text", "")
            nl = None
            if lbl == "hint_infra": nl = relabel_infra(t)
            elif lbl == "hint_tool": nl = relabel_tool(t)
            if nl:
                key = f"{lbl}->{nl}"
                stats[key] = stats.get(key, 0) + 1
                sp = {**sp, 'label': nl}
            spans.append(sp)
        row["spans"] = spans
        rows.append(row)
    Path(out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return stats

import argparse
p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", default=None)
p.add_argument("--preview", action="store_true")
args = p.parse_args()
if args.preview:
    preview(args.input)
else:
    s = process(args.input, args.output or args.input)
    for k, v in sorted(s.items()): print(f'  {k}: {v}')
