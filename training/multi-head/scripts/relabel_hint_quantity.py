#!/usr/bin/env python3
import json, re
from pathlib import Path
RATE = re.compile(
    r"(km/h|m/s|mph|buts?/match|points?/match|tr/min|rpm|MB/s|Mbps|Gbps"
    r"|par\s+(heure|jour|semaine|mois|an|rencontre|match|km))", re.I)
MEASURE = re.compile(
    r"(\bkg\b|\bmg\b|tonnes?|centim\S*tres?|millim\S*tres?"
    r"|kilom\S*tres?\b|m\S*tres?\b|\bcm\b|\bmm\b|\bkm\b|\bm\b"
    r"|hectares?|\bha\b|pieds?\b|pouces?\b|yards?\b"
    r"|litres?\b|\bml\b|\bGo\b|\bMo\b|\bKo\b|\bGB\b|\bMB\b|\bKB\b|GiB|MiB"
    r"|[Mm]egapixels?\b|pixels?\b|\b[Mm][Pp]\b"
    r"|\bHz\b|\bkHz\b|\bMHz\b|\bGHz\b|volts?\b|watts?\b|\bdB\b)", re.I)
COUNT_UNITS = re.compile(
    r"(buts?\b|goals?\b|victoires?\b|morts?\b|bless\S*s?\b|victimes?\b"
    r"|membres?\b|d\S*put\S*s?\b|s\S*nateurs?\b|t\S*moins?\b|experts?\b"
    r"|voix\b|articles?\b|v\S*hicules?\b|voitures?\b|avions?\b|soldats?\b"
    r"|manifestants?\b|pays\b|villes?\b|entreprises?\b|rencontres?\b"
    r"|matchs?\b|participants?\b|candidats?\b|d\S*put\S*s?\b)", re.I)
NB_WORDS = re.compile(
    r"^(z.ro|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix"
    r"|onze|douze|treize|quatorze|quinze|seize|vingt|trente|quarante"
    r"|cinquante|soixante|cent|mille|une dizaine|une vingtaine"
    r"|une trentaine|une quarantaine|une cinquantaine|une soixantaine"
    r"|une centaine|quelques)$", re.I)
SCORE = re.compile(r"^\d+\s*[-]\s*\d+$|^\d+\s+(contre|vs)\.?\s+\d+$", re.I)
PURE_NUM = re.compile(r"^[\d\s,\.]+$")
ORDINAL = re.compile(
    r"(premier|premi.re|deuxi.me|troisi.me|quatri.me|cinqui.me|sixi.me"
    r"|\d+e(?:r|re)?\b|i.me\b)", re.I)
def relabel(t):
    t = t.strip()
    if RATE.search(t): return "hint_rate"
    if MEASURE.search(t): return "hint_measure"
    if SCORE.match(t): return "hint_count"
    if PURE_NUM.match(t): return "hint_count"
    s = t.strip("\"\'")
    if NB_WORDS.match(s): return "hint_count"
    if (re.search(r"\d", t) or NB_WORDS.search(t)) and COUNT_UNITS.search(t):
        return "hint_count"
    if ORDINAL.search(t) and not MEASURE.search(t): return "hint_count"
    return None
def preview(path):
    by = {}
    for line in Path(path).open():
        row = json.loads(line)
        for sp in row.get("spans", []):
            if sp.get("label") == "hint_quantity":
                nl = relabel(sp.get("text", "")) or "hint_quantity (garde)"
                by.setdefault(nl, []).append(sp.get("text", ""))
    for nl, ex in sorted(by.items()):
        print(f"\n{'='*50}\n  -> {nl} ({len(ex)} cas)\n{'='*50}")
        for e in ex[:12]: print(f"    [{e}]")
        if len(ex) > 12: print(f"    ...+{len(ex)-12}")
def process(inp, out):
    stats = {k: 0 for k in ["kept", "hint_rate", "hint_measure", "hint_count"]}
    rows = []
    for line in Path(inp).open():
        row = json.loads(line)
        spans = []
        for sp in row.get("spans", []):
            if sp.get("label") == "hint_quantity":
                nl = relabel(sp.get("text", ""))
                if nl:
                    sp = {**sp, "label": nl}
                    stats[nl] += 1
                else:
                    stats["kept"] += 1
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
    print(f"garde={s['kept']} rate={s['hint_rate']} measure={s['hint_measure']} count={s['hint_count']}")
