#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opml_to_feeds_json.py

Convertit une ou plusieurs listes OPML en config JSON.

Exemples:
  python opml_to_feeds_json.py --lang fr --opml France.opml --out feeds_fr.json
  python opml_to_feeds_json.py --auto-lang-from-filename --dedup --opml France.opml Germany.opml --out feeds_eu.json
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET


def infer_lang_from_filename(path: str) -> str:
    n = path.lower()
    if "france" in n:
        return "fr"
    if "germany" in n or "deutsch" in n:
        return "de"
    if "spain" in n or "espa" in n:
        return "es"
    if "italy" in n or "italia" in n:
        return "it"
    if "portugal" in n:
        return "pt"
    if "united kingdom" in n or "uk" in n or "britain" in n:
        return "en"
    return "en"


def parse_opml(opml_path: str):
    tree = ET.parse(opml_path)
    root = tree.getroot()
    feeds = []
    for outline in root.iter("outline"):
        url = outline.attrib.get("xmlUrl")
        if not url:
            continue
        title = outline.attrib.get("title") or outline.attrib.get("text") or "Unknown"
        feeds.append((title.strip(), url.strip()))
    return feeds


def normalize_source(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opml", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="")
    ap.add_argument("--auto-lang-from-filename", action="store_true")
    ap.add_argument("--dedup", action="store_true")
    args = ap.parse_args()

    out = {"feeds": []}
    seen = set()

    for opml_file in args.opml:
        lang = args.lang.strip() if args.lang.strip() else None
        if args.auto_lang_from_filename and not lang:
            lang = infer_lang_from_filename(opml_file)
        if not lang:
            lang = "en"

        for title, url in parse_opml(opml_file):
            if args.dedup and url in seen:
                continue
            seen.add(url)
            out["feeds"].append({"source": normalize_source(title), "lang": lang, "url": url})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(out['feeds'])} feeds to {args.out}")


if __name__ == "__main__":
    main()
