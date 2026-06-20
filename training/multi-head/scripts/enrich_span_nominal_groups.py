#!/usr/bin/env python3
"""Enrich spans with a conservative nominal group built from Stanza dependencies.

Input JSONL record format:
- {"id": ..., "text": ..., "spans": [{"start": int, "end": int, ...}, ...]}

Output JSONL:
- same records + for each span:
  - nominal_group
  - nominal_group_start
  - nominal_group_end
  - nominal_group_head

The algorithm is conservative:
- starts from words overlapping the span,
- selects a nominal head,
- expands through nominal-only dependency edges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SKIP_LABELS = {"verb_trigger", "pron_subj", "pron_obj"}
HEAD_POS = {"NOUN", "PROPN", "PRON"}
BLOCK_POS = {"VERB", "AUX", "PUNCT"}
NOMINAL_DEPS = {
    "det",
    "amod",
    "nummod",
    "compound",
    "flat",
    "flat:name",
    "flat:foreign",
    "nmod",
    "nmod:poss",
    "appos",
    "case",
    "fixed",
}


def load_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_words(doc) -> list[dict]:
    words = []
    for si, sent in enumerate(doc.sentences, start=1):
        token_offsets: dict[int, tuple[int, int]] = {}
        for tok in sent.tokens:
            for tw in tok.words:
                token_offsets[int(tw.id)] = (int(tok.start_char), int(tok.end_char))

        for w in sent.words:
            if w.start_char is None or w.end_char is None:
                start_char, end_char = token_offsets.get(int(w.id), (None, None))
            else:
                start_char, end_char = int(w.start_char), int(w.end_char)

            if start_char is None or end_char is None:
                # Skip rare malformed token alignments instead of crashing the batch.
                continue

            words.append(
                {
                    "id": f"s{si}_w{w.id}",
                    "sent_id": si,
                    "wid": int(w.id),
                    "text": w.text,
                    "lemma": w.lemma,
                    "upos": w.upos,
                    "head": int(w.head),
                    "deprel": w.deprel,
                    "start": start_char,
                    "end": end_char,
                }
            )
    return words


def overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 <= b0 or b1 <= a0)


def pick_head(covered: list[dict], covered_wids: set[int]) -> dict | None:
    candidates = [w for w in covered if w["head"] == 0 or w["head"] not in covered_wids]
    if not candidates:
        candidates = covered

    nominal = [w for w in candidates if w["upos"] in HEAD_POS]
    if nominal:
        return sorted(nominal, key=lambda x: x["wid"])[0]

    non_block = [w for w in candidates if w["upos"] not in BLOCK_POS]
    if non_block:
        return sorted(non_block, key=lambda x: x["wid"])[0]

    return sorted(candidates, key=lambda x: x["wid"])[0] if candidates else None


def nominal_group_for_span(text: str, span: dict, words: list[dict]) -> tuple[str, int, int, str] | None:
    s0, s1 = int(span["start"]), int(span["end"])
    covered = [w for w in words if overlap(s0, s1, w["start"], w["end"]) and w["upos"] != "PUNCT"]
    if not covered:
        return None

    covered_wids = {w["wid"] for w in covered}
    head = pick_head(covered, covered_wids)
    if head is None:
        return None

    children: dict[int, list[dict]] = {}
    for w in words:
        if w["sent_id"] != head["sent_id"]:
            continue
        children.setdefault(w["head"], []).append(w)

    included_ids = {w["id"] for w in covered}
    queue = [head]
    seen = set()
    while queue:
        parent = queue.pop(0)
        if parent["id"] in seen:
            continue
        seen.add(parent["id"])

        for ch in children.get(parent["wid"], []):
            dep = ch["deprel"]
            if dep in NOMINAL_DEPS and ch["upos"] not in BLOCK_POS:
                if ch["id"] not in included_ids:
                    included_ids.add(ch["id"])
                    queue.append(ch)

    included = [w for w in words if w["id"] in included_ids]
    ng_start = min(w["start"] for w in included)
    ng_end = max(w["end"] for w in included)
    ng_text = text[ng_start:ng_end]
    return ng_text, ng_start, ng_end, head["text"]


def enrich_record(nlp, record: dict, overwrite: bool) -> tuple[dict, int]:
    text = record.get("text", "")
    spans = record.get("spans", [])
    if not text or not spans:
        return record, 0

    doc = nlp(text)
    words = build_words(doc)

    updates = 0
    for sp in spans:
        if sp.get("label") in SKIP_LABELS:
            continue
        if not overwrite and "nominal_group" in sp:
            continue

        ng = nominal_group_for_span(text, sp, words)
        if ng is None:
            continue

        ng_text, ng_start, ng_end, ng_head = ng
        sp["nominal_group"] = ng_text
        sp["nominal_group_start"] = ng_start
        sp["nominal_group_end"] = ng_end
        sp["nominal_group_head"] = ng_head
        updates += 1

    return record, updates


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich spans with nominal groups from Stanza")
    parser.add_argument("--input", required=True, help="Input JSONL with text+spans")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--max-records", type=int, default=0, help="Limit rows (0=all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing nominal_group fields")
    parser.add_argument("--download-models", action="store_true", help="Run stanza.download('fr') before processing")
    args = parser.parse_args()

    try:
        import stanza
    except Exception as exc:
        print("❌ stanza not installed.")
        print("   Install with: pip install stanza")
        print(f"   Details: {exc}")
        return 1

    if args.download_models:
        print("⏳ Downloading Stanza fr models...")
        stanza.download("fr")

    print("⏳ Loading Stanza pipeline (fr: tokenize,pos,lemma,depparse)")
    nlp = stanza.Pipeline(
        lang="fr",
        processors="tokenize,pos,lemma,depparse",
        tokenize_no_ssplit=False,
        verbose=False,
    )
    print("✅ Stanza ready")

    rows = []
    total_updates = 0

    for i, rec in enumerate(load_jsonl(Path(args.input)), start=1):
        if args.max_records > 0 and i > args.max_records:
            break
        out, n_updates = enrich_record(nlp, rec, overwrite=args.overwrite)
        rows.append(out)
        total_updates += n_updates

    write_jsonl(Path(args.output), rows)
    print(f"✅ wrote {len(rows)} records to {args.output}")
    print(f"📌 spans enriched with nominal_group: {total_updates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


