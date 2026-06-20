#!/usr/bin/env python3
"""Build complete nominal head trees from Stanza.

This script keeps all original spans and adds a head-centric nominal layer:
- `nominal_nodes`: unique nominal heads/chunks (existing or synthetic)
- `nominal_edges`: nominal dependency edges between head nodes
- per-span binding fields: `nominal_head_id`, `nominal_head_text`

Goal:
- reuse all current spans,
- add missing nominal heads when no NER span exists,
- recover as much of the nominal structure as Stanza provides.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

SKIP_LABELS = {"verb_trigger"}
HEAD_POS = {"NOUN", "PROPN", "PRON", "ADJ", "NUM"}
BLOCK_POS = {"VERB", "AUX", "PUNCT"}
NOMINAL_DEPREL_MAP = {
    "appos": "APPOS",
    "nmod": "NMOD",
    "nmod:poss": "NMOD",
    "amod": "AMOD",
    "compound": "COMPOUND",
    # flat:name / flat:foreign are internal parts of a proper name — not edges we want
}

# deprels that are flat/internal parts of a proper noun chain — skip as edge and synthetic head
FLAT_DEPRELS = {"flat", "flat:name", "flat:foreign"}
NOMINAL_EXPAND_DEPS = {
    "amod",
    "nummod",
    "compound",
    "flat",
    "flat:name",
    "flat:foreign",
    "nmod",
    "nmod:poss",
    "appos",
    "fixed",
}

FUNC_DEPS = {"fixed"}  # always expand regardless of NER coverage

CASE_DE = {"de", "du", "des", "d'", "d\u2019"}
CASE_LOC = {"\u00e0", "au", "aux", "dans", "en", "sur", "chez", "vers"}


def load_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def fold(s: str) -> str:
    s = norm(s)
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 <= b0 or b1 <= a0)


def build_words(doc) -> list[dict]:
    words = []
    for si, sent in enumerate(doc.sentences, start=1):
        token_offsets = {}
        for tok in sent.tokens:
            for tw in tok.words:
                token_offsets[int(tw.id)] = (int(tok.start_char), int(tok.end_char))

        for w in sent.words:
            if w.start_char is None or w.end_char is None:
                start_char, end_char = token_offsets.get(int(w.id), (None, None))
            else:
                start_char, end_char = int(w.start_char), int(w.end_char)
            if start_char is None or end_char is None:
                continue
            words.append(
                {
                    "sid": si,
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


def find_span_words(span: dict, words: list[dict]) -> list[dict]:
    s0, s1 = int(span["start"]), int(span["end"])
    covered = [w for w in words if overlap(s0, s1, w["start"], w["end"]) and w["upos"] != "PUNCT"]
    if covered:
        return covered

    span_fold = fold(span.get("text", ""))
    if not span_fold:
        return []

    near = []
    global_match = []
    for w in words:
        if w["upos"] == "PUNCT":
            continue
        wf = fold(w.get("text", ""))
        if not wf:
            continue
        if wf in span_fold:
            global_match.append(w)
            if abs(w["start"] - s0) <= 40:
                near.append(w)
    if near:
        return near
    if len(global_match) == 1:
        return global_match
    return []


def pick_head_word(span: dict, words: list[dict]) -> dict | None:
    covered = find_span_words(span, words)
    if not covered:
        return None

    covered_ids = {w["wid"] for w in covered}
    outside = [w for w in covered if w["head"] == 0 or w["head"] not in covered_ids]
    if outside:
        covered = outside

    for upos in ("NOUN", "PROPN", "PRON", "ADJ", "NUM"):
        cand = [w for w in covered if w["upos"] == upos]
        if cand:
            return sorted(cand, key=lambda x: x["wid"])[0]
    return sorted(covered, key=lambda x: x["wid"])[0] if covered else None


FUNC_DEPS = {"det", "case", "fixed"}  # always expand (function words, no NER span ever covers them)


def extract_nominal_chunk(
    head_word: dict,
    sentence_words: list[dict],
    text: str,
    blocked_intervals: list[tuple[int, int]] | None = None,
) -> dict:
    """Build the minimal chunk around head_word.

    blocked_intervals: char intervals of other NER spans.  When set, content
    dependents (amod, nmod, compound…) whose offset is wholly contained in a
    blocked interval are NOT expanded into — because they are a separate NER
    span and their relationship is expressed via a graph edge, not by merging
    them into this chunk text.
    """
    blocked_intervals = blocked_intervals or []

    def is_blocked(w: dict) -> bool:
        ws, we = w["start"], w["end"]
        return any(s <= ws and we <= e for s, e in blocked_intervals)

    by_head = defaultdict(list)
    by_wid = {}
    for w in sentence_words:
        by_head[w["head"]].append(w)
        by_wid[w["wid"]] = w

    keep_ids = {head_word["wid"]}
    queue = [head_word["wid"]]
    seen = set()
    while queue:
        wid = queue.pop(0)
        if wid in seen:
            continue
        seen.add(wid)
        for ch in by_head.get(wid, []):
            if ch["deprel"] not in NOMINAL_EXPAND_DEPS or ch["upos"] == "PUNCT":
                continue
            # Function words (det/case/fixed) always included.
            # Content dependents are skipped when they belong to another NER span.
            if ch["deprel"] not in FUNC_DEPS and is_blocked(ch):
                continue
            if ch["wid"] not in keep_ids:
                keep_ids.add(ch["wid"])
                queue.append(ch["wid"])

    kept = [by_wid[w] for w in sorted(keep_ids) if w in by_wid]
    if not kept:
        kept = [head_word]
    start = min(w["start"] for w in kept)
    end = max(w["end"] for w in kept)
    return {
        "text": text[start:end],
        "start": start,
        "end": end,
        "head_text": head_word["text"],
        "head_lemma": head_word.get("lemma"),
        "head_upos": head_word.get("upos"),
        "sid": head_word["sid"],
        "wid": head_word["wid"],
    }


def semantic_role_for_relation(rel: str, child: dict, sentence_words: list[dict]) -> str:
    if rel == "APPOS":
        return "IDENTITY"
    if rel == "NMOD":
        case_tokens = [w for w in sentence_words if w["head"] == child["wid"] and (w["deprel"] or "").startswith("case")]
        case_lemmas = {norm(w.get("lemma", "")) for w in case_tokens}
        if case_lemmas & CASE_DE:
            return "CONTENT"
        if case_lemmas & CASE_LOC:
            return "LOCATION"
    return "NONE"


def build_nominal_tree_record(record: dict, nlp) -> tuple[dict, dict]:
    text = record.get("text", "")
    spans = record.get("spans", [])
    doc = nlp(text)
    words = build_words(doc)
    by_sid = defaultdict(list)
    for w in words:
        by_sid[w["sid"]].append(w)

    # Pre-build blocked intervals: char offsets of all content NER spans.
    # Used to prevent chunk expansion into dependents that are their own NER spans.
    blocked_intervals: list[tuple[int, int]] = [
        (int(sp["start"]), int(sp["end"]))
        for sp in spans
        if sp.get("label") not in SKIP_LABELS
        and sp.get("start") is not None
        and sp.get("end") is not None
    ]

    head_nodes = {}
    head_node_ids = {}
    next_id = 1

    def ensure_head_node(head_word: dict, source: str) -> str:
        nonlocal next_id
        key = (head_word["sid"], head_word["wid"])
        if key in head_node_ids:
            nid = head_node_ids[key]
            if source not in head_nodes[nid]["sources"]:
                head_nodes[nid]["sources"].append(source)
            return nid
        chunk = extract_nominal_chunk(head_word, by_sid[head_word["sid"]], text, blocked_intervals)
        nid = f"h{next_id}"
        next_id += 1
        head_node_ids[key] = nid
        head_nodes[nid] = {
            "id": nid,
            "text": chunk["text"],
            "start": chunk["start"],
            "end": chunk["end"],
            "head_text": chunk["head_text"],
            "head_lemma": chunk["head_lemma"],
            "head_upos": chunk["head_upos"],
            "token_sid": chunk["sid"],
            "token_wid": chunk["wid"],
            "sources": [source],
            "matched_span_ids": [],
        }
        return nid

    # 1) bind all current spans to a head node
    for idx, sp in enumerate(spans):
        if sp.get("label") in SKIP_LABELS:
            continue
        head = pick_head_word(sp, words)
        if head is None:
            continue
        hid = ensure_head_node(head, "span")
        span_id = f"span_{idx}"
        head_nodes[hid]["matched_span_ids"].append(span_id)
        sp["nominal_head_id"] = hid
        sp["nominal_head_text"] = head["text"]
        sp["nominal_head_start"] = head["start"]
        sp["nominal_head_end"] = head["end"]

    # 2) add missing nominal heads from stanza graph even if no NER span exists
    for w in words:
        # skip flat sub-parts of proper names — the span already covers the full name
        if (w.get("deprel") or "") in FLAT_DEPRELS:
            continue
        rel = NOMINAL_DEPREL_MAP.get(w.get("deprel") or "")
        if w["upos"] not in HEAD_POS:
            continue
        if rel or any(ch["head"] == w["wid"] and (ch["deprel"] in NOMINAL_DEPREL_MAP or ch["deprel"] in NOMINAL_EXPAND_DEPS) for ch in by_sid[w["sid"]]):
            ensure_head_node(w, "synthetic")

    # 3) build all nominal edges between head nodes
    edges = []
    seen = set()
    for nid, node in head_nodes.items():
        sid = node["token_sid"]
        wid = node["token_wid"]
        word = next(w for w in by_sid[sid] if w["wid"] == wid)
        rel = NOMINAL_DEPREL_MAP.get(word.get("deprel") or "")
        if not rel or word["head"] == 0:
            continue
        parent_id = head_node_ids.get((sid, word["head"]))
        if not parent_id or parent_id == nid:
            continue
        sem = semantic_role_for_relation(rel, word, by_sid[sid])
        edge = {
            "source": nid,
            "target": parent_id,
            "syntactic": rel,
            "semantic_role": sem,
        }
        key = (edge["source"], edge["target"], edge["syntactic"], edge["semantic_role"])
        if key not in seen:
            seen.add(key)
            edges.append(edge)

    out = dict(record)
    out["nominal_nodes"] = list(sorted(head_nodes.values(), key=lambda x: (x["start"], x["end"], x["id"])))
    out["nominal_edges"] = edges

    stats = {
        "span_count": len(spans),
        "head_node_count": len(out["nominal_nodes"]),
        "edge_count": len(edges),
        "synthetic_head_count": sum(1 for n in out["nominal_nodes"] if not n.get("matched_span_ids")),
    }
    return out, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build nominal head trees from spans with Stanza")
    parser.add_argument("--input", required=True, help="Input JSONL with text + spans")
    parser.add_argument("--output", required=True, help="Output JSONL with nominal head trees")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--download-models", action="store_true")
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
    nlp = stanza.Pipeline(lang="fr", processors="tokenize,pos,lemma,depparse", tokenize_no_ssplit=False, verbose=False)
    print("✅ Stanza ready")

    rows = []
    total_heads = 0
    total_edges = 0
    total_synth = 0
    for i, rec in enumerate(load_jsonl(Path(args.input)), start=1):
        if args.max_records > 0 and i > args.max_records:
            break
        out, stats = build_nominal_tree_record(rec, nlp)
        rows.append(out)
        total_heads += stats["head_node_count"]
        total_edges += stats["edge_count"]
        total_synth += stats["synthetic_head_count"]

    write_jsonl(Path(args.output), rows)
    print(f"✅ wrote {len(rows)} records to {args.output}")
    print(f"🌳 nominal head nodes: {total_heads}")
    print(f"🔗 nominal head edges: {total_edges}")
    print(f"➕ synthetic heads added: {total_synth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())









