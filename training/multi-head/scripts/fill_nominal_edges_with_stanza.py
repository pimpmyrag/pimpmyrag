#!/usr/bin/env python3
"""Fill graph `edges` from Stanza nominal dependencies.

Inputs:
- graph JSONL: records with {id, text, nodes, edges, events, discourse}
- spans JSONL: records with {id, text, spans} (same ids), used for offsets

Output:
- graph JSONL with `edges` populated/merged from Stanza nominal relations

Conservative behavior:
- does not alter nodes/events/discourse
- creates only nominal edges (APPOS, NMOD, AMOD, COMPOUND)
- keeps existing edges and appends new unique ones
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

NOMINAL_DEPREL_MAP = {
    "appos": "APPOS",
    "nmod": "NMOD",
    "nmod:poss": "NMOD",
    "amod": "AMOD",
    "compound": "COMPOUND",
    "flat": "COMPOUND",
    "flat:name": "COMPOUND",
    "flat:foreign": "COMPOUND",
}

CASE_DE = {"de", "du", "des", "d'", "d’"}
CASE_LOC = {"à", "au", "aux", "dans", "en", "sur", "chez", "vers"}

SKIP_LABELS = {"verb_trigger", "pron_subj", "pron_obj"}


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
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s


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


def map_nodes_to_spans(nodes: list[dict], spans: list[dict]) -> dict[str, dict] | None:
    """Map node id -> span with stable matching by (label,text), ordered by start."""
    by_key = defaultdict(list)
    for sp in sorted(spans, key=lambda x: (x.get("start", 0), x.get("end", 0))):
        if sp.get("label") in SKIP_LABELS:
            continue
        key = (sp.get("label"), norm(sp.get("text", "")))
        by_key[key].append(sp)

    node_to_span = {}
    for n in sorted(nodes, key=lambda x: x.get("id", "")):
        key = (n.get("ner_fine"), norm(n.get("text", "")))
        candidates = by_key.get(key, [])
        if not candidates:
            # cannot map this node reliably
            continue
        sp = candidates.pop(0)
        node_to_span[n["id"]] = sp

    return node_to_span


def pick_head_word(span: dict, words: list[dict]) -> dict | None:
    s0, s1 = int(span["start"]), int(span["end"])
    covered = [w for w in words if overlap(s0, s1, w["start"], w["end"]) and w["upos"] != "PUNCT"]

    # Fallback for imperfect offsets: token text containment near expected span start.
    if not covered:
        span_fold = fold(span.get("text", ""))
        if span_fold:
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
                covered = near
            elif len(global_match) == 1:
                covered = global_match

    if not covered:
        return None

    covered_ids = {w["wid"] for w in covered}
    outside = [w for w in covered if w["head"] == 0 or w["head"] not in covered_ids]
    if outside:
        covered = outside

    # prefer nominal heads
    for upos in ("NOUN", "PROPN", "PRON", "ADJ"):
        cand = [w for w in covered if w["upos"] == upos]
        if cand:
            return sorted(cand, key=lambda x: x["wid"])[0]

    return sorted(covered, key=lambda x: x["wid"])[0]


def semantic_role_for_relation(rel: str, child: dict, parent: dict, words_by_sid_wid: dict, sentence_words: list[dict]) -> str:
    if rel == "APPOS":
        return "IDENTITY"

    if rel == "NMOD":
        # Look for case marker attached to child head (UD: case -> dependent of noun)
        case_tokens = [
            w for w in sentence_words
            if w["head"] == child["wid"] and (w["deprel"] or "").startswith("case")
        ]
        case_lemmas = {norm(w.get("lemma", "")) for w in case_tokens}
        if case_lemmas & CASE_DE:
            return "CONTENT"
        if case_lemmas & CASE_LOC:
            return "LOCATION"

    return "NONE"


def extract_nominal_chunk(head_word: dict, sentence_words: list[dict]) -> dict:
    """Build a conservative candidate chunk around a nominal head token."""
    by_head = defaultdict(list)
    by_wid = {}
    for w in sentence_words:
        by_head[w["head"]].append(w)
        by_wid[w["wid"]] = w

    keep_deps = {
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

    keep_ids = {head_word["wid"]}
    queue = [head_word["wid"]]
    seen = set()
    while queue:
        wid = queue.pop(0)
        if wid in seen:
            continue
        seen.add(wid)
        for ch in by_head.get(wid, []):
            if ch["deprel"] in keep_deps and ch["upos"] != "PUNCT":
                if ch["wid"] not in keep_ids:
                    keep_ids.add(ch["wid"])
                    queue.append(ch["wid"])

    kept = [by_wid[w] for w in sorted(keep_ids) if w in by_wid]
    if not kept:
        kept = [head_word]

    start = min(w["start"] for w in kept)
    end = max(w["end"] for w in kept)
    return {
        "start": start,
        "end": end,
        "head_text": head_word["text"],
        "head_lemma": head_word.get("lemma"),
        "head_upos": head_word.get("upos"),
        "token_count": len(kept),
    }


def build_stanza_edges_for_record(text: str, nodes: list[dict], spans: list[dict], nlp) -> tuple[list[dict], list[dict]]:
    doc = nlp(text)
    words = build_words(doc)
    if not words:
        return [], []

    node_to_span = map_nodes_to_spans(nodes, spans)
    if not node_to_span:
        return [], []

    # node_id -> head word
    node_head = {}
    head_to_node = {}
    for nid, sp in node_to_span.items():
        hw = pick_head_word(sp, words)
        if hw is None:
            continue
        node_head[nid] = hw
        head_to_node[(hw["sid"], hw["wid"])] = nid

    words_by_sid_wid = {(w["sid"], w["wid"]): w for w in words}
    words_by_sid = defaultdict(list)
    for w in words:
        words_by_sid[w["sid"]].append(w)

    edges = []
    pending = []
    for nid, hw in node_head.items():
        rel_raw = hw.get("deprel") or ""
        rel = NOMINAL_DEPREL_MAP.get(rel_raw)
        if not rel:
            continue
        if hw["head"] == 0:
            continue

        parent_key = (hw["sid"], hw["head"])
        parent_word = words_by_sid_wid.get(parent_key)
        if not parent_word:
            continue

        parent_nid = head_to_node.get(parent_key)
        if parent_nid == nid:
            continue

        sem = semantic_role_for_relation(rel, hw, parent_word, words_by_sid_wid, words_by_sid[hw["sid"]])
        if parent_nid:
            edges.append(
                {
                    "type": "nominal",
                    "source": nid,
                    "target": parent_nid,
                    "syntactic": rel,
                    "semantic_role": sem,
                }
            )
        else:
            cand = extract_nominal_chunk(parent_word, words_by_sid[hw["sid"]])
            cand_text = text[cand["start"]:cand["end"]]
            pending.append(
                {
                    "source_node_id": nid,
                    "syntactic": rel,
                    "semantic_role": sem,
                    "missing_side": "target",
                    "target_candidate": {
                        "text": cand_text,
                        "start": cand["start"],
                        "end": cand["end"],
                        "head_text": cand["head_text"],
                        "head_lemma": cand["head_lemma"],
                        "head_upos": cand["head_upos"],
                        "token_count": cand["token_count"],
                    },
                }
            )

    # dedup
    seen = set()
    out = []
    for e in edges:
        k = (e["source"], e["target"], e["syntactic"], e["semantic_role"])
        if k not in seen:
            seen.add(k)
            out.append(e)
    # dedup pending
    p_seen = set()
    p_out = []
    for p in pending:
        tc = p["target_candidate"]
        k = (
            p["source_node_id"],
            p["syntactic"],
            tc["start"],
            tc["end"],
            tc["head_text"],
        )
        if k not in p_seen:
            p_seen.add(k)
            p_out.append(p)

    return out, p_out


def merge_edges(existing: list[dict], new_edges: list[dict]) -> list[dict]:
    merged = []
    seen = set()

    for e in existing + new_edges:
        # support both old/new keys
        source = e.get("source") or e.get("source_id")
        target = e.get("target") or e.get("target_id")
        syntactic = e.get("syntactic") or e.get("syntactic_role") or e.get("relation")
        semantic = e.get("semantic_role") or "NONE"
        if not source or not target or not syntactic:
            continue

        e2 = {
            "type": "nominal",
            "source": source,
            "target": target,
            "syntactic": syntactic,
            "semantic_role": semantic,
        }
        k = (e2["source"], e2["target"], e2["syntactic"], e2["semantic_role"])
        if k not in seen:
            seen.add(k)
            merged.append(e2)

    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill graph nominal edges from Stanza")
    parser.add_argument("--graph-input", required=True, help="Graph JSONL input")
    parser.add_argument("--spans-input", required=True, help="Spans JSONL input (same ids)")
    parser.add_argument("--output", required=True, help="Graph JSONL output with edges")
    parser.add_argument("--max-records", type=int, default=0, help="Limit rows (0=all)")
    parser.add_argument("--replace-edges", action="store_true", help="Replace edges instead of merge")
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

    graph_rows = list(load_jsonl(Path(args.graph_input)))
    spans_by_id = {r.get("id"): r for r in load_jsonl(Path(args.spans_input))}

    out_rows = []
    total_new = 0
    for i, gr in enumerate(graph_rows, start=1):
        if args.max_records > 0 and i > args.max_records:
            break

        rid = gr.get("id")
        sr = spans_by_id.get(rid)
        if not sr:
            out_rows.append(gr)
            continue

        new_edges, pending_links = build_stanza_edges_for_record(gr.get("text", ""), gr.get("nodes", []), sr.get("spans", []), nlp)
        total_new += len(new_edges)

        if args.replace_edges:
            gr["edges"] = new_edges
        else:
            gr["edges"] = merge_edges(gr.get("edges", []), new_edges)

        # Keep unresolved nominal relations for later NER matching.
        gr["pending_nominal_links"] = pending_links

        out_rows.append(gr)

    write_jsonl(Path(args.output), out_rows)
    print(f"✅ wrote {len(out_rows)} records to {args.output}")
    print(f"📌 stanza nominal edges produced: {total_new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

