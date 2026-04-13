#!/usr/bin/env python3
"""
Convert a generic jsonl (e.g. Mistral results) to the dataset.jsonl format
expected by the multi-head training code.

Usage:
  python3 convert_to_dataset_jsonl.py --input <input.jsonl> --output <out.jsonl> [--tokenizer microsoft/deberta-v3-base] [--max-lines N]

Behavior:
- Attempts to map provided labels to the `FINE_LABELS` / `COARSE_LABELS` defined in labels.py.
- Accepts span representations in either token indices (tok_start/tok_end) or char offsets
  (char_start/char_end). If char offsets are provided, converts to token indices using a
  HuggingFace fast tokenizer with add_special_tokens=False so tok indices correspond to the
  training format (the dataset builder expects tok indices without special tokens).
- Drops candidates whose token indices fall outside the tokenized text length.
- Emits a summary of processed / dropped candidates and writes the output jsonl.
"""

import argparse
import json
from typing import Optional
from pathlib import Path

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
from difflib import get_close_matches

# When this script is executed directly the package context may be missing,
# ensure the training/multi-head package folder is on sys.path so we can
# import `labels` module.
this_dir = Path(__file__).resolve().parent
parent_dir = this_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import labels

# Extract label maps from labels module; be robust if NONE variants were removed
FINE2ID = getattr(labels, "FINE2ID", {})
COARSE2ID = getattr(labels, "COARSE2ID", {})
FINE_LABELS = getattr(labels, "FINE_LABELS", [])
COARSE_LABELS = getattr(labels, "COARSE_LABELS", [])
FINE_NONE_ID = getattr(labels, "FINE_NONE_ID", None)
COARSE_NONE_ID = getattr(labels, "COARSE_NONE_ID", None)


def map_label_to_fine_id(label_val: Optional[str]):
    """Try to map arbitrary label value to a fine_id.
    Returns fine_id or FINE_NONE_ID if not found.
    """
    if label_val is None:
        return None
    if isinstance(label_val, int):
        # already an id
        if 0 <= label_val < len(FINE_LABELS):
            return int(label_val)
        return FINE_NONE_ID
    lab = str(label_val).strip()
    if lab in FINE2ID:
        return FINE2ID[lab]
    # try normalization
    lab_low = lab.lower()
    # direct substring match
    for fname in FINE2ID:
        if fname.lower() == lab_low or lab_low in fname.lower() or fname.lower() in lab_low:
            return FINE2ID[fname]
    # try fuzzy matching against known fine labels
    if FINE2ID:
        candidates = get_close_matches(lab, list(FINE2ID.keys()), n=1, cutoff=0.6)
        if candidates:
            return FINE2ID[candidates[0]]
    return None


def map_label_to_coarse_id(label_val: Optional[str]):
    if label_val is None:
        return None
    if isinstance(label_val, int):
        if 0 <= label_val < len(COARSE_LABELS):
            return int(label_val)
        return COARSE_NONE_ID
    lab = str(label_val).strip()
    if lab in COARSE2ID:
        return COARSE2ID[lab]
    lab_low = lab.lower()
    for cname in COARSE2ID:
        if cname.lower() == lab_low or lab_low in cname.lower() or cname.lower() in lab_low:
            return COARSE2ID[cname]
    return None


def char_span_to_token_span(char_start: int, char_end: int, offsets):
    """Map inclusive character span [char_start, char_end] to token indices (start_tok, end_tok)
    offsets is list of (char_start, char_end) per token. Returns (start_tok, end_tok) or None.
    We'll find the first token whose span intersects char_start and last token intersecting char_end.
    """
    # Use overlap heuristic: select tokens whose offset intersects the char span
    start_tok = None
    end_tok = None
    for i, (s, e) in enumerate(offsets):
        if e <= char_start:
            continue
        if s > char_end:
            break
        # token intersects span
        if start_tok is None:
            start_tok = i
        end_tok = i

    if start_tok is None or end_tok is None or start_tok > end_tok:
        return None
    return start_tok, end_tok


def process_file(input_path: Path, output_path: Path, tokenizer_name: str, max_lines: Optional[int] = None, include_coarse: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    total_lines = 0
    total_candidates = 0
    kept_candidates = 0
    dropped_candidates = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for raw in fin:
            if max_lines is not None and total_lines >= max_lines:
                break
            total_lines += 1
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)

            # Try to obtain the source text. The input may be a simple dict with
            # a `text` field, or it may be a model response object where the
            # generated content contains a JSON with `spans` (and maybe no full
            # text). We try several fallbacks.
            text = row.get("text") or row.get("sentence") or row.get("doc") or row.get("body")

            spans_input = None
            # If text is missing, try to parse assistant content
            if text is None:
                # common structure: response.body.choices[0].message.content
                resp = row.get("response") or row.get("output") or {}
                body = resp.get("body") if isinstance(resp, dict) else None
                choices = None
                if isinstance(body, dict):
                    choices = body.get("choices")
                if choices and isinstance(choices, list) and choices:
                    # message.content is often a string containing JSON
                    msg = choices[0].get("message") or choices[0].get("text") or {}
                    content = None
                    if isinstance(msg, dict):
                        content = msg.get("content")
                    elif isinstance(choices[0].get("message"), str):
                        content = choices[0].get("message")
                    if content:
                        try:
                            parsed = json.loads(content)
                            spans_input = parsed.get("spans")
                            # if parsed contains a full text field, prefer it
                            text = parsed.get("text") or parsed.get("sentence")
                        except Exception:
                            # content may be JSON-like but with newlines; try to extract a JSON substring
                            try:
                                start = content.find('{')
                                if start != -1:
                                    parsed = json.loads(content[start:])
                                    spans_input = parsed.get("spans")
                                    text = parsed.get("text") or parsed.get("sentence")
                            except Exception:
                                spans_input = None

            # If spans_input not found yet, also try direct keys on the row
            if spans_input is None:
                spans_input = row.get("spans") or row.get("entities") or row.get("annotations")

            # If still no text but we have spans with embedded 'text', reconstruct a compact text
            # by joining span texts with single spaces. This avoids huge gaps of spaces when
            # original text is missing and still allows mapping spans to token indices by
            # searching span text sequentially in the reconstructed text.
            reconstructed_from_spans = False
            if text is None and spans_input:
                spans_list = [s for s in (spans_input if isinstance(spans_input, list) else []) if isinstance(s, dict)]
                span_texts = [s.get("text", "").strip() for s in spans_list if s.get("text")]
                if span_texts:
                    # join with single space to produce readable text
                    text = " ".join(span_texts)
                    reconstructed_from_spans = True

            if text is None:
                # can't process entries without any text context: skip
                continue

            # Always normalize consecutive whitespace to single spaces to avoid huge gaps
            # (this applies both to original texts and to texts reconstructed from spans).
            text = " ".join(str(text).split())

            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            token_len = len(enc["input_ids"])

            out_candidates = []
            # source candidates may be in multiple places: prefer explicit 'candidates', else use spans_input
            in_candidates = row.get("candidates") or spans_input or row.get("spans") or []
            # If we reconstructed the text from spans, we'll use a sequential search
            # position to map each candidate's 'text' to character offsets inside our
            # reconstructed string.
            search_pos = 0
            for cand in in_candidates:
                total_candidates += 1
                # detect char vs token spans
                tok_start = cand.get("tok_start")
                tok_end = cand.get("tok_end")
                if tok_start is None or tok_end is None:
                    # If the converter reconstructed the text from spans, try to find the
                    # candidate text sequentially in the reconstructed string. This is
                    # robust and avoids relying on original char offsets which are
                    # unavailable relative to our reconstructed text.
                    if reconstructed_from_spans and cand.get("text"):
                        cand_text = str(cand.get("text")).strip()
                        if not cand_text:
                            dropped_candidates += 1
                            continue
                        idx = text.find(cand_text, search_pos)
                        if idx == -1:
                            # try case-insensitive search fallback
                            idx = text.lower().find(cand_text.lower(), search_pos)
                        if idx == -1:
                            dropped_candidates += 1
                            continue
                        cs = idx
                        ce = idx + len(cand_text)
                        mapped = char_span_to_token_span(int(cs), int(ce)-1, offsets)
                        if mapped is None:
                            dropped_candidates += 1
                            continue
                        tok_start, tok_end = mapped
                        search_pos = ce
                    else:
                        # try char spans from source if available
                        cs = cand.get("char_start") or cand.get("start_char") or cand.get("start")
                        ce = cand.get("char_end") or cand.get("end_char") or cand.get("end")
                        if cs is None or ce is None:
                            # cannot map span: drop
                            dropped_candidates += 1
                            continue
                        mapped = char_span_to_token_span(int(cs), int(ce)-1, offsets)
                        if mapped is None:
                            dropped_candidates += 1
                            continue
                        tok_start, tok_end = mapped
                else:
                    tok_start = int(tok_start)
                    tok_end = int(tok_end)
                # validate token indices
                if tok_start < 0 or tok_end < 0 or tok_start >= token_len or tok_end >= token_len or tok_start > tok_end:
                    dropped_candidates += 1
                    continue

                # label mapping heuristics
                # accept various keys: fine_label / fine / fine_label_id or 'label' (from model outputs)
                fine_val = cand.get("fine_label") or cand.get("fine") or cand.get("fine_label_id") or cand.get("label")
                coarse_val = cand.get("coarse_label") or cand.get("coarse") or cand.get("coarse_label_id")
                boundary_val = cand.get("boundary_label") if "boundary_label" in cand else cand.get("boundary")

                fine_id = map_label_to_fine_id(fine_val)
                if fine_id is None:
                    # cannot map fine label -> drop candidate
                    dropped_candidates += 1
                    continue

                coarse_id = None
                if include_coarse:
                    if isinstance(coarse_val, int):
                        ctmp = map_label_to_coarse_id(coarse_val)
                        coarse_id = ctmp if ctmp is not None else None
                    elif coarse_val is not None:
                        ctmp = map_label_to_coarse_id(coarse_val)
                        coarse_id = ctmp if ctmp is not None else None
                    else:
                        # try to infer coarse from fine via labels.fine_to_coarse_id if available
                        if hasattr(labels, "fine_to_coarse_id"):
                            try:
                                coarse_id = labels.fine_to_coarse_id(fine_id)
                            except Exception:
                                coarse_id = None
                        else:
                            coarse_id = None

                if boundary_val is None:
                    # default: presence of candidate => boundary=1
                    boundary_label = 1
                else:
                    try:
                        boundary_label = int(boundary_val)
                    except Exception:
                        boundary_label = 1

                sample_weight = float(cand.get("sample_weight", cand.get("weight", 1.0)))
                neg_type = cand.get("neg_type", cand.get("neg", "unknown"))

                entry = {
                    "tok_start": tok_start,
                    "tok_end": tok_end,
                    "boundary_label": int(boundary_label),
                    "fine_label_id": int(fine_id),
                    "sample_weight": float(sample_weight),
                    "neg_type": neg_type,
                }
                if include_coarse and (coarse_id is not None):
                    entry["coarse_label_id"] = int(coarse_id)
                out_candidates.append(entry)
                kept_candidates += 1

            out_obj = {
                "id": row.get("id") or f"line_{total_lines}",
                "text": text,
                "candidates": out_candidates,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    summary = {
        "lines_read": total_lines,
        "total_candidates": total_candidates,
        "kept_candidates": kept_candidates,
        "dropped_candidates": dropped_candidates,
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="microsoft/deberta-v3-base")
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--no-coarse", action="store_true", help="Do not produce coarse_label_id in output")

    args = parser.parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    include_coarse = not args.no_coarse
    summary = process_file(inp, out, args.tokenizer, args.max_lines, include_coarse=include_coarse)
    print("Conversion terminée:")
    print(summary)

