#!/usr/bin/env python3
"""
Convert model outputs (jsonl with nested response choices) to dataset.jsonl format:

Output schema per line:
{
  "id": "all_000535",
  "text": "...",
  "spans": [ {"label": "hint_norp", "start": 22, "end": 26, "text": "Boks"}, ... ]
}

Usage:
  python3 convert_to_dataset_jsonl_raw.py --input <in.jsonl> --output <out.jsonl> [--max-lines N] [--tokenizer microsoft/deberta-v3-base]

Behavior:
- Prefers an existing `text` field in the input record; otherwise tries to parse
  assistant content JSON (response.body.choices[0].message.content) and uses its
  `text` if present. If the text is still missing, reconstructs text by joining
  span texts with single spaces.
- Accepts span entries with keys (label,start,end,text). Detects whether start
  are 1-based and converts to 0-based. Ensures `end` is exclusive. If offsets do
  not match the substring, attempts to find the span text inside `text` (first
  occurrence after previous span) and adjusts start/end accordingly.
- Verifies labels against `training/multi-head/labels.py` FINE_LABELS and applies
  fuzzy matching; if mapping fails, keeps the original label but logs a warning.
"""

import argparse
import json
from pathlib import Path
import sys
from difflib import get_close_matches

# ensure module path
this_dir = Path(__file__).resolve().parent
parent_dir = this_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import labels

FINE_LABELS = getattr(labels, "FINE_LABELS", [])


def normalize_whitespace(s: str) -> str:
    return " ".join(s.split())


def map_label(label: str):
    if label in FINE_LABELS:
        return label
    # try exact lowercase
    for f in FINE_LABELS:
        if f.lower() == label.lower():
            return f
    # fuzzy
    cand = get_close_matches(label, FINE_LABELS, n=1, cutoff=0.6)
    if cand:
        return cand[0]
    return label


def extract_spans_from_parsed(parsed):
    # parsed expected to be dict containing 'spans' list
    spans = parsed.get("spans") if isinstance(parsed, dict) else None
    if not spans:
        return []
    out = []
    for s in spans:
        if not isinstance(s, dict):
            continue
        label = s.get("label") or s.get("fine_label") or s.get("type")
        start = s.get("start")
        end = s.get("end")
        text = s.get("text")
        out.append({"label": label, "start": start, "end": end, "text": text})
    return out


def parse_row(raw):
    row = json.loads(raw)
    # id selection
    rid = row.get("id") or row.get("custom_id") or row.get("doc_id") or row.get("uid") or row.get("_id") or None
    # prefer top-level text
    text = row.get("text") or row.get("sentence") or row.get("body") or None

    spans = row.get("spans") or row.get("entities") or None

    # try assistant-model nested structure
    if text is None or spans is None:
        resp = row.get("response") or row.get("output") or None
        if isinstance(resp, dict):
            body = resp.get("body") if isinstance(resp.get("body"), dict) else resp
            choices = None
            if isinstance(body, dict):
                choices = body.get("choices")
            if choices and isinstance(choices, list) and choices:
                choice = choices[0]
                msg = choice.get("message") or choice.get("text") or choice
                content = None
                if isinstance(msg, dict):
                    content = msg.get("content")
                elif isinstance(msg, str):
                    content = msg
                if content:
                    # try parse content as JSON
                    try:
                        parsed = json.loads(content)
                    except Exception:
                        # try to find first { ... }
                        try:
                            start = content.find('{')
                            if start != -1:
                                parsed = json.loads(content[start:])
                            else:
                                parsed = {}
                        except Exception:
                            parsed = {}
                    if text is None:
                        text = parsed.get("text") or parsed.get("sentence")
                    if spans is None:
                        spans = extract_spans_from_parsed(parsed)

    # If still no text but parsed spans include texts, reconstruct compact text
    reconstructed = False
    if text is None and spans:
        span_texts = [s.get("text", "") for s in spans if s.get("text")]
        if span_texts:
            text = " ".join(t.strip() for t in span_texts)
            reconstructed = True

    if text is None:
        # cannot produce dataset entry without any text
        return None

    text = normalize_whitespace(text)

    # normalize spans: ensure list of dicts label/start/end/text
    spans_list = []
    if spans and isinstance(spans, list):
        for s in spans:
            if not isinstance(s, dict):
                continue
            lab = s.get("label") or s.get("fine_label") or s.get("type")
            st = s.get("start")
            ed = s.get("end")
            tx = s.get("text")
            spans_list.append({"label": lab, "start": st, "end": ed, "text": tx})

    return {"id": rid, "text": text, "spans": spans_list, "reconstructed": reconstructed}


def detect_and_fix_offsets(item):
    # item: dict id,text,spans
    text = item["text"]
    spans = item["spans"]
    if not spans:
        return []
    # detect if starts are 1-based: if min start >=1
    starts = [s["start"] for s in spans if isinstance(s.get("start"), int)]
    one_based = False
    if starts and min(starts) >= 1:
        one_based = True
    fixed = []
    cursor = 0
    for s in spans:
        lab = s.get("label")
        st = s.get("start")
        ed = s.get("end")
        tx = s.get("text")
        if isinstance(st, int) and isinstance(ed, int):
            if one_based:
                st0 = st - 1
                ed0 = ed - 1
            else:
                st0 = st
                ed0 = ed
            # interpret ed as exclusive if ed - st == len(text)? check
            # if ed0 <= st0: treat as invalid
            if ed0 <= st0:
                st0 = None
                ed0 = None
            else:
                # if substring doesn't match, we'll try to find tx sequentially
                try:
                    sub = text[st0:ed0]
                    if tx is not None and normalize_whitespace(str(tx)) != normalize_whitespace(sub):
                        # mismatch -> try to find tx in text after cursor
                        if tx:
                            idx = text.find(str(tx), cursor)
                            if idx == -1:
                                idx = text.lower().find(str(tx).lower(), cursor)
                            if idx != -1:
                                st0 = idx
                                ed0 = idx + len(str(tx))
                                cursor = ed0
                            else:
                                # keep original but mark as best-effort
                                pass
                        else:
                            pass
                    else:
                        cursor = ed0
                except Exception:
                    st0 = None
                    ed0 = None
        else:
            # no numeric offsets; try to find text in the big string
            if tx:
                idx = text.find(str(tx), cursor)
                if idx == -1:
                    idx = text.lower().find(str(tx).lower(), cursor)
                if idx != -1:
                    st0 = idx
                    ed0 = idx + len(str(tx))
                    cursor = ed0
                else:
                    st0 = None
                    ed0 = None
            else:
                st0 = None
                ed0 = None

        # if still None try global find
        if (st0 is None or ed0 is None) and tx:
            idx = text.find(str(tx))
            if idx == -1:
                idx = text.lower().find(str(tx).lower())
            if idx != -1:
                st0 = idx
                ed0 = idx + len(str(tx))
        # final check
        if st0 is None or ed0 is None:
            # skip this span (could not map)
            continue
        fixed.append({"label": map_label(str(lab)) if lab else lab, "start": int(st0), "end": int(ed0), "text": text[st0:ed0]})
    return fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-lines", type=int, default=None)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    wrote = 0
    bad_spans = 0
    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for raw in fin:
            if args.max_lines is not None and total >= args.max_lines:
                break
            total += 1
            raw = raw.strip()
            if not raw:
                continue
            parsed = parse_row(raw)
            if parsed is None:
                continue
            fixed = detect_and_fix_offsets(parsed)
            if not fixed:
                # still write entry with empty spans
                obj = {"id": parsed.get("id") or f"line_{total:06d}", "text": parsed["text"], "spans": []}
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                wrote += 1
                continue
            obj = {"id": parsed.get("id") or f"line_{total:06d}", "text": parsed["text"], "spans": fixed}
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            wrote += 1
            bad_spans += max(0, len(parsed.get("spans", [])) - len(fixed))

    print(f"Processed {total} lines, wrote {wrote} entries, dropped {bad_spans} spans")


if __name__ == '__main__':
    main()

