#!/usr/bin/env python3
"""
Validate a converted dataset.jsonl produced by convert_to_dataset_jsonl.py

Usage:
  python3 validate_converted.py --input path/to/converted.jsonl --tokenizer microsoft/deberta-v3-base --out-bad bad_examples.jsonl --max-sample 100

Outputs a concise summary to stdout and writes a JSONL of bad examples if requested.
"""
import argparse
import json
from pathlib import Path
import sys

# Ensure package imports work when script is executed directly
this_dir = Path(__file__).resolve().parent
parent_dir = this_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import labels
from transformers import AutoTokenizer

# Extract constants from labels module; provide a fallback for fine_to_coarse_id
FINE_LABELS = getattr(labels, "FINE_LABELS")
COARSE_LABELS = getattr(labels, "COARSE_LABELS")
if hasattr(labels, "fine_to_coarse_id"):
    fine_to_coarse_id = labels.fine_to_coarse_id
else:
    # build inverse mapping from COARSE_TO_FINE if available
    if hasattr(labels, "COARSE_TO_FINE"):
        COARSE_TO_FINE = labels.COARSE_TO_FINE
        def fine_to_coarse_id(fine_id: int) -> int:
            for coarse_id, fine_ids in COARSE_TO_FINE.items():
                if fine_id in fine_ids:
                    return coarse_id
            # fallback to NONE index if available
            return getattr(labels, "COARSE_NONE_ID", len(COARSE_LABELS)-1)
    else:
        # last-resort: map everything to NONE
        def fine_to_coarse_id(fine_id: int) -> int:
            return getattr(labels, "COARSE_NONE_ID", len(COARSE_LABELS)-1)


def validate_file(path: Path, tokenizer_name: str, max_bad: int = 100):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    lines = 0
    candidates = 0
    bad_spans = 0
    bad_labels = 0
    incompat = 0
    missing_fields = 0
    bad_examples = []

    with path.open("r", encoding="utf-8") as fin:
        for raw in fin:
            lines += 1
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception as e:
                missing_fields += 1
                if len(bad_examples) < max_bad:
                    bad_examples.append({"id": None, "error": f"invalid_json: {e}", "raw": raw})
                continue

            text = obj.get("text", "")
            cand_list = obj.get("candidates")
            if cand_list is None:
                missing_fields += 1
                if len(bad_examples) < max_bad:
                    bad_examples.append({"id": obj.get("id"), "error": "missing_candidates", "obj": obj})
                continue

            enc = tokenizer(text, add_special_tokens=False)
            tlen = len(enc["input_ids"]) if enc and "input_ids" in enc else 0

            for cand in cand_list:
                candidates += 1
                ts = cand.get("tok_start")
                te = cand.get("tok_end")
                fine = cand.get("fine_label_id")
                coarse = cand.get("coarse_label_id")

                bad = False
                msgs = []
                if ts is None or te is None:
                    bad_spans += 1
                    bad = True
                    msgs.append("missing_tok_start_end")
                else:
                    try:
                        ts_i = int(ts)
                        te_i = int(te)
                        if ts_i < 0 or te_i < 0 or ts_i > te_i or te_i >= tlen:
                            bad_spans += 1
                            bad = True
                            msgs.append(f"tok_range_out_of_bounds: {ts_i}-{te_i} >= token_len {tlen}")
                    except Exception:
                        bad_spans += 1
                        bad = True
                        msgs.append("tok_start_end_not_int")

                if fine is None or coarse is None:
                    bad_labels += 1
                    bad = True
                    msgs.append("missing_label_ids")
                else:
                    try:
                        fine_i = int(fine)
                        coarse_i = int(coarse)
                        if not (0 <= fine_i < len(FINE_LABELS)):
                            bad_labels += 1
                            bad = True
                            msgs.append(f"fine_out_of_range: {fine_i}")
                        if not (0 <= coarse_i < len(COARSE_LABELS)):
                            bad_labels += 1
                            bad = True
                            msgs.append(f"coarse_out_of_range: {coarse_i}")
                        # compatibility check
                        try:
                            expected_coarse = fine_to_coarse_id(fine_i)
                            if expected_coarse != coarse_i:
                                incompat += 1
                                bad = True
                                msgs.append(f"incompatible fine->{fine_i} expects coarse {expected_coarse}, got {coarse_i}")
                        except Exception as e:
                            bad_labels += 1
                            bad = True
                            msgs.append(f"fine_to_coarse_error: {e}")
                    except Exception:
                        bad_labels += 1
                        bad = True
                        msgs.append("label_ids_not_int")

                if bad and len(bad_examples) < max_bad:
                    ex = {
                        "id": obj.get("id"),
                        "text_sample": text[:240],
                        "candidate": cand,
                        "token_len": tlen,
                        "msgs": msgs,
                    }
                    bad_examples.append(ex)

    summary = {
        "lines": lines,
        "candidates": candidates,
        "bad_spans": bad_spans,
        "bad_labels": bad_labels,
        "incompat_fine_to_coarse": incompat,
        "missing_entries": missing_fields,
        "bad_examples_reported": len(bad_examples),
    }
    return summary, bad_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", default="microsoft/deberta-v3-base")
    parser.add_argument("--out-bad", default=None, help="Write bad examples to this JSONL path")
    parser.add_argument("--max-sample", type=int, default=100, help="Max bad examples to store")
    parser.add_argument("--require-coarse", action="store_true", help="Treat missing coarse_label_id as an error")

    args = parser.parse_args()
    inp = Path(args.input)
    if not inp.exists():
        print(f"ERROR: input missing: {inp}")
        raise SystemExit(2)

    summary, bad_examples = validate_file(inp, args.tokenizer, max_bad=args.max_sample)
    if args.require_coarse:
        # count missing coarse as errors by re-checking examples briefly
        missing_coarse = 0
        for ex in bad_examples:
            cand = ex.get("candidate", {})
            if "coarse_label_id" not in cand:
                missing_coarse += 1
        if missing_coarse:
            print(f"Note: {missing_coarse} bad examples are missing coarse_label_id (require-coarse enabled)")
    print("Validation summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.out_bad:
        outp = Path(args.out_bad)
        with outp.open("w", encoding="utf-8") as fout:
            for ex in bad_examples:
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(bad_examples)} bad examples to {outp}")
    elif bad_examples:
        print("Bad examples (first few):")
        for ex in bad_examples[:10]:
            print(json.dumps(ex, ensure_ascii=False))

