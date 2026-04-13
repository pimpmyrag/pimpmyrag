#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import re

def bilou_tag(tokens, spans, text):
    """
    tokens: list of (token, start_char, end_char)
    spans: list of dicts: {"label": "...", "start": ..., "end": ...}
    """

    # Init all O
    tags = ["O"] * len(tokens)

    for sp in spans:
        lab = sp["label"]
        s = sp["start"]
        e = sp["end"]

        # collect token indices overlapping the span
        tok_idxs = []
        for i, (_, ts, te) in enumerate(tokens):
            if te > s and ts < e:
                tok_idxs.append(i)

        if not tok_idxs:
            continue

        if len(tok_idxs) == 1:
            tags[tok_idxs[0]] = f"U-{lab}"
        else:
            tags[tok_idxs[0]] = f"B-{lab}"
            for k in tok_idxs[1:-1]:
                tags[k] = f"I-{lab}"
            tags[tok_idxs[-1]] = f"L-{lab}"

    return tags


def tokenize_whitespace(text):
    tokens = []
    for match in re.finditer(r"\S+", text):
        tok = match.group(0)
        start = match.start()
        end = match.end()
        tokens.append((tok, start, end))
    return tokens


def convert_jsonl_to_bilou(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            text = obj["text"]
            spans = obj.get("spans", [])

            # whitespace-based tokens (HF-friendly)
            tokens = tokenize_whitespace(text)

            # bilou tags
            tags = bilou_tag(tokens, spans, text)

            # emit
            for (tok, _, _), lab in zip(tokens, tags):
                fout.write(f"{tok}\t{lab}\n")
            fout.write("\n")

if __name__ == "__main__":
    convert_jsonl_to_bilou(sys.argv[1], sys.argv[2])