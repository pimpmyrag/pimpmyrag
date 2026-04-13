#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import re
import unicodedata
from transformers import AutoTokenizer

MODEL = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

HTML_TAG = re.compile(r"<[^>]+>")
BRACKETS = re.compile(r"[\[\]]+")
SPACE = re.compile(r"\s+")

def clean_text(t):
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("’", "'")
    t = HTML_TAG.sub(" ", t)
    t = BRACKETS.sub(" ", t)
    t = SPACE.sub(" ", t).strip()
    return t

def convert(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            obj = json.loads(line)
            raw = obj["text"]
            spans = obj.get("spans", [])

            text = clean_text(raw)

            # 1) Tokenisation XLM-R
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False
            )
            tokens = enc.tokens()
            offsets = enc["offset_mapping"]

            # 2) init BILOU
            labels = ["O"] * len(tokens)

            # 3) assign labels
            for sp in spans:
                s, e = sp["start"], sp["end"]
                lab = sp["label"]

                idxs = []
                for i, (ts, te) in enumerate(offsets):
                    if te > s and ts < e:
                        idxs.append(i)

                if not idxs:
                    continue

                if len(idxs) == 1:
                    labels[idxs[0]] = f"U-{lab}"
                else:
                    labels[idxs[0]] = f"B-{lab}"
                    for k in idxs[1:-1]:
                        labels[k] = f"I-{lab}"
                    labels[idxs[-1]] = f"L-{lab}"

            # 4) remove ▁ (just for readability)
            for tok, lab in zip(tokens, labels):
                fout.write(f"{tok.replace('▁','')}\t{lab}\n")
            fout.write("\n")

    print("✅ BILOU propre généré (segmentation XLM-R).")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])