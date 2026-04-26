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

# Mapping hint_* → labels coarses pour RoBERTa NER
# RoBERTa est un tagger coarse ; le raffinement hint_* est géré par DeBERTa.
HINT_TO_COARSE = {
    # PER
    "hint_person_name":   "PER",
    "hint_person_role":   "PER",
    "hint_group_role":    "PER",
    "hint_norp":          "PER",
    # LOC
    "hint_gpe":           "LOC",
    "hint_loc_generic":   "LOC",
    "hint_fac_name":      "LOC",
    "hint_infra":         "LOC",
    # ORG
    "hint_org_name":      "ORG",
    # TIME
    "hint_time_date":     "TIME",
    "hint_time_clock":    "TIME",
    "hint_time_duration": "TIME",
    # EVENT
    "hint_event_named":   "EVENT",
    "hint_event_nominal": "EVENT",
    # OBJECT
    "hint_object_generic": "OBJECT",
    "hint_object_name":    "OBJECT",
    "hint_vehicle":        "OBJECT",
    "hint_weapon":         "OBJECT",
    "hint_tool":           "OBJECT",
    "hint_food":           "OBJECT",
    "hint_substance":      "OBJECT",
    "hint_quantity":       "OBJECT",
}

def map_label(lab: str) -> str:
    """Mappe un label hint_* vers son label coarse. Passe les labels déjà coarses tels quels."""
    return HINT_TO_COARSE.get(lab, lab)

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

            # Tokenisation XLM-R EXACTE
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False
            )
            tokens = enc.tokens()
            offsets = enc["offset_mapping"]

            # Init labels
            labels = ["O"] * len(tokens)

            # Assign BILOU avec mapping hint_* → coarse
            for sp in spans:
                s, e = sp["start"], sp["end"]
                lab = map_label(sp["label"])

                idxs = [i for i,(ts,te) in enumerate(offsets) if te > s and ts < e]

                if not idxs:
                    continue

                if len(idxs) == 1:
                    labels[idxs[0]] = f"U-{lab}"
                else:
                    labels[idxs[0]] = f"B-{lab}"
                    for k in idxs[1:-1]:
                        labels[k] = f"I-{lab}"
                    labels[idxs[-1]] = f"L-{lab}"

            # ÉCRITURE : garder les tokens EXACTS, y compris ▁
            for tok, lab in zip(tokens, labels):
              if tok == "▁":
                continue  # skip invalid standalone underscore tokens
              fout.write(f"{tok}\t{lab}\n")
            fout.write("\n")
    print("✅ BILOU généré (tokens XLM-R EXACTS, sans modification).")

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
