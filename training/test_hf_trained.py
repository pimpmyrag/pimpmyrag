#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# -----------------------------
# CONFIG
# -----------------------------
MODEL_DIR = "outputs/xml_ner_bilou"     # <-- ton dossier
LABELS = [
    "O",
    "B-PER","I-PER","L-PER","U-PER",
    "B-LOC","I-LOC","L-LOC","U-LOC",
    "B-OBJECT","I-OBJECT","L-OBJECT","U-OBJECT",
    "B-ORG","I-ORG","L-ORG","U-ORG",
    "B-TIME","I-TIME","L-TIME","U-TIME",
    "B-EVENT","I-EVENT","L-EVENT","U-EVENT"
]

# -----------------------------
# INPUT TEXT
# -----------------------------
if len(sys.argv) > 1:
    text = " ".join(sys.argv[1:])
else:
    text = "Les policiers ont tiré sur les manifestants au fond de la place de la République à 2h44 du matin."

print("Loading model…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

# -----------------------------
# TOKENIZE THE INPUT
# -----------------------------
enc = tokenizer(
    text,
    return_tensors="pt",
    return_offsets_mapping=True,
    add_special_tokens=True
)

input_ids = enc["input_ids"]
offsets = enc["offset_mapping"][0].tolist()

# -----------------------------
# RUN INFERENCE
# -----------------------------
with torch.no_grad():
    outputs = model(**enc)
logits = outputs.logits[0]

pred_ids = torch.argmax(logits, dim=-1).tolist()
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# -----------------------------
# PRINT TOKENS + LABELS
# -----------------------------
print("\n--- TOKENS + LABELS ---")
for t, pid in zip(tokens, pred_ids):
    print(f"{t:20s} {LABELS[pid]}")

# -----------------------------
# RECONSTRUCT ENTITIES (BILOU)
# -----------------------------
print("\n--- ENTITIES ---")
i = 0
while i < len(pred_ids):
    lab = LABELS[pred_ids[i]]
    if lab == "O":
        i += 1
        continue

    tag, typ = lab.split("-", 1)

    # U-XXX
    if tag == "U":
        start_char, end_char = offsets[i]
        print(f"{typ}: {text[start_char:end_char]}")
        i += 1
        continue

    # B-XXX
    if tag == "B":
        start_idx = i
        i += 1

        while i < len(pred_ids):
            next_lab = LABELS[pred_ids[i]]

            if next_lab == f"I-{typ}":
                i += 1
                continue

            if next_lab == f"L-{typ}":
                end_idx = i
                i += 1
                break

            # Wrong continuation → stop entity
            break

        # char offsets from subword positions
        start_char = offsets[start_idx][0]
        end_char = offsets[end_idx][1]

        print(f"{typ}: {text[start_char:end_char]}")
        continue

    # If tag is weird (I- or L- without B-), skip
    i += 1