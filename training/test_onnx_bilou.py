#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import sys

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MODEL_DIR = "outputs/xml_ner_bilou"          # dossier où il y a config.json / tokenizer.json
ONNX_PATH = "outputs/xml_ner_bilou/xmlr_bilou.onnx"     # chemin vers ton export ONNX

LABELS = [
    "O",
    "B-PER","I-PER","L-PER","U-PER",
    "B-LOC","I-LOC","L-LOC","U-LOC",
    "B-OBJECT","I-OBJECT","L-OBJECT","U-OBJECT",
    "B-ORG","I-ORG","L-ORG","U-ORG",
    "B-TIME","I-TIME","L-TIME","U-TIME",
    "B-EVENT","I-EVENT","L-EVENT","U-EVENT"
]

# ------------------------------------------------------------
# INPUT TEXT (tu peux changer)
# ------------------------------------------------------------
if len(sys.argv) > 1:
    text = " ".join(sys.argv[1:])
else:
    text = "Les policiers ont tiré sur les manifestants au fond de la place de la République à 2h44 du matin."


# ------------------------------------------------------------
# LOAD ONNX + TOKENIZER
# ------------------------------------------------------------
print("Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

print("Loading ONNX…")
session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

# ------------------------------------------------------------
# TOKENIZE EXACTLY LIKE HF TRAINING
# ------------------------------------------------------------
enc = tok(
    text,
    return_tensors="np",
    padding="max_length",
    truncation=True,
    max_length=256
)

inputs = {
    "input_ids": enc["input_ids"],
    "attention_mask": enc["attention_mask"]
}

# ------------------------------------------------------------
# RUN ONNX
# ------------------------------------------------------------
print("Running inference…")
logits = session.run(["logits"], inputs)[0]         # shape: [1, seq, labels]
pred = np.argmax(logits, axis=-1)[0]               # seq → id label
tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])

# ------------------------------------------------------------
# DECODE BILOU (naïf, token-level)
# ------------------------------------------------------------
print("\n--- TOKENS + LABELS ---")
for t, lab_id in zip(tokens, pred):
    lab = LABELS[lab_id]
    print(f"{t:15s}  {lab}")

# ------------------------------------------------------------
# SIMPLE SPAN GROUPING (BILOU)
# ------------------------------------------------------------
entities = []
i = 0
while i < len(pred):
    lab = LABELS[pred[i]]
    if lab == "O":
        i += 1
        continue

    tag, etype = lab.split("-",1)

    if tag == "U":
        word = tok.decode([enc["input_ids"][0][i]])
        entities.append((etype, word))
        i += 1

    elif tag == "B":
        start = i
        i += 1
        while i < len(pred):
            lab2 = LABELS[pred[i]]
            if lab2 == f"I-{etype}":
                i += 1
                continue
            if lab2 == f"L-{etype}":
                end = i
                i += 1
                break
            break
        ids = enc["input_ids"][0][start:end+1]
        span_text = tok.decode(ids)
        entities.append((etype, span_text))

    else:
        i += 1

print("\n--- ENTITIES ---")
for etype, txt in entities:
    print(f"{etype:6s} → {txt}")