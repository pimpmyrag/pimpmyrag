#!/usr/bin/env python3
import json, random

def extract_spans(line, label):
    d = json.loads(line)
    text = d.get('text', '')
    return [text[sp['start']:sp['end']] for sp in d.get('spans', []) if sp.get('label') == label]

samples_doc = []
samples_inst = []

with open('data/train_v5.jsonl') as f:
    for line in f:
        samples_doc.extend(extract_spans(line, 'hint_document'))
        samples_inst.extend(extract_spans(line, 'hint_inst_name'))

random.seed(42)

print(f"=== hint_document : {len(samples_doc)} occurrences ===")
for s in random.sample(samples_doc, min(60, len(samples_doc))):
    print(f"  DOC | {s}")

print()
print(f"=== hint_inst_name : {len(samples_inst)} occurrences ===")
for s in random.sample(samples_inst, min(60, len(samples_inst))):
    print(f"  INST | {s}")

