#!/usr/bin/env python3
import json, random

with open('data/wikinews_rare_candidates_clean.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]

print(f'Total: {len(lines)} phrases\n')

bad = no_preds = short = garbled = 0
for obj in lines:
    if 'text' not in obj or 'predictions' not in obj:
        bad += 1
    if not obj.get('predictions'):
        no_preds += 1
    if len(obj.get('text','')) < 15:
        short += 1
    t = obj.get('text','')
    if any(x in t for x in ['__NOTOC__','colspan','rowspan','{|','|}','[[',']]','{{','}}']):
        garbled += 1

print(f'Structure invalide: {bad}')
print(f'Sans prédictions:   {no_preds}')
print(f'Trop courtes (<15): {short}')
print(f'Markup résiduel:    {garbled}')

random.seed(99)
samples = random.sample(lines, 10)
print()
for s in samples:
    preds = [f"{p.get('text','')} → {p.get('fine','')}" for p in s.get('predictions',[])]
    print(f"[{s['id']}] {s['text'][:130]}")
    for p in preds[:6]:
        print(f"    {p}")
    print()

