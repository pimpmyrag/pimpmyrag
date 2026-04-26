#!/usr/bin/env python3
"""Spot-check : affiche N phrases avec spans en contexte pour review humaine."""
import json, sys, random, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data/wikinews_claude_annotated_valid.jsonl")
parser.add_argument("--n", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

with open(args.input) as f:
    lines = [json.loads(l) for l in f if l.strip()]

random.seed(args.seed)
samples = random.sample(lines, min(args.n, len(lines)))

for i, obj in enumerate(samples, 1):
    text = obj["text"]
    spans = sorted(obj.get("spans", []), key=lambda s: s["start"])

    # Build annotated text
    parts = []
    last = 0
    for s in spans:
        start, end = s["start"], s["end"]
        if start > last:
            parts.append(text[last:start])
        parts.append(f"[«{text[start:end]}» → {s['label']}]")
        last = end
    if last < len(text):
        parts.append(text[last:])

    annotated = "".join(parts)
    print(f"\n{'─'*70}")
    print(f"#{i} [{obj.get('id','')}]")
    print(f"  {annotated}")
    labels = [s["label"] for s in spans]
    print(f"  Labels: {', '.join(labels)}")

