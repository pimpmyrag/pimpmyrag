#!/usr/bin/env python3
import sys

filename = sys.argv[1]
bad = False
current_tokens = []
current_labels = []

def check_seq(tokens, labels, line_num):
    global bad
    n = len(tokens)
    for i in range(n):
        t = tokens[i]
        l = labels[i]
        if "-" in l:
            prefix, typ = l.split("-", 1)
        else:
            prefix = l
            typ = None

        # Check valid label
        if prefix not in ["B","I","L","U","O"]:
            print(f"[ERROR] Invalid prefix {prefix} at line {line_num}: {l}")
            bad = True

        if prefix in ["I", "L"]:
            if i == 0 or not labels[i-1].endswith("-"+typ) or labels[i-1][0] not in ["B","I"]:
                print(f"[ERROR] I/L without valid previous B/I at line {line_num}: {l}")
                bad = True

        if prefix == "B":
            # if next is invalid
            if i+1 < n and labels[i+1].startswith("L-") and not labels[i+1].endswith("-"+typ):
                print(f"[ERROR] B mismatch with next L at line {line_num}: {l} then {labels[i+1]}")
                bad = True

# Read CONLL
with open(filename, "r", encoding="utf-8") as fh:
    tokens = []
    labels = []
    line_num = 0
    for line in fh:
        line_num += 1
        line = line.strip()
        if not line:
            if tokens:
                check_seq(tokens, labels, line_num)
                tokens, labels = [], []
            continue
        if "\t" not in line:
            print(f"[ERROR] No tab at line {line_num}: {line}")
            bad = True
            continue
        tok, tag = line.split("\t")
        tokens.append(tok)
        labels.append(tag)

    if tokens:
        check_seq(tokens, labels, line_num)

if not bad:
    print("✅ Dataset valid.")
else:
    print("❌ Dataset has errors.")