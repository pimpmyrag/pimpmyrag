#!/usr/bin/env python3
"""Valide les offsets des spans Claude : text[start:end] == span.text"""
import json, sys
from collections import Counter

VALID_LABELS = {
    "hint_person_name","hint_person_role","hint_norp","hint_group_role",
    "hint_org_name","hint_gpe","hint_fac_name","hint_loc_generic",
    "hint_infra","hint_weapon","hint_vehicle","hint_substance",
    "hint_food","hint_tool","hint_object_generic","hint_object_name",
    "hint_event_nominal","hint_event_named","hint_time_date",
    "hint_time_clock","hint_time_duration","hint_quantity","hint_measure",
    "hint_percentage","hint_count","hint_money","hint_rate",
    "hint_law","hint_work_of_art","hint_concept","hint_disease","hint_language",
}

INPUT = sys.argv[1] if len(sys.argv) > 1 else "data/wikinews_claude_annotated.jsonl"
OUTPUT = INPUT.replace(".jsonl", "_valid.jsonl")

n_total = n_ok = n_bad_offset = n_bad_label = n_fallback = 0
n_spans_total = n_spans_ok = n_spans_fixed = n_spans_dropped = 0
label_stats = Counter()
errors = []

with open(INPUT) as f, open(OUTPUT, "w") as out:
    for lineno, line in enumerate(f, 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        n_total += 1

        if obj.get("_fallback"):
            n_fallback += 1

        text = obj["text"]
        valid_spans = []

        for s in obj.get("spans", []):
            n_spans_total += 1
            label = s.get("label", "")
            start = s.get("start", 0)
            end = s.get("end", 0)
            span_text = s.get("text", "")

            # Check label
            if label not in VALID_LABELS:
                n_bad_label += 1
                n_spans_dropped += 1
                continue

            # Check offset
            actual = text[start:end]
            if actual == span_text:
                valid_spans.append(s)
                n_spans_ok += 1
                label_stats[label] += 1
            else:
                # Try to fix by searching
                idx = text.find(span_text)
                if idx >= 0:
                    s["start"] = idx
                    s["end"] = idx + len(span_text)
                    valid_spans.append(s)
                    n_spans_fixed += 1
                    label_stats[label] += 1
                else:
                    n_spans_dropped += 1
                    n_bad_offset += 1
                    if len(errors) < 20:
                        errors.append(f"  L{lineno} [{label}] expected='{span_text}' got='{actual}' ({start}:{end})")

        obj["spans"] = valid_spans
        if "_fallback" in obj:
            del obj["_fallback"]
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        if valid_spans:
            n_ok += 1

print(f"\n{'='*60}")
print(f"📝 Phrases:    {n_total} (dont {n_fallback} fallback)")
print(f"✅ Avec spans valides: {n_ok}")
print(f"\n📊 Spans:")
print(f"  Total:      {n_spans_total}")
print(f"  OK:         {n_spans_ok}")
print(f"  Fixés:      {n_spans_fixed}")
print(f"  Supprimés:  {n_spans_dropped} (offset={n_bad_offset}, label={n_bad_label})")
print(f"  Taux:       {(n_spans_ok+n_spans_fixed)/max(n_spans_total,1)*100:.1f}% conservés")

if errors:
    print(f"\n🔍 Exemples d'erreurs d'offset:")
    for e in errors:
        print(e)

print(f"\n📊 Labels validés:")
for label, count in label_stats.most_common():
    print(f"  {label:<25} {count:>6}")

print(f"\n💾 → {OUTPUT}")

