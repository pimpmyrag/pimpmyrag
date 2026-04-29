"""Analyse la distribution du dataset train_v4_claude.jsonl et estime les splits utilisables."""
import json, sys, argparse
from collections import Counter

def analyze(path: str):
    lines = open(path).readlines()
    n = len(lines)
    print(f"Phrases total : {n}")

    ner_counts   = Counter()
    role_counts  = Counter()
    svo_counts   = Counter()
    voice_counts = Counter()
    cert_counts  = Counter()
    spans_per_sent = []
    ner_per_sent   = []
    verb_per_sent  = []
    sentences_with_ner  = 0
    sentences_with_verb = 0
    sentences_with_role = 0
    fallback_count = 0
    empty_count    = 0

    for line in lines:
        d = json.loads(line)
        spans = d.get("spans", [])
        if d.get("_fallback"):
            fallback_count += 1
        spans_per_sent.append(len(spans))
        ner   = [s for s in spans if s.get("label","").startswith("hint_")]
        verbs = [s for s in spans if s.get("label") == "verb_trigger"]
        with_role = [s for s in spans if s.get("svo_role") and s["svo_role"] != "NONE"]
        ner_per_sent.append(len(ner))
        verb_per_sent.append(len(verbs))
        if ner:       sentences_with_ner  += 1
        if verbs:     sentences_with_verb += 1
        if with_role: sentences_with_role += 1
        if not spans: empty_count += 1
        for s in ner:
            ner_counts[s["label"]] += 1
        for s in spans:
            r = s.get("svo_role","")
            if r: role_counts[r] += 1
            l = s.get("label","")
            if l in ("verb_trigger","pron_subj","pron_obj"):
                svo_counts[l] += 1
            if l == "verb_trigger":
                voice_counts[s.get("voice","?")] += 1
                cert_counts[s.get("certainty","?")] += 1

    total_ner = sum(ner_counts.values())

    print(f"\n── Qualité générale ───────────────────────────────────")
    print(f"  Fallback (sans annot Claude) : {fallback_count:>5} ({100*fallback_count/n:.1f}%)")
    print(f"  Phrases sans aucun span      : {empty_count:>5} ({100*empty_count/n:.1f}%)")
    print(f"  Phrases avec ≥1 NER          : {sentences_with_ner:>5} ({100*sentences_with_ner/n:.1f}%)")
    print(f"  Phrases avec ≥1 verb_trigger : {sentences_with_verb:>5} ({100*sentences_with_verb/n:.1f}%)")
    print(f"  Phrases avec ≥1 rôle SVO     : {sentences_with_role:>5} ({100*sentences_with_role/n:.1f}%)")
    print(f"  Spans/phrase (moy)           : {sum(spans_per_sent)/n:.2f}")
    print(f"  NER/phrase (moy)             : {sum(ner_per_sent)/n:.2f}")
    print(f"  Verb/phrase (moy)            : {sum(verb_per_sent)/n:.2f}")

    print(f"\n── NER ({total_ner} spans, {len(ner_counts)} labels) ──────────────────────")
    max_c = max(ner_counts.values())
    for label, c in sorted(ner_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(36 * c / max_c)
        print(f"  {label:<28} {c:>6}  {bar}")

    rares = [(l,c) for l,c in ner_counts.items() if c < 1000]
    print(f"\n⚠️  Labels rares (<1000 ex) : {len(rares)}")
    for l,c in sorted(rares, key=lambda x: x[1]):
        print(f"   {l:<28} {c:>6}")

    imb = max(ner_counts.values()) / max(1, min(ner_counts.values()))
    print(f"\n  Ratio imbalance max/min : {imb:.0f}x")

    print(f"\n── SVO ────────────────────────────────────────────────")
    for k,v in sorted(svo_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<20} {v}")

    print(f"\n── Rôles ──────────────────────────────────────────────")
    for k,v in role_counts.most_common():
        print(f"  {k:<20} {v}")

    print(f"\n── Voix ───────────────────────────────────────────────")
    for k,v in voice_counts.most_common():
        print(f"  {k:<12} {v}")

    print(f"\n── Modalité ───────────────────────────────────────────")
    for k,v in cert_counts.most_common():
        print(f"  {k:<12} {v}")

    # ── Estimation splits ──────────────────────────────────────
    print(f"\n── Estimation de splits depuis {n} phrases ────────────")
    min_ner = min(ner_counts.values())
    min_label = min(ner_counts, key=ner_counts.get)
    for name, tr, va, te in [
        ("90 / 5 / 5",  0.90, 0.05, 0.05),
        ("85 / 7.5/7.5",0.85, 0.075,0.075),
        ("80 / 10 / 10",0.80, 0.10, 0.10),
    ]:
        nt = int(n * tr)
        nv = int(n * va)
        nte = n - nt - nv
        min_val = int(min_ner * va)
        ok = "✅" if min_val >= 20 else "⚠️ "
        print(f"  {name} → train:{nt:>6}  val:{nv:>5}  test:{nte:>5}  "
              f"| '{min_label}'×val≈{min_val}  {ok}")

    print(f"\n📋 Verdict :")
    if n >= 15000 and min_ner >= 400:
        print(f"  ✅ Dataset viable pour entraînement — split 90/5/5 recommandé")
        print(f"  ✅ Toutes les têtes NER, SVO, morpho ont assez d'exemples")
        if fallback_count / n > 0.05:
            print(f"  ⚠️  {fallback_count} fallbacks ({100*fallback_count/n:.1f}%) sans annot SVO/morpho")
            print(f"     → légère faiblesse sur les têtes verb_trigger/rôles prévue")
    else:
        print(f"  ⚠️  Dataset marginal — vérifier les classes rares avant de lancer le training")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/train_v4_claude.jsonl")
    args = p.parse_args()
    analyze(args.input)

