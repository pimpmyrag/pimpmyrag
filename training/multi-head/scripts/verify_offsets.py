#!/usr/bin/env python3
"""
Vérifie les offsets de spans dans les fichiers JSONL du dataset.
Corrige les offsets décalés si possible, supprime les spans invalides.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

def verify_file(path: Path, fix: bool = False):
    lines = path.read_text().splitlines()
    total_spans = 0
    ok_spans = 0
    fixed_spans = 0
    dropped_spans = 0
    fixed_items = []

    errors_by_type = defaultdict(int)

    for i, line in enumerate(lines):
        if not line.strip():
            continue
        item = json.loads(line)
        text = item.get("text", "")
        new_spans = []
        for s in item.get("spans", []):
            total_spans += 1
            start = s.get("start", -1)
            end = s.get("end", -1)
            span_text = s.get("text", "")

            # Vérif bornes
            if not (0 <= start < end <= len(text)):
                errors_by_type["out_of_bounds"] += 1
                if fix and span_text:
                    idx = text.find(span_text)
                    if idx >= 0:
                        new_spans.append({**s, "start": idx, "end": idx + len(span_text)})
                        fixed_spans += 1
                        continue
                dropped_spans += 1
                continue

            # Vérif cohérence texte
            extracted = text[start:end]
            if extracted != span_text and span_text:
                errors_by_type["text_mismatch"] += 1
                if fix:
                    idx = text.find(span_text)
                    if idx >= 0:
                        new_spans.append({**s, "start": idx, "end": idx + len(span_text)})
                        fixed_spans += 1
                        continue
                    else:
                        # Essayer strip
                        idx = text.find(span_text.strip())
                        if idx >= 0:
                            t = span_text.strip()
                            new_spans.append({**s, "start": idx, "end": idx + len(t), "text": t})
                            fixed_spans += 1
                            continue
                dropped_spans += 1
                continue

            ok_spans += 1
            new_spans.append(s)

        if fix:
            fixed_items.append({**item, "spans": new_spans})

    print(f"\n📄 {path.name}")
    print(f"   Phrases    : {len(lines)}")
    print(f"   Spans total: {total_spans}")
    print(f"   ✅ OK       : {ok_spans}")
    if errors_by_type:
        for k, v in errors_by_type.items():
            print(f"   ⚠️  {k}: {v}")
    if fix:
        print(f"   🔧 Corrigés : {fixed_spans}")
        print(f"   🗑️  Supprimés: {dropped_spans}")
    else:
        print(f"   ❌ Invalides: {dropped_spans + fixed_spans} (relancer avec --fix pour corriger)")

    if fix and (fixed_spans > 0 or dropped_spans > 0):
        out_path = path.with_suffix(".fixed.jsonl") if not path.stem.endswith("_v2") else path
        with open(out_path, "w") as f:
            for item in fixed_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"   💾 Sauvegardé: {out_path}")

    return total_spans, ok_spans, fixed_spans, dropped_spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Fichiers JSONL à vérifier (défaut: train_v2, val_v2, test_v2)")
    parser.add_argument("--fix", action="store_true", help="Corriger les offsets et sauvegarder")
    parser.add_argument("--data-dir", default="data", help="Répertoire data")
    args = parser.parse_args()

    base = Path(args.data_dir)
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = [base / "train_v2.jsonl", base / "val_v2.jsonl", base / "test_v2.jsonl"]

    total_t, total_ok, total_fix, total_drop = 0, 0, 0, 0
    for f in files:
        if not f.exists():
            print(f"⚠️  Fichier introuvable: {f}")
            continue
        t, ok, fx, dr = verify_file(f, fix=args.fix)
        total_t += t; total_ok += ok; total_fix += fx; total_drop += dr

    print(f"\n{'═'*50}")
    print(f"  TOTAL spans    : {total_t}")
    print(f"  ✅ OK           : {total_ok} ({100*total_ok/max(total_t,1):.1f}%)")
    if args.fix:
        print(f"  🔧 Corrigés    : {total_fix}")
        print(f"  🗑️  Supprimés   : {total_drop}")
    else:
        print(f"  ❌ Problèmes   : {total_fix + total_drop}")
    print(f"{'═'*50}")


if __name__ == "__main__":
    main()

