#!/usr/bin/env python3
"""
Reconstruit train/val/test en fusionnant le dataset existant
avec les nouvelles annotations Claude (rare_candidates_annotated.jsonl).

Stratégie :
- Dédupe par texte
- Valide les spans (start/end cohérents, label valide)
- Re-split stratifié sur les labels rares pour équilibrer val/test
- Rapport de distribution final

Usage :
  python3 scripts/rebuild_dataset.py \
    --existing-train data/train.jsonl \
    --existing-val   data/val.jsonl \
    --existing-test  data/test.jsonl \
    --new-annotations data/rare_candidates_annotated.jsonl \
    --out-train data/train_v2.jsonl \
    --out-val   data/val_v2.jsonl \
    --out-test  data/test_v2.jsonl \
    --val-ratio 0.10 \
    --test-ratio 0.10
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

VALID_LABELS = {
    "hint_person_name", "hint_person_role", "hint_norp", "hint_group_role",
    "hint_org_name", "hint_gpe", "hint_fac_name", "hint_loc_generic",
    "hint_infra", "hint_weapon", "hint_vehicle", "hint_substance",
    "hint_food", "hint_tool", "hint_object_generic", "hint_object_name",
    "hint_event_nominal", "hint_event_named", "hint_time_date",
    "hint_time_clock", "hint_time_duration", "hint_quantity", "hint_measure",
    "hint_percentage", "hint_count", "hint_money", "hint_rate",
    "hint_law", "hint_work_of_art", "hint_concept", "hint_disease", "hint_language",
}

RARE_LABELS = {
    "hint_concept", "hint_language", "hint_law", "hint_work_of_art",
    "hint_tool", "hint_disease", "hint_food", "hint_money",
    "hint_count", "hint_substance", "hint_rate", "hint_percentage",
    "hint_time_clock", "hint_object_name", "hint_weapon", "hint_infra",
    "hint_fac_name", "hint_measure",
}


def validate_spans(item: dict) -> dict:
    """Filtre les spans invalides (label inconnu, start/end incohérents)."""
    text = item.get("text", "")
    valid = []
    for s in item.get("spans", []):
        label = s.get("label", "")
        start = s.get("start", -1)
        end = s.get("end", -1)
        span_text = s.get("text", "")
        if label not in VALID_LABELS:
            continue
        if not (0 <= start < end <= len(text)):
            continue
        # Vérifier cohérence texte si présent
        if span_text and text[start:end] != span_text:
            # Tenter de corriger l'offset via recherche du texte
            idx = text.find(span_text)
            if idx >= 0:
                s = {**s, "start": idx, "end": idx + len(span_text)}
            else:
                continue
        valid.append(s)
    return {**item, "spans": valid}


def load_jsonl(path: str) -> list[dict]:
    items = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    except FileNotFoundError:
        print(f"  ⚠️  {path} introuvable")
    return items


def write_jsonl(items: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            # Ne garder que les clés utiles
            record = {
                "id": item.get("id", ""),
                "text": item.get("text", ""),
                "spans": item.get("spans", []),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_distribution(items: list[dict], title: str):
    counts = Counter()
    for item in items:
        for s in item.get("spans", []):
            counts[s["label"]] += 1
    total_spans = sum(counts.values())
    print(f"\n{'─'*60}")
    print(f"  {title} — {len(items)} phrases, {total_spans} spans")
    print(f"{'─'*60}")
    for label in sorted(VALID_LABELS):
        cnt = counts.get(label, 0)
        marker = "🔴" if label in RARE_LABELS and cnt < 50 else ("🟡" if label in RARE_LABELS and cnt < 150 else "✅")
        bar = "█" * min(cnt // 30, 40)
        print(f"  {marker} {label:<25} {cnt:>5}  {bar}")


def stratified_split(items: list[dict], val_ratio: float, test_ratio: float, seed: int = 42) -> tuple:
    """
    Split stratifié sur les labels rares :
    pour chaque label rare, on garantit un minimum de représentation dans val/test.
    """
    rng = random.Random(seed)

    # Indexer les items par label rare qu'ils contiennent
    label_to_items = defaultdict(list)
    for i, item in enumerate(items):
        labels_in = {s["label"] for s in item.get("spans", [])} & RARE_LABELS
        for lbl in labels_in:
            label_to_items[lbl].append(i)

    val_indices = set()
    test_indices = set()

    # Pour chaque label rare, réserver un quota minimum dans val/test
    for lbl in RARE_LABELS:
        candidates = label_to_items[lbl]
        rng.shuffle(candidates)
        n_val = max(1, int(len(candidates) * val_ratio))
        n_test = max(1, int(len(candidates) * test_ratio))
        for idx in candidates[:n_val]:
            val_indices.add(idx)
        for idx in candidates[n_val:n_val + n_test]:
            test_indices.add(idx)

    # Le reste va en train
    remaining = [i for i in range(len(items)) if i not in val_indices and i not in test_indices]
    # Compléter val/test avec le ratio global
    rng.shuffle(remaining)
    n_val_extra = max(0, int(len(items) * val_ratio) - len(val_indices))
    n_test_extra = max(0, int(len(items) * test_ratio) - len(test_indices))
    val_indices.update(remaining[:n_val_extra])
    test_indices.update(remaining[n_val_extra:n_val_extra + n_test_extra])
    train_indices = set(range(len(items))) - val_indices - test_indices

    train = [items[i] for i in sorted(train_indices)]
    val = [items[i] for i in sorted(val_indices)]
    test = [items[i] for i in sorted(test_indices)]
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-train", default="data/train.jsonl")
    parser.add_argument("--existing-val",   default="data/val.jsonl")
    parser.add_argument("--existing-test",  default="data/test.jsonl")
    parser.add_argument("--new-annotations", default="data/rare_candidates_annotated.jsonl")
    parser.add_argument("--out-train", default="data/train_v2.jsonl")
    parser.add_argument("--out-val",   default="data/val_v2.jsonl")
    parser.add_argument("--out-test",  default="data/test_v2.jsonl")
    parser.add_argument("--val-ratio",  type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--min-spans",  type=int,   default=1, help="Phrases avec au moins N spans")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--no-resplit", action="store_true",
                        help="Conserver les splits existants, ajouter les nouveaux en train uniquement")
    args = parser.parse_args()

    # ── 1. Charger tout ─────────────────────────────────────────────
    print("📂 Chargement des données...")
    existing_train = load_jsonl(args.existing_train)
    existing_val   = load_jsonl(args.existing_val)
    existing_test  = load_jsonl(args.existing_test)
    new_items      = load_jsonl(args.new_annotations)

    print(f"  Existant : train={len(existing_train)}, val={len(existing_val)}, test={len(existing_test)}")
    print(f"  Nouvelles annotations Claude : {len(new_items)}")

    # ── 2. Valider les spans ─────────────────────────────────────────
    print("\n🔍 Validation des spans...")
    all_existing = existing_train + existing_val + existing_test
    all_existing = [validate_spans(i) for i in all_existing]
    new_items = [validate_spans(i) for i in new_items]

    # Filtrer les fallbacks sans spans utiles
    new_items = [i for i in new_items if not i.get("_fallback") or len(i.get("spans", [])) > 0]
    new_items = [i for i in new_items if len(i.get("spans", [])) >= args.min_spans]

    # ── 3. Déduplication globale ─────────────────────────────────────
    print("\n🔁 Déduplication...")
    seen_texts = set()
    unique_existing = []
    for item in all_existing:
        t = item["text"].strip()
        if t not in seen_texts:
            seen_texts.add(t)
            unique_existing.append(item)

    unique_new = []
    dupes_new = 0
    for item in new_items:
        t = item["text"].strip()
        if t not in seen_texts:
            seen_texts.add(t)
            unique_new.append(item)
        else:
            dupes_new += 1

    print(f"  Existant après dédup : {len(unique_existing)} (inchangé)")
    print(f"  Nouvelles après dédup : {len(unique_new)} (+{dupes_new} doublons ignorés)")

    # ── 4. Combiner et splitter ──────────────────────────────────────
    if args.no_resplit:
        # Mode conservateur : les anciens splits restent, les nouveaux → train
        print("\n📐 Mode --no-resplit : nouveaux items → train uniquement")
        final_train = unique_existing + unique_new  # val/test déjà dans existing
        # Reconstruire correctement
        existing_val_texts = {i["text"].strip() for i in existing_val}
        existing_test_texts = {i["text"].strip() for i in existing_test}
        final_train = [i for i in (existing_train + unique_new)
                       if i["text"].strip() not in existing_val_texts
                       and i["text"].strip() not in existing_test_texts]
        final_val  = existing_val
        final_test = existing_test
    else:
        # Mode resplit total : on repart de zéro
        print(f"\n📐 Re-split stratifié (val={args.val_ratio:.0%}, test={args.test_ratio:.0%})...")
        all_items = unique_existing + unique_new
        random.seed(args.seed)
        random.shuffle(all_items)
        final_train, final_val, final_test = stratified_split(
            all_items, args.val_ratio, args.test_ratio, seed=args.seed
        )

    print(f"  → train={len(final_train)}, val={len(final_val)}, test={len(final_test)}")

    # ── 5. Rapport de distribution ───────────────────────────────────
    print_distribution(final_train, "TRAIN")
    print_distribution(final_val,   "VAL")
    print_distribution(final_test,  "TEST")

    # Vérifier les labels rares dans val et test
    print("\n⚠️  Labels rares dans VAL (objectif ≥ 30) :")
    val_counts = Counter(s["label"] for i in final_val for s in i.get("spans", []))
    test_counts = Counter(s["label"] for i in final_test for s in i.get("spans", []))
    for lbl in sorted(RARE_LABELS):
        vc = val_counts.get(lbl, 0)
        tc = test_counts.get(lbl, 0)
        flag = "✅" if vc >= 30 and tc >= 30 else "⚠️ "
        print(f"  {flag} {lbl:<25} val={vc:>4}  test={tc:>4}")

    # ── 6. Écriture ──────────────────────────────────────────────────
    print(f"\n💾 Écriture...")
    write_jsonl(final_train, args.out_train)
    write_jsonl(final_val,   args.out_val)
    write_jsonl(final_test,  args.out_test)
    print(f"  ✅ {args.out_train} ({len(final_train)} phrases)")
    print(f"  ✅ {args.out_val}   ({len(final_val)} phrases)")
    print(f"  ✅ {args.out_test}  ({len(final_test)} phrases)")

    # Estimation gain potentiel
    n_new_rare = sum(
        1 for i in unique_new
        if any(s["label"] in RARE_LABELS for s in i.get("spans", []))
    )
    print(f"\n📈 {n_new_rare} nouvelles phrases avec labels rares ajoutées au pool")
    print("   → Attendre +1 à +3 pts F1 sur les labels rares après re-training")


if __name__ == "__main__":
    main()

