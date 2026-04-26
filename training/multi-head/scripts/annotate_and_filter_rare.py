#!/usr/bin/env python3
"""
Annote un JSONL de phrases avec le best checkpoint multi-head NER,
puis filtre les phrases contenant des labels rares pour préparer
l'annotation humaine.

Sortie : JSONL avec prédictions du modèle, trié par label rare.
+ fichier stats JSON mis à jour régulièrement.
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

# Ajouter le parent au path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_model_sentences_v2 import (
    load_model_and_tokenizer,
    predict_texts_batch,
    pick_device,
    dedupe_overlaps,
)
from labels import FINE_LABELS

# Labels rares à remonter (avec leur count actuel dans le val set)
RARE_LABELS = {
    "hint_concept":        12,
    "hint_language":        17,
    "hint_law":             15,
    "hint_work_of_art":     22,
    "hint_tool":            24,
    "hint_disease":         24,
    "hint_food":            28,
    "hint_money":           30,
    "hint_count":           32,
    "hint_substance":       37,
    "hint_rate":            37,
    "hint_percentage":      38,
    "hint_time_clock":      41,
    "hint_object_name":     43,
    "hint_weapon":          48,
    "hint_infra":           52,
    "hint_fac_name":        71,
    "hint_measure":         85,
}

TARGET_PER_LABEL = 1000


def write_stats(stats_path: str, label_counts: Counter, all_label_stats: Counter, done: int, total: int, n_collected: int):
    """Écrit un JSON de stats mis à jour régulièrement."""
    stats = {
        "progress": f"{done}/{total} ({done/total*100:.1f}%)",
        "collected_sentences": n_collected,
        "rare_labels": {
            label: {"found": label_counts.get(label, 0), "target": TARGET_PER_LABEL, "saturated": label_counts.get(label, 0) >= TARGET_PER_LABEL}
            for label in sorted(RARE_LABELS.keys())
        },
        "all_labels": {label: count for label, count in all_label_stats.most_common()},
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL de phrases (id, text)")
    parser.add_argument("--output", required=True, help="JSONL de sortie avec prédictions")
    parser.add_argument("--stats", default=None, help="Fichier JSON de stats (mis à jour en temps réel)")
    # ...existing args...
    parser.add_argument("--checkpoint", default="checkpoint_best_multitask.pt")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--tau-boundary", type=float, default=0.40)
    parser.add_argument("--tau-none", type=float, default=0.99)
    parser.add_argument("--min-fine-prob", type=float, default=0.30)
    args = parser.parse_args()

    stats_path = args.stats or args.output.replace(".jsonl", ".stats.json")

    device = pick_device(args.device)
    print(f"✅ device = {device}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        device=device,
    )

    # Charger les phrases
    sentences = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sentences.append(obj)
            if args.max_sentences and len(sentences) >= args.max_sentences:
                break

    print(f"📝 {len(sentences)} phrases chargées")

    # Compteur par label rare
    label_counts = Counter()
    n_collected = 0
    all_label_stats = Counter()

    # Écriture au fil de l'eau
    out_file = open(args.output, "w", encoding="utf-8")

    texts = [s["text"] for s in sentences]
    all_saturated = False

    for batch_start in range(0, len(texts), args.batch_size):
        # Early stop si tous les labels rares sont saturés
        if all(label_counts.get(l, 0) >= TARGET_PER_LABEL for l in RARE_LABELS):
            print(f"\n🎯 Tous les labels rares ont atteint {TARGET_PER_LABEL} — arrêt anticipé à {batch_start}/{len(texts)}")
            all_saturated = True
            break

        batch_texts = texts[batch_start : batch_start + args.batch_size]
        batch_preds = predict_texts_batch(
            model=model,
            tokenizer=tokenizer,
            texts=batch_texts,
            device=device,
            max_length=128,
            max_span_len=12,
            tau_boundary=args.tau_boundary,
            tau_none=args.tau_none,
            tau_coarse=0.0,
            tau_fine=0.0,
            topk_coarse=2,
            min_char_len=2,
            enforce_word_boundaries=True,
        )

        for i, preds in enumerate(batch_preds):
            idx = batch_start + i
            sentence = sentences[idx]

            # Filtrer par fine_prob minimale
            preds = [p for p in preds if p.get("fine_prob", 0) >= args.min_fine_prob]
            preds = dedupe_overlaps(preds, allow_nested=True)

            if not preds:
                continue

            # Compter TOUS les labels (stats globales)
            fine_labels_in_sentence = set()
            for p in preds:
                fl = p.get("fine", "")
                all_label_stats[fl] += 1
                if fl in RARE_LABELS:
                    fine_labels_in_sentence.add(fl)

            # Garder la phrase si elle contient au moins un label rare pas encore saturé
            has_useful_rare = False
            for fl in fine_labels_in_sentence:
                if label_counts[fl] < TARGET_PER_LABEL:
                    has_useful_rare = True
                    break

            if has_useful_rare:
                record = {
                    "id": sentence["id"],
                    "text": sentence["text"],
                    "source_title": sentence.get("source_title", ""),
                    "predictions": preds,
                    "rare_labels": sorted(fine_labels_in_sentence),
                }
                # Écriture immédiate
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()
                n_collected += 1
                for fl in fine_labels_in_sentence:
                    label_counts[fl] += 1

        # Progress
        done = min(batch_start + args.batch_size, len(texts))
        if done % (args.batch_size * 10) == 0 or done == len(texts):
            # Résumé temps réel des labels rares
            rare_summary = " | ".join(
                f"{l.replace('hint_','')[:8]}={label_counts.get(l,0)}"
                for l in sorted(RARE_LABELS.keys())
                if label_counts.get(l, 0) > 0
            )
            pct = done / len(texts) * 100
            print(f"  {done}/{len(texts)} ({pct:.1f}%) — {n_collected} phrases | {rare_summary}")

            # Sauver stats JSON
            write_stats(stats_path, label_counts, all_label_stats, done, len(texts), n_collected)

    out_file.close()

    # Stats finales
    write_stats(stats_path, label_counts, all_label_stats, len(texts), len(texts), n_collected)

    print(f"\n✅ {n_collected} phrases avec labels rares → {args.output}")
    print(f"📊 Stats JSON → {stats_path}")

    print(f"\n📊 Distribution des labels rares collectés :")
    for label in sorted(RARE_LABELS.keys()):
        count = label_counts.get(label, 0)
        bar = "█" * min(count // 10, 50)
        status = "✅" if count >= TARGET_PER_LABEL else "⚠️"
        print(f"  {status} {label:<25} {count:>5} / {TARGET_PER_LABEL}  {bar}")

    print(f"\n📊 Distribution GLOBALE de tous les labels prédits :")
    for label, count in all_label_stats.most_common():
        print(f"  {label:<25} {count:>6}")

    # Résumé des labels NON rares (pour voir l'équilibre)
    non_rare = {l: c for l, c in all_label_stats.items() if l not in RARE_LABELS}
    if non_rare:
        print(f"\n📊 Labels fréquents (non ciblés) :")
        for label, count in sorted(non_rare.items(), key=lambda x: -x[1]):
            print(f"  {label:<25} {count:>6}")


if __name__ == "__main__":
    main()

