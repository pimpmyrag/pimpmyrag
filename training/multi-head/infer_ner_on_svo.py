"""
infer_ner_on_svo.py
====================
Parcourt un fichier SVO silver (textes bruts + spans svo_*/pron_*),
lance l'inférence NER avec le modèle local, et injecte les spans hint_*
prédits dans chaque exemple.

Résultat : un fichier JSONL avec spans svo_* + pron_* + hint_* (prédits),
prêt à être passé dans filter_rare_svo.py pour le scoring de rareté.

Usage
─────
  python infer_ner_on_svo.py \\
      --input   data/train_svo_obl.jsonl \\
      --output  data/train_svo_obl_ner.jsonl \\
      --checkpoint  checkpoint_best_multitask.pt \\
      --model-name  microsoft/deberta-v3-base \\
      --batch-size  32
"""

import argparse
import json
import time
from pathlib import Path

import torch

import torch
from test_model_sentences_v3 import (
    predict_texts_batch,
    post_process_dynamic,
    dedupe_overlaps,
    pick_device,
)
from test_model_sentences_v3 import load_model_and_tokenizer as _load_model_and_tokenizer
from multitask_model import SpanMultiTaskModel
from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Conversion prédiction → span JSONL
# ─────────────────────────────────────────────────────────────────────────────

def ner_pred_to_span(pred: dict) -> dict:
    """Convertit une prédiction NER du modèle en span au format JSONL du dataset."""
    return {
        "label":      pred["fine"],
        "start":      pred["char_start"],
        "end":        pred["char_end"],
        "text":       pred["text"],
        # métadonnées de confiance (utiles pour audit, ignorées par filter_rare_svo)
        "_predicted": True,
        "_score":     pred.get("score"),
        "_coarse":    pred.get("coarse"),
    }


def load_model_and_tokenizer_compat(model_name, checkpoint_path, tokenizer_path, device):
    """Chargement avec filtrage des shape mismatches (ex: gender_head 3->2 classes)."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_name, use_fast=True)
    if getattr(tokenizer, "model_max_length", None) is None or tokenizer.model_max_length > 100000:
        tokenizer.model_max_length = 128

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    num_coarse = state["coarse_head.weight"].shape[0]

    model = SpanMultiTaskModel(model_name=model_name, num_coarse=num_coarse).to(device).float()

    # Supprimer les clés avec shape mismatch (ex: gender_head 3->2)
    model_shapes = {k: v.shape for k, v in model.state_dict().items()}
    for key in list(state.keys()):
        if key in model_shapes and state[key].shape != model_shapes[key]:
            print(f"[INFER] Shape mismatch ignoré : {key} {state[key].shape} → {model_shapes[key]}")
            del state[key]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[INFER] Clés manquantes (ignorées) : {len(missing)}")
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process(args):
    device = pick_device(args.device)
    print(f"[INFER] device = {device}")

    print(f"[INFER] Chargement du modèle {args.checkpoint}…")
    model, tokenizer = load_model_and_tokenizer_compat(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        device=device,
    )

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else \
                  input_path.with_name(input_path.stem + "_ner.jsonl")

    # Chargement en mémoire (les SVO silver sont généralement < 100k exemples)
    print(f"[INFER] Lecture de {input_path.name}…")
    examples = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if args.only_with_obl:
                if not any(sp["label"] == "svo_iobj" for sp in ex.get("spans", [])):
                    continue
            examples.append(ex)

    n_total = len(examples)
    qualifier = " (filtrés : contiennent svo_iobj)" if args.only_with_obl else ""
    print(f"[INFER] {n_total} exemples chargés{qualifier}.")

    # Paramètres inférence NER
    infer_kwargs = dict(
        max_length=args.max_length,
        max_span_len=args.max_span_len,
        tau_boundary=args.tau_boundary,
        tau_none=args.tau_none,
        tau_coarse=args.tau_coarse,
        tau_fine=args.tau_fine,
        topk_coarse=args.topk_coarse,
        min_char_len=2,
        enforce_word_boundaries=True,
        tau_svo_boundary=0.50,   # on s'en fout ici, on ne garde que le NER
    )

    t_start = time.time()
    n_done  = 0
    n_spans_added = 0

    with open(output_path, "w", encoding="utf-8") as f_out:
        for batch_start in range(0, n_total, args.batch_size):
            batch_exs   = examples[batch_start: batch_start + args.batch_size]
            batch_texts = [ex["text"] for ex in batch_exs]

            results = predict_texts_batch(
                model=model, tokenizer=tokenizer,
                texts=batch_texts, device=device,
                **infer_kwargs,
            )

            for ex, res in zip(batch_exs, results):
                ner_preds = res["ner"]

                # Post-processing selon le flag
                if args.post_process:
                    ner_preds = post_process_dynamic(ner_preds)
                else:
                    ner_preds = dedupe_overlaps(ner_preds)

                ner_spans = [ner_pred_to_span(p) for p in ner_preds]
                n_spans_added += len(ner_spans)

                # Fusionner : spans existants (svo_* / pron_*) + hints prédits
                merged_spans = ex.get("spans", []) + ner_spans

                out_row = {**ex, "spans": merged_spans}
                f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            n_done += len(batch_exs)

            # Log périodique
            now     = time.time()
            elapsed = now - t_start
            rate    = n_done / elapsed if elapsed > 0 else 0
            if (batch_start // args.batch_size) % 20 == 0:
                print(f"  [{n_done:>6}/{n_total}]  {rate:.1f} ex/s  "
                      f"| spans hint_* ajoutés jusqu'ici : {n_spans_added}")

    elapsed_total = time.time() - t_start
    print(f"\n[INFER] ✅ {n_total} exemples traités en {elapsed_total:.1f}s "
          f"({n_total/elapsed_total:.1f} ex/s)")
    print(f"[INFER]    {n_spans_added} spans hint_* injectés "
          f"(~{n_spans_added/n_total:.1f} par exemple en moyenne)")
    print(f"[INFER]    → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inject NER predictions into SVO silver JSONL")
    parser.add_argument("--input",          required=True,
                        help="Fichier SVO silver source (.jsonl)")
    parser.add_argument("--output",         default="",
                        help="Fichier de sortie (défaut: <input>_ner.jsonl)")
    parser.add_argument("--model-name",     default="microsoft/deberta-v3-base")
    parser.add_argument("--checkpoint",     default="checkpoint_best_multitask.pt")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--device",         choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--batch-size",     type=int, default=32)
    parser.add_argument("--max-length",     type=int, default=128)
    parser.add_argument("--max-span-len",   type=int, default=8)
    parser.add_argument("--tau-boundary",   type=float, default=0.50)
    parser.add_argument("--tau-none",       type=float, default=0.99)
    parser.add_argument("--tau-coarse",     type=float, default=0.00)
    parser.add_argument("--tau-fine",       type=float, default=0.00)
    parser.add_argument("--topk-coarse",    type=int,   default=2)
    parser.add_argument("--post-process",   action="store_true",
                        help="Appliquer post_process_dynamic (NMS + seuils dynamiques)")
    parser.add_argument("--only-with-obl",  action="store_true",
                        help="Ne traiter que les exemples contenant au moins un svo_iobj")
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()

