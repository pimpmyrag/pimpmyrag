#!/usr/bin/env python3
"""
annotate_frwiki_with_model.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pré-annote frwiki_silver_raw.jsonl avec le modèle multitask.

  - Tau généreux (0.40) pour capturer les labels rares
  - Compatible avec les modèles entraînés avec ignore_coarse_none=True
    (coarse_head à 9 classes, NONE absent du tenseur)
  - Sortie : data/frwiki_preannotated.jsonl (format v8.1)

Usage :
  python annotate_frwiki_with_model.py
  python annotate_frwiki_with_model.py --checkpoint /tmp/checkpoint_best_ep60.pt --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from multitask_model import SpanMultiTaskModel
from labels import (
    COARSE_LABELS, FINE_LABELS, COARSE2ID,
    SYN_LABELS, NUM_SYN,
    VOICE_LABELS, NUM_VOICE,
    GENDER_LABELS, NUM_GENDER, GENDER_NONE_ID,
    NUMBER_LABELS, NUM_NUMBER, NUMBER_NONE_ID,
    ROLE_LABELS, NUM_ROLE, ROLE_NONE_ID, ID2ROLE,
)

# ── Chemins ───────────────────────────────────────────────────────────────────
INPUT_RAW        = ROOT / "data" / "frwiki_silver_raw.jsonl"
OUTPUT_PREANNO   = ROOT / "data" / "frwiki_preannotated.jsonl"

RARE_SVO_ROLES   = {"OBLIQUE_AGENT", "APPOS"}

# Seuils généreux — on veut capturer les labels rares
TAU_BOUNDARY     = 0.40
TAU_NONE         = 0.99   # utilisé seulement si NONE présent dans le modèle
MIN_SCORE        = 0.03
TAU_SVO_BOUNDARY = 0.40
MAX_SPAN_LEN     = 12
MAX_LENGTH       = 128
IOU_THRESHOLD    = 0.60

# ── Utilitaires ───────────────────────────────────────────────────────────────

def pick_device(forced=None):
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_secrets():
    path = ROOT / ".secrets.env"
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k, v)


def clean_text(text: str) -> str:
    return re.sub(r"\*\*", "", text)


def clean_char_span(text, s, e):
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def span_iou(a, b):
    inter = max(0, min(a["char_end"], b["char_end"]) - max(a["char_start"], b["char_start"]))
    if inter == 0:
        return 0.0
    union = (a["char_end"] - a["char_start"]) + (b["char_end"] - b["char_start"]) - inter
    return inter / union


def dedupe_overlaps(preds):
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    kept = []
    for p in preds:
        if not any(span_iou(p, k) >= IOU_THRESHOLD and p["coarse"] == k["coarse"] for k in kept):
            kept.append(p)
    return sorted(kept, key=lambda x: x["char_start"])


# ── Chargement modèle ─────────────────────────────────────────────────────────

def load_model(checkpoint: str, model_name: str, device: str):
    print(f"📦 Chargement {checkpoint} sur {device} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if getattr(tokenizer, "model_max_length", None) is None or tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = MAX_LENGTH
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("ema_state") or ckpt.get("model_state") or ckpt
    num_coarse = state["coarse_head.weight"].shape[0]
    model = SpanMultiTaskModel(model_name=model_name, num_coarse=num_coarse).to(device).float()
    model.load_state_dict(state, strict=False)
    model.eval()

    # ── Détecte si le modèle a été entraîné sans NONE (nocoarsenone) ──────────
    none_in_labels = COARSE2ID.get("NONE", -1)
    has_none_class = (none_in_labels >= 0) and (none_in_labels < num_coarse)
    none_idx = none_in_labels if has_none_class else -1

    print(f"✅ Modèle chargé | num_coarse={num_coarse} (labels.py en a {len(COARSE_LABELS)})")
    if not has_none_class:
        print(f"   ⚠️  Modèle nocoarsenone : NONE absent du coarse_head (index {none_in_labels} hors bornes)")
    else:
        print(f"   NONE class à l'index {none_idx}")

    return model, tokenizer, none_idx


# ── Inférence batch ───────────────────────────────────────────────────────────

def build_batch_candidates(tokenizer, texts):
    enc = tokenizer(texts, return_offsets_mapping=True, add_special_tokens=True,
                    truncation=True, padding=True, max_length=MAX_LENGTH)
    input_ids      = torch.tensor(enc["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
    offsets_batch  = enc["offset_mapping"]
    spans_per_example, meta_flat = [], []
    for ex_idx, (text, offsets) in enumerate(zip(texts, offsets_batch)):
        tok_positions = [i for i, (s, e) in enumerate(offsets) if e > s]
        spans, metas = [], []
        for si, tok_start in enumerate(tok_positions):
            for tok_end in tok_positions[si: si + MAX_SPAN_LEN]:
                cs, ce = clean_char_span(text, offsets[tok_start][0], offsets[tok_end][1])
                if ce <= cs:
                    continue
                if cs > 0 and text[cs - 1].isalnum():
                    continue
                if ce < len(text) and text[ce].isalnum():
                    continue
                span_text = text[cs:ce]
                if len(span_text.strip()) < 2 or all(not c.isalnum() for c in span_text):
                    continue
                spans.append({"tok_start": tok_start, "tok_end": tok_end})
                metas.append({"tok_start": tok_start, "tok_end": tok_end,
                               "char_start": cs, "char_end": ce,
                               "text": span_text, "example_idx": ex_idx})
        spans_per_example.append(spans)
        meta_flat.extend(metas)
    return input_ids, attention_mask, spans_per_example, meta_flat


@torch.no_grad()
def predict_batch(model, tokenizer, texts, device, none_idx: int):
    """
    none_idx : index de la classe NONE dans coarse_head.
               -1 si le modèle a été entraîné sans NONE (nocoarsenone).
    """
    if not texts:
        return []
    input_ids, attention_mask, spans_per_example, meta_flat = build_batch_candidates(tokenizer, texts)
    if not meta_flat:
        return [{"ner": [], "svo": []} for _ in texts]

    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    outputs = model({"input_ids": input_ids, "attention_mask": attention_mask,
                     "spans": spans_per_example})

    b_probs      = F.softmax(outputs["boundary_logits"], dim=-1)
    c_probs      = F.softmax(outputs["coarse_logits"], dim=-1)
    f_logits     = outputs["fine_logits"]
    role_probs   = F.softmax(outputs["role_logits"], dim=-1)
    gender_probs = F.softmax(outputs["gender_logits"], dim=-1)
    number_probs = F.softmax(outputs["number_logits"], dim=-1)
    svob_probs   = F.softmax(outputs["svo_boundary_logits"], dim=-1)
    syn_probs    = F.softmax(outputs["syn_logits"], dim=-1)
    voice_probs  = F.softmax(outputs["voice_logits"], dim=-1)
    coarse_fine_mask = model.coarse_fine_mask.to(device)

    n = len(texts)
    per_ner = [[] for _ in range(n)]
    per_svo = [[] for _ in range(n)]

    for i, m in enumerate(meta_flat):
        ex_idx = m["example_idx"]
        p_ent  = float(b_probs[i, 1])
        if p_ent < TAU_BOUNDARY:
            continue

        coarse_row = c_probs[i]

        # ── Filtre NONE : seulement si le modèle a cette classe ──────────────
        if none_idx >= 0:
            p_none = float(coarse_row[none_idx])
            if p_none >= TAU_NONE:
                continue

        # ── Top-k coarse (en excluant NONE si présent) ───────────────────────
        k = min(4, coarse_row.numel())
        top_vals, top_idxs = torch.topk(coarse_row, k=k)
        best, best_score = None, -1.0

        for coarse_prob, coarse_idx_t in zip(top_vals.tolist(), top_idxs.tolist()):
            coarse_idx = int(coarse_idx_t)
            if coarse_idx == none_idx:   # none_idx == -1 → jamais vrai, OK
                continue
            allowed = coarse_fine_mask[coarse_idx]
            if not allowed.any():
                continue
            masked   = f_logits[i].clone().masked_fill(~allowed, -1e9)
            fp       = F.softmax(masked.unsqueeze(0), dim=-1)[0]
            fine_idx = int(torch.argmax(fp))
            fine_prob = float(fp[fine_idx])
            score    = p_ent * coarse_prob * fine_prob
            if score > best_score and score >= MIN_SCORE:
                best_score = score
                role_idx   = int(torch.argmax(role_probs[i]))
                role_label = ID2ROLE.get(role_idx, "NONE") if (0 <= role_idx != ROLE_NONE_ID) else "NONE"
                g_idx = int(torch.argmax(gender_probs[i]))
                n_idx = int(torch.argmax(number_probs[i]))
                gender = GENDER_LABELS[g_idx] if g_idx < GENDER_NONE_ID else None
                number = NUMBER_LABELS[n_idx] if n_idx < NUMBER_NONE_ID else None
                if gender == "N":
                    gender = None
                best = {
                    "char_start": m["char_start"], "char_end": m["char_end"],
                    "text":       m["text"],
                    "coarse":     COARSE_LABELS[coarse_idx],
                    "fine":       FINE_LABELS[fine_idx],
                    "score":      round(score, 4),
                    "svo_role":   role_label,
                    "gender":     gender,
                    "number":     number,
                }
        if best:
            per_ner[ex_idx].append(best)

        # ── SVO boundary ─────────────────────────────────────────────────────
        p_svob = float(svob_probs[i, 1])
        if p_svob >= TAU_SVO_BOUNDARY:
            syn_idx   = int(torch.argmax(syn_probs[i]))
            syn_label = SYN_LABELS[syn_idx] if syn_idx < NUM_SYN else "verb_trigger"
            v_idx     = int(torch.argmax(voice_probs[i]))
            voice     = VOICE_LABELS[v_idx] if v_idx < NUM_VOICE else None
            per_svo[ex_idx].append({
                "char_start": m["char_start"], "char_end": m["char_end"],
                "text": m["text"], "syn_label": syn_label, "voice": voice,
            })

    results = []
    for i in range(n):
        results.append({
            "ner": dedupe_overlaps(per_ner[i]),
            "svo": sorted(per_svo[i], key=lambda x: x["char_start"]),
        })
    return results


def to_dataset_spans(ner_preds, svo_preds):
    spans = []
    for p in ner_preds:
        sp = {"label": p["fine"], "start": p["char_start"], "end": p["char_end"],
              "text": p["text"], "_score": p["score"]}
        if p.get("svo_role") and p["svo_role"] != "NONE":
            sp["svo_role"] = p["svo_role"]
        if p.get("gender"):
            sp["gender"] = p["gender"]
        if p.get("number"):
            sp["number"] = p["number"]
        spans.append(sp)
    for s in svo_preds:
        sp = {"label": s["syn_label"], "start": s["char_start"], "end": s["char_end"],
              "text": s["text"]}
        if s.get("voice"):
            sp["voice"] = s["voice"]
        spans.append(sp)
    return sorted(spans, key=lambda x: x["start"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pré-annote frwiki_silver_raw.jsonl avec le modèle multitask"
    )
    parser.add_argument("--checkpoint", default="/tmp/checkpoint_best_ep60.pt")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device",     default=None)
    parser.add_argument("--input",      default=str(INPUT_RAW),
                        help="Fichier source JSONL (défaut: frwiki_silver_raw.jsonl)")
    parser.add_argument("--output",     default=str(OUTPUT_PREANNO),
                        help="Fichier de sortie JSONL (défaut: frwiki_preannotated.jsonl)")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Autoriser l'écrasement si le fichier de sortie existe déjà")
    args = parser.parse_args()

    load_secrets()
    device = pick_device(args.device)
    model, tokenizer, none_idx = load_model(args.checkpoint, args.model_name, device)

    input_path  = Path(args.input)
    output_path = Path(args.output)

    # ── Protection contre l'écrasement accidentel ─────────────────────────────
    if output_path.exists() and not args.overwrite:
        size = output_path.stat().st_size / 1024
        lines = sum(1 for _ in open(output_path))
        print(f"❌ STOP : {output_path} existe déjà ({lines} lignes, {size:.0f} KB)")
        print(f"   Utilisez --overwrite pour écraser, ou choisissez un autre --output")
        sys.exit(1)

    print(f"📂 Input  : {input_path}")
    print(f"💾 Output : {output_path}")

    rows = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"\n📝 {len(rows)} phrases à pré-annoter → {output_path}\n")

    svo_role_counts  = Counter()
    ner_label_counts = Counter()
    t0 = time.perf_counter()

    with open(output_path, "w", encoding="utf-8") as out_f:
        for batch_start in range(0, len(rows), args.batch_size):
            batch = rows[batch_start: batch_start + args.batch_size]
            texts = [clean_text(r["text"]) for r in batch]
            try:
                results = predict_batch(model, tokenizer, texts, device, none_idx)
            except Exception as e:
                print(f"\n  ⚠️  Erreur batch {batch_start}: {e}")
                results = [{"ner": [], "svo": []} for _ in batch]

            for row, text, res in zip(batch, texts, results):
                spans = to_dataset_spans(res["ner"], res["svo"])
                for sp in spans:
                    if sp.get("svo_role"):
                        svo_role_counts[sp["svo_role"]] += 1
                    ner_label_counts[sp["label"]] += 1

                entry = {
                    "id":    row["article"].replace(" ", "_") + f"__{batch_start}",
                    "text":  text,
                    "spans": spans,
                    "_meta": {
                        "source":       "frwiki",
                        "category":     row["category"],
                        "target_label": row["target_label"],
                        "article":      row["article"],
                    },
                }
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            elapsed = time.perf_counter() - t0
            done    = min(batch_start + args.batch_size, len(rows))
            speed   = done / elapsed
            eta     = (len(rows) - done) / speed if speed > 0 else 0
            rare    = sum(svo_role_counts.get(r, 0) for r in RARE_SVO_ROLES)
            print(f"\r  {done:>6}/{len(rows)}  {speed:.1f} ph/s  ETA {eta:.0f}s  "
                  f"svo_rare={rare}", end="", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n\n✅ Pré-annotation terminée en {elapsed:.1f}s → {output_path}")

    print(f"\nSVO rôles détectés :")
    for role, cnt in sorted(svo_role_counts.items(), key=lambda x: -x[1]):
        marker = "  ← RARE" if role in RARE_SVO_ROLES else ""
        print(f"  {role:<20} {cnt:>5}{marker}")

    print(f"\nTop NER labels détectés :")
    for label, cnt in ner_label_counts.most_common(15):
        print(f"  {label:<30} {cnt:>6}")


if __name__ == "__main__":
    main()

