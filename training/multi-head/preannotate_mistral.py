#!/usr/bin/env python3
"""
preannotate_mistral.py
~~~~~~~~~~~~~~~~~~~~~~
Pré-annote mistral_targeted_generations_r1.jsonl avec le modèle en cours de training.

Sortie :  data/preannotated_mistral_r1.jsonl
  - format identique au dataset d'entraînement (id, text, spans)
  - chaque span contient : label (fine), start, end, text,
    + optionnel : svo_role, gender, number
  - champs méta conservés pour review Claude : target_label, difficulty, why, job_id

Usage :
  python preannotate_mistral.py
  python preannotate_mistral.py --checkpoint checkpoint_best_multitask.pt --batch-size 16
"""

import argparse, json, re, sys, time
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer

# ── Chemins relatifs au script ─────────────────────────────────────────────────
ROOT = Path(__file__).parent
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

# ── Paramètres de décodage NER ─────────────────────────────────────────────────
# On est généreux (tau bas) pour la pré-annotation — Claude corrigera
TAU_BOUNDARY    = 0.40
TAU_NONE        = 0.99
TAU_COARSE      = 0.00
TAU_FINE        = 0.00
TAU_SVO_BOUNDARY = 0.45

MAX_SPAN_LEN    = 12
MAX_LENGTH      = 128
IOU_THRESHOLD   = 0.60


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def clean_text(text: str) -> str:
    """Enlève les marqueurs Markdown bold (**) utilisés dans les générations Mistral."""
    return re.sub(r'\*\*', '', text)


# ── Géométrie spans ────────────────────────────────────────────────────────────

def clean_char_span(text, s, e):
    while s < e and text[s].isspace(): s += 1
    while e > s and text[e-1].isspace(): e -= 1
    return s, e


def span_iou(a, b):
    inter = max(0, min(a["char_end"], b["char_end"]) - max(a["char_start"], b["char_start"]))
    if inter == 0: return 0.0
    union = (a["char_end"]-a["char_start"]) + (b["char_end"]-b["char_start"]) - inter
    return inter / union


def adjusted_score(p):
    return float(p["score"])


def dedupe_overlaps(preds):
    seen = set()
    uniq = []
    for p in preds:
        key = (p["char_start"], p["char_end"], p["fine"])
        if key in seen: continue
        seen.add(key)
        uniq.append(p)
    # NMS : garder meilleur score si fort overlap (même coarse group)
    preds = sorted(uniq, key=adjusted_score, reverse=True)
    kept = []
    for p in preds:
        drop = False
        for k in kept:
            if span_iou(p, k) >= IOU_THRESHOLD and p["coarse"] == k["coarse"]:
                drop = True; break
        if not drop:
            kept.append(p)
    return sorted(kept, key=lambda x: x["char_start"])


# ── Chargement ─────────────────────────────────────────────────────────────────

def load_model(checkpoint: str, model_name: str, device: str):
    print(f"📦 Chargement modèle depuis {checkpoint} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if getattr(tokenizer, "model_max_length", None) is None or tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = MAX_LENGTH

    model = SpanMultiTaskModel(model_name=model_name).to(device).float()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    # Supporte EMA (ema_state) ou model_state direct
    state = ckpt.get("ema_state") or ckpt.get("model_state") or ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"✅ Modèle chargé  |  device={device}")
    return model, tokenizer


# ── Construction des spans candidats ──────────────────────────────────────────

def build_batch_candidates(tokenizer, texts: List[str]):
    enc = tokenizer(
        texts, return_offsets_mapping=True, add_special_tokens=True,
        truncation=True, padding=True, max_length=MAX_LENGTH,
    )
    input_ids      = torch.tensor(enc["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
    offsets_batch  = enc["offset_mapping"]

    spans_per_example = []
    meta_flat         = []

    for ex_idx, (text, offsets) in enumerate(zip(texts, offsets_batch)):
        tok_positions = [i for i, (s, e) in enumerate(offsets) if e > s]
        spans, metas = [], []
        for si, tok_start in enumerate(tok_positions):
            for tok_end in tok_positions[si: si + MAX_SPAN_LEN]:
                cs, ce = clean_char_span(text, offsets[tok_start][0], offsets[tok_end][1])
                if ce <= cs: continue
                # word boundary gauche
                if cs > 0 and text[cs-1].isalnum(): continue
                # word boundary droite
                if ce < len(text) and text[ce].isalnum(): continue
                span_text = text[cs:ce]
                if len(span_text.strip()) < 2 or all(not c.isalnum() for c in span_text): continue
                spans.append({"tok_start": tok_start, "tok_end": tok_end})
                metas.append({
                    "tok_start": tok_start, "tok_end": tok_end,
                    "char_start": cs, "char_end": ce,
                    "text": span_text, "example_idx": ex_idx,
                })
        spans_per_example.append(spans)
        meta_flat.extend(metas)

    return input_ids, attention_mask, spans_per_example, meta_flat


# ── Décodage ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_batch(model, tokenizer, texts: List[str], device: str) -> List[Dict]:
    if not texts:
        return []
    input_ids, attention_mask, spans_per_example, meta_flat = build_batch_candidates(tokenizer, texts)
    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model({"input_ids": input_ids, "attention_mask": attention_mask, "spans": spans_per_example})

    import torch.nn.functional as F
    b_probs     = F.softmax(outputs["boundary_logits"], dim=-1)
    c_probs     = F.softmax(outputs["coarse_logits"], dim=-1)
    f_logits    = outputs["fine_logits"]
    role_probs  = F.softmax(outputs["role_logits"], dim=-1)
    gender_probs= F.softmax(outputs["gender_logits"], dim=-1)
    number_probs= F.softmax(outputs["number_logits"], dim=-1)
    svob_probs  = F.softmax(outputs["svo_boundary_logits"], dim=-1)
    syn_probs   = F.softmax(outputs["syn_logits"], dim=-1)
    voice_probs = F.softmax(outputs["voice_logits"], dim=-1)

    coarse_fine_mask = model.coarse_fine_mask.to(device)
    none_idx = COARSE2ID["NONE"]

    n = len(texts)
    per_ner = [[] for _ in range(n)]
    per_svo = [[] for _ in range(n)]

    for i, m in enumerate(meta_flat):
        ex_idx = m["example_idx"]

        # ── NER ──────────────────────────────────────────────────────────────
        p_ent = float(b_probs[i, 1])
        if p_ent >= TAU_BOUNDARY:
            coarse_row = c_probs[i]
            p_none = float(coarse_row[none_idx])
            if p_none < TAU_NONE:
                top_vals, top_idxs = torch.topk(coarse_row, k=min(4, coarse_row.numel()))
                best, best_score = None, -1.0
                for coarse_prob, coarse_idx_t in zip(top_vals.tolist(), top_idxs.tolist()):
                    coarse_idx = int(coarse_idx_t)
                    if coarse_idx == none_idx or coarse_prob < TAU_COARSE: continue
                    allowed = coarse_fine_mask[coarse_idx]
                    if not allowed.any(): continue
                    masked = f_logits[i].clone().masked_fill(~allowed, -1e9)
                    fp = F.softmax(masked.unsqueeze(0), dim=-1)[0]
                    fine_idx = int(torch.argmax(fp))
                    fine_prob = float(fp[fine_idx])
                    if fine_prob < TAU_FINE: continue
                    score = p_ent * coarse_prob * fine_prob
                    if score > best_score:
                        best_score = score
                        # rôle SVO sur ce span NER
                        role_idx = int(torch.argmax(role_probs[i]))
                        role_label = ID2ROLE.get(role_idx, "NONE") if role_idx < ROLE_NONE_ID else "NONE"
                        # morpho
                        g_idx = int(torch.argmax(gender_probs[i]))
                        n_idx = int(torch.argmax(number_probs[i]))
                        gender = GENDER_LABELS[g_idx] if g_idx < GENDER_NONE_ID else None
                        number = NUMBER_LABELS[n_idx] if n_idx < NUMBER_NONE_ID else None
                        # normalise N → None
                        if gender == "N": gender = None

                        best = {
                            "char_start": m["char_start"], "char_end": m["char_end"],
                            "tok_start": m["tok_start"], "tok_end": m["tok_end"],
                            "text": m["text"],
                            "coarse": COARSE_LABELS[coarse_idx],
                            "fine": FINE_LABELS[fine_idx],
                            "boundary_prob": round(p_ent, 4),
                            "coarse_prob": round(float(coarse_prob), 4),
                            "fine_prob": round(fine_prob, 4),
                            "score": round(score, 4),
                            "svo_role": role_label,
                            "gender": gender,
                            "number": number,
                        }
                if best:
                    per_ner[ex_idx].append(best)

        # ── SVO boundary (verb + pronoms) ─────────────────────────────────
        p_svob = float(svob_probs[i, 1])
        if p_svob >= TAU_SVO_BOUNDARY:
            syn_idx = int(torch.argmax(syn_probs[i]))
            syn_label = SYN_LABELS[syn_idx] if syn_idx < NUM_SYN else "verb_trigger"
            v_idx = int(torch.argmax(voice_probs[i]))
            voice = VOICE_LABELS[v_idx] if v_idx < NUM_VOICE else None
            g_idx = int(torch.argmax(gender_probs[i]))
            n_idx2 = int(torch.argmax(number_probs[i]))
            gender = GENDER_LABELS[g_idx] if g_idx < GENDER_NONE_ID else None
            number = NUMBER_LABELS[n_idx2] if n_idx2 < NUMBER_NONE_ID else None
            if gender == "N": gender = None
            per_svo[ex_idx].append({
                "char_start": m["char_start"], "char_end": m["char_end"],
                "text": m["text"],
                "syn_label": syn_label,
                "voice": voice,
                "gender": gender, "number": number,
                "svo_boundary_prob": round(p_svob, 4),
            })

    results = []
    for i in range(n):
        ner = dedupe_overlaps(per_ner[i])
        svo = sorted(per_svo[i], key=lambda x: x["char_start"])
        results.append({"ner": ner, "svo": svo})
    return results


# ── Conversion → format dataset ───────────────────────────────────────────────

def to_dataset_spans(ner_preds: List[Dict], svo_preds: List[Dict]) -> List[Dict]:
    """
    Convertit les prédictions brutes en spans au format dataset d'entraînement.
    NER spans → label/start/end/text + svo_role/gender/number optionnels
    SVO spans (verbes/pronoms) → label=syn_label + voice optionnel
    """
    spans = []

    for p in ner_preds:
        span: Dict[str, Any] = {
            "label": p["fine"],
            "start": p["char_start"],
            "end":   p["char_end"],
            "text":  p["text"],
            # scores pour la review (non utilisés à l'entraînement mais utiles pour Claude)
            "_score": p["score"],
            "_boundary_prob": p["boundary_prob"],
        }
        if p.get("svo_role") and p["svo_role"] != "NONE":
            span["svo_role"] = p["svo_role"]
        if p.get("gender"):
            span["gender"] = p["gender"]
        if p.get("number"):
            span["number"] = p["number"]
        spans.append(span)

    for s in svo_preds:
        span = {
            "label": s["syn_label"],   # verb_trigger / pron_subj / pron_obj
            "start": s["char_start"],
            "end":   s["char_end"],
            "text":  s["text"],
            "_svo_boundary_prob": s["svo_boundary_prob"],
        }
        if s.get("voice"):
            span["voice"] = s["voice"]
        if s.get("gender"):
            span["gender"] = s["gender"]
        if s.get("number"):
            span["number"] = s["number"]
        spans.append(span)

    return sorted(spans, key=lambda x: x["start"])


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="data/mistral_targeted_generations_r1.jsonl")
    parser.add_argument("--output",     default="data/preannotated_mistral_r1.jsonl")
    parser.add_argument("--checkpoint", default="checkpoint_best_multitask.pt")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    device = args.device or pick_device()
    model, tokenizer = load_model(args.checkpoint, args.model_name, device)

    input_path  = ROOT / args.input
    output_path = ROOT / args.output

    print(f"📂 Input  : {input_path}")
    print(f"💾 Output : {output_path}")

    rows = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"📝 {len(rows)} phrases à pré-annoter")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output_path, "w", encoding="utf-8")

    total_spans = 0
    t0 = time.perf_counter()

    for batch_start in range(0, len(rows), args.batch_size):
        batch = rows[batch_start: batch_start + args.batch_size]
        # Nettoyage des marqueurs Markdown **bold**
        texts = [clean_text(r["text"]) for r in batch]

        try:
            results = predict_batch(model, tokenizer, texts, device)
        except Exception as e:
            print(f"  ⚠️  Erreur batch {batch_start}: {e}")
            results = [{"ner": [], "svo": []} for _ in batch]

        for row, text, res in zip(batch, texts, results):
            spans = to_dataset_spans(res["ner"], res["svo"])
            total_spans += len([s for s in spans if not s["label"].startswith("verb_")])

            entry = {
                "id": row["job_id"],
                "text": text,
                "spans": spans,
                # méta pour review Claude
                "_meta": {
                    "target_label": row["target_label"],
                    "difficulty":   row["difficulty"],
                    "why":          row["why"],
                    "primary_span": row["primary_span_text"],
                    "theme":        row["theme"],
                    "style":        row["style"],
                    "source":       row.get("source", ""),
                }
            }
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        elapsed = time.perf_counter() - t0
        done = min(batch_start + args.batch_size, len(rows))
        speed = done / elapsed
        eta   = (len(rows) - done) / speed if speed > 0 else 0
        print(f"\r  {done}/{len(rows)}  |  {speed:.1f} phrases/s  |  ETA {eta:.0f}s", end="", flush=True)

    out_f.close()
    elapsed = time.perf_counter() - t0
    print(f"\n\n✅ {len(rows)} phrases pré-annotées en {elapsed:.1f}s")
    print(f"   {total_spans} spans NER générés")
    print(f"   → {output_path}")


if __name__ == "__main__":
    main()

