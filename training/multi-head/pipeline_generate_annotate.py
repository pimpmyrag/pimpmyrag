#!/usr/bin/env python3
"""
pipeline_generate_annotate.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pipeline adaptatif : génère avec Mistral → pré-annote avec DeBERTa → compte les confirmations.
S'arrête quand chaque label rare a TARGET_PER_LABEL exemples confirmés supplémentaires.

Confirmation = le modèle prédit le bon label (ou le bon svo_role) sur le primary_span.

Usage :
  python pipeline_generate_annotate.py
  python pipeline_generate_annotate.py --target 1000 --batch-gen 4 --device cpu
"""
from __future__ import annotations

import argparse, json, os, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer
from dotenv import dotenv_values

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

# ── Import des metas depuis r2 ─────────────────────────────────────────────────
from generate_targeted_mistral_r2 import (
    NER_LABEL_META, SVO_ROLE_META, STYLE_VARIANTS,
    build_messages_ner, build_messages_svo,
    extract_json_payload, call_mistral,
    validate_ner_items, validate_svo_items,
    load_api_key, slugify,
)

# ── Config inférence ───────────────────────────────────────────────────────────
TAU_BOUNDARY     = 0.55
TAU_NONE         = 0.95
MIN_SCORE        = 0.05
TAU_SVO_BOUNDARY = 0.50
MAX_SPAN_LEN     = 12
MAX_LENGTH       = 128
IOU_THRESHOLD    = 0.50

API_URL = "https://api.mistral.ai/v1/chat/completions"


# ─────────────────────────────────────────────────────────────────────────────
#  Labels cibles (NER + rôles SVO)
# ─────────────────────────────────────────────────────────────────────────────

ALL_NER_LABELS  = list(NER_LABEL_META.keys())
ALL_SVO_ROLES   = list(SVO_ROLE_META.keys())

def all_targets() -> list[str]:
    return ALL_NER_LABELS + [f"svo:{r}" for r in ALL_SVO_ROLES]


# ─────────────────────────────────────────────────────────────────────────────
#  Comptage des exemples déjà présents dans le dataset courant
# ─────────────────────────────────────────────────────────────────────────────

def count_existing(data_dir: Path) -> Counter:
    counts: Counter = Counter()
    for split in ("train", "val", "test"):
        p = data_dir / f"{split}_v7.0.jsonl"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                for sp in row.get("spans", []):
                    lbl = sp.get("label", "")
                    if lbl in ALL_NER_LABELS:
                        counts[lbl] += 1
                    role = sp.get("svo_role", "")
                    if role in ALL_SVO_ROLES:
                        counts[f"svo:{role}"] += 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
#  Générateur de jobs infinis (cycle sur les thèmes + variation de style)
# ─────────────────────────────────────────────────────────────────────────────

_job_cursor: dict[str, int] = defaultdict(int)  # label → indice thème courant

def next_job(label: str, sentences_per_job: int) -> dict[str, Any]:
    """Génère le prochain job pour un label, en cyclant sur les thèmes."""
    if label.startswith("svo:"):
        role = label[4:]
        meta = SVO_ROLE_META[role]
        themes = meta["themes"]
        idx = _job_cursor[label] % (len(themes) * len(STYLE_VARIANTS))
        t_idx = idx % len(themes)
        s_idx = (idx // len(themes)) % len(STYLE_VARIANTS)
        _job_cursor[label] += 1
        job = {
            "job_id": f"svo_{role.lower()}__{slugify(themes[t_idx])}__dyn{_job_cursor[label]}",
            "job_type": "svo_role",
            "target_role": role,
            "description": meta["description"],
            "svo_hint": meta["svo_hint"],
            "entity_types": meta["entity_types"],
            "theme": themes[t_idx],
            "style": STYLE_VARIANTS[s_idx],
            "sentences_per_job": sentences_per_job,
        }
    else:
        meta = NER_LABEL_META[label]
        themes = meta["themes"]
        idx = _job_cursor[label] % (len(themes) * len(STYLE_VARIANTS))
        t_idx = idx % len(themes)
        s_idx = (idx // len(themes)) % len(STYLE_VARIANTS)
        _job_cursor[label] += 1
        job = {
            "job_id": f"{label}__{slugify(themes[t_idx])}__dyn{_job_cursor[label]}",
            "job_type": "ner",
            "target_label": label,
            "description": meta["description"],
            "confusions": meta["confusions"],
            "theme": themes[t_idx],
            "style": STYLE_VARIANTS[s_idx],
            "sentences_per_job": sentences_per_job,
        }
    return job


# ─────────────────────────────────────────────────────────────────────────────
#  Chargement modèle
# ─────────────────────────────────────────────────────────────────────────────

def pick_device(forced=None):
    if forced: return forced
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def load_model(checkpoint: str, model_name: str, device: str):
    print(f"📦 Chargement modèle {checkpoint} sur {device} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if getattr(tokenizer, "model_max_length", None) is None or tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = MAX_LENGTH
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("ema_state") or ckpt.get("model_state") or ckpt
    num_coarse = state["coarse_head.weight"].shape[0]
    model = SpanMultiTaskModel(model_name=model_name, num_coarse=num_coarse).to(device).float()
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"✅ Modèle chargé | num_coarse={num_coarse}")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
#  Inférence (identique à preannotate_mistral.py)
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    return re.sub(r'\*\*', '', text)

def clean_char_span(text, s, e):
    while s < e and text[s].isspace(): s += 1
    while e > s and text[e-1].isspace(): e -= 1
    return s, e

def span_iou(a, b):
    inter = max(0, min(a["char_end"], b["char_end"]) - max(a["char_start"], b["char_start"]))
    if inter == 0: return 0.0
    union = (a["char_end"]-a["char_start"]) + (b["char_end"]-b["char_start"]) - inter
    return inter / union

def dedupe_overlaps(preds):
    seen = set()
    uniq = [p for k in [None] for p in preds if not seen.add((p["char_start"], p["char_end"], p["fine"])) and (p["char_start"], p["char_end"], p["fine"]) not in seen or True]
    # Simplifi : juste dédupliquer
    seen2 = set()
    uniq2 = []
    for p in preds:
        k = (p["char_start"], p["char_end"], p["fine"])
        if k not in seen2:
            seen2.add(k)
            uniq2.append(p)
    preds = sorted(uniq2, key=lambda x: x["score"], reverse=True)
    kept = []
    for p in preds:
        if not any(span_iou(p, k) >= IOU_THRESHOLD and p["coarse"] == k["coarse"] for k in kept):
            kept.append(p)
    return sorted(kept, key=lambda x: x["char_start"])

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
                if ce <= cs: continue
                if cs > 0 and text[cs-1].isalnum(): continue
                if ce < len(text) and text[ce].isalnum(): continue
                span_text = text[cs:ce]
                if len(span_text.strip()) < 2 or all(not c.isalnum() for c in span_text): continue
                spans.append({"tok_start": tok_start, "tok_end": tok_end})
                metas.append({"tok_start": tok_start, "tok_end": tok_end,
                               "char_start": cs, "char_end": ce,
                               "text": span_text, "example_idx": ex_idx})
        spans_per_example.append(spans)
        meta_flat.extend(metas)
    return input_ids, attention_mask, spans_per_example, meta_flat

@torch.no_grad()
def predict_batch(model, tokenizer, texts, device):
    if not texts: return []
    import torch.nn.functional as F
    input_ids, attention_mask, spans_per_example, meta_flat = build_batch_candidates(tokenizer, texts)
    if not meta_flat:
        return [{"ner": [], "svo": []} for _ in texts]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    outputs = model({"input_ids": input_ids, "attention_mask": attention_mask, "spans": spans_per_example})
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
    none_idx = COARSE2ID["NONE"]
    n = len(texts)
    per_ner = [[] for _ in range(n)]
    per_svo = [[] for _ in range(n)]
    for i, m in enumerate(meta_flat):
        ex_idx = m["example_idx"]
        p_ent = float(b_probs[i, 1])
        if p_ent >= TAU_BOUNDARY:
            coarse_row = c_probs[i]
            p_none = float(coarse_row[none_idx])
            if p_none < TAU_NONE:
                top_vals, top_idxs = torch.topk(coarse_row, k=min(4, coarse_row.numel()))
                best, best_score = None, -1.0
                for coarse_prob, coarse_idx_t in zip(top_vals.tolist(), top_idxs.tolist()):
                    coarse_idx = int(coarse_idx_t)
                    if coarse_idx == none_idx: continue
                    allowed = coarse_fine_mask[coarse_idx]
                    if not allowed.any(): continue
                    masked = f_logits[i].clone().masked_fill(~allowed, -1e9)
                    fp = F.softmax(masked.unsqueeze(0), dim=-1)[0]
                    fine_idx = int(torch.argmax(fp))
                    fine_prob = float(fp[fine_idx])
                    score = p_ent * coarse_prob * fine_prob
                    if score > best_score and score >= MIN_SCORE:
                        best_score = score
                        role_idx = int(torch.argmax(role_probs[i]))
                        role_label = ID2ROLE.get(role_idx, "NONE") if (0 <= role_idx != ROLE_NONE_ID) else "NONE"
                        g_idx = int(torch.argmax(gender_probs[i]))
                        n_idx = int(torch.argmax(number_probs[i]))
                        gender = GENDER_LABELS[g_idx] if g_idx < GENDER_NONE_ID else None
                        number = NUMBER_LABELS[n_idx] if n_idx < NUMBER_NONE_ID else None
                        if gender == "N": gender = None
                        best = {
                            "char_start": m["char_start"], "char_end": m["char_end"],
                            "text": m["text"],
                            "coarse": COARSE_LABELS[coarse_idx],
                            "fine": FINE_LABELS[fine_idx],
                            "score": round(score, 4),
                            "svo_role": role_label,
                            "gender": gender,
                            "number": number,
                        }
                if best:
                    per_ner[ex_idx].append(best)
        p_svob = float(svob_probs[i, 1])
        if p_svob >= TAU_SVO_BOUNDARY:
            syn_idx = int(torch.argmax(syn_probs[i]))
            syn_label = SYN_LABELS[syn_idx] if syn_idx < NUM_SYN else "verb_trigger"
            v_idx = int(torch.argmax(voice_probs[i]))
            voice = VOICE_LABELS[v_idx] if v_idx < NUM_VOICE else None
            per_svo[ex_idx].append({
                "char_start": m["char_start"], "char_end": m["char_end"],
                "text": m["text"], "syn_label": syn_label, "voice": voice,
            })
    results = []
    for i in range(n):
        results.append({"ner": dedupe_overlaps(per_ner[i]),
                         "svo": sorted(per_svo[i], key=lambda x: x["char_start"])})
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Vérification de confirmation sur le primary_span
# ─────────────────────────────────────────────────────────────────────────────

def is_confirmed(row: dict, ner_preds: list, target: str) -> bool:
    """
    Vérifie si le modèle a prédit le bon label/rôle sur ou près du primary_span.
    - NER label  : fine == target_label et overlap avec primary_span
    - SVO role   : svo_role == role et overlap avec primary_span
    """
    text = row["text"]
    span_text = row["primary_span_text"]
    # Cherche la position du primary_span dans le texte
    pos = text.find(span_text)
    if pos < 0:
        return False
    span_start, span_end = pos, pos + len(span_text)

    if target.startswith("svo:"):
        role = target[4:]
        for p in ner_preds:
            overlap = min(p["char_end"], span_end) - max(p["char_start"], span_start)
            if overlap > 0 and p.get("svo_role") == role:
                return True
        return False
    else:
        for p in ner_preds:
            overlap = min(p["char_end"], span_end) - max(p["char_start"], span_start)
            if overlap > 0 and p["fine"] == target:
                return True
        return False


def to_dataset_span(p: dict) -> dict:
    span = {"label": p["fine"], "start": p["char_start"], "end": p["char_end"], "text": p["text"],
            "_score": p["score"]}
    if p.get("svo_role") and p["svo_role"] != "NONE": span["svo_role"] = p["svo_role"]
    if p.get("gender"): span["gender"] = p["gender"]
    if p.get("number"): span["number"] = p["number"]
    return span


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      default="checkpoint_best_multitask.pt")
    parser.add_argument("--model-name",      default="microsoft/deberta-v3-base")
    parser.add_argument("--device",          default=None)
    parser.add_argument("--target",          type=int, default=1000,
                        help="Nombre d'exemples confirmés supplémentaires par label")
    parser.add_argument("--batch-gen",       type=int, default=4,
                        help="Phrases générées par appel Mistral")
    parser.add_argument("--batch-infer",     type=int, default=8,
                        help="Batch size pour l'inférence DeBERTa")
    parser.add_argument("--model",           default="mistral-large-latest",
                        help="Modèle Mistral (mistral-large-latest ou mistral-small-latest)")
    parser.add_argument("--delay",           type=float, default=2.0,
                        help="Délai entre appels Mistral (rate limit)")
    parser.add_argument("--output",          default="data/pipeline_generated.jsonl",
                        help="Fichier de sortie (pré-annoté confirmé)")
    parser.add_argument("--raw-output",      default="data/pipeline_raw.jsonl",
                        help="Toutes les phrases générées (y compris non confirmées)")
    parser.add_argument("--max-attempts",    type=int, default=500,
                        help="Tentatives max par label avant abandon")
    parser.add_argument("--labels",          default=None,
                        help="Sous-ensemble de labels séparés par virgule (défaut: tous)")
    parser.add_argument("--api-key",         default=None)
    args = parser.parse_args()

    device     = pick_device(args.device)
    model, tok = load_model(args.checkpoint, args.model_name, device)
    api_key    = load_api_key(args.api_key)

    # Labels à cibler
    if args.labels:
        targets = [x.strip() for x in args.labels.split(",")]
    else:
        targets = all_targets()

    # Comptage existant dans le dataset
    data_dir = ROOT / "data"
    existing = count_existing(data_dir)
    print("\n📊 Exemples existants par label :")
    for t in targets:
        print(f"  {t:<35} {existing.get(t, 0):5d} existants  →  cible +{args.target}")

    # Compteurs de nouvelles confirmations (session courante)
    confirmed_counts: Counter = Counter()
    attempt_counts:   Counter = Counter()

    # Reprise depuis l'output existant
    output_path  = ROOT / args.output
    raw_path     = ROOT / args.raw_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    lbl = row.get("_confirmed_label", "")
                    if lbl in targets: confirmed_counts[lbl] += 1
                    if lbl.startswith("svo:"):
                        role = lbl[4:]
                        if f"svo:{role}" in targets: confirmed_counts[f"svo:{role}"] += 1
                except Exception: pass
        print(f"\n♻️  Reprise : {sum(confirmed_counts.values())} confirmations déjà présentes")

    out_f = open(output_path, "a", encoding="utf-8")
    raw_f = open(raw_path,    "a", encoding="utf-8")

    t0 = time.perf_counter()
    total_gen = 0
    total_ok  = 0
    iteration = 0

    print("\n🚀 Démarrage pipeline...\n")

    while True:
        # Labels qui n'ont pas encore leur quota
        todo = [t for t in targets
                if confirmed_counts[t] < args.target
                and attempt_counts[t] < args.max_attempts]

        if not todo:
            break  # tous les labels sont complets (ou abandonnés)

        # Priorité : label le plus en retard (ratio confirmé/cible le plus bas)
        todo.sort(key=lambda t: confirmed_counts[t] / args.target)
        # Prend les 2 labels les plus en retard pour varier
        pick = todo[:2]
        random.shuffle(pick)
        label = pick[0]

        iteration += 1
        attempt_counts[label] += 1

        # Génère des phrases Mistral pour ce label
        job = next_job(label, args.batch_gen)
        try:
            if job["job_type"] == "ner":
                messages = build_messages_ner(job)
            else:
                messages = build_messages_svo(job)
            content = call_mistral(api_key, args.model, messages, 90, 3)
            payload = extract_json_payload(content)
            if job["job_type"] == "ner":
                rows = validate_ner_items(job, payload)
            else:
                rows = validate_svo_items(job, payload)
        except Exception as e:
            print(f"  ⚠️  Mistral erreur [{label}]: {e}")
            time.sleep(5)
            continue

        if not rows:
            print(f"  ⚠️  0 phrases valides pour {label}")
            continue

        # Filtre les phrases qui contiennent le nom du label dans le texte (artefact Mistral)
        rows = [r for r in rows if r["target_label"] not in r["text"]]
        if not rows:
            print(f"  ⚠️  toutes les phrases filtrées (label dans texte) pour {label}")
            continue

        # Écrit le raw
        for row in rows:
            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        raw_f.flush()
        total_gen += len(rows)

        # Inférence DeBERTa par batches
        for batch_start in range(0, len(rows), args.batch_infer):
            batch = rows[batch_start: batch_start + args.batch_infer]
            texts = [clean_text(r["text"]) for r in batch]
            try:
                results = predict_batch(model, tok, texts, device)
            except Exception as e:
                print(f"  ⚠️  Inférence erreur: {e}")
                continue

            for row, text, res in zip(batch, texts, results):
                # Détermine le target de confirmation
                conf_target = (f"svo:{row['target_svo_role']}"
                               if "target_svo_role" in row
                               else row["target_label"])

                confirmed = is_confirmed({"text": text, "primary_span_text": row["primary_span_text"]},
                                         res["ner"], conf_target)

                # Spans au format dataset
                ner_spans = [to_dataset_span(p) for p in res["ner"]]
                svo_spans = [{"label": s["syn_label"], "start": s["char_start"], "end": s["char_end"],
                               "text": s["text"], **({"voice": s["voice"]} if s.get("voice") else {})}
                              for s in res["svo"]]
                all_spans = sorted(ner_spans + svo_spans, key=lambda x: x["start"])

                entry = {
                    "id":     row["job_id"],
                    "text":   text,
                    "spans":  all_spans,
                    "_confirmed": confirmed,
                    "_confirmed_label": conf_target if confirmed else None,
                    "_meta": {
                        "target_label":  row["target_label"],
                        "target_svo_role": row.get("target_svo_role"),
                        "primary_span":  row["primary_span_text"],
                        "difficulty":    row["difficulty"],
                        "why":           row["why"],
                        "theme":         row["theme"],
                        "source":        row.get("source", ""),
                    }
                }

                if confirmed:
                    confirmed_counts[conf_target] += 1
                    total_ok += 1
                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    out_f.flush()

        # Affichage état
        elapsed = time.perf_counter() - t0
        done_labels    = sum(1 for t in targets if confirmed_counts[t] >= args.target)
        dropped_labels = sum(1 for t in targets if attempt_counts[t] >= args.max_attempts
                             and confirmed_counts[t] < args.target)

        # Barre de progression courte
        status = "  ".join(
            f"{t.split(':')[-1][:12]}:{confirmed_counts[t]}/{args.target}"
            for t in targets
        )
        print(f"\r[{iteration:4d}] {elapsed:5.0f}s | gen={total_gen:5d} ok={total_ok:5d} "
              f"done={done_labels}/{len(targets)} | {status}", end="", flush=True)

        if done_labels + dropped_labels >= len(targets):
            break

        time.sleep(args.delay)

    out_f.close()
    raw_f.close()

    elapsed = time.perf_counter() - t0
    print(f"\n\n{'='*70}")
    print(f"✅ Pipeline terminé en {elapsed:.0f}s")
    print(f"   Phrases générées : {total_gen}")
    print(f"   Confirmées       : {total_ok}")
    print(f"\n📊 Résultat final par label :")
    for t in targets:
        status = "✅" if confirmed_counts[t] >= args.target else ("❌ max_attempts" if attempt_counts[t] >= args.max_attempts else "⚠️")
        print(f"  {status}  {t:<35} {confirmed_counts[t]:5d}/{args.target}")
    print(f"\n  → {output_path}")


if __name__ == "__main__":
    main()

