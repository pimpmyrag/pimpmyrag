"""
test_model_sentences_v3.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Version 3 : NER rich + têtes SVO / voice / svo_boundary.

Nouveautés vs v2 :
  - predict_text / predict_texts_batch retournent aussi les spans SVO/voice/svo_boundary
  - affichage séparé des entités NER et des spans SVO
  - option --show-svo pour afficher les spans SVO
  - option --tau-svo-boundary pour filtrer les spans verbe/pronom
  - reconstruction des triplets (S, V, O) par phrase
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from multitask_model import SpanMultiTaskModel
from labels import COARSE_LABELS, FINE_LABELS, COARSE2ID, SVO_LABELS, NUM_SVO, NUM_VOICE


# ──────────────────────────────────────────────────────────
#  Device
# ──────────────────────────────────────────────────────────

def pick_device(forced: str | None = None) -> str:
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ──────────────────────────────────────────────────────────
#  Seuils & config NER (repris de v2)
# ──────────────────────────────────────────────────────────

FINE_THRESHOLDS = {
    "hint_person_name":    0.90,
    "hint_org_name":       0.90,
    "hint_gpe":            0.70,
    "hint_fac_name":       0.70,
    "hint_time_date":      0.70,
    "hint_time_clock":     0.70,
    "hint_person_role":    0.95,
    "hint_group_role":     0.95,
    "hint_event_nominal":  0.97,
    "hint_object_generic": 0.97,
}
DEFAULT_FINE_THRESHOLD = 0.80

LONG_SPAN_TYPES = {"hint_org_name", "hint_gpe", "hint_fac_name", "hint_time_date", "hint_time_clock"}
SHORT_SPAN_TYPES = {"hint_person_role", "hint_group_role"}

MAX_TOK_LEN_PER_FINE = {
    "hint_person_name":     6,
    "hint_person_role":     4,
    "hint_group_role":      4,
    "hint_gpe":             5,
    "hint_org_name":        8,
    "hint_fac_name":        7,
    "hint_time_date":       6,
    "hint_time_clock":      5,
    "hint_event_nominal":   6,
    "hint_object_generic":  5,
    "hint_percentage":      4,
    "hint_money":           6,
    "hint_measure":         6,
    "hint_count":           5,
    "hint_quantity":        5,
    "hint_rate":            7,
    "hint_substance":       4,
}

LABEL_CONFIG = {
    "hint_person_name":    {"base_tau": 0.97, "len_soft_cap": 4, "len_penalty": 0.20, "len_bonus": 0.00},
    "hint_event_nominal":  {"base_tau": 0.97, "len_soft_cap": 2, "len_penalty": 0.00, "len_bonus": 0.05, "floor_tau": 0.55},
    "hint_event_named":    {"base_tau": 0.80, "len_soft_cap": 2, "len_penalty": 0.00, "len_bonus": 0.05, "floor_tau": 0.55},
    "hint_object_generic": {"base_tau": 0.97, "len_soft_cap": 2, "len_penalty": 0.00, "len_bonus": 0.04, "floor_tau": 0.55},
    "hint_quantity":       {"base_tau": 0.80, "len_soft_cap": 4, "len_penalty": 0.05, "len_bonus": 0.00},
    "hint_measure":        {"base_tau": 0.80, "len_soft_cap": 5, "len_penalty": 0.03, "len_bonus": 0.02},
    "hint_percentage":     {"base_tau": 0.75, "len_soft_cap": 3, "len_penalty": 0.05, "len_bonus": 0.00},
    "hint_count":          {"base_tau": 0.80, "len_soft_cap": 4, "len_penalty": 0.05, "len_bonus": 0.00},
    "hint_money":          {"base_tau": 0.75, "len_soft_cap": 5, "len_penalty": 0.03, "len_bonus": 0.02},
    "hint_rate":           {"base_tau": 0.80, "len_soft_cap": 4, "len_penalty": 0.05, "len_bonus": 0.00},
}
DEFAULT_CONFIG = {"base_tau": 0.80, "len_soft_cap": 3, "len_penalty": 0.02, "len_bonus": 0.02}

# Labels SVO → emoji pour affichage
SVO_EMOJI = {
    "svo_verb":    "🔵",
    "svo_subject": "🟢",
    "svo_object":  "🔴",
    "svo_iobj":    "🟠",
    "pron_subj":   "🟢",
    "pron_obj":    "🔴",
}

VOICE_LABELS = ["ACTIVE", "PASSIVE"]


# ──────────────────────────────────────────────────────────
#  Utilitaires géométriques
# ──────────────────────────────────────────────────────────

def tok_len(p: Dict) -> int:
    return int(p["tok_end"]) - int(p["tok_start"]) + 1


def char_len(p: Dict) -> int:
    return int(p["char_end"]) - int(p["char_start"])


def span_iou(a: Dict, b: Dict) -> float:
    inter = max(0, min(a["char_end"], b["char_end"]) - max(a["char_start"], b["char_start"]))
    if inter == 0:
        return 0.0
    union = (a["char_end"] - a["char_start"]) + (b["char_end"] - b["char_start"]) - inter
    return inter / union


def clean_char_span(text: str, s: int, e: int) -> Tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def is_left_word_boundary(text: str, pos: int) -> bool:
    return pos == 0 or not text[pos - 1].isalnum()


def is_right_word_boundary(text: str, pos: int) -> bool:
    return pos == len(text) or not text[pos].isalnum()


# ──────────────────────────────────────────────────────────
#  Seuils dynamiques NER
# ──────────────────────────────────────────────────────────

def adjusted_score(p: Dict) -> float:
    cfg = LABEL_CONFIG.get(p["fine"], DEFAULT_CONFIG)
    L = tok_len(p)
    bonus = cfg.get("len_bonus", 0.0) * math.log1p(L)
    excess = max(0, L - cfg.get("len_soft_cap", 3))
    penalty = cfg.get("len_penalty", 0.0) * (excess ** 2)
    return float(p["score"]) + bonus - penalty


def dynamic_tau(p: Dict) -> float:
    cfg = LABEL_CONFIG.get(p["fine"], DEFAULT_CONFIG)
    base = cfg["base_tau"]
    L = tok_len(p)
    if "floor_tau" in cfg and L >= cfg["len_soft_cap"]:
        return max(cfg["floor_tau"], base - 0.35)
    if p["fine"] == "hint_person_name" and L > cfg["len_soft_cap"]:
        return min(0.98, base + 0.20)
    return base


def pass_dynamic_threshold(p: Dict) -> bool:
    return float(p["fine_prob"]) >= dynamic_tau(p)


# ──────────────────────────────────────────────────────────
#  Post-traitement NER
# ──────────────────────────────────────────────────────────

def post_process_dynamic(preds: List[Dict], iou_threshold: float = 0.60, allow_nested: bool = True) -> List[Dict]:
    preds = [p for p in preds if pass_dynamic_threshold(p)]
    if not preds:
        return preds
    preds = sorted(preds, key=adjusted_score, reverse=True)
    kept = []
    for p in preds:
        drop = False
        for k in kept:
            iou = span_iou(p, k)
            if iou < iou_threshold:
                continue
            if p["fine"] == k["fine"]:
                if adjusted_score(p) <= adjusted_score(k):
                    drop = True
                    break
            else:
                if not allow_nested:
                    if adjusted_score(p) <= adjusted_score(k):
                        drop = True
                        break
                else:
                    nested = (
                        (p["char_start"] >= k["char_start"] and p["char_end"] <= k["char_end"]) or
                        (k["char_start"] >= p["char_start"] and k["char_end"] <= p["char_end"])
                    )
                    if not nested:
                        if p["coarse"] in {"EVENT", "OBJECT", "VALUE"} and k["coarse"] in {"EVENT", "OBJECT", "VALUE"}:
                            if adjusted_score(p) <= adjusted_score(k):
                                drop = True
                                break
        if not drop:
            kept.append(p)
    return kept


def dedupe_overlaps(preds: List[Dict], allow_nested: bool = True) -> List[Dict]:
    seen = set()
    uniq = []
    for p in preds:
        key = (p["char_start"], p["char_end"], p.get("fine"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    if allow_nested:
        return uniq
    kept = []
    for p in uniq:
        if not any(not (p["char_end"] <= q["char_start"] or q["char_end"] <= p["char_start"]) for q in kept):
            kept.append(p)
    return kept


# ──────────────────────────────────────────────────────────
#  Chargement modèle
# ──────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str, checkpoint_path: str, tokenizer_path: str | None, device: str):
    tokenizer_source = tokenizer_path or model_name
    print(f"Chargement tokenizer depuis: {tokenizer_source}")

    tokenizer = None
    last_exc = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        print("✅ AutoTokenizer fast chargé")
    except Exception as e:
        last_exc = e

    if tokenizer is None and tokenizer_path is not None:
        tok_json = Path(tokenizer_path) / "tokenizer.json"
        if tok_json.exists():
            try:
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=str(tok_json),
                    unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]",
                    cls_token="[CLS]", mask_token="[MASK]",
                )
                print("✅ PreTrainedTokenizerFast chargé")
            except Exception as e:
                last_exc = e

    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
            print("✅ AutoTokenizer slow chargé")
        except Exception as e:
            last_exc = e

    if tokenizer is None:
        raise RuntimeError(f"Impossible de charger le tokenizer depuis {tokenizer_source}\n{repr(last_exc)}")

    if getattr(tokenizer, "model_max_length", None) is None or tokenizer.model_max_length > 100000:
        tokenizer.model_max_length = 128

    model = SpanMultiTaskModel(model_name=model_name).to(device).float()
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, tokenizer


# ──────────────────────────────────────────────────────────
#  Construction des spans candidats
# ──────────────────────────────────────────────────────────

def build_candidate_spans(tokenizer, text: str, max_length: int, max_span_len: int,
                          min_char_len: int = 2, enforce_word_boundaries: bool = True):
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True,
                    truncation=True, max_length=max_length)
    input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long)
    attention_mask = torch.tensor([enc["attention_mask"]], dtype=torch.long)
    offsets = enc["offset_mapping"]
    text_token_positions = [i for i, (s, e) in enumerate(offsets) if e > s]
    spans, meta = [], []
    for start_idx, tok_start in enumerate(text_token_positions):
        for tok_end in text_token_positions[start_idx: start_idx + max_span_len]:
            s_char, e_char = clean_char_span(text, offsets[tok_start][0], offsets[tok_end][1])
            if e_char <= s_char:
                continue
            if enforce_word_boundaries:
                if not is_left_word_boundary(text, s_char):
                    continue
                if not is_right_word_boundary(text, e_char):
                    continue
            span_text = text[s_char:e_char]
            if len(span_text.strip()) < min_char_len or all(not ch.isalnum() for ch in span_text):
                continue
            spans.append({"tok_start": tok_start, "tok_end": tok_end})
            meta.append({"tok_start": tok_start, "tok_end": tok_end,
                          "char_start": s_char, "char_end": e_char, "text": span_text})
    return input_ids, attention_mask, [spans], meta


def build_batch_candidate_spans(tokenizer, texts: List[str], max_length: int, max_span_len: int,
                                 min_char_len: int = 2, enforce_word_boundaries: bool = True):
    enc = tokenizer(texts, return_offsets_mapping=True, add_special_tokens=True,
                    truncation=True, padding=True, max_length=max_length)
    input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
    offsets_batch = enc["offset_mapping"]
    spans_per_example, meta_flat = [], []
    for ex_idx, (text, offsets) in enumerate(zip(texts, offsets_batch)):
        text_token_positions = [i for i, (s, e) in enumerate(offsets) if e > s]
        spans, metas = [], []
        for start_idx, tok_start in enumerate(text_token_positions):
            for tok_end in text_token_positions[start_idx: start_idx + max_span_len]:
                s_char, e_char = clean_char_span(text, offsets[tok_start][0], offsets[tok_end][1])
                if e_char <= s_char:
                    continue
                if enforce_word_boundaries:
                    if not (s_char == 0 or not text[s_char - 1].isalnum()):
                        continue
                    if not (e_char == len(text) or not text[e_char].isalnum()):
                        continue
                span_text = text[s_char:e_char]
                if len(span_text.strip()) < min_char_len or all(not ch.isalnum() for ch in span_text):
                    continue
                spans.append({"tok_start": tok_start, "tok_end": tok_end})
                metas.append({"tok_start": tok_start, "tok_end": tok_end,
                               "char_start": s_char, "char_end": e_char,
                               "text": span_text, "example_idx": ex_idx})
        spans_per_example.append(spans)
        meta_flat.extend(metas)
    return input_ids, attention_mask, spans_per_example, meta_flat


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


# ──────────────────────────────────────────────────────────
#  Prédiction (single + batch) — NER + SVO + voice
# ──────────────────────────────────────────────────────────

def _decode_spans(outputs, meta_flat, model, device,
                  tau_boundary, tau_none, tau_coarse, tau_fine, topk_coarse,
                  tau_svo_boundary,
                  per_example_ner, per_example_svo):
    """
    Décode les logits pour tous les spans et remplit per_example_ner / per_example_svo.
    """
    b_probs  = softmax_probs(outputs["boundary_logits"])
    c_probs  = softmax_probs(outputs["coarse_logits"])
    f_logits = outputs["fine_logits"]
    svob_probs = softmax_probs(outputs["svo_boundary_logits"])
    svo_probs  = softmax_probs(outputs["svo_logits"])
    voice_probs = softmax_probs(outputs["voice_logits"])

    coarse_fine_mask = model.coarse_fine_mask.to(device)
    none_idx = COARSE2ID["NONE"]

    for i, m in enumerate(meta_flat):
        ex_idx = m.get("example_idx", 0)

        # ── NER ──────────────────────────────────────────────────
        p_ent = float(b_probs[i, 1].item())
        if p_ent >= tau_boundary:
            coarse_row = c_probs[i]
            p_none = float(coarse_row[none_idx].item())
            top_vals, top_idxs = torch.topk(coarse_row, k=min(topk_coarse, coarse_row.numel()))
            chosen = None
            chosen_score = -1.0
            for coarse_prob, coarse_idx_t in zip(top_vals.tolist(), top_idxs.tolist()):
                coarse_idx = int(coarse_idx_t)
                if coarse_idx == none_idx or p_none >= tau_none or coarse_prob < tau_coarse:
                    continue
                allowed = coarse_fine_mask[coarse_idx]
                if not allowed.any():
                    continue
                masked = f_logits[i].clone().masked_fill(~allowed, -1e9)
                fine_probs = softmax_probs(masked.unsqueeze(0))[0]
                fine_idx = int(torch.argmax(fine_probs).item())
                fine_prob = float(fine_probs[fine_idx].item())
                if fine_prob < tau_fine:
                    continue
                score = p_ent * coarse_prob * fine_prob
                if score > chosen_score:
                    chosen_score = score
                    chosen = {
                        "text": m["text"], "char_start": m["char_start"], "char_end": m["char_end"],
                        "tok_start": m["tok_start"], "tok_end": m["tok_end"],
                        "boundary_prob": round(p_ent, 4),
                        "coarse": COARSE_LABELS[coarse_idx], "coarse_prob": round(float(coarse_prob), 4),
                        "fine": FINE_LABELS[fine_idx], "fine_prob": round(fine_prob, 4),
                        "score": round(score, 4),
                        "example_idx": ex_idx,
                    }
            if chosen is not None:
                tk = chosen["tok_end"] - chosen["tok_start"] + 1
                max_tok = MAX_TOK_LEN_PER_FINE.get(chosen["fine"])
                if max_tok is None or tk <= max_tok:
                    per_example_ner[ex_idx].append(chosen)

        # ── SVO boundary (verbe / pronom) ─────────────────────────
        p_svob = float(svob_probs[i, 1].item())
        if p_svob >= tau_svo_boundary:
            # rôle SVO
            svo_idx = int(torch.argmax(svo_probs[i]).item())
            svo_label = SVO_LABELS[svo_idx] if svo_idx < len(SVO_LABELS) else "?"
            svo_prob = float(svo_probs[i, svo_idx].item())

            # voice (pertinent surtout pour svo_verb)
            voice_idx = int(torch.argmax(voice_probs[i]).item())
            voice_label = VOICE_LABELS[voice_idx] if voice_idx < NUM_VOICE else "?"
            voice_prob = float(voice_probs[i, voice_idx].item())

            per_example_svo[ex_idx].append({
                "text": m["text"], "char_start": m["char_start"], "char_end": m["char_end"],
                "tok_start": m["tok_start"], "tok_end": m["tok_end"],
                "svo_boundary_prob": round(p_svob, 4),
                "svo_role": svo_label, "svo_prob": round(svo_prob, 4),
                "voice": voice_label, "voice_prob": round(voice_prob, 4),
                "example_idx": ex_idx,
            })


def predict_text(model, tokenizer, text: str, device: str,
                 max_length: int = 128, max_span_len: int = 8,
                 tau_boundary: float = 0.70, tau_none: float = 0.50,
                 tau_coarse: float = 0.45, tau_fine: float = 0.00,
                 topk_coarse: int = 2, min_char_len: int = 2,
                 enforce_word_boundaries: bool = True,
                 tau_svo_boundary: float = 0.50):
    input_ids, attention_mask, spans, meta = build_candidate_spans(
        tokenizer, text, max_length=max_length, max_span_len=max_span_len,
        min_char_len=min_char_len, enforce_word_boundaries=enforce_word_boundaries)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        outputs = model({"input_ids": input_ids, "attention_mask": attention_mask, "spans": spans})

    per_example_ner = [[]]
    per_example_svo = [[]]
    # ajouter example_idx=0 dans meta
    for m in meta:
        m["example_idx"] = 0
    _decode_spans(outputs, meta, model, device,
                  tau_boundary, tau_none, tau_coarse, tau_fine, topk_coarse,
                  tau_svo_boundary, per_example_ner, per_example_svo)

    ner = sorted(per_example_ner[0], key=lambda x: (x["score"], x["char_end"] - x["char_start"]), reverse=True)
    svo = sorted(per_example_svo[0], key=lambda x: x["char_start"])
    return ner, svo


def predict_texts_batch(model, tokenizer, texts: List[str], device: str,
                        max_length: int = 128, max_span_len: int = 8,
                        tau_boundary: float = 0.70, tau_none: float = 0.50,
                        tau_coarse: float = 0.45, tau_fine: float = 0.00,
                        topk_coarse: int = 4, min_char_len: int = 2,
                        enforce_word_boundaries: bool = True,
                        tau_svo_boundary: float = 0.50):
    input_ids, attention_mask, spans_per_example, meta_flat = build_batch_candidate_spans(
        tokenizer, texts, max_length, max_span_len, min_char_len, enforce_word_boundaries)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        outputs = model({"input_ids": input_ids, "attention_mask": attention_mask, "spans": spans_per_example})

    n = len(texts)
    per_example_ner = [[] for _ in range(n)]
    per_example_svo = [[] for _ in range(n)]
    _decode_spans(outputs, meta_flat, model, device,
                  tau_boundary, tau_none, tau_coarse, tau_fine, topk_coarse,
                  tau_svo_boundary, per_example_ner, per_example_svo)

    results = []
    for i in range(n):
        ner = sorted(per_example_ner[i], key=lambda x: (x["score"], x["char_end"] - x["char_start"]), reverse=True)
        svo = sorted(per_example_svo[i], key=lambda x: x["char_start"])
        results.append({"ner": ner, "svo": svo})
    return results


# ──────────────────────────────────────────────────────────
#  Reconstruction des triplets SVO
# ──────────────────────────────────────────────────────────

def build_svo_triplets(svo_spans: List[Dict]) -> List[Dict]:
    """
    Regroupe les spans SVO en triplets (S, V, O) de façon greedy :
    - un verbe ancre un triplet
    - on lui associe le sujet le plus proche (à gauche) et l'objet le plus proche (à droite)
    Heuristique simple, suffisante pour la visualisation.
    """
    verbs    = [s for s in svo_spans if s["svo_role"] in ("svo_verb",)]
    subjects = [s for s in svo_spans if s["svo_role"] in ("svo_subject", "pron_subj")]
    objects  = [s for s in svo_spans if s["svo_role"] in ("svo_object", "svo_iobj", "pron_obj")]

    triplets = []
    for v in verbs:
        # sujet = dernier sujet dont char_end <= verb.char_start
        subj_candidates = [s for s in subjects if s["char_end"] <= v["char_start"]]
        subj = max(subj_candidates, key=lambda x: x["char_start"], default=None)

        # objet = premier objet dont char_start >= verb.char_end
        obj_candidates = [o for o in objects if o["char_start"] >= v["char_end"]]
        obj = min(obj_candidates, key=lambda x: x["char_start"], default=None)

        triplets.append({
            "verb": v,
            "subject": subj,
            "object": obj,
            "voice": v.get("voice"),
            "voice_prob": v.get("voice_prob"),
        })

    return triplets


# ──────────────────────────────────────────────────────────
#  Affichage
# ──────────────────────────────────────────────────────────

def print_ner(preds: List[Dict]):
    if not preds:
        print("  (aucune entité détectée)")
        return
    for p in preds:
        print(
            f"  [{p['char_start']:>4}:{p['char_end']:<4}] "
            f"{p['text']!r:<35} | coarse={p['coarse']:<8} ({p['coarse_prob']:.3f}) "
            f"| fine={p['fine']:<22} ({p['fine_prob']:.3f}) "
            f"| p_ent={p['boundary_prob']:.3f} | score={p['score']:.4f} "
            f"| tok=[{p.get('tok_start')},{p.get('tok_end')}]"
        )


def print_svo(svo_spans: List[Dict]):
    if not svo_spans:
        print("  (aucun span verbe/pronom détecté)")
        return
    for s in svo_spans:
        role = s["svo_role"]
        emoji = SVO_EMOJI.get(role, "⚪")
        voice_info = ""
        if role == "svo_verb":
            voice_info = f" | voice={s['voice']} ({s['voice_prob']:.3f})"
        print(
            f"  {emoji} [{s['char_start']:>4}:{s['char_end']:<4}] "
            f"{s['text']!r:<30} | role={role:<12} ({s['svo_prob']:.3f})"
            f" | p_svob={s['svo_boundary_prob']:.3f}{voice_info}"
            f" | tok=[{s.get('tok_start')},{s.get('tok_end')}]"
        )


def print_triplets(triplets: List[Dict]):
    if not triplets:
        print("  (aucun triplet reconstruit)")
        return
    for t in triplets:
        v = t["verb"]
        s = t["subject"]
        o = t["object"]
        voice = t.get("voice", "?")
        s_text = f'"{s["text"]}"' if s else "∅"
        o_text = f'"{o["text"]}"' if o else "∅"
        print(f"  {s_text:>30}  →[{voice}]→  \"{v['text']}\"  →  {o_text}")


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test v3 : NER rich + SVO + voice")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--checkpoint", default="checkpoint_best_multitask.pt")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-span-len", type=int, default=12)
    parser.add_argument("--tau-boundary", type=float, default=0.50)
    parser.add_argument("--tau-none", type=float, default=0.99)
    parser.add_argument("--tau-coarse", type=float, default=0.00)
    parser.add_argument("--tau-fine", type=float, default=0.00)
    parser.add_argument("--tau-svo-boundary", type=float, default=0.50,
                        help="Seuil pour détecter un span verbe/pronom (svo_boundary_head)")
    parser.add_argument("--topk-coarse", type=int, default=2)
    parser.add_argument("--min-char-len", type=int, default=2)
    parser.add_argument("--allow-midword", action="store_true")
    parser.add_argument("--no-nested", action="store_true")
    parser.add_argument("--show-svo", action="store_true", default=True,
                        help="Afficher les spans SVO/voice (activé par défaut)")
    parser.add_argument("--no-svo", action="store_true",
                        help="Masquer les spans SVO")
    parser.add_argument("--show-triplets", action="store_true", default=True,
                        help="Afficher les triplets (S,V,O) reconstruits")
    parser.add_argument("--no-triplets", action="store_true")
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--input-text", default=None,
                        help="Texte passé directement en argument (alternative à --text)")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--post-process", action="store_true",
                        help="Appliquer post_process_dynamic sur les entités NER")

    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"✅ device = {device}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        device=device,
    )

    texts = list(args.text)
    if args.input_text:
        texts.append(args.input_text)
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            texts.extend([line.rstrip("\n") for line in f if line.strip()])

    if not texts:
        raise ValueError("Aucun texte fourni. Utilise --text ou --input-file.")

    show_svo      = args.show_svo and not args.no_svo
    show_triplets = args.show_triplets and not args.no_triplets

    import time
    total_time = 0.0
    all_outputs = []

    common_kwargs = dict(
        max_length=args.max_length,
        max_span_len=args.max_span_len,
        tau_boundary=args.tau_boundary,
        tau_none=args.tau_none,
        tau_coarse=args.tau_coarse,
        tau_fine=args.tau_fine,
        topk_coarse=args.topk_coarse,
        min_char_len=args.min_char_len,
        enforce_word_boundaries=(not args.allow_midword),
        tau_svo_boundary=args.tau_svo_boundary,
    )

    use_batch = args.batch_size and args.batch_size > 1 and len(texts) > 1

    if use_batch:
        for batch_start in range(0, len(texts), args.batch_size):
            batch_texts = texts[batch_start: batch_start + args.batch_size]
            t0 = time.perf_counter()
            batch_results = predict_texts_batch(model=model, tokenizer=tokenizer,
                                                 texts=batch_texts, device=device, **common_kwargs)
            total_time += time.perf_counter() - t0

            for i, res in enumerate(batch_results):
                idx = batch_start + i + 1
                text = batch_texts[i]
                ner = res["ner"]
                svo = res["svo"]

                if args.post_process:
                    ner = post_process_dynamic(ner, allow_nested=(not args.no_nested))
                else:
                    ner = dedupe_overlaps(ner, allow_nested=(not args.no_nested))

                print("\n" + "=" * 100)
                print(f"📝 TEXTE #{idx}")
                print(text)

                print("\n🏷️  NER")
                print_ner(ner)

                if show_svo:
                    print("\n🔗 SVO / Verbes / Pronoms")
                    print_svo(svo)

                if show_triplets:
                    triplets = build_svo_triplets(svo)
                    print("\n🔺 Triplets (S, V, O)")
                    print_triplets(triplets)

                all_outputs.append({"text": text, "ner": ner, "svo": svo})
    else:
        for idx, text in enumerate(texts, start=1):
            t0 = time.perf_counter()
            ner, svo = predict_text(model=model, tokenizer=tokenizer, text=text, device=device, **common_kwargs)
            total_time += time.perf_counter() - t0

            if args.post_process:
                ner = post_process_dynamic(ner, allow_nested=(not args.no_nested))
            else:
                ner = dedupe_overlaps(ner, allow_nested=(not args.no_nested))

            print("\n" + "=" * 100)
            print(f"📝 TEXTE #{idx}")
            print(text)

            print("\n🏷️  NER")
            print_ner(ner)

            if show_svo:
                print("\n🔗 SVO / Verbes / Pronoms")
                print_svo(svo)

            if show_triplets:
                triplets = build_svo_triplets(svo)
                print("\n🔺 Triplets (S, V, O)")
                print_triplets(triplets)

            all_outputs.append({"text": text, "ner": ner, "svo": svo})

    print("\n" + "#" * 60)
    print(f"⏱  Temps total inference : {total_time:.3f}s pour {len(texts)} phrases (batch_size={args.batch_size})")
    print("#" * 60)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Résultats JSON écrits dans {args.json_out}")


if __name__ == "__main__":
    main()

