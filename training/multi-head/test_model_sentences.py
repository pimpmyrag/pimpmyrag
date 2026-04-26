import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from multitask_model import SpanMultiTaskModel
from labels import COARSE_LABELS, FINE_LABELS, COARSE2ID


def pick_device(forced: str | None = None) -> str:
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"



FINE_THRESHOLDS = {
    # Basiques : permissifs (recall first)
    "hint_person_name":    0.90,
    "hint_org_name":       0.90,
    "hint_gpe":            0.70,
    "hint_fac_name":       0.70,
    "hint_time_date":      0.70,
    "hint_time_clock":     0.70,

    # Rôles : un peu plus strict
    "hint_person_role":    0.95,
    "hint_group_role":     0.95,

    # Dangereux (beaucoup de bruit)
    "hint_event_nominal":  0.97,
    "hint_object_generic": 0.97,
}

DEFAULT_FINE_THRESHOLD = 0.80

LONG_SPAN_TYPES = {
    "hint_org_name",
    "hint_gpe",
    "hint_fac_name",
    "hint_time_date",
    "hint_time_clock",
}

SHORT_SPAN_TYPES = {
    "hint_person_role",
    "hint_group_role",
}

def len_span(s):
    return s.end_char - s.start_char

def is_bad_nominal(span, text):
    if span.fine not in {"hint_event_nominal", "hint_object_generic"}:
        return False

    span_text = text[span.start_char:span.end_char]

    # 1 token, minuscules, pas de complément
    if span_text.islower() and " " not in span_text:
        return True

    return False

def post_process(spans, text):
    # 1. seuils par label
    spans = [s for s in spans if pass_fine_threshold(s)]

    # 2. NMS type-aware
    spans = nms_spans(spans)

    # 3. filtre nominaux isolés
    spans = [s for s in spans if not is_bad_nominal(s, text)]

    return spans


def nms_spans(spans, iou_threshold=0.6):
    """
    spans : list[Span] déjà filtrés par seuil
    """
    # Trier par score décroissant
    spans = sorted(spans, key=lambda s: s.score, reverse=True)

    kept = []

    for s in spans:
        discard = False

        for k in kept:
            iou = span_iou(s, k)
            if iou < iou_threshold:
                continue

            # --- même label fin ---
            if s.fine == k.fine:
                # Types longs → garder le plus long
                if s.fine in LONG_SPAN_TYPES:
                    discard = (len_span(s) <= len_span(k))
                # Types courts → garder le plus court
                elif s.fine in SHORT_SPAN_TYPES:
                    discard = (len_span(s) >= len_span(k))
                # Autres → garder meilleur score
                else:
                    discard = (s.score <= k.score)

            # --- types différents mais même coarse ---
            elif s.coarse == k.coarse:
                # EVENT / OBJECT : on est conservateur
                if s.coarse in {"EVENT", "OBJECT"}:
                    discard = True
                else:
                    # PER/LOC/ORG/TIME : garder les deux si nested utile
                    discard = False

            if discard:
                break

        if not discard:
            kept.append(s)

    return kept

def pass_fine_threshold(span):
    tau = FINE_THRESHOLDS.get(span.fine, DEFAULT_FINE_THRESHOLD)
    return span.p_fine >= tau


def span_iou(a, b):
    inter = max(0, min(a.end_char, b.end_char) - max(a.start_char, b.start_char))
    if inter == 0:
        return 0.0
    union = (a.end_char - a.start_char) + (b.end_char - b.start_char) - inter
    return inter / union

def load_model_and_tokenizer(model_name: str, checkpoint_path: str, tokenizer_path: str | None, device: str):
    """
    Chargement robuste du tokenizer :
      1) AutoTokenizer fast
      2) si tokenizer_path local contient tokenizer.json, chargement direct via PreTrainedTokenizerFast
      3) fallback AutoTokenizer slow
    """
    tokenizer_source = tokenizer_path or model_name
    print(f"Chargement tokenizer depuis: {tokenizer_source}")

    tokenizer = None
    last_exc = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        print("✅ AutoTokenizer fast chargé")
    except Exception as e:
        last_exc = e
        print(f"⚠️ AutoTokenizer fast a échoué: {repr(e)}")

    if tokenizer is None and tokenizer_path is not None:
        tok_json = Path(tokenizer_path) / "tokenizer.json"
        if tok_json.exists():
            try:
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=str(tok_json),
                    unk_token="[UNK]",
                    sep_token="[SEP]",
                    pad_token="[PAD]",
                    cls_token="[CLS]",
                    mask_token="[MASK]",
                )
                print("✅ PreTrainedTokenizerFast chargé directement depuis tokenizer.json")
            except Exception as e:
                last_exc = e
                print(f"⚠️ Chargement direct tokenizer.json a échoué: {repr(e)}")

    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
            print("✅ AutoTokenizer slow chargé")
        except Exception as e:
            last_exc = e

    if tokenizer is None:
        raise RuntimeError(
            f"Impossible de charger le tokenizer depuis {tokenizer_source}\n"
            f"Dernière erreur: {repr(last_exc)}"
        )

    # Sécurité : certains tokenizers ont model_max_length = 1e30
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


def is_left_word_boundary(text: str, pos: int) -> bool:
    return pos == 0 or not text[pos - 1].isalnum()


def is_right_word_boundary(text: str, pos: int) -> bool:
    return pos == len(text) or not text[pos].isalnum()


def clean_char_span(text: str, s_char: int, e_char: int) -> Tuple[int, int]:
    # Trim espaces de tête / fin
    while s_char < e_char and text[s_char].isspace():
        s_char += 1
    while e_char > s_char and text[e_char - 1].isspace():
        e_char -= 1
    return s_char, e_char


def build_candidate_spans(
    tokenizer,
    text: str,
    max_length: int,
    max_span_len: int,
    min_char_len: int = 2,
    enforce_word_boundaries: bool = True,
):
    """
    Génère tous les spans candidats jusqu'à max_span_len tokens.
    Empêche la coupe au milieu des mots si enforce_word_boundaries=True.

    Retourne:
      - input_ids torch [1, L]
      - attention_mask torch [1, L]
      - spans list[list[{tok_start,tok_end}]]
      - meta list[dict] aligné sur l'ordre des spans
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )

    input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long)
    attention_mask = torch.tensor([enc["attention_mask"]], dtype=torch.long)
    offsets = enc["offset_mapping"]

    # On ne garde que les tokens texte réels (pas [CLS]/[SEP] = (0,0))
    text_token_positions = []
    for i, (s, e) in enumerate(offsets):
        if e > s:
            text_token_positions.append(i)

    spans = []
    meta = []

    for start_idx_in_list, tok_start in enumerate(text_token_positions):
        for tok_end in text_token_positions[start_idx_in_list : start_idx_in_list + max_span_len]:
            s_char = offsets[tok_start][0]
            e_char = offsets[tok_end][1]

            s_char, e_char = clean_char_span(text, s_char, e_char)
            if e_char <= s_char:
                continue

            if enforce_word_boundaries:
                if not is_left_word_boundary(text, s_char):
                    continue
                if not is_right_word_boundary(text, e_char):
                    continue

            span_text = text[s_char:e_char]
            if len(span_text.strip()) < min_char_len:
                continue
            if all(not ch.isalnum() for ch in span_text):
                continue

            spans.append({"tok_start": tok_start, "tok_end": tok_end})
            meta.append(
                {
                    "tok_start": tok_start,
                    "tok_end": tok_end,
                    "char_start": s_char,
                    "char_end": e_char,
                    "text": span_text,
                }
            )

    return input_ids, attention_mask, [spans], meta


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def build_batch_candidate_spans(
    tokenizer,
    texts: List[str],
    max_length: int,
    max_span_len: int,
    min_char_len: int = 2,
    enforce_word_boundaries: bool = True,
):
    """
    Construire les tensors batchés et listes de spans/metas pour une liste de textes.
    Retourne (input_ids, attention_mask, spans_per_example, meta_flat)
    meta_flat contient des dicts avec 'char_start','char_end','text','example_idx','tok_start','tok_end'.
    """
    enc = tokenizer(
        texts,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
    offsets_batch = enc["offset_mapping"]

    spans_per_example = []
    meta_flat: List[Dict[str, Any]] = []

    for ex_idx, (text, offsets) in enumerate(zip(texts, offsets_batch)):
        # repérer les tokens qui correspondent au texte (offset e>s)
        text_token_positions = [i for i, (s, e) in enumerate(offsets) if e > s]

        spans = []
        metas = []
        for start_idx_in_list, tok_start in enumerate(text_token_positions):
            for tok_end in text_token_positions[start_idx_in_list : start_idx_in_list + max_span_len]:
                s_char = offsets[tok_start][0]
                e_char = offsets[tok_end][1]

                s_char, e_char = clean_char_span(text, s_char, e_char)
                if e_char <= s_char:
                    continue

                if enforce_word_boundaries:
                    if not (s_char == 0 or not text[s_char - 1].isalnum()):
                        continue
                    if not (e_char == len(text) or not text[e_char].isalnum()):
                        continue

                span_text = text[s_char:e_char]
                if len(span_text.strip()) < min_char_len:
                    continue
                if all(not ch.isalnum() for ch in span_text):
                    continue

                spans.append({"tok_start": tok_start, "tok_end": tok_end})
                metas.append(
                    {
                        "tok_start": tok_start,
                        "tok_end": tok_end,
                        "char_start": s_char,
                        "char_end": e_char,
                        "text": span_text,
                        "example_idx": ex_idx,
                    }
                )

        spans_per_example.append(spans)
        meta_flat.extend(metas)

    return input_ids, attention_mask, spans_per_example, meta_flat


def predict_texts_batch(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    max_length: int = 128,
    max_span_len: int = 8,
    tau_boundary: float = 0.70,
    tau_none: float = 0.50,
    tau_coarse: float = 0.45,
    tau_fine: float = 0.00,
    topk_coarse: int = 4,
    min_char_len: int = 2,
    enforce_word_boundaries: bool = True,
):
    """Prédit pour un lot de textes en une seule passe modèle. Retourne une liste de listes de prédictions par texte."""
    input_ids, attention_mask, spans_per_example, meta_flat = build_batch_candidate_spans(
        tokenizer, texts, max_length, max_span_len, min_char_len, enforce_word_boundaries
    )

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "spans": spans_per_example,
            }
        )

    b_logits = outputs["boundary_logits"]
    c_logits = outputs["coarse_logits"]
    f_logits = outputs["fine_logits"]

    b_probs = softmax_probs(b_logits)
    c_probs = softmax_probs(c_logits)
    coarse_fine_mask = model.coarse_fine_mask.to(device)
    none_idx = COARSE2ID["NONE"]

    # résultats par exemple
    per_example_results: List[List[Dict[str, Any]]] = [[] for _ in texts]

    # itérer sur chaque span (meta_flat ordonné) et assigner à l'exemple correspondant
    for i, m in enumerate(meta_flat):
        p_ent = float(b_probs[i, 1].item())
        if p_ent < tau_boundary:
            continue

        coarse_row = c_probs[i]
        p_none = float(coarse_row[none_idx].item())
        top_vals, top_idxs = torch.topk(coarse_row, k=min(topk_coarse, coarse_row.numel()))

        chosen = None
        chosen_score = -1.0

        for coarse_prob, coarse_idx_t in zip(top_vals.tolist(), top_idxs.tolist()):
            coarse_idx = int(coarse_idx_t)
            if coarse_idx == none_idx:
                continue
            if p_none >= tau_none:
                continue
            if coarse_prob < tau_coarse:
                continue

            allowed = coarse_fine_mask[coarse_idx]
            if not allowed.any():
                continue

            masked = f_logits[i].clone()
            masked = masked.masked_fill(~allowed, -1e9)
            fine_probs = softmax_probs(masked.unsqueeze(0))[0]
            fine_idx = int(torch.argmax(fine_probs).item())
            fine_prob = float(fine_probs[fine_idx].item())

            if fine_prob < tau_fine:
                continue

            score = p_ent * coarse_prob * fine_prob
            if score > chosen_score:
                chosen_score = score
                chosen = {
                    "text": m["text"],
                    "char_start": m["char_start"],
                    "char_end": m["char_end"],
                    "tok_start": m["tok_start"],
                    "tok_end": m["tok_end"],
                    "boundary_prob": round(p_ent, 4),
                    "coarse": COARSE_LABELS[coarse_idx],
                    "coarse_prob": round(float(coarse_prob), 4),
                    "fine": FINE_LABELS[fine_idx],
                    "fine_prob": round(fine_prob, 4),
                    "score": round(score, 4),
                    "example_idx": m["example_idx"],
                }

        if chosen is not None:
            per_example_results[chosen["example_idx"]].append(chosen)

    # trier chaque liste par score décroissant puis longueur
    for lst in per_example_results:
        lst.sort(key=lambda x: (x["score"], x["char_end"] - x["char_start"]), reverse=True)

    return per_example_results


def predict_text(
    model,
    tokenizer,
    text: str,
    device: str,
    max_length: int = 128,
    max_span_len: int = 8,
    tau_boundary: float = 0.70,
    tau_none: float = 0.50,
    tau_coarse: float = 0.45,
    tau_fine: float = 0.00,
    topk_coarse: int = 2,
    min_char_len: int = 2,
    enforce_word_boundaries: bool = True,
):
    input_ids, attention_mask, spans, meta = build_candidate_spans(
        tokenizer,
        text,
        max_length=max_length,
        max_span_len=max_span_len,
        min_char_len=min_char_len,
        enforce_word_boundaries=enforce_word_boundaries,
    )

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "spans": spans,
            }
        )

    b_logits = outputs["boundary_logits"]
    c_logits = outputs["coarse_logits"]
    f_logits = outputs["fine_logits"]

    b_probs = softmax_probs(b_logits)
    c_probs = softmax_probs(c_logits)
    coarse_fine_mask = model.coarse_fine_mask.to(device)
    none_idx = COARSE2ID["NONE"]

    results = []

    for i, m in enumerate(meta):
        p_ent = float(b_probs[i, 1].item())
        if p_ent < tau_boundary:
            continue

        coarse_row = c_probs[i]
        p_none = float(coarse_row[none_idx].item())
        top_vals, top_idxs = torch.topk(coarse_row, k=min(topk_coarse, coarse_row.numel()))

        chosen = None
        chosen_score = -1.0

        for coarse_prob, coarse_idx_t in zip(top_vals.tolist(), top_idxs.tolist()):
            coarse_idx = int(coarse_idx_t)
            if coarse_idx == none_idx:
                continue
            if p_none >= tau_none:
                continue
            if coarse_prob < tau_coarse:
                continue

            allowed = coarse_fine_mask[coarse_idx]
            if not allowed.any():
                continue

            masked = f_logits[i].clone()
            masked = masked.masked_fill(~allowed, -1e9)
            fine_probs = softmax_probs(masked.unsqueeze(0))[0]
            fine_idx = int(torch.argmax(fine_probs).item())
            fine_prob = float(fine_probs[fine_idx].item())

            if fine_prob < tau_fine:
                continue

            score = p_ent * coarse_prob * fine_prob
            if score > chosen_score:
                chosen_score = score
                chosen = {
                    "text": m["text"],
                    "char_start": m["char_start"],
                    "char_end": m["char_end"],
                    "tok_start": m["tok_start"],
                    "tok_end": m["tok_end"],
                    "boundary_prob": round(p_ent, 4),
                    "coarse": COARSE_LABELS[coarse_idx],
                    "coarse_prob": round(float(coarse_prob), 4),
                    "fine": FINE_LABELS[fine_idx],
                    "fine_prob": round(fine_prob, 4),
                    "score": round(score, 4),
                }

        if chosen is not None:
            results.append(chosen)

    # Tri : score décroissant puis span plus long d'abord
    results.sort(key=lambda x: (x["score"], x["char_end"] - x["char_start"]), reverse=True)
    return results


def dedupe_overlaps(preds: List[Dict[str, Any]], allow_nested: bool = True) -> List[Dict[str, Any]]:
    """
    Déduplication simple :
      - supprime les doublons exacts (même start/end/fine)
      - si allow_nested=False, garde le meilleur score sur les overlaps
    """
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
        overlap = False
        for q in kept:
            if not (p["char_end"] <= q["char_start"] or q["char_end"] <= p["char_start"]):
                overlap = True
                break
        if not overlap:
            kept.append(p)
    return kept


def prefer_longest_by_confidence(
    preds: List[Dict[str, Any]],
    margin: float = 0.05,
    min_boundary: float = 0.0,
    group_by: str = "fine",
    margins_tiers: Optional[List[tuple[int, float]]] = None,
    trust_fine_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Remplace les spans imbriqués du même type par le span englobant le plus long si
    celui-ci a un score supérieur aux spans contenus par au moins `margin`.

    - preds: liste de prédictions (doit contenir 'char_start','char_end','score','boundary_prob','fine','coarse')
    - margin: différence minimale entre score_long et max_score_subspans pour préférer le long
    - min_boundary: score minimal de boundary pour accepter un span englobant
    - group_by: "fine" ou "coarse" — regrouper par label fin ou coarse
    Retourne une nouvelle liste de prédictions filtrée.
    """
    if group_by not in ("fine", "coarse"):
        group_by = "fine"

    def margin_for_length(length: int) -> float:
        if margins_tiers:
            for max_len, m in margins_tiers:
                if length <= max_len:
                    return m
        return margin

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for p in preds:
        key = p.get(group_by)
        groups.setdefault(key, []).append(p)

    kept_global = []

    for _, group in groups.items():
        group_sorted = sorted(group, key=lambda x: ((x["char_end"] - x["char_start"]), x["score"]), reverse=True)
        removed = set()

        for long_span in group_sorted:
            if id(long_span) in removed:
                continue
            sL = long_span["char_start"]
            eL = long_span["char_end"]
            if long_span.get("boundary_prob", 0.0) < min_boundary:
                continue

            contained = [
                p
                for p in group_sorted
                if p is not long_span and p["char_start"] >= sL and p["char_end"] <= eL
            ]
            if not contained:
                continue

            max_sub_score = max((p["score"] for p in contained), default=0.0)

            if trust_fine_threshold is not None:
                fine_prob_long = float(long_span.get("fine_prob", 0.0))
                if fine_prob_long >= trust_fine_threshold:
                    for p in contained:
                        removed.add(id(p))
                    continue

            len_long = eL - sL
            effective_margin = margin_for_length(len_long)

            if long_span["score"] >= max_sub_score + effective_margin:
                for p in contained:
                    removed.add(id(p))
            else:
                removed.add(id(long_span))

        for p in group_sorted:
            if id(p) not in removed:
                kept_global.append(p)

    kept_global = sorted(
        kept_global,
        key=lambda x: (x["score"], -(x["char_end"] - x["char_start"])),
        reverse=True,
    )
    return kept_global


def _can_merge_by_char(text: str, a: Dict[str, Any], b: Dict[str, Any], max_gap_chars: int = 1) -> bool:
    """Autorise le merge si overlap ou quasi-adjacent (gap <= max_gap_chars) et sans mots dans le gap."""
    # a avant b
    if a["char_end"] <= b["char_start"]:
        gap = text[a["char_end"] : b["char_start"]]
        if (b["char_start"] - a["char_end"]) > max_gap_chars:
            return False
        return not any(ch.isalnum() for ch in gap)

    # b avant a
    if b["char_end"] <= a["char_start"]:
        gap = text[b["char_end"] : a["char_start"]]
        if (a["char_start"] - b["char_end"]) > max_gap_chars:
            return False
        return not any(ch.isalnum() for ch in gap)

    # overlap
    return True


def merge_longest_spans_same_candidate(
    preds: List[Dict[str, Any]],
    text: str,
    max_tokens: int = 12,
    group_by: str = "fine",
    max_gap_chars: int = 1,
    agg: str = "max",
) -> List[Dict[str, Any]]:
    """
    Merge greedy des spans du même candidat (fine/coarse) en partant des plus longs,
    sans dépasser max_tokens (calculé via tok_start/tok_end).

    - group_by: "fine" ou "coarse"
    - max_gap_chars: autorise un merge si les spans sont séparés par <= N chars (souvent 0 ou 1)
    - agg: "max" ou "mean" pour agréger score/probas
    """
    if group_by not in ("fine", "coarse"):
        group_by = "fine"

    if not preds:
        return preds

    # Si pas de tok_start/tok_end, on ne merge pas
    if any(("tok_start" not in p or "tok_end" not in p) for p in preds):
        return preds

    def tok_len(p: Dict[str, Any]) -> int:
        return int(p["tok_end"]) - int(p["tok_start"]) + 1

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for p in preds:
        groups.setdefault(p.get(group_by), []).append(p)

    merged_all: List[Dict[str, Any]] = []

    for label, spans in groups.items():
        spans_sorted = sorted(spans, key=lambda x: (tok_len(x), x["score"]), reverse=True)
        used = [False] * len(spans_sorted)

        for i in range(len(spans_sorted)):
            if used[i]:
                continue

            cur = dict(spans_sorted[i])
            changed = True

            while changed:
                changed = False
                for j in range(len(spans_sorted)):
                    if used[j] or j == i:
                        continue

                    other = spans_sorted[j]
                    if other.get(group_by) != label:
                        continue

                    if not _can_merge_by_char(text, cur, other, max_gap_chars=max_gap_chars):
                        continue

                    merged_tok_start = min(int(cur["tok_start"]), int(other["tok_start"]))
                    merged_tok_end = max(int(cur["tok_end"]), int(other["tok_end"]))
                    merged_tok_len = merged_tok_end - merged_tok_start + 1

                    if merged_tok_len > max_tokens:
                        continue

                    # Appliquer merge
                    cur["tok_start"] = merged_tok_start
                    cur["tok_end"] = merged_tok_end
                    cur["char_start"] = min(int(cur["char_start"]), int(other["char_start"]))
                    cur["char_end"] = max(int(cur["char_end"]), int(other["char_end"]))

                    s, e = clean_char_span(text, cur["char_start"], cur["char_end"])
                    cur["char_start"], cur["char_end"] = s, e
                    cur["text"] = text[s:e]

                    # Agrégation des scores/probas
                    keys = ("score", "boundary_prob", "coarse_prob", "fine_prob")
                    if agg == "mean":
                        for k in keys:
                            cur[k] = round((float(cur.get(k, 0.0)) + float(other.get(k, 0.0))) / 2.0, 4)
                    else:
                        for k in keys:
                            cur[k] = round(max(float(cur.get(k, 0.0)), float(other.get(k, 0.0))), 4)

                    used[j] = True
                    changed = True

            used[i] = True
            merged_all.append(cur)

    merged_all.sort(key=lambda x: (x["score"], x["char_end"] - x["char_start"]), reverse=True)
    return merged_all

import math
from typing import List, Dict


# ==============================
# Configuration par label
# ==============================

LABEL_CONFIG = {
    # noms propres : courts, précis
    "hint_person_name": {
        "base_tau": 0.97,
        "len_soft_cap": 4,     # tokens
        "len_penalty": 0.20,
        "len_bonus": 0.00,
    },

    # événements / objets : informatifs, souvent longs
    "hint_event_nominal": {
        "base_tau": 0.97,
        "len_soft_cap": 2,
        "len_penalty": 0.00,
        "len_bonus": 0.05,
        "floor_tau": 0.55,
    },
    "hint_event_named": {
        "base_tau": 0.80,
        "len_soft_cap": 2,
        "len_penalty": 0.00,
        "len_bonus": 0.05,
        "floor_tau": 0.55,
    },
    "hint_object_generic": {
        "base_tau": 0.97,
        "len_soft_cap": 2,
        "len_penalty": 0.00,
        "len_bonus": 0.04,
        "floor_tau": 0.55,
    },
}

DEFAULT_CONFIG = {
    "base_tau": 0.80,
    "len_soft_cap": 3,
    "len_penalty": 0.02,
    "len_bonus": 0.02,
}


# ==============================
# Utilitaires
# ==============================

def tok_len(p: Dict) -> int:
    return int(p["tok_end"]) - int(p["tok_start"]) + 1


def char_len(p: Dict) -> int:
    return int(p["char_end"]) - int(p["char_start"])


def span_iou(a: Dict, b: Dict) -> float:
    inter = max(
        0,
        min(a["char_end"], b["char_end"]) -
        max(a["char_start"], b["char_start"])
    )
    if inter == 0:
        return 0.0
    union = (
        (a["char_end"] - a["char_start"]) +
        (b["char_end"] - b["char_start"]) -
        inter
    )
    return inter / union




# ==============================
# Score ajusté (clé du système)
# ==============================

def adjusted_score(p: Dict) -> float:
    cfg = LABEL_CONFIG.get(p["fine"], DEFAULT_CONFIG)
    L = tok_len(p)

    bonus = cfg.get("len_bonus", 0.0) * math.log1p(L)

    excess = max(0, L - cfg.get("len_soft_cap", 3))
    penalty = cfg.get("len_penalty", 0.0) * (excess ** 2)

    return float(p["score"]) + bonus - penalty


def _can_merge_contiguous_or_overlap(text: str, a: Dict[str, Any], b: Dict[str, Any], max_gap_chars: int = 1) -> bool:
    """
    Merge autorisé si overlap OU contigu (gap <= max_gap_chars) et le gap ne contient pas de caractère alphanumérique.
    -> 100% agnostique langue (isalnum), pas de lexique.
    """
    # overlap
    if not (a["char_end"] <= b["char_start"] or b["char_end"] <= a["char_start"]):
        return True

    # a avant b
    if a["char_end"] <= b["char_start"]:
        gap_len = b["char_start"] - a["char_end"]
        if gap_len > max_gap_chars:
            return False
        gap = text[a["char_end"]:b["char_start"]]
        return not any(ch.isalnum() for ch in gap)

    # b avant a
    if b["char_end"] <= a["char_start"]:
        gap_len = a["char_start"] - b["char_end"]
        if gap_len > max_gap_chars:
            return False
        gap = text[b["char_end"]:a["char_start"]]
        return not any(ch.isalnum() for ch in gap)

    return False


def merge_contiguous_spans_same_candidate(
    preds: List[Dict[str, Any]],
    text: str,
    max_tokens: int = 12,
    group_by: str = "fine",
    max_gap_chars: int = 1,
    agg: str = "max",
    skip_merge_fines: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    Merge greedy des spans du même candidat (fine/coarse) MAIS uniquement si contigu ou overlap.
    Ne merge PAS certains labels fins (par défaut: hint_person_name).

    - group_by: \"fine\" ou \"coarse\"
    - max_gap_chars: contigu = gap <= max_gap_chars et gap sans alnum
    - max_tokens: limite dure sur (tok_end - tok_start + 1) après merge
    - agg: \"max\" ou \"mean\" sur score/probas
    """
    if group_by not in ("fine", "coarse"):
        group_by = "fine"

    if not preds:
        return preds

    if skip_merge_fines is None:
        skip_merge_fines = {"hint_person_name", "hint_person_role", "hint_event_named"}  # demandé : ne jamais merger person_name

    # Si pas de tok_start/tok_end, on ne merge pas
    if any(("tok_start" not in p or "tok_end" not in p) for p in preds):
        return preds

    def tok_len(p: Dict[str, Any]) -> int:
        return int(p["tok_end"]) - int(p["tok_start"]) + 1

    # Séparer ce qu’on ne veut jamais merger (ex: person_name)
    no_merge = [p for p in preds if p.get("fine") in skip_merge_fines]
    mergeable = [p for p in preds if p.get("fine") not in skip_merge_fines]

    # Grouper les mergeables
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for p in mergeable:
        groups.setdefault(p.get(group_by), []).append(p)

    merged_all: List[Dict[str, Any]] = []

    for label, spans in groups.items():
        # tri longest-first (puis score)
        spans_sorted = sorted(spans, key=lambda x: (tok_len(x), x["score"]), reverse=True)
        used = [False] * len(spans_sorted)

        for i in range(len(spans_sorted)):
            if used[i]:
                continue

            cur = dict(spans_sorted[i])
            used[i] = True

            changed = True
            while changed:
                changed = False
                for j in range(len(spans_sorted)):
                    if used[j]:
                        continue
                    other = spans_sorted[j]

                    # même label (fine ou coarse)
                    if other.get(group_by) != label:
                        continue

                    # condition strict : overlap ou contigu
                    if not _can_merge_contiguous_or_overlap(text, cur, other, max_gap_chars=max_gap_chars):
                        continue

                    # check max_tokens
                    merged_tok_start = min(int(cur["tok_start"]), int(other["tok_start"]))
                    merged_tok_end = max(int(cur["tok_end"]), int(other["tok_end"]))
                    merged_tok_len = merged_tok_end - merged_tok_start + 1
                    if merged_tok_len > max_tokens:
                        continue

                    # merge
                    cur["tok_start"] = merged_tok_start
                    cur["tok_end"] = merged_tok_end
                    cur["char_start"] = min(int(cur["char_start"]), int(other["char_start"]))
                    cur["char_end"] = max(int(cur["char_end"]), int(other["char_end"]))

                    # refresh text
                    s, e = clean_char_span(text, cur["char_start"], cur["char_end"])
                    cur["char_start"], cur["char_end"] = s, e
                    cur["text"] = text[s:e]

                    # agrégation
                    keys = ("score", "boundary_prob", "coarse_prob", "fine_prob")
                    if agg == "mean":
                        for k in keys:
                            cur[k] = round((float(cur.get(k, 0.0)) + float(other.get(k, 0.0))) / 2.0, 4)
                    else:
                        for k in keys:
                            cur[k] = round(max(float(cur.get(k, 0.0)), float(other.get(k, 0.0))), 4)

                    used[j] = True
                    changed = True

            merged_all.append(cur)

    # re-ranger + réinjecter les spans non mergeables (person_name)
    out = merged_all + no_merge
    out.sort(key=lambda x: (x["score"], x["char_end"] - x["char_start"]), reverse=True)
    return out

def _contains(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (a["char_start"] <= b["char_start"]) and (b["char_end"] <= a["char_end"])


def merge_contiguous_spans_same_candidate_guarded(
    preds: List[Dict[str, Any]],
    text: str,
    max_tokens: int = 12,
    group_by: str = "fine",
    max_gap_chars: int = 1,
    agg: str = "max",
) -> List[Dict[str, Any]]:
    """
    Merge greedy same candidate (fine/coarse) UNIQUEMENT si overlap/contigu,
    et garde-fou spécial PERSON_NAME :
      - interdit de produire un span person_name qui englobe un (ou plusieurs) person_name
        ayant un score plus élevé.
    """
    if group_by not in ("fine", "coarse"):
        group_by = "fine"
    if not preds:
        return preds

    if any(("tok_start" not in p or "tok_end" not in p) for p in preds):
        return preds

    # Liste de tous les person_name (référence pour le garde-fou)
    all_person_names = [p for p in preds if p.get("fine") == "hint_person_name"]

    def tok_len(p: Dict[str, Any]) -> int:
        return int(p["tok_end"]) - int(p["tok_start"]) + 1

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for p in preds:
        groups.setdefault(p.get(group_by), []).append(p)

    merged_all: List[Dict[str, Any]] = []

    for label, spans in groups.items():
        spans_sorted = sorted(spans, key=lambda x: (tok_len(x), x["score"]), reverse=True)
        used = [False] * len(spans_sorted)

        for i in range(len(spans_sorted)):
            if used[i]:
                continue

            cur = dict(spans_sorted[i])
            used[i] = True

            changed = True
            while changed:
                changed = False
                for j in range(len(spans_sorted)):
                    if used[j] or j == i:
                        continue
                    other = spans_sorted[j]
                    if other.get(group_by) != label:
                        continue

                    # --- condition stricte : overlap ou contigu (gap <= max_gap_chars, gap sans alnum)
                    if not _can_merge_by_char(text, cur, other, max_gap_chars=max_gap_chars):
                        continue

                    merged_tok_start = min(int(cur["tok_start"]), int(other["tok_start"]))
                    merged_tok_end = max(int(cur["tok_end"]), int(other["tok_end"]))
                    merged_tok_len = merged_tok_end - merged_tok_start + 1
                    if merged_tok_len > max_tokens:
                        continue

                    # Simuler le span mergé (sans l’appliquer encore)
                    merged_candidate = dict(cur)
                    merged_candidate["tok_start"] = merged_tok_start
                    merged_candidate["tok_end"] = merged_tok_end
                    merged_candidate["char_start"] = min(int(cur["char_start"]), int(other["char_start"]))
                    merged_candidate["char_end"] = max(int(cur["char_end"]), int(other["char_end"]))

                    s, e = clean_char_span(text, merged_candidate["char_start"], merged_candidate["char_end"])
                    merged_candidate["char_start"], merged_candidate["char_end"] = s, e

                    # Agréger score/probas (comme dans ton code actuel)
                    keys = ("score", "boundary_prob", "coarse_prob", "fine_prob")
                    if agg == "mean":
                        merged_score = (float(cur.get("score", 0.0)) + float(other.get("score", 0.0))) / 2.0
                        merged_candidate["score"] = round(merged_score, 4)
                        for k in ("boundary_prob", "coarse_prob", "fine_prob"):
                            merged_candidate[k] = round((float(cur.get(k, 0.0)) + float(other.get(k, 0.0))) / 2.0, 4)
                    else:
                        merged_score = max(float(cur.get("score", 0.0)), float(other.get("score", 0.0)))
                        merged_candidate["score"] = round(merged_score, 4)
                        for k in ("boundary_prob", "coarse_prob", "fine_prob"):
                            merged_candidate[k] = round(max(float(cur.get(k, 0.0)), float(other.get(k, 0.0))), 4)

                    merged_candidate["text"] = text[s:e]

                    # --- GARDE-FOU PERSON_NAME ---
                    # Si on est en train de produire/étendre un hint_person_name,
                    # on interdit si le span mergé englobe un person_name mieux scoré (ou plusieurs).
                    if merged_candidate.get("fine") == "hint_person_name":
                        continue
#                         if _would_swallow_better_person_name(
#                             merged_candidate,
#                             all_person_names=all_person_names,
#                             merged_score=merged_candidate["score"],
#                             text=text,
#                             max_gap_chars=max_gap_chars,
#                         ):
#                             continue

                    # Appliquer le merge (validé)
                    cur = merged_candidate
                    used[j] = True
                    changed = True

            merged_all.append(cur)

    merged_all.sort(key=lambda x: (x["score"], x["char_end"] - x["char_start"]), reverse=True)
    return merged_all

def _would_swallow_better_person_name(
    merged_span: Dict[str, Any],
    all_person_names: List[Dict[str, Any]],
    merged_score: float,
    text: str,
    max_gap_chars: int = 1,
    eps: float = 1e-6,
) -> bool:
    """
    RÈGLE (comme tu la veux) :
    - Pour un merge PERSON_NAME, on AUTORISE le span long seulement si :
        (a) il contient au moins deux spans hint_person_name,
        (b) ces deux spans sont contigus (gap <= max_gap_chars et gap sans alnum),
        (c) merged_score > score_des_deux (+ eps).
    - Sinon -> on BLOQUE le merge (retour True).

    100% agnostique langue : uniquement offsets + score + isalnum sur le gap.
    """

    s, e = merged_span["char_start"], merged_span["char_end"]

    # 1) récupérer les person_name contenus et dédoublonner par (start,end) en gardant le meilleur score
    by_range = {}
    for pn in all_person_names:
        if pn.get("fine") != "hint_person_name":
            continue
        if pn["char_start"] >= s and pn["char_end"] <= e:
            key = (pn["char_start"], pn["char_end"])
            if key not in by_range or float(pn["score"]) > float(by_range[key]["score"]):
                by_range[key] = pn

    contained = sorted(by_range.values(), key=lambda x: (x["char_start"], x["char_end"]))

    # Il faut au moins 2 PN contenus, sinon on bloque
    if len(contained) < 2:
        return True

    # 2) chercher une paire CONTIGUË (ou quasi) dans ces spans
    # contigu = gap <= max_gap_chars et le gap ne contient pas de caractère alphanumérique
    def is_contiguous(a, b) -> bool:
        gap_len = b["char_start"] - a["char_end"]
        if gap_len < 0:
            # overlap : on ne considère pas ça "contigu" pour ton cas (A et B séparés)
            return False
        if gap_len > max_gap_chars:
            return False
        gap = text[a["char_end"]:b["char_start"]]
        return not any(ch.isalnum() for ch in gap)

    # 3) pour chaque paire contiguë, vérifier que merged_score bat LES DEUX
    for i in range(len(contained) - 1):
        a = contained[i]
        b = contained[i + 1]
        if not is_contiguous(a, b):
            continue

        if float(merged_score) > float(a["score"]) + eps and float(merged_score) > float(b["score"]) + eps:
            # ✅ on a trouvé une paire contiguë et le long bat les deux -> on autorise le merge
            return False

    # Aucune paire contiguë battue par merged_score -> on bloque
    return True


# ==============================
# Seuil dynamique
# ==============================

def dynamic_tau(p: Dict) -> float:
    cfg = LABEL_CONFIG.get(p["fine"], DEFAULT_CONFIG)
    base = cfg["base_tau"]

    L = tok_len(p)

    # EVENT / OBJECT longs → seuil plus bas
    if "floor_tau" in cfg and L >= cfg["len_soft_cap"]:
        return max(cfg["floor_tau"], base - 0.35)

    # PERSON_NAME trop longs → seuil plus haut
    if p["fine"] == "hint_person_name" and L > cfg["len_soft_cap"]:
        return min(0.98, base + 0.20)

    return base


def pass_dynamic_threshold(p: Dict) -> bool:
    return float(p["fine_prob"]) >= dynamic_tau(p)


# ==============================
# NMS géométrique + score ajusté
# ==============================

def post_process_dynamic(
    preds: List[Dict],
    iou_threshold: float = 0.60,
    allow_nested: bool = True,
) -> List[Dict]:

    # 1) seuil dynamique
    preds = [p for p in preds if pass_dynamic_threshold(p)]
    if not preds:
        return preds

    # 2) tri par score ajusté
    preds = sorted(preds, key=adjusted_score, reverse=True)

    kept = []

    for p in preds:
        drop = False
        for k in kept:
            iou = span_iou(p, k)
            if iou < iou_threshold:
                continue

            # même label → compétition directe
            if p["fine"] == k["fine"]:
                if adjusted_score(p) <= adjusted_score(k):
                    drop = True
                    break

            else:
                # labels différents
                if not allow_nested:
                    if adjusted_score(p) <= adjusted_score(k):
                        drop = True
                        break
                else:
                    # nested autorisé uniquement si inclusion claire
                    nested = (
                        (p["char_start"] >= k["char_start"] and p["char_end"] <= k["char_end"]) or
                        (k["char_start"] >= p["char_start"] and k["char_end"] <= p["char_end"])
                    )
                    if not nested:
                        if p["coarse"] in {"EVENT", "OBJECT"} and k["coarse"] in {"EVENT", "OBJECT"}:
                            if adjusted_score(p) <= adjusted_score(k):
                                drop = True
                                break

        if not drop:
            kept.append(p)

    return kept


def main():
    parser = argparse.ArgumentParser(description="Teste le modèle multitête sur des phrases et affiche les labels.")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--checkpoint", default="checkpoint_best_multitask.pt")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-span-len", type=int, default=12)
    parser.add_argument("--tau-boundary", type=float, default=0.90)
    parser.add_argument("--tau-none", type=float, default=0.99, help="Seuil de rejet si p(NONE) >= tau_none")
    parser.add_argument("--tau-coarse", type=float, default=0.00, help="Seuil de confiance coarse top-1")
    parser.add_argument("--tau-fine", type=float, default=0.00)
    parser.add_argument("--topk-coarse", type=int, default=2, help="1 = top-1 coarse, 2 = beam coarse top-2")
    parser.add_argument("--min-char-len", type=int, default=2, help="Longueur minimale d'un span en caractères")
    parser.add_argument("--allow-midword", action="store_true", help="Autorise les spans qui coupent au milieu d'un mot (déconseillé)")
    parser.add_argument("--no-nested", action="store_true", help="Supprime les overlaps au lieu de garder les spans imbriqués")
    parser.add_argument("--text", action="append", default=[], help="Phrase à tester. Option répétable.")
    parser.add_argument("--input-file", default=None, help="Fichier texte (une phrase par ligne)")
    parser.add_argument("--json-out", default=None, help="Si renseigné, écrit aussi les prédictions en JSON.")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille de batch pour inference (1 = sentence par sentence). Utiliser 16 ou 32 pour accélérer.")

    # Post-traitement longest
    parser.add_argument("--prefer-longest", action="store_true", help="Préfère le span englobant le plus long s'il a une confiance nettement supérieure")
    parser.add_argument("--longest-margin", type=float, default=0.10, help="Marge minimale (score) pour préférer le long au contenu")
    parser.add_argument("--longest-min-boundary", type=float, default=0.10, help="Seuil minimal de boundary pour considérer un span englobant")
    parser.add_argument("--longest-group-by", choices=["fine", "coarse"], default="fine", help="Grouper la comparaison par label 'fine' ou 'coarse'")
    parser.add_argument("--longest-trust-fine-threshold", type=float, default=0.99, help="Si le fine_prob de l'englobant >= seuil, il gagne automatiquement")

    # Nouveau: merge greedy longest-first avec cap tokens
    parser.add_argument("--merge-longest", action="store_true",
                        help="Merge greedy des spans du même label (longest-first) sans dépasser un max de tokens")
    parser.add_argument("--merge-max-tokens", type=int, default=12,
                        help="Nombre maximum de tokens (tok_end-tok_start+1) autorisé pour un span mergé")
    parser.add_argument("--merge-group-by", choices=["fine", "coarse"], default="fine",
                        help="Définit ce qui signifie 'même candidat' : même fine ou même coarse")
    parser.add_argument("--merge-max-gap-chars", type=int, default=1,
                        help="Autorise le merge si les spans sont séparés par <= N chars de gap (ponctuation/espaces)")
    parser.add_argument("--merge-agg", choices=["max", "mean"], default="max",
                        help="Agrégation des scores/probas lors d'un merge")

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
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            texts.extend([line.rstrip("\n") for line in f if line.strip()])

    if not texts:
        raise ValueError("Aucun texte fourni. Utilise --text ou --input-file.")

    all_outputs = []

    import time
    total_decode_time = 0.0

    if args.batch_size and args.batch_size > 1 and len(texts) > 1:
        for batch_start in range(0, len(texts), args.batch_size):
            batch_texts = texts[batch_start : batch_start + args.batch_size]
            t0 = time.perf_counter()
            batch_preds = predict_texts_batch(
                model=model,
                tokenizer=tokenizer,
                texts=batch_texts,
                device=device,
                max_length=args.max_length,
                max_span_len=args.max_span_len,
                tau_boundary=args.tau_boundary,
                tau_none=args.tau_none,
                tau_coarse=args.tau_coarse,
                tau_fine=args.tau_fine,
                topk_coarse=args.topk_coarse,
                min_char_len=args.min_char_len,
                enforce_word_boundaries=(not args.allow_midword),
            )
            t1 = time.perf_counter()
            total_decode_time += (t1 - t0)

            for i, preds in enumerate(batch_preds, start=batch_start + 1):
                text = texts[i - 1]
#                 preds = post_process_dynamic(preds, iou_threshold=0.60, allow_nested=(not args.no_nested))

                if args.prefer_longest:
                    preds = prefer_longest_by_confidence(
                        preds,
                        margin=args.longest_margin,
                        min_boundary=args.longest_min_boundary,
                        group_by=args.longest_group_by,
                        trust_fine_threshold=args.longest_trust_fine_threshold,
                    )

                if args.merge_longest:
                    preds = merge_contiguous_spans_same_candidate_guarded(
                            preds,
                            text=text,
                            max_tokens=args.merge_max_tokens,
                            group_by=args.merge_group_by,
                            max_gap_chars=args.merge_max_gap_chars,
                            agg=args.merge_agg,
                        )

                print("\n" + "=" * 100)
                print(f"📝 TEXTE #{i}")
                print(text)
                print("-" * 100)

                if not preds:
                    print("Aucune entité détectée avec les seuils actuels.")
                else:
                    for p in preds:
                        print(
                            f"[{p['char_start']:>4}:{p['char_end']:<4}] "
                            f"{p['text']!r:<35} | coarse={p['coarse']:<6} ({p['coarse_prob']:.4f}) "
                            f"| fine={p['fine']:<22} ({p['fine_prob']:.4f}) "
                            f"| p_ent={p['boundary_prob']:.4f} | score={p['score']:.4f} "
                            f"| tok=[{p.get('tok_start')},{p.get('tok_end')}]"
                        )

                all_outputs.append({"text": text, "predictions": preds})
    else:
        for idx, text in enumerate(texts, start=1):
            t0 = time.perf_counter()
            preds = predict_text(
                model=model,
                tokenizer=tokenizer,
                text=text,
                device=device,
                max_length=args.max_length,
                max_span_len=args.max_span_len,
                tau_boundary=args.tau_boundary,
                tau_none=args.tau_none,
                tau_coarse=args.tau_coarse,
                tau_fine=args.tau_fine,
                topk_coarse=args.topk_coarse,
                min_char_len=args.min_char_len,
                enforce_word_boundaries=(not args.allow_midword),
            )
            t1 = time.perf_counter()
            total_decode_time += (t1 - t0)

            preds = dedupe_overlaps(preds, allow_nested=(not args.no_nested))

            if args.prefer_longest:
                preds = prefer_longest_by_confidence(
                    preds,
                    margin=args.longest_margin,
                    min_boundary=args.longest_min_boundary,
                    group_by=args.longest_group_by,
                    trust_fine_threshold=args.longest_trust_fine_threshold,
                )

            if args.merge_longest:
                preds = merge_longest_spans_same_candidate(
                    preds,
                    text=text,
                    max_tokens=args.merge_max_tokens,
                    group_by=args.merge_group_by,
                    max_gap_chars=args.merge_max_gap_chars,
                    agg=args.merge_agg,
                )

            print("\n" + "=" * 100)
            print(f"📝 TEXTE #{idx}")
            print(text)
            print("-" * 100)

            if not preds:
                print("Aucune entité détectée avec les seuils actuels.")
            else:
                for p in preds:
                    print(
                        f"[{p['char_start']:>4}:{p['char_end']:<4}] "
                        f"{p['text']!r:<35} | coarse={p['coarse']:<6} ({p['coarse_prob']:.4f}) "
                        f"| fine={p['fine']:<22} ({p['fine_prob']:.4f}) "
                        f"| p_ent={p['boundary_prob']:.4f} | score={p['score']:.4f} "
                        f"| tok=[{p.get('tok_start')},{p.get('tok_end')}]"
                    )

            all_outputs.append({"text": text, "predictions": preds})

    print("\n" + "#" * 40)
    print(
        f"Temps total passé en décodage/inference : {total_decode_time:.3f} s pour {len(texts)} phrases (batch_size={args.batch_size})"
    )
    print("#" * 40)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Résultats JSON écrits dans {args.json_out}")


if __name__ == "__main__":
    main()