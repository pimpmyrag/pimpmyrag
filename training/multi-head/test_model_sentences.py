import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

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


def clean_char_span(text: str, s_char: int, e_char: int) -> tuple[int, int]:
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
        for tok_end in text_token_positions[start_idx_in_list:start_idx_in_list + max_span_len]:
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

            spans.append({
                "tok_start": tok_start,
                "tok_end": tok_end,
            })
            meta.append({
                "tok_start": tok_start,
                "tok_end": tok_end,
                "char_start": s_char,
                "char_end": e_char,
                "text": span_text,
            })

    return input_ids, attention_mask, [spans], meta


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


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
    tau_fine: float = 0.70,
    topk_coarse: int = 1,
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
        outputs = model({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "spans": spans,
        })

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
        key = (p["char_start"], p["char_end"], p["fine"])
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


def main():
    parser = argparse.ArgumentParser(description="Teste le modèle multitête sur des phrases et affiche les labels.")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--checkpoint", default="checkpoint_best_multitask.pt")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-span-len", type=int, default=8)
    parser.add_argument("--tau-boundary", type=float, default=0.70)
    parser.add_argument("--tau-none", type=float, default=0.50, help="Seuil de rejet si p(NONE) >= tau_none")
    parser.add_argument("--tau-coarse", type=float, default=0.45, help="Seuil de confiance coarse top-1")
    parser.add_argument("--tau-fine", type=float, default=0.00)
    parser.add_argument("--topk-coarse", type=int, default=1, help="1 = top-1 coarse, 2 = beam coarse top-2")
    parser.add_argument("--min-char-len", type=int, default=2, help="Longueur minimale d'un span en caractères")
    parser.add_argument("--allow-midword", action="store_true", help="Autorise les spans qui coupent au milieu d'un mot (déconseillé)")
    parser.add_argument("--no-nested", action="store_true", help="Supprime les overlaps au lieu de garder les spans imbriqués")
    parser.add_argument("--text", action="append", default=[], help="Phrase à tester. Option répétable.")
    parser.add_argument("--input-file", default=None, help="Fichier texte (une phrase par ligne)")
    parser.add_argument("--json-out", default=None, help="Si renseigné, écrit aussi les prédictions en JSON.")
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

    for idx, text in enumerate(texts, start=1):
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
        preds = dedupe_overlaps(preds, allow_nested=(not args.no_nested))

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
                    f"| p_ent={p['boundary_prob']:.4f} | score={p['score']:.4f}"
                )

        all_outputs.append({
            "text": text,
            "predictions": preds,
        })

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Résultats JSON écrits dans {args.json_out}")


if __name__ == "__main__":
    main()