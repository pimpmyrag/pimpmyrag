# build_multitask_dataset.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set

from transformers import AutoTokenizer

from labels import (
    FINE2ID, FINE_NONE_ID,
    fine_label_to_coarse_id,
    COARSE_NONE_ID,
    SYN2ID, SYN_NONE_ID, ALL_SYN_LABELS,
    ROLE2ID, ROLE_NONE_ID,
    VOICE2ID, VOICE_NONE_ID,
    CERTAINTY2ID, CERTAINTY_NONE_ID,
    GENDER2ID, GENDER_NONE_ID,
    NUMBER2ID, NUMBER_NONE_ID,
    PERSON2ID, PERSON_NONE_ID,
    # compat aliases
    SVO2ID, SVO_NONE_ID, ALL_SVO_LABELS,
)

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def tokenize_with_offsets(tokenizer, text: str):
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    return enc["input_ids"], enc["offset_mapping"]

def char_span_to_token_span(offsets: List[Tuple[int, int]], start: int, end: int):
    """
    Convertit un span char -> span token couvrant la zone [start, end)
    même si les frontières char ne tombent pas exactement sur des offsets token.
    """
    tok_start = None
    tok_end = None

    for i, (s, e) in enumerate(offsets):
        if e <= start:
            continue
        if s >= end:
            break
        if tok_start is None:
            tok_start = i
        tok_end = i

    if tok_start is None or tok_end is None:
        return None
    return tok_start, tok_end

def token_span_to_char_span(offsets: List[Tuple[int, int]], tok_start: int, tok_end: int):
    return offsets[tok_start][0], offsets[tok_end][1]

def spans_overlap(a_start, a_end, b_start, b_end):
    return not (a_end <= b_start or b_end <= a_start)

def token_span_iou(a, b):
    a0, a1 = a
    b0, b1 = b
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return inter / union if union > 0 else 0.0

def build_gold_candidates(row, tokenizer):
    text = row["text"]
    input_ids, offsets = tokenize_with_offsets(tokenizer, text)

    gold_candidates = []
    gold_token_spans = []
    gold_char_spans = set()

    for sp in row.get("spans", []):
        label = sp["label"]
        start = sp["start"]
        end   = sp["end"]

        tok_span = char_span_to_token_span(offsets, start, end)
        if tok_span is None:
            continue
        tok_start, tok_end = tok_span

        # ── Spans syntaxiques v4 : verb_trigger, pron_subj, pron_obj ────────
        if label in ALL_SYN_LABELS:
            syn_id = SYN2ID[label]

            # Voice + certainty sur verb_trigger
            voice_id     = VOICE_NONE_ID
            certainty_id = CERTAINTY_NONE_ID
            if label == "verb_trigger":
                voice_str = sp.get("voice", "")
                voice_id  = VOICE2ID.get(voice_str, VOICE_NONE_ID)
                cert_str  = sp.get("certainty", "")
                certainty_id = CERTAINTY2ID.get(cert_str, CERTAINTY_NONE_ID)

            # Rôle SVO du pronom + gov_verb_tok_start
            role_id          = ROLE2ID.get(sp.get("svo_role", "NONE"), ROLE_NONE_ID)
            gov_verb_tok_start = -1
            gvs = sp.get("gov_verb_start")
            if gvs is not None:
                v_tok = char_span_to_token_span(offsets, gvs, gvs + 1)
                if v_tok is not None:
                    gov_verb_tok_start = v_tok[0]

            # Conversion du champ person (peut être int ou str dans différentes versions du dataset)
            person_raw = sp.get("person")
            person_id = PERSON2ID.get(str(person_raw) if person_raw is not None else "", PERSON_NONE_ID)

            cand = {
                "char_start":          start,
                "char_end":            end,
                "tok_start":           tok_start,
                "tok_end":             tok_end,
                "boundary_label":      0,            # pas un span NER
                "svo_boundary_label":  1,            # span syntaxique
                "coarse_label_id":     COARSE_NONE_ID,
                "fine_label_id":       FINE_NONE_ID,
                "syn_label_id":        syn_id,
                "role_label_id":       role_id,
                "voice_label_id":      voice_id,
                "certainty_label_id":  certainty_id,
                "gender_label_id":     GENDER2ID.get(sp.get("gender"), GENDER_NONE_ID),
                "number_label_id":     NUMBER2ID.get(sp.get("number"), NUMBER_NONE_ID),
                "person_label_id":     person_id,
                "gov_verb_tok_start":  gov_verb_tok_start,
                "mod_of_tok_start":    -1,
                "neg_type":            "syn_gold",
                "sample_weight":       1.0,
                "text":                sp.get("text", text[start:end]),
            }
            gold_candidates.append(cand)
            gold_token_spans.append((tok_start, tok_end))
            gold_char_spans.add((start, end))
            continue

        # ── Spans NER classiques (hint_*) — portent maintenant svo_role ─────
        if label not in FINE2ID:
            # Ignorer les labels inconnus (anciens labels Stanza si mélange de données)
            continue

        fine_id   = FINE2ID[label]
        coarse_id = fine_label_to_coarse_id(label)

        # Rôle SVO annoté sur ce span NER
        role_id = ROLE2ID.get(sp.get("svo_role", "NONE"), ROLE_NONE_ID)

        # Pointeur vers le verbe gouverneur
        gov_verb_tok_start = -1
        gvs = sp.get("gov_verb_start")
        if gvs is not None:
            v_tok = char_span_to_token_span(offsets, gvs, gvs + 1)
            if v_tok is not None:
                gov_verb_tok_start = v_tok[0]

        # Pointeur vers le span dont ce span est modificateur nominal
        mod_of_tok_start = -1
        mos = sp.get("mod_of_start")
        if mos is not None:
            m_tok = char_span_to_token_span(offsets, mos, mos + 1)
            if m_tok is not None:
                mod_of_tok_start = m_tok[0]

        cand = {
            "char_start":          start,
            "char_end":            end,
            "tok_start":           tok_start,
            "tok_end":             tok_end,
            "boundary_label":      1,            # span NER
            "svo_boundary_label":  0,
            "coarse_label_id":     coarse_id,
            "fine_label_id":       fine_id,
            "syn_label_id":        SYN_NONE_ID,
            "role_label_id":       role_id,
            "voice_label_id":      VOICE_NONE_ID,
            "certainty_label_id":  CERTAINTY_NONE_ID,
            "gender_label_id":     GENDER2ID.get(sp.get("gender"), GENDER_NONE_ID),
            "number_label_id":     NUMBER2ID.get(sp.get("number"), NUMBER_NONE_ID),
            "person_label_id":     PERSON_NONE_ID,   # person sur pronoms seulement
            "gov_verb_tok_start":  gov_verb_tok_start,
            "mod_of_tok_start":    mod_of_tok_start,
            "neg_type":            "gold",
            "sample_weight":       1.0,
            "text":                sp.get("text", text[start:end]),
        }
        gold_candidates.append(cand)
        gold_token_spans.append((tok_start, tok_end))
        gold_char_spans.add((start, end))

    return text, input_ids, offsets, gold_candidates, gold_token_spans, gold_char_spans

def _make_negative(neg_type: str, char_start: int, char_end: int,
                   tok_start: int, tok_end: int, weight: float = 1.0) -> dict:
    """Crée un candidat négatif avec tous les champs v4 à NONE/sentinel."""
    return {
        "char_start":          char_start,
        "char_end":            char_end,
        "tok_start":           tok_start,
        "tok_end":             tok_end,
        "boundary_label":      0,
        "svo_boundary_label":  0,
        "coarse_label_id":     COARSE_NONE_ID,
        "fine_label_id":       FINE_NONE_ID,
        "syn_label_id":        SYN_NONE_ID,
        "role_label_id":       ROLE_NONE_ID,
        "voice_label_id":      VOICE_NONE_ID,
        "certainty_label_id":  CERTAINTY_NONE_ID,
        "gender_label_id":     GENDER_NONE_ID,
        "number_label_id":     NUMBER_NONE_ID,
        "person_label_id":     PERSON_NONE_ID,
        "gov_verb_tok_start":  -1,
        "mod_of_tok_start":    -1,
        "neg_type":            neg_type,
        "sample_weight":       weight,
        "text":                None,
    }


def generate_hard_negatives(offsets, gold_candidates, gold_char_spans, max_per_gold=6):
    """
    Hard negatives = variantes de frontières autour des spans golds.
    """
    n_tokens = len(offsets)
    out = []
    seen = set()

    for gc in gold_candidates:
        l = gc["tok_start"]
        r = gc["tok_end"]

        proposals = []
        for dl in [-2, -1, 0, 1, 2]:
            for dr in [-2, -1, 0, 1, 2]:
                nl = l + dl
                nr = r + dr
                if nl < 0 or nr < 0 or nl >= n_tokens or nr >= n_tokens or nl > nr:
                    continue
                if nl == l and nr == r:
                    continue

                cstart, cend = token_span_to_char_span(offsets, nl, nr)
                if (cstart, cend) in gold_char_spans:
                    continue

                # On veut plutôt des spans "proches" d'un gold
                iou = token_span_iou((l, r), (nl, nr))
                if iou <= 0.0:
                    continue

                proposals.append((nl, nr, cstart, cend, iou))

        # garder les plus proches
        proposals.sort(key=lambda x: x[-1], reverse=True)
        kept = 0
        for nl, nr, cstart, cend, iou in proposals:
            key = (cstart, cend)
            if key in seen:
                continue
            seen.add(key)
            out.append(_make_negative("hard_neg", cstart, cend, nl, nr))
            kept += 1
            if kept >= max_per_gold:
                break

    return out

def generate_soft_negatives(offsets, gold_token_spans, gold_char_spans, num_soft=20, max_span_len=8, seed=13):
    """
    Soft negatives = spans aléatoires non-overlap.
    """
    rnd = random.Random(seed)
    n_tokens = len(offsets)
    out = []
    seen = set()

    attempts = 0
    max_attempts = num_soft * 50

    while len(out) < num_soft and attempts < max_attempts:
        attempts += 1
        if n_tokens == 0:
            break

        l = rnd.randint(0, n_tokens - 1)
        span_len = rnd.randint(1, max_span_len)
        r = min(n_tokens - 1, l + span_len - 1)

        cstart, cend = token_span_to_char_span(offsets, l, r)
        key = (cstart, cend)
        if key in seen or key in gold_char_spans:
            continue

        # pas de recouvrement avec les golds
        overlap = False
        for gl, gr in gold_token_spans:
            if token_span_iou((l, r), (gl, gr)) > 0.0:
                overlap = True
                break
        if overlap:
            continue

        seen.add(key)
        out.append(_make_negative("soft_neg", cstart, cend, l, r, weight=0.35))

    return out


def generate_englobant_negatives(offsets, gold_candidates, gold_char_spans, max_per_gold=3, max_span_len=12):
    """
    Englobant negatives = spans larges qui CONTIENNENT un gold mais débordent
    suffisamment pour ne plus être une entité valide.
    Enseigne au boundary head que 'Simon Bolivar est considéré comme le Libérateur'
    n'est PAS une entité même si ça contient 'Simon Bolivar'.
    """
    n_tokens = len(offsets)
    out = []
    seen = set()

    for gc in gold_candidates:
        l = gc["tok_start"]
        r = gc["tok_end"]
        gold_len = r - l + 1

        proposals = []
        # Étendre à gauche et/ou à droite de 2 à 6 tokens au total
        for expand_left in range(0, 5):
            for expand_right in range(0, 5):
                total_expand = expand_left + expand_right
                if total_expand < 2:
                    # Au moins 2 tokens d'expansion pour être un vrai englobant
                    continue
                nl = l - expand_left
                nr = r + expand_right
                if nl < 0 or nr >= n_tokens or nr - nl + 1 > max_span_len:
                    continue

                cstart, cend = token_span_to_char_span(offsets, nl, nr)
                if (cstart, cend) in gold_char_spans:
                    continue

                new_len = nr - nl + 1
                # Plus l'expansion est grande, plus c'est un bon négatif
                expansion_ratio = new_len / max(1, gold_len)
                proposals.append((nl, nr, cstart, cend, expansion_ratio))

        # Trier par expansion décroissante (les plus larges d'abord = les plus informatifs)
        proposals.sort(key=lambda x: x[-1], reverse=True)
        kept = 0
        for nl, nr, cstart, cend, _ in proposals:
            key = (cstart, cend)
            if key in seen:
                continue
            seen.add(key)
            out.append(_make_negative("englobant_neg", cstart, cend, nl, nr, weight=1.5))
            kept += 1
            if kept >= max_per_gold:
                break

    return out


def generate_multi_entity_negatives(gold_candidates, gold_char_spans, max_negatives=5):
    """
    Multi-entity negatives = spans qui englobent 2+ entités adjacentes.
    Enseigne que 'Winston Churchill et Franklin D. Roosevelt' n'est PAS un seul span.
    """
    out = []
    seen = set()

    # Trier les golds par position
    sorted_golds = sorted(gold_candidates, key=lambda g: g["tok_start"])

    for i in range(len(sorted_golds) - 1):
        g1 = sorted_golds[i]
        g2 = sorted_golds[i + 1]

        # Vérifier qu'ils sont proches (gap <= 5 tokens)
        gap = g2["tok_start"] - g1["tok_end"]
        if gap > 5 or gap < 0:
            continue

        nl = g1["tok_start"]
        nr = g2["tok_end"]
        cstart = g1["char_start"]
        cend = g2["char_end"]

        key = (cstart, cend)
        if key in seen or key in gold_char_spans:
            continue
        seen.add(key)
        seen.add(key)
        out.append(_make_negative("multi_entity_neg", cstart, cend, nl, nr, weight=2.0))

        if len(out) >= max_negatives:
            break

    return out


def make_multitask_row(row, tokenizer, hard_per_gold=6, soft_factor=2.0, max_span_len=8, seed=13):
    text, input_ids, offsets, gold_candidates, gold_token_spans, gold_char_spans = build_gold_candidates(row, tokenizer)

    num_soft = max(1, int(len(gold_candidates) * soft_factor))
    hard_negs = generate_hard_negatives(
        offsets,
        gold_candidates,
        gold_char_spans,
        max_per_gold=hard_per_gold
    )
    soft_negs = generate_soft_negatives(
        offsets,
        gold_token_spans,
        gold_char_spans,
        num_soft=num_soft,
        max_span_len=max_span_len,
        seed=seed
    )
    englobant_negs = generate_englobant_negatives(
        offsets,
        gold_candidates,
        gold_char_spans,
        max_per_gold=3,
        max_span_len=max_span_len,
    )
    multi_ent_negs = generate_multi_entity_negatives(
        gold_candidates,
        gold_char_spans,
        max_negatives=5,
    )

    candidates = gold_candidates + hard_negs + soft_negs + englobant_negs + multi_ent_negs

    # renseigner le texte du span si absent
    for c in candidates:
        if c["text"] is None:
            c["text"] = text[c["char_start"]:c["char_end"]]

    # Appliquer le _source_weight de merge_silver.py si présent
    source_weight = row.get("_source_weight", 1.0)
    if source_weight != 1.0:
        for c in candidates:
            c["sample_weight"] = c.get("sample_weight", 1.0) * source_weight

    return {
        "id": row["id"],
        "text": text,
        "candidates": candidates,
        "meta": {
            "num_gold": len(gold_candidates),
            "num_hard_neg": len(hard_negs),
            "num_soft_neg": len(soft_negs),
            "num_englobant_neg": len(englobant_negs),
            "num_multi_entity_neg": len(multi_ent_negs),
            "num_tokens": len(input_ids),
        }
    }

def export_head_views(rows, out_prefix: str):
    """
    Exporte des vues séparées si tu veux inspecter / debugger chaque tête.
    """
    boundary_rows = []
    coarse_rows = []
    fine_rows = []

    for row in rows:
        boundary_rows.append({
            "id": row["id"],
            "text": row["text"],
            "spans": [
                {
                    "start": c["char_start"],
                    "end": c["char_end"],
                    "text": c["text"],
                    "label": c["boundary_label"],
                    "neg_type": c["neg_type"],
                    "sample_weight": c["sample_weight"],
                }
                for c in row["candidates"]
            ]
        })
        coarse_rows.append({
            "id": row["id"],
            "text": row["text"],
            "spans": [
                {
                    "start": c["char_start"],
                    "end": c["char_end"],
                    "text": c["text"],
                    "label_id": c["coarse_label_id"],
                    "neg_type": c["neg_type"],
                    "sample_weight": c["sample_weight"],
                }
                for c in row["candidates"]
            ]
        })
        fine_rows.append({
            "id": row["id"],
            "text": row["text"],
            "spans": [
                {
                    "start": c["char_start"],
                    "end": c["char_end"],
                    "text": c["text"],
                    "label_id": c["fine_label_id"],
                    "neg_type": c["neg_type"],
                    "sample_weight": c["sample_weight"],
                }
                for c in row["candidates"]
            ]
        })

    write_jsonl(out_prefix + ".boundary.jsonl", boundary_rows)
    write_jsonl(out_prefix + ".coarse.jsonl", coarse_rows)
    write_jsonl(out_prefix + ".fine.jsonl", fine_rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="dataset JSONL source (positifs uniquement)")
    parser.add_argument("--output", required=True, help="dataset JSONL multitask de sortie")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--tokenizer-path", default=None, help="si tu veux utiliser ton tokenizer local fast")
    parser.add_argument("--hard-per-gold", type=int, default=6)
    parser.add_argument("--soft-factor", type=float, default=2.0, help="nb soft neg = soft_factor * nb gold")
    parser.add_argument("--max-span-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--export-head-views-prefix", default=None, help="si défini, exporte aussi boundary/coarse/fine séparés")
    args = parser.parse_args()

    random.seed(args.seed)

    tokenizer_source = args.tokenizer_path or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    rows = []
    for row in load_jsonl(args.input):
        mt_row = make_multitask_row(
            row,
            tokenizer=tokenizer,
            hard_per_gold=args.hard_per_gold,
            soft_factor=args.soft_factor,
            max_span_len=args.max_span_len,
            seed=args.seed
        )
        rows.append(mt_row)

    write_jsonl(args.output, rows)
    print(f"✅ dataset multitask écrit dans {args.output} ({len(rows)} lignes)")

    if args.export_head_views_prefix:
        export_head_views(rows, args.export_head_views_prefix)
        print(f"✅ vues séparées écrites avec préfixe {args.export_head_views_prefix}")

if __name__ == "__main__":
    main()
