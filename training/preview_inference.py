#!/usr/bin/env python3
"""
Preview inference for token-classification model trained with BILOU tags.
Loads a model directory (the output of training/train_ner.py) and runs a few example sentences,
printing token-level predicted tags, offsets and reconstructed entities.

Usage:
  python training/preview_inference.py --model_dir outputs/ner --sent "Les policiers ont tiré sur les manifestants" --sent "Emmanuel Macron a parlé à Paris"

Or provide a file with one sentence per line:
  python training/preview_inference.py --model_dir outputs/ner --input_file data/sample_sents.txt

"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


DEFAULT_SENTS = [
    "Les policiers ont tiré sur les manifestants au fond de la place de la République à 2h44 du matin.",
    "Emmanuel Macron a tenu une réunion à Paris.",
    "Apple a présenté un nouveau produit lors d'un événement à Cupertino.",
]


def device_name():
    if torch.cuda.is_available():
        return 'cuda'
    # MPS (mac) if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def bilou_to_spans(tokens: List[str], offsets: List[Tuple[int,int]], labels: List[str]):
    """Convert BILOU token-level labels into character spans and text.
    tokens: token strings
    offsets: list of (start,end) offsets per token
    labels: list of label strings (eg 'B-PER','I-PER','L-PER','U-PER','O')
    Returns list of dict {label, start, end, text, token_indices}
    """
    res = []
    cur = None
    cur_tokens = []
    cur_start = None
    cur_end = None
    cur_label = None

    for i, lab in enumerate(labels):
        if lab == 'O' or lab is None:
            # close any open
            if cur is not None:
                res.append({
                    'label': cur_label,
                    'start': cur_start,
                    'end': cur_end,
                    'tokens': cur_tokens,
                })
                cur = None
                cur_tokens = []
                cur_start = None
                cur_end = None
                cur_label = None
            continue
        if '-' not in lab:
            # unknown format, skip
            continue
        prefix, typ = lab.split('-', 1)
        if prefix == 'U':
            s, e = offsets[i]
            res.append({'label': typ, 'start': s, 'end': e, 'tokens': [i]})
            # ensure close
            cur = None
        elif prefix == 'B':
            # start a span
            s, e = offsets[i]
            cur = True
            cur_label = typ
            cur_tokens = [i]
            cur_start = s
            cur_end = e
        elif prefix == 'I':
            if cur is None:
                # treat as B
                s, e = offsets[i]
                cur = True
                cur_label = typ
                cur_tokens = [i]
                cur_start = s
                cur_end = e
            else:
                # continue
                cur_tokens.append(i)
                cur_end = offsets[i][1]
        elif prefix == 'L':
            if cur is None:
                # treat as single
                s, e = offsets[i]
                res.append({'label': typ, 'start': s, 'end': e, 'tokens': [i]})
            else:
                cur_tokens.append(i)
                cur_end = offsets[i][1]
                res.append({'label': cur_label, 'start': cur_start, 'end': cur_end, 'tokens': list(cur_tokens)})
                cur = None
                cur_tokens = []
                cur_start = None
                cur_end = None
                cur_label = None
        else:
            # unknown prefix
            continue
    # close any leftover
    if cur is not None:
        res.append({'label': cur_label, 'start': cur_start, 'end': cur_end, 'tokens': list(cur_tokens)})
    return res


def run_preview(model_dir: str, sents: List[str], max_len: int = 512):
    device = torch.device(device_name())
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # determine id2label
    EXPECTED_LABEL_LIST = ["O"] + [f"B-{l}" for l in ["PER","LOC","OBJECT","ORG","TIME","EVENT"]] + [f"I-{l}" for l in ["PER","LOC","OBJECT","ORG","TIME","EVENT"]] + [f"L-{l}" for l in ["PER","LOC","OBJECT","ORG","TIME","EVENT"]] + [f"U-{l}" for l in ["PER","LOC","OBJECT","ORG","TIME","EVENT"]]
    id2label = None
    raw = getattr(model.config, 'id2label', None)
    if raw:
        try:
            if isinstance(raw, dict):
                # check if existing mapping contains real BILOU tags or placeholder LABEL_x
                values = list(raw.values())
                has_real = any((isinstance(v, str) and (v == 'O' or v.startswith('B-') or v.startswith('I-') or v.startswith('L-') or v.startswith('U-'))) for v in values)
                if has_real:
                    id2label = {int(k): v for k, v in raw.items()}
                else:
                    # override with expected BILOU labels
                    id2label = {i: EXPECTED_LABEL_LIST[i] for i in range(len(EXPECTED_LABEL_LIST))}
            else:
                id2label = raw
        except Exception:
            id2label = {i: EXPECTED_LABEL_LIST[i] for i in range(len(EXPECTED_LABEL_LIST))}
    else:
        id2label = {i: EXPECTED_LABEL_LIST[i] for i in range(len(EXPECTED_LABEL_LIST))}

    for sent in sents:
        print('\n' + '='*80)
        print('SENTENCE:', sent)
        enc = tokenizer(sent, return_offsets_mapping=True, truncation=True, max_length=max_len, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        offsets = enc['offset_mapping'][0].tolist()

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits.cpu()
            preds = logits.argmax(-1).squeeze().tolist()

        ids = input_ids.squeeze().cpu().tolist()
        tokstr = tokenizer.convert_ids_to_tokens(ids)

        # build display rows
        rows = []
        labels = []
        offs = []
        for i, (t, offs_i, pid) in enumerate(zip(tokstr, offsets, preds)):
            # skip special tokens which usually have offset (0,0)
            s_off, e_off = offs_i
            lab = id2label.get(int(pid), 'O')
            rows.append((i, t, s_off, e_off, lab))
            labels.append(lab)
            offs.append((s_off, e_off))

        # pretty print table
        print(f"{'i':>3} {'token':>20} {'offs':>12} {'label':>10}")
        for i, t, s_off, e_off, lab in rows:
            print(f"{i:3d} {t:>20} {s_off:3d}-{e_off:<3d} {lab:>10}")

        # reconstruct spans from BILOU tags
        spans = bilou_to_spans(tokstr, offs, labels)
        print('\nExtracted spans:')
        for sp in spans:
            st = sp['start']
            en = sp['end']
            text_slice = sent[st:en] if st is not None and en is not None and st < len(sent) else ''
            print(f" - {sp['label']}: [{st},{en}] '{text_slice}' tokens={sp['tokens']}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', required=True)
    p.add_argument('--input_file', help='one sentence per line')
    p.add_argument('--sent', action='append', help='sentence to infer (can be repeated)')
    args = p.parse_args()

    sents = []
    if args.input_file:
        sents = [l.strip() for l in Path(args.input_file).read_text(encoding='utf-8').splitlines() if l.strip()]
    if args.sent:
        sents.extend(args.sent)
    if not sents:
        sents = DEFAULT_SENTS

    run_preview(args.model_dir, sents)

