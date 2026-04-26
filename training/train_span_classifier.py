#!/usr/bin/env python3
"""
Train a span classification head on top of a pretrained transformer (DeBERTa v3 recommended).
Input: JSONL file with fields: text, span_start, span_end, span_text, label (label in {NONE, PER, LOC, OBJECT, ORG, TIME, EVENT})

The model will be trained to classify each candidate span. We implement a simple dataset loader and a Trainer pipeline.

Usage:
  python training/train_span_classifier.py --input spans.jsonl --output_dir outputs/spanclf --model_name microsoft/deberta-v3-base

"""
from __future__ import annotations
import argparse
import json
from typing import List, Dict
import datasets
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

LABELS = ["NONE","PER","LOC","OBJECT","ORG","TIME","EVENT"]
LABEL2ID = {l:i for i,l in enumerate(LABELS)}


def read_jsonl(path: str) -> List[Dict]:
    res = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line=line.strip()
            if not line: continue
            res.append(json.loads(line))
    return res


def prepare_examples(records, tokenizer, max_length=256):
    texts = [r['text'] for r in records]
    spans = [(r['span_start'], r['span_end']) for r in records]
    inputs = tokenizer(texts, padding=True, truncation=True, return_offsets_mapping=True, max_length=max_length)
    labels = []
    # For each record, compute token indices of span using offsets
    for i, off in enumerate(inputs['offset_mapping']):
        s,e = spans[i]
        # find first token with offset start >= s and last token with offset end <= e (robustification could be added)
        token_start = 0
        token_end = 0
        for idx, (ts,te) in enumerate(off):
            if ts >= s and token_start == 0:
                token_start = idx
            if te <= e:
                token_end = idx
        # store token start/end as input features (we'll stringify as text for a simple sequence classification)
        # For sequence classifier, we'll concatenate span text with sentence: "[SPAN] span_text [SEP] full_text"
        labels.append(LABEL2ID.get(records[i].get('label','NONE'), 0))
    # Create a simple dataset where input text is span + special sep + context
    examples = []
    for i, r in enumerate(records):
        span_text = r['span_text']
        ctx = r['text']
        inp = f"[SPAN] {span_text} [CONTEXT] {ctx}"
        examples.append({ 'text': inp, 'label': LABEL2ID.get(r.get('label','NONE'), 0) })
    return examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--model_name', default='microsoft/deberta-v3-base')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=3)
    args = p.parse_args()

    records = read_jsonl(args.input)
    print('Loaded', len(records), 'candidate spans')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    examples = prepare_examples(records, tokenizer)
    ds = Dataset.from_list(examples)
    def transform(ex):
        return tokenizer(ex['text'], truncation=True, padding='max_length')
    tokenized = ds.map(transform, batched=True)
    tokenized = tokenized.rename_column('label', 'labels')
    tokenized.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

    config = AutoConfig.from_pretrained(args.model_name, num_labels=len(LABELS))
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_total_limit=2,
        fp16=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__=='__main__':
    main()

