# multitask_dataset.py
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class MultiTaskSpanDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = row["text"]

        enc = self.tokenizer(
            text,
            return_offsets_mapping=False,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Longueur réelle de la séquence tokenisée AVEC special tokens
        seq_len = len(input_ids)

        # Hypothèse standard BERT-like: [CLS] ... [SEP]
        # Les spans du builder sont calculés SANS special tokens.
        # Donc indices valides "sans special tokens" = [0, seq_len - 3]
        # car après shift +1, le dernier token de texte doit rester < seq_len - 1 ([SEP])
        actual_text_token_len = max(0, seq_len - 2)

        candidates = []
        invalid_count = 0

        for c in row["candidates"]:
            ts = c["tok_start"]
            te = c["tok_end"]

            # Validation stricte AVANT ajout des labels
            if not isinstance(ts, int) or not isinstance(te, int):
                invalid_count += 1
                continue

            if ts < 0 or te < 0 or ts > te:
                invalid_count += 1
                continue

            # important : on vérifie par rapport à la longueur réelle tokenisée
            # et pas seulement max_length - 2
            if te >= actual_text_token_len:
                invalid_count += 1
                continue

            candidates.append({
                # décalage +1 à cause du token spécial initial
                "tok_start": ts + 1,
                "tok_end": te + 1,
                "boundary_label": c["boundary_label"],
                "coarse_label_id": c["coarse_label_id"],
                "fine_label_id": c["fine_label_id"],
                "sample_weight": c.get("sample_weight", 1.0),
                "neg_type": c.get("neg_type", "unknown"),
            })

        return {
            "id": row["id"],
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "candidates": candidates,
            "invalid_candidate_count": invalid_count,
        }


def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate_fn(batch):
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids = []
        attention_mask = []
        spans = []

        boundary_labels = []
        coarse_labels = []
        fine_labels = []
        sample_weights = []

        ids = []
        invalid_candidate_count = 0

        for item in batch:
            ids.append(item["id"])
            invalid_candidate_count += item.get("invalid_candidate_count", 0)

            ids_tensor = item["input_ids"]
            att_tensor = item["attention_mask"]

            pad_len = max_len - ids_tensor.size(0)
            if pad_len > 0:
                ids_tensor = torch.cat([
                    ids_tensor,
                    torch.full((pad_len,), pad_id, dtype=torch.long)
                ])
                att_tensor = torch.cat([
                    att_tensor,
                    torch.zeros(pad_len, dtype=torch.long)
                ])

            input_ids.append(ids_tensor)
            attention_mask.append(att_tensor)

            sample_spans = []
            for c in item["candidates"]:
                sample_spans.append({
                    "tok_start": c["tok_start"],
                    "tok_end": c["tok_end"],
                })
                boundary_labels.append(c["boundary_label"])
                coarse_labels.append(c["coarse_label_id"])
                fine_labels.append(c["fine_label_id"])
                sample_weights.append(c["sample_weight"])

            spans.append(sample_spans)

        return {
            "ids": ids,
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "spans": spans,
            "boundary_labels": torch.tensor(boundary_labels, dtype=torch.long),
            "coarse_labels": torch.tensor(coarse_labels, dtype=torch.long),
            "fine_labels": torch.tensor(fine_labels, dtype=torch.long),
            "sample_weights": torch.tensor(sample_weights, dtype=torch.float32),
            "invalid_candidate_count": invalid_candidate_count,
        }

    return collate_fn
