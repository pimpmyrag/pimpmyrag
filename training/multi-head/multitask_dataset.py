# multitask_dataset.py
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from labels import SVO_NONE_ID, SYN_NONE_ID, ROLE_NONE_ID, ROLE2ID, VOICE_NONE_ID, CERTAINTY_NONE_ID, GENDER_NONE_ID, NUMBER_NONE_ID, PERSON_NONE_ID, ROLE_COARSE_NONE_ID


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
                "tok_start":           ts + 1,
                "tok_end":             te + 1,
                "boundary_label":      c["boundary_label"],
                "svo_boundary_label":  c.get("svo_boundary_label", 0),
                "coarse_label_id":     c["coarse_label_id"],
                "fine_label_id":       c["fine_label_id"],
                "syn_label_id":        c.get("syn_label_id", c.get("svo_label_id", SYN_NONE_ID)),
                "role_label_id":       c.get("role_label_id",
                                           ROLE2ID.get(c.get("svo_role", ""), ROLE_NONE_ID)),
                "voice_label_id":      c.get("voice_label_id", VOICE_NONE_ID),
                "certainty_label_id":  c.get("certainty_label_id", CERTAINTY_NONE_ID),
                "gender_label_id":     c.get("gender_label_id", GENDER_NONE_ID),
                "number_label_id":     c.get("number_label_id", NUMBER_NONE_ID),
                "person_label_id":     c.get("person_label_id", PERSON_NONE_ID),
                # +1 décalage CLS ; -1 = non supervisé ; clamp à max_length-1 (bornes DeBERTa)
                "gov_verb_tok_start":  min(c["gov_verb_tok_start"] + 1, self.max_length - 1)
                                       if c.get("gov_verb_tok_start", -1) >= 0 else -1,
                "mod_of_tok_start":    min(c["mod_of_tok_start"] + 1, self.max_length - 1)
                                       if c.get("mod_of_tok_start", -1) >= 0 else -1,
                "sample_weight":       c.get("sample_weight", 1.0),
                "neg_type":            c.get("neg_type", "unknown"),
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
        svo_boundary_labels = []
        syn_labels = []
        role_labels = []
        role_coarse_labels = []
        voice_labels = []
        certainty_labels = []
        gender_labels = []
        number_labels = []
        person_labels = []
        gov_verb_labels = []
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
                svo_boundary_labels.append(c.get("svo_boundary_label", 0))
                coarse_labels.append(c["coarse_label_id"])
                fine_labels.append(c["fine_label_id"])
                syn_labels.append(c.get("syn_label_id", SYN_NONE_ID))
                role_labels.append(c.get("role_label_id", ROLE_NONE_ID))
                role_coarse_labels.append(c.get("role_coarse_label_id", ROLE_COARSE_NONE_ID))
                voice_labels.append(c.get("voice_label_id", VOICE_NONE_ID))
                certainty_labels.append(c.get("certainty_label_id", CERTAINTY_NONE_ID))
                gender_labels.append(c.get("gender_label_id", GENDER_NONE_ID))
                number_labels.append(c.get("number_label_id", NUMBER_NONE_ID))
                person_labels.append(c.get("person_label_id", PERSON_NONE_ID))
                gov_verb_labels.append(c.get("gov_verb_tok_start", -1))
                sample_weights.append(c.get("sample_weight", 1.0))

            spans.append(sample_spans)

        return {
            "ids": ids,
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "spans": spans,
            "boundary_labels":      torch.tensor(boundary_labels,    dtype=torch.long),
            "svo_boundary_labels":  torch.tensor(svo_boundary_labels, dtype=torch.long),
            "coarse_labels":        torch.tensor(coarse_labels,       dtype=torch.long),
            "fine_labels":          torch.tensor(fine_labels,         dtype=torch.long),
            "syn_labels":           torch.tensor(syn_labels,          dtype=torch.long),
            "role_labels":          torch.tensor(role_labels,         dtype=torch.long),
            "role_coarse_labels":   torch.tensor(role_coarse_labels,  dtype=torch.long),
            "voice_labels":         torch.tensor(voice_labels,        dtype=torch.long),
            "certainty_labels":     torch.tensor(certainty_labels,    dtype=torch.long),
            "gender_labels":        torch.tensor(gender_labels,       dtype=torch.long),
            "number_labels":        torch.tensor(number_labels,       dtype=torch.long),
            "person_labels":        torch.tensor(person_labels,       dtype=torch.long),
            "gov_verb_labels":      torch.tensor(gov_verb_labels,     dtype=torch.long),
            "sample_weights":       torch.tensor(sample_weights,      dtype=torch.float32),
            "invalid_candidate_count": invalid_candidate_count,
        }

    return collate_fn
