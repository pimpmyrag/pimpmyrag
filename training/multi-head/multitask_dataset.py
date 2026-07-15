# multitask_dataset.py
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from labels import (SVO_NONE_ID, SYN_NONE_ID, ROLE_NONE_ID, ROLE2ID, VOICE_NONE_ID, CERTAINTY_NONE_ID,
                    GENDER_NONE_ID, NUMBER_NONE_ID, PERSON_NONE_ID, ROLE_COARSE_NONE_ID, ROLE_OBLIQUE_NONE_ID,
                    SEMANTIC_ROLE_SKIP_ID,
                    VERB_FAMILY_NONE_ID, VERB_FAMILY_FINE_NONE_ID, VERB_POLARITY_NONE_ID,
                    VERB_ASPECT_NONE_ID, VERB_SOURCE_NONE_ID,
                    NOMINAL_RELATION_NONE_ID)
from labels_v9 import (ANIMACY_NONE_ID, LIVING_NONE_ID, ABSTRACT_NONE_ID,
                       DYNAMICITY_NONE_ID, WORK_NONE_ID)


class MultiTaskSpanDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.max_length = max_length

        # Pré-tokenisation vectorisée en __init__ — évite la tokenisation répétée à chaque __getitem__
        # Sur 31k phrases courtes (NER), batch_encode_plus est ~50x plus rapide que appels séquentiels.
        # Élimine le bottleneck CPU/DataLoader qui plafonnait le GPU à ~17% VRAM utilisée.
        print(f" Pré-tokenisation de {len(self.rows)} phrases...", flush=True)
        texts = [row["text"] for row in self.rows]
        encodings = tokenizer(
            texts,
            return_offsets_mapping=False,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        self.input_ids_list     = [torch.tensor(ids,  dtype=torch.long) for ids  in encodings["input_ids"]]
        self.attention_mask_list = [torch.tensor(mask, dtype=torch.long) for mask in encodings["attention_mask"]]
        print(f"✅ Pré-tokenisation terminée.", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        input_ids      = self.input_ids_list[idx]
        attention_mask = self.attention_mask_list[idx]

        # Longueur réelle de la séquence tokenisée AVEC special tokens
        seq_len = input_ids.size(0)

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
                "tok_start":             ts + 1,
                "tok_end":               te + 1,
                "boundary_label":        c["boundary_label"],
                "svo_boundary_label":    c.get("svo_boundary_label", 0),
                "coarse_label_id":       c["coarse_label_id"],
                "fine_label_id":         c["fine_label_id"],
                "syn_label_id":          c.get("syn_label_id", c.get("svo_label_id", SYN_NONE_ID)),
                "role_label_id":         c.get("role_label_id",
                                              ROLE2ID.get(c.get("svo_role", ""), ROLE_NONE_ID)),
                "role_coarse_label_id":  c.get("role_coarse_label_id", ROLE_COARSE_NONE_ID),
                "semantic_role_label_id": c.get("semantic_role_label_id", SEMANTIC_ROLE_SKIP_ID),
                "role_oblique_label_id": c.get("role_oblique_label_id", ROLE_OBLIQUE_NONE_ID),
                "voice_label_id":        c.get("voice_label_id", VOICE_NONE_ID),   # FIX: était absent → VOICE_NONE_ID forcé même si JSON l'avait
                "certainty_label_id":    c.get("certainty_label_id", CERTAINTY_NONE_ID),
                "gender_label_id":       c.get("gender_label_id", GENDER_NONE_ID),
                "number_label_id":       c.get("number_label_id", NUMBER_NONE_ID),
                "person_label_id":       c.get("person_label_id", PERSON_NONE_ID),
                # +1 décalage CLS ; -1 = non supervisé ; clamp à max_length-1 (bornes DeBERTa)
                "gov_verb_tok_start":    min(c["gov_verb_tok_start"] + 1, self.max_length - 1)
                                         if c.get("gov_verb_tok_start", -1) >= 0 else -1,
                "mod_of_tok_start":      min(c["mod_of_tok_start"] + 1, self.max_length - 1)
                                         if c.get("mod_of_tok_start", -1) >= 0 else -1,
                # Nominal parent pointer (v8.22) — +1 décalage CLS ; -1 = NO_PARENT
                "nominal_parent_tok_start": min(c["nominal_parent_tok_start"] + 1, self.max_length - 1)
                                            if c.get("nominal_parent_tok_start", -1) >= 0 else -1,
                "nominal_relation_label_id": c.get("nominal_relation_label_id", NOMINAL_RELATION_NONE_ID),
                # verbfam labels (verb_trigger uniquement ; NONE_ID pour les autres spans)
                "verb_family_label_id":      c.get("verb_family_label_id", VERB_FAMILY_NONE_ID),
                "verb_family_fine_label_id": c.get("verb_family_fine_label_id", VERB_FAMILY_FINE_NONE_ID),
                "verb_polarity_label_id":    c.get("verb_polarity_label_id", VERB_POLARITY_NONE_ID),
                "verb_aspect_label_id":      c.get("verb_aspect_label_id", VERB_ASPECT_NONE_ID),
                "verb_source_label_id":      c.get("verb_source_label_id", VERB_SOURCE_NONE_ID),
                "sample_weight":         c.get("sample_weight", 1.0),
                "neg_type":              c.get("neg_type", "unknown"),
            })

        return {
            "id": row["id"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "candidates": candidates,
            "invalid_candidate_count": invalid_count,
        }


class CollateFn:
    """Callable picklable (compatible multiprocessing DataLoader num_workers>0)."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        pad_id = self.pad_id
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
        semantic_role_labels = []
        role_oblique_labels = []
        voice_labels = []
        certainty_labels = []
        gender_labels = []
        number_labels = []
        person_labels = []
        gov_verb_labels = []
        nominal_parent_labels    = []
        nominal_relation_labels  = []
        verb_family_labels      = []
        verb_family_fine_labels = []
        verb_polarity_labels    = []
        verb_aspect_labels      = []
        verb_source_labels      = []
        animacy_labels    = []
        living_labels     = []
        abstract_labels   = []
        dynamicity_labels = []
        work_labels       = []
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
                semantic_role_labels.append(c.get("semantic_role_label_id", SEMANTIC_ROLE_SKIP_ID))
                role_oblique_labels.append(c.get("role_oblique_label_id", ROLE_OBLIQUE_NONE_ID))
                voice_labels.append(c.get("voice_label_id", VOICE_NONE_ID))
                certainty_labels.append(c.get("certainty_label_id", CERTAINTY_NONE_ID))
                gender_labels.append(c.get("gender_label_id", GENDER_NONE_ID))
                number_labels.append(c.get("number_label_id", NUMBER_NONE_ID))
                person_labels.append(c.get("person_label_id", PERSON_NONE_ID))
                gov_verb_labels.append(c.get("gov_verb_tok_start", -1))
                nominal_parent_labels.append(c.get("nominal_parent_tok_start", -1))
                nominal_relation_labels.append(c.get("nominal_relation_label_id", NOMINAL_RELATION_NONE_ID))
                verb_family_labels.append(c.get("verb_family_label_id", VERB_FAMILY_NONE_ID))
                verb_family_fine_labels.append(c.get("verb_family_fine_label_id", VERB_FAMILY_FINE_NONE_ID))
                verb_polarity_labels.append(c.get("verb_polarity_label_id", VERB_POLARITY_NONE_ID))
                verb_aspect_labels.append(c.get("verb_aspect_label_id", VERB_ASPECT_NONE_ID))
                verb_source_labels.append(c.get("verb_source_label_id", VERB_SOURCE_NONE_ID))
                animacy_labels.append(c.get("animacy_label_id", ANIMACY_NONE_ID))
                living_labels.append(c.get("living_label_id", LIVING_NONE_ID))
                abstract_labels.append(c.get("abstract_label_id", ABSTRACT_NONE_ID))
                dynamicity_labels.append(c.get("dynamicity_label_id", DYNAMICITY_NONE_ID))
                work_labels.append(c.get("work_label_id", WORK_NONE_ID))
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
            "semantic_role_labels": torch.tensor(semantic_role_labels, dtype=torch.long),
            "role_oblique_labels":  torch.tensor(role_oblique_labels, dtype=torch.long),
            "voice_labels":         torch.tensor(voice_labels,        dtype=torch.long),
            "certainty_labels":     torch.tensor(certainty_labels,    dtype=torch.long),
            "gender_labels":        torch.tensor(gender_labels,       dtype=torch.long),
            "number_labels":        torch.tensor(number_labels,       dtype=torch.long),
            "person_labels":        torch.tensor(person_labels,       dtype=torch.long),
            "gov_verb_labels":      torch.tensor(gov_verb_labels,     dtype=torch.long),
            "nominal_parent_labels":   torch.tensor(nominal_parent_labels,   dtype=torch.long),
            "nominal_relation_labels": torch.tensor(nominal_relation_labels, dtype=torch.long),
            "verb_family_labels":      torch.tensor(verb_family_labels,      dtype=torch.long),
            "verb_family_fine_labels": torch.tensor(verb_family_fine_labels, dtype=torch.long),
            "verb_polarity_labels":    torch.tensor(verb_polarity_labels,    dtype=torch.long),
            "verb_aspect_labels":      torch.tensor(verb_aspect_labels,      dtype=torch.long),
            "verb_source_labels":      torch.tensor(verb_source_labels,      dtype=torch.long),
            "animacy_labels":          torch.tensor(animacy_labels,    dtype=torch.long),
            "living_labels":           torch.tensor(living_labels,     dtype=torch.long),
            "abstract_labels":         torch.tensor(abstract_labels,   dtype=torch.long),
            "dynamicity_labels":       torch.tensor(dynamicity_labels, dtype=torch.long),
            "work_labels":             torch.tensor(work_labels,       dtype=torch.long),
            "sample_weights":       torch.tensor(sample_weights,      dtype=torch.float32),
            "invalid_candidate_count": invalid_candidate_count,
        }


def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return CollateFn(pad_id)
