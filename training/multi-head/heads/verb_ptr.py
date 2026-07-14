"""Tête VERB_PTR — pointeur argument→verbe gouverneur (attention bilinéaire).

Couches propres : `model.verb_ptr_query` (span_hidden_dim -> 64),
`model.verb_ptr_key` (hidden_size -> 64). Le score est un produit bilinéaire
span_query · token_key sur toute la séquence -> logits [N, seq].
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from labels import ROLE_COARSE_NONE_ID
from .base import Head


class VerbPtrHead(Head):
    task_key = "verb_ptr"
    jsonl_keys = {
        "verb_ptr_acc": "verb_ptr_acc",
        "verb_ptr_n": "verb_ptr_n",
    }

    def __init__(self, query_linear, key_linear):
        self.query_linear = query_linear
        self.key_linear = key_linear

    @classmethod
    def from_model(cls, model) -> "VerbPtrHead":
        return cls(model.verb_ptr_query, model.verb_ptr_key)

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        span_h = features["span_h"]
        hidden = features["hidden"]
        span_batch_idx = features["span_batch_idx"]
        if span_h.size(0) == 0:
            return {"verb_ptr_logits": torch.zeros((0, hidden.size(1)), device=hidden.device)}

        ptr_queries = self.query_linear(span_h)          # [N, 64]
        ptr_keys = self.key_linear(hidden)                # [B, seq, 64]
        gathered_keys = ptr_keys[span_batch_idx]          # [N, seq, 64]
        verb_ptr_logits = torch.bmm(
            gathered_keys, ptr_queries.unsqueeze(-1)
        ).squeeze(-1)                                     # [N, seq]
        return {"verb_ptr_logits": verb_ptr_logits}

    def compute_loss(self, outputs, labels, sample_weights, class_weights=None, **kwargs: Any) -> torch.Tensor:
        vptr_logits = outputs["verb_ptr_logits"]
        device = vptr_logits.device
        gov_verb_labels = labels["gov_verb_labels"].to(device=device, dtype=torch.long)
        role_coarse_labels = labels["role_coarse_labels"].to(device=device, dtype=torch.long)
        seq_len = vptr_logits.size(1)
        ptr_mask = (
            (gov_verb_labels >= 0) & (gov_verb_labels < seq_len)
            & (role_coarse_labels >= 0) & (role_coarse_labels < ROLE_COARSE_NONE_ID)
        )
        if not (ptr_mask.any() and vptr_logits.size(0) > 0):
            return torch.tensor(0.0, device=device)
        loss_per = F.cross_entropy(vptr_logits[ptr_mask], gov_verb_labels[ptr_mask], reduction="none")
        return (loss_per * sample_weights[ptr_mask]).mean()

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        vptr_logits_cpu = outputs["verb_ptr_logits"].detach().cpu()
        ptr_pred = vptr_logits_cpu.argmax(dim=-1).tolist()
        seq_len_ptr = vptr_logits_cpu.size(1) if vptr_logits_cpu.numel() else 0

        if span_indices is not None:
            si_cpu = span_indices.detach().cpu().to(dtype=torch.long)
            gov_verb_true = labels["gov_verb_labels"].detach().cpu()[si_cpu].tolist()
        else:
            gov_verb_true = labels["gov_verb_labels"].detach().cpu().tolist()

        out_true, out_pred = [], []
        for gvt, gvp in zip(gov_verb_true, ptr_pred):
            if gvt >= 0 and gvt < seq_len_ptr:
                out_true.append(gvt)
                out_pred.append(gvp)
        return out_true, out_pred

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        acc = (
            sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
            if y_true else 0.0
        )
        return {"verb_ptr_acc": acc, "verb_ptr_n": len(y_true)}

