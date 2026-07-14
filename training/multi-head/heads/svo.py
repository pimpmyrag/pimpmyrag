"""Tête SVO (syn_head) — sous-type syntaxique verb_trigger / pron_subj / pron_obj.

Note : cette tête n'a pas de métriques de suivi historiques dans le JSONL
(seule la loss était utilisée). On garde `compute_metrics` disponible pour
un futur monitoring, mais `jsonl_keys` reste vide pour ne pas changer le
format existant.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from .base import Head
from .common import safe_macro_f1, safe_report, gather_by_span_indices


class SvoHead(Head):
    task_key = "svo"
    jsonl_keys: dict[str, str] = {}

    def __init__(self, linear):
        self.linear = linear

    @classmethod
    def from_model(cls, model) -> "SvoHead":
        return cls(model.syn_head)

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        logits = self.linear(features["span_h"])
        return {"syn_logits": logits, "svo_logits": logits}

    def compute_loss(self, outputs, labels, sample_weights, class_weights=None, **kwargs: Any) -> torch.Tensor:
        device = outputs["syn_logits"].device
        syn_logits = outputs["syn_logits"]
        syn_labels = labels["syn_labels"].to(device=device, dtype=torch.long)
        syn_mask = (syn_labels >= 0) & (syn_labels < syn_logits.size(-1))
        if not syn_mask.any():
            return torch.tensor(0.0, device=device)
        loss_per = F.cross_entropy(syn_logits[syn_mask], syn_labels[syn_mask], reduction="none")
        return (loss_per * sample_weights[syn_mask]).mean()

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        syn_pred_full = outputs["syn_logits"].argmax(dim=-1).detach().cpu().tolist()
        syn_true_full = gather_by_span_indices(labels["syn_labels"], span_indices)
        out_true, out_pred = [], []
        for st, sp in zip(syn_true_full, syn_pred_full):
            if st >= 0:
                out_true.append(st)
                out_pred.append(sp)
        return out_true, out_pred

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        return {"svo_macro_f1": safe_macro_f1(y_true, y_pred)} if y_true else {"svo_macro_f1": 0.0}

