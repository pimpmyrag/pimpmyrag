"""Tête ROLE (legacy, 12 labels) — conservée pour compatibilité checkpoints.

lambda_role=0.0 par défaut → aucun gradient réel, mais la tête reste
chargeable/sauvegardable (state_dict) et calculable pour diagnostic.
Couche propre : `model.role_head`.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from labels import ROLE_LABELS, NUM_ROLE, ROLE_NONE_ID
from .base import Head
from .common import safe_macro_f1, safe_report, gather_by_span_indices


class RoleLegacyHead(Head):
    task_key = "role"
    jsonl_keys = {"role_macro_f1": "role_f1"}

    def __init__(self, linear):
        self.linear = linear

    @classmethod
    def from_model(cls, model) -> "RoleLegacyHead":
        return cls(model.role_head)

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {"role_logits": self.linear(features["span_h_role"])}

    def compute_loss(self, outputs, labels, sample_weights, class_weights=None, **kwargs: Any) -> torch.Tensor:
        device = outputs["role_logits"].device
        role_logits = outputs["role_logits"]
        role_labels = labels["role_labels"].to(device=device, dtype=torch.long)
        role_mask = (role_labels >= 0) & (role_labels != ROLE_NONE_ID)
        if not role_mask.any():
            return torch.tensor(0.0, device=device)
        loss_per = F.cross_entropy(role_logits[role_mask], role_labels[role_mask], reduction="none")
        return (loss_per * sample_weights[role_mask]).mean()

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        role_pred = outputs["role_logits"].argmax(dim=-1).detach().cpu().tolist()
        role_true = gather_by_span_indices(labels["role_labels"], span_indices)
        out_true, out_pred = [], []
        for rt, rp in zip(role_true, role_pred):
            if rt >= 0 and rt != ROLE_NONE_ID:
                out_true.append(rt)
                out_pred.append(rp)
        return out_true, out_pred

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        present = set(y_true)
        return {
            "role_macro_f1": safe_macro_f1(
                y_true, y_pred, labels=[l for l in range(NUM_ROLE) if l in present]
            ) if y_true else 0.0,
            "role_report": safe_report(
                y_true, y_pred,
                labels=[l for l in range(NUM_ROLE) if l != ROLE_NONE_ID and l in present],
                target_names=[ROLE_LABELS[l] for l in range(NUM_ROLE) if l != ROLE_NONE_ID and l in present],
            ) if y_true else "N/A",
        }

