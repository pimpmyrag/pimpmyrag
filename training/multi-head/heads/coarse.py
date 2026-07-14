"""Tête COARSE — 6 familles NER + NONE.

Couche propre : `model.coarse_head` (Linear span_hidden_dim -> num_coarse).
Métriques calculées en POSITIVE-ONLY (boundary gold == 1), NONE exclu.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from labels import COARSE_LABELS, COARSE_NONE_ID
from .base import Head
from .common import safe_macro_f1, safe_report, gather_by_span_indices


class CoarseHead(Head):
    task_key = "coarse"
    jsonl_keys = {
        "coarse_macro_f1": "coarse_f1",
    }

    def __init__(self, linear):
        self.linear = linear

    @classmethod
    def from_model(cls, model) -> "CoarseHead":
        return cls(model.coarse_head)

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {"coarse_logits": self.linear(features["span_h_coarse"])}

    def compute_loss(
        self,
        outputs,
        labels,
        sample_weights,
        class_weights: Optional[torch.Tensor] = None,
        focal_coarse_gamma: float = 0.0,
        ignore_coarse_none: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        device = outputs["coarse_logits"].device
        c_logits = outputs["coarse_logits"]
        coarse_labels = labels["coarse_labels"].to(device=device, dtype=torch.long)

        if ignore_coarse_none:
            pos_mask_c = (coarse_labels != COARSE_NONE_ID)
            if not pos_mask_c.any():
                return torch.tensor(0.0, device=device)
            loss_per = F.cross_entropy(
                c_logits[pos_mask_c], coarse_labels[pos_mask_c],
                weight=class_weights, reduction="none")
            if focal_coarse_gamma > 0.0:
                p_t = F.softmax(c_logits[pos_mask_c].detach(), dim=-1).gather(
                    1, coarse_labels[pos_mask_c].unsqueeze(1)).squeeze(1)
                loss_per = loss_per * (1.0 - p_t) ** focal_coarse_gamma
            return (loss_per * sample_weights[pos_mask_c]).mean()

        loss_per = F.cross_entropy(c_logits, coarse_labels, weight=class_weights, reduction="none")
        if focal_coarse_gamma > 0.0:
            pos_mask = (coarse_labels != COARSE_NONE_ID)
            if pos_mask.any():
                p_t = F.softmax(c_logits[pos_mask].detach(), dim=-1).gather(
                    1, coarse_labels[pos_mask].unsqueeze(1)).squeeze(1)
                loss_per = loss_per.clone()
                loss_per[pos_mask] = loss_per[pos_mask] * (1.0 - p_t) ** focal_coarse_gamma
        return (loss_per * sample_weights).mean()

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        """Positive-only : ne garde que les spans avec boundary gold == 1 (fourni via context)."""
        c_pred = outputs["coarse_logits"].argmax(dim=-1).detach().cpu().tolist()
        c_true = gather_by_span_indices(labels["coarse_labels"], span_indices)
        b_true = (context or {}).get("boundary_true")
        if b_true is None:
            return c_true, c_pred
        out_true, out_pred = [], []
        for bt, ct, cp in zip(b_true, c_true, c_pred):
            if bt == 1:
                out_true.append(ct)
                out_pred.append(cp)
        return out_true, out_pred

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        labels_range = list(range(len(COARSE_LABELS) - 1))  # exclut NONE
        return {
            "coarse_macro_f1": safe_macro_f1(y_true, y_pred, labels=labels_range),
            "coarse_report": safe_report(
                y_true, y_pred, labels=labels_range, target_names=COARSE_LABELS[:-1]
            ) if y_true else "N/A",
        }

