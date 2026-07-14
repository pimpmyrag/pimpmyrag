"""Tête BOUNDARY — détection binaire span-entité vs non-entité.

Couche propre : `model.boundary_head` (Linear span_hidden_dim -> 2).
Reçoit en plus l'évidence NER (coarse/fine softmax détachés) via
`model.boundary_ner_evidence_head` si `boundary_aux_from_ner` est actif
(cf. SpanMultiTaskModel.__init__ / forward — logique cross-tête conservée
dans le modèle car elle combine 3 têtes : boundary + coarse + fine).
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from .base import Head
from .common import safe_macro_f1, safe_report, gather_by_span_indices


class BoundaryHead(Head):
    task_key = "boundary"
    jsonl_keys = {
        "boundary_f1": "boundary_f1",
    }

    def __init__(self, linear):
        self.linear = linear

    @classmethod
    def from_model(cls, model) -> "BoundaryHead":
        return cls(model.boundary_head)

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {"boundary_logits_base": self.linear(features["span_h_ner"])}

    def compute_loss(
        self,
        outputs,
        labels,
        sample_weights,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 0.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        device = outputs["boundary_logits"].device
        b_logits = outputs["boundary_logits"]
        boundary_labels = labels["boundary_labels"].to(device=device, dtype=torch.long)

        loss_per = F.cross_entropy(b_logits, boundary_labels, weight=class_weights, reduction="none")
        if focal_gamma > 0.0:
            p_t = F.softmax(b_logits.detach(), dim=-1).gather(1, boundary_labels.unsqueeze(1)).squeeze(1)
            loss_per = loss_per * (1.0 - p_t) ** focal_gamma
        return (loss_per * sample_weights).mean()

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        b_pred = outputs["boundary_logits"].argmax(dim=-1).detach().cpu().tolist()
        b_true = gather_by_span_indices(labels["boundary_labels"], span_indices)
        return b_true, b_pred

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        return {
            "boundary_f1": safe_macro_f1(y_true, y_pred),
            "boundary_report": safe_report(y_true, y_pred) if y_true else "N/A",
        }

