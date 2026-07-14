"""Tête SVO_BOUNDARY — détection binaire verbe/pronom déclencheur SVO."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from .base import Head
from .common import safe_macro_f1, safe_report, gather_by_span_indices


class SvoBoundaryHead(Head):
    task_key = "svo_boundary"
    jsonl_keys = {"svo_boundary_f1": "svo_boundary_f1"}

    def __init__(self, linear):
        self.linear = linear

    @classmethod
    def from_model(cls, model) -> "SvoBoundaryHead":
        return cls(model.svo_boundary_head)

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {"svo_boundary_logits": self.linear(features["span_h"])}

    def compute_loss(self, outputs, labels, sample_weights, class_weights=None, **kwargs: Any) -> torch.Tensor:
        device = outputs["svo_boundary_logits"].device
        svo_b_logits = outputs["svo_boundary_logits"]
        svo_boundary_labels = labels["svo_boundary_labels"].to(device=device, dtype=torch.long)
        loss_per = F.cross_entropy(svo_b_logits, svo_boundary_labels, reduction="none")
        return (loss_per * sample_weights).mean()

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        svob_pred = outputs["svo_boundary_logits"].argmax(dim=-1).detach().cpu().tolist()
        svob_true = gather_by_span_indices(labels["svo_boundary_labels"], span_indices)
        return svob_true, svob_pred

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        return {
            "svo_boundary_f1": safe_macro_f1(y_true, y_pred),
            "svo_boundary_report": safe_report(
                y_true, y_pred, labels=[0, 1], target_names=["non_verb", "verb_trigger"]
            ) if y_true else "N/A",
        }

