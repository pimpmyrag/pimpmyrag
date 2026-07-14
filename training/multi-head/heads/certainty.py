"""Tête CERTAINTY — certain / probable / doute (verb_trigger uniquement)."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from labels import NUM_CERTAINTY
from .base import Head
from .common import safe_macro_f1, safe_report, gather_by_span_indices


class CertaintyHead(Head):
    task_key = "certainty"
    jsonl_keys = {"certainty_macro_f1": "certainty_f1"}

    def __init__(self, linear):
        self.linear = linear

    @classmethod
    def from_model(cls, model) -> "CertaintyHead":
        return cls(model.certainty_head)

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {"certainty_logits": self.linear(features["span_h"])}

    def compute_loss(self, outputs, labels, sample_weights, class_weights=None, **kwargs: Any) -> torch.Tensor:
        device = outputs["certainty_logits"].device
        cert_logits = outputs["certainty_logits"]
        certainty_labels = labels["certainty_labels"].to(device=device, dtype=torch.long)
        cert_mask = (certainty_labels >= 0) & (certainty_labels < cert_logits.size(-1))
        if not cert_mask.any():
            return torch.tensor(0.0, device=device)
        loss_per = F.cross_entropy(
            cert_logits[cert_mask], certainty_labels[cert_mask], weight=class_weights, reduction="none"
        )
        return (loss_per * sample_weights[cert_mask]).mean()

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        cert_pred = outputs["certainty_logits"].argmax(dim=-1).detach().cpu().tolist()
        cert_true = gather_by_span_indices(labels["certainty_labels"], span_indices)
        out_true, out_pred = [], []
        for ct, cp in zip(cert_true, cert_pred):
            if ct < NUM_CERTAINTY:
                out_true.append(ct)
                out_pred.append(cp)
        return out_true, out_pred

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        present = set(y_true)
        return {
            "certainty_macro_f1": safe_macro_f1(
                y_true, y_pred, labels=[l for l in range(NUM_CERTAINTY) if l in present]
            ) if y_true else 0.0,
        }

