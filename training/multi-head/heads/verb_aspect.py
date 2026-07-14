"""Tête VERB_ASPECT — accompli / inaccompli (verb_trigger uniquement)."""
from __future__ import annotations

from typing import Any, Optional

import torch

from labels import NUM_VERB_ASPECT, VERB_ASPECT_NONE_ID
from .base import Head
from .common import safe_macro_f1, verbfam_ce_loss


class VerbAspectHead(Head):
    task_key = "verb_aspect"
    jsonl_keys = {"verb_aspect_macro_f1": "verb_aspect_f1"}

    def __init__(self, linear):
        self.linear = linear

    @classmethod
    def from_model(cls, model) -> "VerbAspectHead":
        return cls(model.verb_aspect_head)

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {"verb_aspect_logits": self.linear(features["span_h_vf"])}

    def compute_loss(
        self, outputs, labels, sample_weights,
        class_weights: Optional[torch.Tensor] = None, **kwargs: Any,
    ) -> torch.Tensor:
        logits = outputs["verb_aspect_logits"]
        device = logits.device
        vt_mask = labels["svo_boundary_labels"].to(device=device, dtype=torch.long) == 1
        return verbfam_ce_loss(
            logits, labels.get("verb_aspect_labels"), VERB_ASPECT_NONE_ID, vt_mask, device,
            weight=class_weights,
        )

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        vasp_pred_raw = outputs["verb_aspect_logits"].argmax(dim=-1).detach().cpu().tolist()
        n_all = outputs["verb_aspect_logits"].size(0)
        verb_aspect_labels = labels.get("verb_aspect_labels")
        if span_indices is not None:
            si_cpu = span_indices.detach().cpu().to(dtype=torch.long)
            vasp_true = (
                verb_aspect_labels.detach().cpu()[si_cpu].tolist()
                if verb_aspect_labels is not None else [VERB_ASPECT_NONE_ID] * len(si_cpu)
            )
        else:
            vasp_true = (
                verb_aspect_labels.detach().cpu().tolist()
                if verb_aspect_labels is not None else [VERB_ASPECT_NONE_ID] * n_all
            )
        out_true, out_pred = [], []
        for vat, vap in zip(vasp_true, vasp_pred_raw):
            if vat < VERB_ASPECT_NONE_ID:
                out_true.append(vat)
                out_pred.append(vap)
        return out_true, out_pred

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        present = set(y_true)
        return {
            "verb_aspect_macro_f1": safe_macro_f1(
                y_true, y_pred, labels=[l for l in range(NUM_VERB_ASPECT) if l in present]
            ) if y_true else 0.0,
        }

