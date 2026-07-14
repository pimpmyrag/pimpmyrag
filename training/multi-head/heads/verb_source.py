"""Tête VERB_SOURCE — source de l'assertion (locuteur/tiers/générique) (verb_trigger uniquement)."""
from __future__ import annotations

from typing import Any, Optional

import torch

from labels import NUM_VERB_SOURCE, VERB_SOURCE_NONE_ID
from .base import Head
from .common import safe_macro_f1, verbfam_ce_loss


class VerbSourceHead(Head):
    task_key = "verb_source"
    jsonl_keys = {"verb_source_macro_f1": "verb_source_f1"}

    def __init__(self, linear):
        self.linear = linear

    @classmethod
    def from_model(cls, model) -> "VerbSourceHead":
        return cls(model.verb_source_head)

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {"verb_source_logits": self.linear(features["span_h_vf"])}

    def compute_loss(
        self, outputs, labels, sample_weights,
        class_weights: Optional[torch.Tensor] = None, **kwargs: Any,
    ) -> torch.Tensor:
        logits = outputs["verb_source_logits"]
        device = logits.device
        vt_mask = labels["svo_boundary_labels"].to(device=device, dtype=torch.long) == 1
        return verbfam_ce_loss(
            logits, labels.get("verb_source_labels"), VERB_SOURCE_NONE_ID, vt_mask, device,
            weight=class_weights,
        )

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        vsrc_pred_raw = outputs["verb_source_logits"].argmax(dim=-1).detach().cpu().tolist()
        n_all = outputs["verb_source_logits"].size(0)
        verb_source_labels = labels.get("verb_source_labels")
        if span_indices is not None:
            si_cpu = span_indices.detach().cpu().to(dtype=torch.long)
            vsrc_true = (
                verb_source_labels.detach().cpu()[si_cpu].tolist()
                if verb_source_labels is not None else [VERB_SOURCE_NONE_ID] * len(si_cpu)
            )
        else:
            vsrc_true = (
                verb_source_labels.detach().cpu().tolist()
                if verb_source_labels is not None else [VERB_SOURCE_NONE_ID] * n_all
            )
        out_true, out_pred = [], []
        for vst, vsp in zip(vsrc_true, vsrc_pred_raw):
            if vst < VERB_SOURCE_NONE_ID:
                out_true.append(vst)
                out_pred.append(vsp)
        return out_true, out_pred

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        present = set(y_true)
        return {
            "verb_source_macro_f1": safe_macro_f1(
                y_true, y_pred, labels=[l for l in range(NUM_VERB_SOURCE) if l in present]
            ) if y_true else 0.0,
        }

