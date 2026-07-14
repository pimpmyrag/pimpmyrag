"""Tête VERB_FAMILY — 12 familles sémantiques du verbe (verb_trigger uniquement).

Couche propre : `model.verb_family_head` (Linear 256 -> NUM_VERB_FAMILY).
Prend en entrée `span_h_vf` = features issues de `model.verb_family_mlp`
(trunk partagé avec family_fine/polarity/aspect/source, calculé une seule
fois côté modèle et injecté dans `features`).
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from labels import VERB_FAMILY_LABELS, NUM_VERB_FAMILY, VERB_FAMILY_NONE_ID
from .base import Head
from .common import safe_macro_f1, safe_report, verbfam_ce_loss


class VerbFamilyHead(Head):
    task_key = "verb_family"
    jsonl_keys = {"verb_family_macro_f1": "verb_family_f1"}

    def __init__(self, linear):
        self.linear = linear

    @classmethod
    def from_model(cls, model) -> "VerbFamilyHead":
        return cls(model.verb_family_head)

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {"verb_family_logits": self.linear(features["span_h_vf"])}

    def compute_loss(
        self, outputs, labels, sample_weights,
        class_weights: Optional[torch.Tensor] = None, **kwargs: Any,
    ) -> torch.Tensor:
        logits = outputs["verb_family_logits"]
        device = logits.device
        vt_mask = labels["svo_boundary_labels"].to(device=device, dtype=torch.long) == 1
        return verbfam_ce_loss(
            logits, labels.get("verb_family_labels"), VERB_FAMILY_NONE_ID, vt_mask, device,
            weight=class_weights,
        )

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        vfam_pred_raw = outputs["verb_family_logits"].argmax(dim=-1).detach().cpu().tolist()
        n_all = outputs["verb_family_logits"].size(0)
        verb_family_labels = labels.get("verb_family_labels")
        if span_indices is not None:
            si_cpu = span_indices.detach().cpu().to(dtype=torch.long)
            vfam_true = (
                verb_family_labels.detach().cpu()[si_cpu].tolist()
                if verb_family_labels is not None else [VERB_FAMILY_NONE_ID] * len(si_cpu)
            )
        else:
            vfam_true = (
                verb_family_labels.detach().cpu().tolist()
                if verb_family_labels is not None else [VERB_FAMILY_NONE_ID] * n_all
            )

        out_true, out_pred = [], []
        for vft, vfp in zip(vfam_true, vfam_pred_raw):
            if vft < VERB_FAMILY_NONE_ID:
                out_true.append(vft)
                out_pred.append(vfp)
        return out_true, out_pred

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        present = set(y_true)
        return {
            "verb_family_macro_f1": safe_macro_f1(
                y_true, y_pred, labels=[l for l in range(NUM_VERB_FAMILY) if l in present]
            ) if y_true else 0.0,
            "verb_family_report": safe_report(
                y_true, y_pred,
                labels=[l for l in range(NUM_VERB_FAMILY) if l in present],
                target_names=[VERB_FAMILY_LABELS[l] for l in range(NUM_VERB_FAMILY) if l in present],
            ) if y_true else "N/A",
        }

