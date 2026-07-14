"""Tête VERB_FAMILY_FINE — 38 sous-catégories, masquées (soft-mask) par verb_family.

Couche propre : `model.verb_family_fine_head` (Linear 256 -> NUM_VERB_FAMILY_FINE).
Le masquage cascade verb_family -> verb_family_fine (soft, via
`model.verb_family_fine_mask`) est calculé dans le modèle car il agrège deux
têtes ; la loss utilise les logits RAW (non masqués) comme pour fine/coarse NER.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from labels import VERB_FAMILY_FINE_LABELS, NUM_VERB_FAMILY_FINE, VERB_FAMILY_FINE_NONE_ID
from .base import Head
from .common import safe_macro_f1, safe_report, verbfam_ce_loss


class VerbFamilyFineHead(Head):
    task_key = "verb_family_fine"
    jsonl_keys = {"verb_family_fine_macro_f1": "verb_family_fine_f1"}

    def __init__(self, linear):
        self.linear = linear

    @classmethod
    def from_model(cls, model) -> "VerbFamilyFineHead":
        return cls(model.verb_family_fine_head)

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {"verb_family_fine_logits_raw": self.linear(features["span_h_vf"])}

    def compute_loss(self, outputs, labels, sample_weights, class_weights=None, **kwargs: Any) -> torch.Tensor:
        logits = outputs["verb_family_fine_logits_raw"]
        device = logits.device
        vt_mask = labels["svo_boundary_labels"].to(device=device, dtype=torch.long) == 1
        return verbfam_ce_loss(
            logits, labels.get("verb_family_fine_labels"), VERB_FAMILY_FINE_NONE_ID, vt_mask, device,
        )

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        # métriques calculées sur la version MASQUÉE (comportement d'inférence réel)
        vfam_fine_pred_raw = outputs["verb_family_fine_logits"].argmax(dim=-1).detach().cpu().tolist()
        n_all = outputs["verb_family_fine_logits"].size(0)
        verb_family_fine_labels = labels.get("verb_family_fine_labels")
        if span_indices is not None:
            si_cpu = span_indices.detach().cpu().to(dtype=torch.long)
            vfam_fine_true = (
                verb_family_fine_labels.detach().cpu()[si_cpu].tolist()
                if verb_family_fine_labels is not None else [VERB_FAMILY_FINE_NONE_ID] * len(si_cpu)
            )
        else:
            vfam_fine_true = (
                verb_family_fine_labels.detach().cpu().tolist()
                if verb_family_fine_labels is not None else [VERB_FAMILY_FINE_NONE_ID] * n_all
            )

        out_true, out_pred = [], []
        for vft, vfp in zip(vfam_fine_true, vfam_fine_pred_raw):
            if vft < VERB_FAMILY_FINE_NONE_ID:
                out_true.append(vft)
                out_pred.append(vfp)
        return out_true, out_pred

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        present = set(y_true)
        return {
            "verb_family_fine_macro_f1": safe_macro_f1(
                y_true, y_pred, labels=[l for l in range(NUM_VERB_FAMILY_FINE) if l in present]
            ) if y_true else 0.0,
            "verb_family_fine_report": safe_report(
                y_true, y_pred,
                labels=[l for l in range(NUM_VERB_FAMILY_FINE) if l in present],
                target_names=[VERB_FAMILY_FINE_LABELS[l] for l in range(NUM_VERB_FAMILY_FINE) if l in present],
            ) if y_true else "N/A",
        }

