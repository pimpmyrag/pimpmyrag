"""Tête ROLE_COARSE — rôle syntaxique SUBJ/OBJ/OBLIQ/APPOS (+ OTHER hors-loss).

Couche propre : `model.role_coarse_head`.
Inclut aussi le diagnostic `role_coarse_from_role` (dérivé du role_head legacy
via logsumexp par groupe, calculé côté modèle car il agrège une AUTRE tête) —
gardé ici car c'est une comparaison directe avec role_coarse, donc un
diagnostic propre à cette tête.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from labels import (
    ROLE_COARSE_LABELS, NUM_ROLE_COARSE, ROLE_COARSE_NONE_ID, ROLE_COARSE_OTHER_ID,
)
from .base import Head
from .common import safe_macro_f1, safe_report, gather_by_span_indices


class RoleCoarseHead(Head):
    task_key = "role_coarse"
    jsonl_keys = {
        "role_coarse_macro_f1": "role_coarse_f1",
        "role_coarse_from_role_macro_f1": "role_coarse_from_role_f1",
    }

    def __init__(self, linear):
        self.linear = linear
        self._from_role_true: list = []
        self._from_role_pred: list = []

    @classmethod
    def from_model(cls, model) -> "RoleCoarseHead":
        return cls(model.role_coarse_head)

    def reset_epoch(self) -> None:
        self._from_role_true = []
        self._from_role_pred = []

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {"role_coarse_logits": self.linear(features["span_h_role"])}

    def compute_loss(
        self, outputs, labels, sample_weights,
        class_weights: Optional[torch.Tensor] = None, **kwargs: Any,
    ) -> torch.Tensor:
        device = outputs["role_coarse_logits"].device
        rc_logits = outputs["role_coarse_logits"]
        role_coarse_labels = labels["role_coarse_labels"].to(device=device, dtype=torch.long)
        rc_mask = (
            (role_coarse_labels >= 0)
            & (role_coarse_labels < rc_logits.size(-1))
            & (role_coarse_labels != ROLE_COARSE_OTHER_ID)
        )
        if not rc_mask.any():
            return torch.tensor(0.0, device=device)
        _w = class_weights.to(device) if class_weights is not None else None
        loss_per = F.cross_entropy(
            rc_logits[rc_mask], role_coarse_labels[rc_mask], weight=_w, reduction="none"
        )
        return (loss_per * sample_weights[rc_mask]).mean()

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        rc_pred = outputs["role_coarse_logits"].argmax(dim=-1).detach().cpu().tolist()
        rc_from_role_pred = outputs["role_coarse_from_role_logits"].argmax(dim=-1).detach().cpu().tolist()
        rc_true = gather_by_span_indices(labels["role_coarse_labels"], span_indices)

        out_true, out_pred = [], []
        for rct, rcp, rcfr in zip(rc_true, rc_pred, rc_from_role_pred):
            if 0 <= rct < ROLE_COARSE_NONE_ID and rct != ROLE_COARSE_OTHER_ID:
                out_true.append(rct)
                out_pred.append(rcp)
                self._from_role_true.append(rct)
                self._from_role_pred.append(rcfr)
        return out_true, out_pred

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        present = set(y_true)
        metrics = {
            "role_coarse_macro_f1": safe_macro_f1(
                y_true, y_pred, labels=[l for l in range(NUM_ROLE_COARSE) if l in present]
            ) if y_true else 0.0,
            "role_coarse_report": safe_report(
                y_true, y_pred,
                labels=[l for l in range(NUM_ROLE_COARSE) if l != ROLE_COARSE_NONE_ID and l in present],
                target_names=[ROLE_COARSE_LABELS[l] for l in range(NUM_ROLE_COARSE)
                              if l != ROLE_COARSE_NONE_ID and l in present],
            ) if y_true else "N/A",
            "role_coarse_from_role_macro_f1": safe_macro_f1(
                self._from_role_true, self._from_role_pred,
                labels=[l for l in range(4) if l in set(self._from_role_true)],
            ) if self._from_role_true else 0.0,
        }
        return metrics

