"""Pseudo-tête COMPAT — cohérence inter-têtes (pas une tête supervisée à proprement
parler : combine boundary + coarse + role_coarse pour imposer 2 contraintes) :

  A) role -> boundary : un span participant à un rôle SVO est forcément une entité NER.
  B) boundary <-> coarse : P(boundary=1) doit s'aligner avec P(coarse != NONE).

Pas de logits propres, pas de métriques dédiées (jsonl_keys vide).
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from labels import ROLE_COARSE_NONE_ID, COARSE_NONE_ID
from .base import Head


class CompatHead(Head):
    task_key = "compat"
    jsonl_keys: dict[str, str] = {}

    def __init__(self):
        pass

    @classmethod
    def from_model(cls, model) -> "CompatHead":
        return cls()

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {}

    def compute_loss(
        self, outputs, labels, sample_weights,
        class_weights: Optional[torch.Tensor] = None, lambda_compat: float = 0.0, **kwargs: Any,
    ) -> torch.Tensor:
        device = outputs["boundary_logits"].device
        b_logits = outputs["boundary_logits"]
        c_logits = outputs["coarse_logits"]
        role_coarse_labels = labels["role_coarse_labels"].to(device=device, dtype=torch.long)

        role_active_mask = (role_coarse_labels >= 0) & (role_coarse_labels < ROLE_COARSE_NONE_ID)
        if lambda_compat > 0.0 and role_active_mask.any():
            forced_boundary = torch.ones(role_active_mask.sum(), device=device, dtype=torch.long)
            loss_compat_rb = (
                F.cross_entropy(b_logits[role_active_mask], forced_boundary, reduction="none")
                * sample_weights[role_active_mask]
            ).mean()
        else:
            loss_compat_rb = torch.tensor(0.0, device=device)

        if lambda_compat > 0.0 and b_logits.size(0) > 0:
            p_boundary_pos = torch.softmax(b_logits.detach(), dim=-1)[:, 1]
            p_coarse_entity = 1.0 - torch.softmax(c_logits, dim=-1)[:, COARSE_NONE_ID]
            loss_compat_bc = F.mse_loss(p_coarse_entity, p_boundary_pos)
        else:
            loss_compat_bc = torch.tensor(0.0, device=device)

        return loss_compat_rb + loss_compat_bc

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        return [], []

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        return {}

