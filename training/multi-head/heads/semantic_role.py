"""Tête SEMANTIC_ROLE — 19 rôles sémantiques (AGENT/PATIENT/CONTENT/SOURCE/LOCATION/...).

Couche propre : `model.semantic_role_head`.
Supervisée sur tous les spans NER (pas seulement OBLIQUE), sauf SKIP_ID.
Deux jeux de métriques :
  - `semantic_role` : TOUS les spans supervisés (rot valide, != SKIP), aligné sur la
    supervision réelle (la tête est entraînée sur tous les spans, pas seulement OBLIQ).
    ⚠️ Avant v8.24b ce score filtrait role_coarse GOLD == OBLIQ, ce qui excluait de fait
    AGENT (SUBJ) et PATIENT (OBJ) → f1 artificiellement ~0.1 sur ces classes majeures.
  - `semantic_role_cascaded` : conditionné sur role_coarse_from_role PRÉDIT == OBLIQ
    (diagnostic cascade legacy, sous-ensemble OBLIQ uniquement) — stateful ici.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from labels import (
    SEMANTIC_ROLE_LABELS, NUM_SEMANTIC_ROLE,
    SEMANTIC_ROLE_SKIP_ID, SEMANTIC_ROLE_NONE_ID,
    ROLE_COARSE2ID,
)
from .base import Head
from .common import safe_macro_f1, safe_report, gather_by_span_indices

_OBLIQ_RC = ROLE_COARSE2ID["OBLIQ"]


class SemanticRoleHead(Head):
    task_key = "semantic_role"
    jsonl_keys = {
        "semantic_role_macro_f1": "semantic_role_f1",
        "semantic_role_cascaded_macro_f1": "semantic_role_cascaded_f1",
    }

    def __init__(self, linear):
        self.linear = linear
        self._cascaded_true: list = []
        self._cascaded_pred: list = []

    @classmethod
    def from_model(cls, model) -> "SemanticRoleHead":
        return cls(model.semantic_role_head)

    def reset_epoch(self) -> None:
        self._cascaded_true = []
        self._cascaded_pred = []

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {"semantic_role_logits": self.linear(features["span_h_role"])}

    def compute_loss(
        self, outputs, labels, sample_weights,
        class_weights: Optional[torch.Tensor] = None, **kwargs: Any,
    ) -> torch.Tensor:
        device = outputs["semantic_role_logits"].device
        sr_logits = outputs["semantic_role_logits"]
        semantic_role_labels = labels["semantic_role_labels"].to(device=device, dtype=torch.long)
        sr_mask = (
            (semantic_role_labels >= 0)
            & (semantic_role_labels < sr_logits.size(-1))
            & (semantic_role_labels != SEMANTIC_ROLE_SKIP_ID)
        )
        if not sr_mask.any():
            return torch.tensor(0.0, device=device)
        _w = class_weights.to(device) if class_weights is not None else None
        loss_per = F.cross_entropy(
            sr_logits[sr_mask], semantic_role_labels[sr_mask], weight=_w, reduction="none"
        )
        return (loss_per * sample_weights[sr_mask]).mean()

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        ro_pred = outputs["semantic_role_logits"].argmax(dim=-1).detach().cpu().tolist()
        ro_true = gather_by_span_indices(labels["semantic_role_labels"], span_indices)
        rc_from_role_pred = outputs["role_coarse_from_role_logits"].argmax(dim=-1).detach().cpu().tolist()

        out_true, out_pred = [], []
        # Métrique principale : TOUS les spans supervisés (rot valide, hors SKIP).
        # PAS de filtre role_coarse == OBLIQ : sinon AGENT (SUBJ) / PATIENT (OBJ) sont exclus.
        for rot, rop in zip(ro_true, ro_pred):
            if rot >= 0 and rot < SEMANTIC_ROLE_SKIP_ID:
                out_true.append(rot)
                out_pred.append(rop)

        # Diagnostic cascade legacy : conditionné sur role_coarse PRÉDIT == OBLIQ.
        for rot, rop, rcfr in zip(ro_true, ro_pred, rc_from_role_pred):
            if rcfr == _OBLIQ_RC and rot >= 0 and rot < SEMANTIC_ROLE_SKIP_ID:
                self._cascaded_true.append(rot)
                self._cascaded_pred.append(rop)

        return out_true, out_pred

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        # Exclure NONE (id=18) et SKIP (id=19) des labels actifs dans les métriques
        active_ids = [l for l in range(NUM_SEMANTIC_ROLE) if l != SEMANTIC_ROLE_NONE_ID]
        present = set(y_true)
        present_cascaded = set(self._cascaded_true)
        return {
            "semantic_role_macro_f1": safe_macro_f1(
                y_true, y_pred, labels=[l for l in active_ids if l in present]
            ) if y_true else 0.0,
            "semantic_role_report": safe_report(
                y_true, y_pred,
                labels=[l for l in active_ids if l in present],
                target_names=[SEMANTIC_ROLE_LABELS[l] for l in active_ids if l in present],
            ) if y_true else "N/A",
            "semantic_role_cascaded_macro_f1": safe_macro_f1(
                self._cascaded_true, self._cascaded_pred,
                labels=[l for l in active_ids if l in present_cascaded],
            ) if self._cascaded_true else 0.0,
        }

