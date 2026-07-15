"""Tête ATTRIBUTES — 5 attributs binaires transverses (v9), agrégés sous un
seul lambda_attributes (cf. TASK_KEYS = "attributes" dans loss_weighting.py).

Prédits sur les spans NER (même population que morpho) :
    animacy    (inanimate/animate)   — humain + animal
    living     (non_living/living)   — + végétal (biologique)
    abstract   (concrete/abstract)   — ex-famille coarse ABSTRACT
    dynamicity (stative/dynamic)     — supervisé sur EVENT uniquement (sinon NONE)
    work       (non_work/work)       — ex-famille coarse WORK

Cinq couches propres sur le modèle : `model.animacy_head`, `model.living_head`,
`model.abstract_head`, `model.dynamicity_head`, `model.work_head`.

Le gold des 5 attributs est DÉRIVÉ du label fine (labels_v9.derive_attributes)
au build → zéro ré-annotation. Le sentinel NONE (= nombre de classes) masque
automatiquement les spans non supervisés (négatifs, ou dynamicity hors EVENT).
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from labels_v9 import (
    NUM_ANIMACY, NUM_LIVING, NUM_ABSTRACT, NUM_DYNAMICITY, NUM_WORK,
)
from .base import Head
from .common import safe_macro_f1, gather_by_span_indices


# (label_key, logits_key, num_classes) — ordre stable
_ATTRS = [
    ("animacy_labels",    "animacy_logits",    NUM_ANIMACY),
    ("living_labels",     "living_logits",     NUM_LIVING),
    ("abstract_labels",   "abstract_logits",   NUM_ABSTRACT),
    ("dynamicity_labels", "dynamicity_logits", NUM_DYNAMICITY),
    ("work_labels",       "work_logits",       NUM_WORK),
]


class AttributesHead(Head):
    task_key = "attributes"
    jsonl_keys = {
        "animacy_macro_f1":    "animacy_f1",
        "living_macro_f1":     "living_f1",
        "abstract_macro_f1":   "abstract_f1",
        "dynamicity_macro_f1": "dynamicity_f1",
        "work_macro_f1":       "work_f1",
    }

    def __init__(self, animacy_linear, living_linear, abstract_linear,
                 dynamicity_linear, work_linear):
        self.linears = {
            "animacy_logits":    animacy_linear,
            "living_logits":     living_linear,
            "abstract_logits":   abstract_linear,
            "dynamicity_logits": dynamicity_linear,
            "work_logits":       work_linear,
        }
        # accumulateurs internes pour les attributs secondaires (animacy = primaire)
        self._acc_true: dict[str, list] = {}
        self._acc_pred: dict[str, list] = {}
        self.reset_epoch()

    @classmethod
    def from_model(cls, model) -> "AttributesHead":
        return cls(
            model.animacy_head, model.living_head, model.abstract_head,
            model.dynamicity_head, model.work_head,
        )

    def reset_epoch(self) -> None:
        self._acc_true = {lk: [] for lk, _, _ in _ATTRS}
        self._acc_pred = {lk: [] for lk, _, _ in _ATTRS}

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        span_h = features["span_h"]
        return {logit_key: lin(span_h) for logit_key, lin in self.linears.items()}

    def compute_loss(self, outputs, labels, sample_weights,
                     class_weights=None, **kwargs: Any) -> torch.Tensor:
        device = outputs["animacy_logits"].device
        total = torch.tensor(0.0, device=device)
        for label_key, logit_key, num_cls in _ATTRS:
            logits = outputs[logit_key]
            y = labels[label_key].to(device=device, dtype=torch.long)
            mask = (y >= 0) & (y < logits.size(-1))   # exclut le sentinel NONE (= num_cls)
            if mask.any():
                total = total + (
                    F.cross_entropy(logits[mask], y[mask], reduction="none")
                    * sample_weights[mask]
                ).mean()
        return total

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        primary_true, primary_pred = [], []
        for i, (label_key, logit_key, num_cls) in enumerate(_ATTRS):
            pred = outputs[logit_key].argmax(dim=-1).detach().cpu().tolist()
            true = gather_by_span_indices(labels[label_key], span_indices)
            for t, p in zip(true, pred):
                if t < num_cls:   # span supervisé pour cet attribut
                    self._acc_true[label_key].append(t)
                    self._acc_pred[label_key].append(p)
                    if i == 0:  # animacy = métrique primaire renvoyée au flux générique
                        primary_true.append(t)
                        primary_pred.append(p)
        return primary_true, primary_pred

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        out = {}
        name_map = {
            "animacy_labels": "animacy", "living_labels": "living",
            "abstract_labels": "abstract", "dynamicity_labels": "dynamicity",
            "work_labels": "work",
        }
        for label_key, _, num_cls in _ATTRS:
            t = self._acc_true[label_key]
            p = self._acc_pred[label_key]
            present = set(t)
            out[f"{name_map[label_key]}_macro_f1"] = (
                safe_macro_f1(t, p, labels=[l for l in range(num_cls) if l in present])
                if t else 0.0
            )
        return out

