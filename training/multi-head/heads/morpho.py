"""Tête MORPHO — genre + nombre + personne (agrégés sous un seul lambda_morpho,
cf. TASK_KEYS = "morpho" dans loss_weighting.py). Trois couches propres :
`model.gender_head`, `model.number_head`, `model.person_head`.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from labels import NUM_GENDER, NUM_NUMBER, NUM_PERSON
from .base import Head
from .common import safe_macro_f1, gather_by_span_indices


class MorphoHead(Head):
    task_key = "morpho"
    jsonl_keys = {
        "gender_macro_f1": "gender_f1",
        "number_macro_f1": "number_f1",
        "person_macro_f1": "person_f1",
    }

    def __init__(self, gender_linear, number_linear, person_linear):
        self.gender_linear = gender_linear
        self.number_linear = number_linear
        self.person_linear = person_linear
        self._number_true: list = []
        self._number_pred: list = []
        self._person_true: list = []
        self._person_pred: list = []

    @classmethod
    def from_model(cls, model) -> "MorphoHead":
        return cls(model.gender_head, model.number_head, model.person_head)

    def reset_epoch(self) -> None:
        self._number_true = []
        self._number_pred = []
        self._person_true = []
        self._person_pred = []

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        span_h = features["span_h"]
        return {
            "gender_logits": self.gender_linear(span_h),
            "number_logits": self.number_linear(span_h),
            "person_logits": self.person_linear(span_h),
        }

    def compute_loss(self, outputs, labels, sample_weights, class_weights=None, **kwargs: Any) -> torch.Tensor:
        device = outputs["gender_logits"].device
        g_logits, n_logits, p_logits = outputs["gender_logits"], outputs["number_logits"], outputs["person_logits"]
        gender_labels = labels["gender_labels"].to(device=device, dtype=torch.long)
        number_labels = labels["number_labels"].to(device=device, dtype=torch.long)
        person_labels = labels["person_labels"].to(device=device, dtype=torch.long)

        gender_mask = (gender_labels >= 0) & (gender_labels < g_logits.size(-1))
        number_mask = (number_labels >= 0) & (number_labels < n_logits.size(-1))
        person_mask = (person_labels >= 0) & (person_labels < p_logits.size(-1))

        loss_gender = (
            F.cross_entropy(g_logits[gender_mask], gender_labels[gender_mask], reduction="none")
            * sample_weights[gender_mask]
        ).mean() if gender_mask.any() else torch.tensor(0.0, device=device)
        loss_number = (
            F.cross_entropy(n_logits[number_mask], number_labels[number_mask], reduction="none")
            * sample_weights[number_mask]
        ).mean() if number_mask.any() else torch.tensor(0.0, device=device)
        loss_person = (
            F.cross_entropy(p_logits[person_mask], person_labels[person_mask], reduction="none")
            * sample_weights[person_mask]
        ).mean() if person_mask.any() else torch.tensor(0.0, device=device)

        return loss_gender + loss_number + loss_person

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        gender_pred = outputs["gender_logits"].argmax(dim=-1).detach().cpu().tolist()
        number_pred = outputs["number_logits"].argmax(dim=-1).detach().cpu().tolist()
        person_pred = outputs["person_logits"].argmax(dim=-1).detach().cpu().tolist()
        gender_true = gather_by_span_indices(labels["gender_labels"], span_indices)
        number_true = gather_by_span_indices(labels["number_labels"], span_indices)
        person_true = gather_by_span_indices(labels["person_labels"], span_indices)

        out_gender_true, out_gender_pred = [], []
        for gt, gp, nt, np_, pt, pp in zip(
            gender_true, gender_pred, number_true, number_pred, person_true, person_pred
        ):
            if gt < NUM_GENDER:
                out_gender_true.append(gt)
                out_gender_pred.append(gp)
            if nt < NUM_NUMBER:
                self._number_true.append(nt)
                self._number_pred.append(np_)
            if pt < NUM_PERSON:
                self._person_true.append(pt)
                self._person_pred.append(pp)
        return out_gender_true, out_gender_pred

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        present_g = set(y_true)
        present_n = set(self._number_true)
        present_p = set(self._person_true)
        return {
            "gender_macro_f1": safe_macro_f1(
                y_true, y_pred, labels=[l for l in range(NUM_GENDER) if l in present_g]
            ) if y_true else 0.0,
            "number_macro_f1": safe_macro_f1(
                self._number_true, self._number_pred,
                labels=[l for l in range(NUM_NUMBER) if l in present_n],
            ) if self._number_true else 0.0,
            "person_macro_f1": safe_macro_f1(
                self._person_true, self._person_pred,
                labels=[l for l in range(NUM_PERSON) if l in present_p],
            ) if self._person_true else 0.0,
        }

