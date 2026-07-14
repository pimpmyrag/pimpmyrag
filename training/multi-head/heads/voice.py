"""Tête VOICE — ACTIVE / PASSIVE (verb_trigger uniquement)."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from labels import NUM_VOICE
from .base import Head
from .common import safe_macro_f1, safe_report, gather_by_span_indices


class VoiceHead(Head):
    task_key = "voice"
    jsonl_keys = {"voice_macro_f1": "voice_f1"}

    def __init__(self, linear):
        self.linear = linear

    @classmethod
    def from_model(cls, model) -> "VoiceHead":
        return cls(model.voice_head)

    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {"voice_logits": self.linear(features["span_h"])}

    def compute_loss(self, outputs, labels, sample_weights, class_weights=None, **kwargs: Any) -> torch.Tensor:
        device = outputs["voice_logits"].device
        voice_logits = outputs["voice_logits"]
        voice_labels = labels["voice_labels"].to(device=device, dtype=torch.long)
        voice_mask = (voice_labels >= 0) & (voice_labels < voice_logits.size(-1))
        if not voice_mask.any():
            return torch.tensor(0.0, device=device)
        loss_per = F.cross_entropy(voice_logits[voice_mask], voice_labels[voice_mask], reduction="none")
        return (loss_per * sample_weights[voice_mask]).mean()

    def collect(self, outputs, labels, span_indices, context: Optional[dict] = None):
        voice_pred = outputs["voice_logits"].argmax(dim=-1).detach().cpu().tolist()
        voice_true = gather_by_span_indices(labels["voice_labels"], span_indices)
        out_true, out_pred = [], []
        for vt, vp in zip(voice_true, voice_pred):
            if vt < NUM_VOICE:
                out_true.append(vt)
                out_pred.append(vp)
        return out_true, out_pred

    def compute_metrics(self, y_true, y_pred, split_name=None) -> dict:
        return {"voice_macro_f1": safe_macro_f1(y_true, y_pred) if y_true else 0.0}

