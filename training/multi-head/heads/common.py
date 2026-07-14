"""Helpers partagés par les têtes (metrics, focal loss, masking utilitaires)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report


def safe_macro_f1(y_true: list, y_pred: list, labels: Optional[list] = None) -> float:
    if not y_true:
        return 0.0
    return f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)


def safe_report(
    y_true: list,
    y_pred: list,
    labels: Optional[list] = None,
    target_names: Optional[list] = None,
) -> str:
    if not y_true:
        return "N/A"
    return classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, digits=3, zero_division=0
    )


def masked_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Cross-entropy pondérée par sample_weights, appliquée seulement où `mask` est vrai.
    Supporte une focal loss optionnelle (gamma>0) et des class_weights optionnels.
    Retourne un tenseur scalaire 0.0 si le mask est vide."""
    if device is None:
        device = logits.device
    if not mask.any():
        return torch.tensor(0.0, device=device)

    loss_per = F.cross_entropy(
        logits[mask], labels[mask], weight=class_weights, reduction="none"
    )
    if focal_gamma > 0.0:
        p_t = F.softmax(logits[mask].detach(), dim=-1).gather(1, labels[mask].unsqueeze(1)).squeeze(1)
        loss_per = loss_per * (1.0 - p_t) ** focal_gamma
    return (loss_per * sample_weights[mask]).mean()


def verbfam_ce_loss(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor],
    none_id: int,
    vt_mask: torch.Tensor,
    device: torch.device,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CE loss sur les spans verb_trigger (`vt_mask`) avec label valide (!= none_id).
    Utilisé par les 5 têtes VerbFam (family/family_fine/polarity/aspect/source)."""
    if not vt_mask.any() or labels is None:
        return torch.tensor(0.0, device=device)
    labels = labels.to(device=device, dtype=torch.long)
    m = vt_mask & (labels != none_id)
    if not m.any():
        return torch.tensor(0.0, device=device)
    w = weight.to(device) if weight is not None else None
    return F.cross_entropy(logits[m], labels[m], weight=w, reduction="mean")


def gather_by_span_indices(tensor: torch.Tensor, span_indices: Optional[torch.Tensor]) -> list:
    """Retourne la liste des valeurs de `tensor` alignées sur `span_indices` (ou tout si None)."""
    if span_indices is not None:
        si_cpu = span_indices.detach().cpu().to(dtype=torch.long)
        return tensor.detach().cpu()[si_cpu].tolist()
    return tensor.detach().cpu().tolist()

