"""Interface commune `Head` — implémentée par chaque tête (un fichier par tête).

Design :
    - Les couches nn.Module (nn.Linear, nn.Sequential, ...) restent déclarées
      et possédées par `SpanMultiTaskModel` (même noms qu'avant le refactor)
      pour rester 100% compatible avec les checkpoints .pt déjà entraînés.
    - Chaque `Head` reçoit ces couches par injection (`from_model(model)`)
      et encapsule uniquement la LOGIQUE : forward applicatif, loss, métriques,
      dump JSONL. Ce n'est PAS un nn.Module (pas de double-enregistrement de
      paramètres).
"""
from __future__ import annotations

import abc
from typing import Any, Optional

import torch


class Head(abc.ABC):
    """Interface qu'une tête du modèle multi-tâche doit implémenter."""

    #: clé identifiant la tête — doit matcher TASK_KEYS de loss_weighting.py
    task_key: str = "base"

    #: mapping metric_interne -> clé JSONL compacte (défaut = pas de dump)
    jsonl_keys: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    @abc.abstractmethod
    def from_model(cls, model) -> "Head":
        """Construit la tête à partir des couches nn.Module déjà déclarées sur `model`."""

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def forward(self, features: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Calcule les logits de cette tête à partir des features partagées.

        `features` contient au minimum : span_h, span_h_ner, span_h_role,
        hidden, span_indices, span_batch_idx, span_positions, device, N.
        Retourne un dict fusionné ensuite dans `outputs` du modèle.
        """

    # ------------------------------------------------------------------
    # Loss (retourne la loss RAW, non pondérée par lambda — la pondération
    # /ramp/dynamic-weighting reste gérée globalement par loss_weighting.py)
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        sample_weights: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Retourne un tenseur scalaire : la loss RAW de cette tête."""

    def reset_epoch(self) -> None:
        """Hook optionnel : réinitialise un état interne accumulé (diagnostics
        cross-head par exemple). No-op par défaut."""
        return None

    # ------------------------------------------------------------------
    # Collecte des prédictions pour les métriques d'epoch
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def collect(
        self,
        outputs: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        span_indices: Optional[torch.Tensor],
        context: Optional[dict] = None,
    ) -> tuple[list, list]:
        """Extrait (y_true, y_pred) pour ce batch — accumulés sur l'epoch entière."""

    # ------------------------------------------------------------------
    # Métriques finales
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def compute_metrics(
        self, y_true: list, y_pred: list, split_name: Optional[str] = None
    ) -> dict:
        """Calcule les métriques finales (F1 macro, reports texte...) pour cette tête."""

    # ------------------------------------------------------------------
    # Dump JSONL compact
    # ------------------------------------------------------------------
    def dump_metrics(self, metrics: dict) -> dict:
        """Filtre + renomme les métriques de cette tête pour le JSONL compact.
        Comportement par défaut : applique `jsonl_keys`. Surchargeable si besoin."""
        out = {}
        for src, dst in self.jsonl_keys.items():
            if src in metrics:
                out[dst] = metrics[src]
        return out

