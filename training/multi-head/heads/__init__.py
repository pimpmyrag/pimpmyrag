"""Package `heads` — une tête = un fichier.

Chaque tête du modèle multi-tâche (boundary, coarse, fine, svo_boundary,
role_coarse, semantic_role, voice, certainty, morpho, verb_family, ...)
implémente l'interface `Head` définie dans `heads.base` :

    - forward(features)               -> dict de logits
    - compute_loss(outputs, labels, sample_weights, class_weights, **kw) -> loss RAW (non pondérée)
    - collect(outputs, labels, span_indices, context) -> (y_true, y_pred) pour accumulation epoch
    - compute_metrics(y_true, y_pred, split_name)      -> dict métriques (F1, reports, ...)
    - dump_metrics(metrics)                            -> dict compact pour le JSONL de suivi

`build_all_heads(model)` construit le registre ordonné des têtes en réutilisant
les couches nn.Module déjà déclarées sur `SpanMultiTaskModel` (préserve les noms
de paramètres -> compatible avec les checkpoints existants).
"""
from __future__ import annotations

from .base import Head
from .boundary import BoundaryHead
from .coarse import CoarseHead
from .fine import FineHead
from .svo_boundary import SvoBoundaryHead
from .svo import SvoHead
from .role_coarse import RoleCoarseHead
from .semantic_role import SemanticRoleHead
from .role_legacy import RoleLegacyHead
from .voice import VoiceHead
from .certainty import CertaintyHead
from .morpho import MorphoHead
from .verb_ptr import VerbPtrHead
from .verb_family import VerbFamilyHead
from .verb_family_fine import VerbFamilyFineHead
from .verb_polarity import VerbPolarityHead
from .verb_aspect import VerbAspectHead
from .verb_source import VerbSourceHead
from .attributes import AttributesHead
from .compat import CompatHead

# Ordre = ordre d'affichage / d'agrégation des métriques (aligné sur TASK_KEYS
# de loss_weighting.py, "compat" en dernier car ce n'est pas une tête supervisée).
HEAD_CLASSES = [
    BoundaryHead,
    CoarseHead,
    FineHead,
    SvoBoundaryHead,
    SvoHead,
    RoleCoarseHead,
    SemanticRoleHead,
    RoleLegacyHead,
    VoiceHead,
    CertaintyHead,
    MorphoHead,
    VerbPtrHead,
    VerbFamilyHead,
    VerbFamilyFineHead,
    VerbPolarityHead,
    VerbAspectHead,
    VerbSourceHead,
    AttributesHead,
    CompatHead,
]


def build_all_heads(model) -> dict[str, Head]:
    """Instancie toutes les têtes en leur passant les couches nn.Module du modèle."""
    heads: dict[str, Head] = {}
    for cls in HEAD_CLASSES:
        head = cls.from_model(model)
        heads[head.task_key] = head
    return heads


__all__ = ["Head", "HEAD_CLASSES", "build_all_heads"]

