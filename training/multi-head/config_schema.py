"""
config_schema.py — Modèle de haut niveau du système d'entraînement multitask pimpmyrag.

Ce module définit les types abstraits (enums + dataclasses).
Chaque fichier JSON de config est une *instanciation* de MultiTaskConfig.

Pas de logique ici — uniquement la structure.
Le reader (config_reader.py) se chargera de désérialiser JSON → ces types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union


# ─────────────────────────────────────────────────────────────
#  Enums — types fermés
# ─────────────────────────────────────────────────────────────

class HeadType(str, Enum):
    """Type de tête de prédiction."""
    SPAN_BINARY           = "span_binary"           # classification binaire sur span (boundary, svo_boundary)
    SPAN_MULTICLASS       = "span_multiclass"        # classification N classes sur span (coarse, fine, roles…)
    SPAN_MULTICLASS_MULTI = "span_multiclass_multi"  # plusieurs têtes multiclass partageant le même input (morpho)
    TOKEN_POINTER         = "token_pointer"          # pointeur span → token dans la séquence (verb_ptr)
    SPAN_BILINEAR         = "span_bilinear"          # compatibilité bilinéaire entre paires de spans (compat)


class InputTensor(str, Enum):
    """Tenseur d'entrée d'une tête."""
    SPAN_H         = "span_h"          # représentation brute du span (pooling sur ses tokens)
    SPAN_H_ROLE    = "span_h_role"     # span_h + verb_ctx_proj(soft_attn_verb) — conditionné par le verbe
    ENCODER_HIDDEN = "encoder_hidden"  # séquence complète de l'encodeur (pour token_pointer)


class LossType(str, Enum):
    """Type de fonction de loss."""
    CROSS_ENTROPY = "cross_entropy"
    FOCAL         = "focal"          # focal loss — nécessite gamma
    BCE           = "bce"            # binary cross-entropy


class GateType(str, Enum):
    """Condition d'activation d'une tête ou d'une phase."""
    ALWAYS           = "always"            # toujours actif
    METRIC_THRESHOLD = "metric_threshold"  # activé quand métrique W&B ≥ seuil
    EPOCH_THRESHOLD  = "epoch_threshold"   # activé à partir de l'epoch N


class PoolingType(str, Enum):
    """Stratégie de pooling pour construire la représentation d'un span."""
    MEAN       = "mean"        # moyenne des tokens du span
    MAX        = "max"         # max pooling
    FIRST_LAST = "first_last"  # concaténation premier + dernier token


class ClassWeightStrategy(str, Enum):
    """Stratégie de pondération des classes dans la loss."""
    AUTO = "auto"  # calculé depuis la distribution du dataset (CWP)
    NONE = "none"  # pas de pondération


# ─────────────────────────────────────────────────────────────
#  Composants de base
# ─────────────────────────────────────────────────────────────

@dataclass
class LossConfig:
    type:  LossType
    gamma: Optional[float] = None  # requis si type == FOCAL


@dataclass
class LambdaRamp:
    """Montée progressive du lambda sur ramp_epochs epochs."""
    start:       float
    target:      float
    ramp_epochs: int


# Un lambda peut être fixe (float) ou progressif (LambdaRamp)
LambdaSchedule = Union[float, LambdaRamp]


@dataclass
class GateConfig:
    """Condition d'activation d'une tête ou d'une phase curriculum."""
    type:      GateType
    metric:    Optional[str]   = None  # ex: "val/boundary_f1"
    threshold: Optional[float] = None  # seuil à franchir
    epoch:     Optional[int]   = None  # epoch de déclenchement


# ─────────────────────────────────────────────────────────────
#  Architecture — têtes
# ─────────────────────────────────────────────────────────────

@dataclass
class SubHeadConfig:
    """Sous-tête dans un HeadType.SPAN_MULTICLASS_MULTI (ex: morpho → gender + number + person)."""
    name:      str   # identifiant, ex: "gender"
    label_set: str   # clé dans DatasetConfig.label_sets


@dataclass
class HeadConfig:
    """Définition complète d'une tête de prédiction."""
    name:      str           # identifiant unique, ex: "boundary"
    type:      HeadType
    input:     InputTensor   # tenseur d'entrée
    loss:      LossConfig
    lambda_:   LambdaSchedule  # "lambda" est réservé en Python
    gate:      GateConfig

    # Optionnels selon le type
    label_set:  Optional[str]              = None  # clé dans DatasetConfig.label_sets
    seq_input:  Optional[InputTensor]      = None  # pour TOKEN_POINTER : séquence encodeur
    sub_heads:  Optional[list[SubHeadConfig]] = None  # pour SPAN_MULTICLASS_MULTI

    # Pondération des classes
    class_weights:       ClassWeightStrategy = ClassWeightStrategy.AUTO
    class_weight_power:  float               = 0.5
    ignore_none:         bool                = False  # exclure la classe NONE de la loss

    # Architecture interne de la tête (MLP)
    num_layers: int           = 1    # couches du MLP de tête
    hidden_dim: Optional[int] = None # dim interne (None = span_hidden_dim)


# ─────────────────────────────────────────────────────────────
#  Architecture — modules contextuels
# ─────────────────────────────────────────────────────────────

@dataclass
class VerbCtxConfig:
    """
    Module de conditionnement de la représentation de span par le verbe gouverneur.
    Produit span_h_role = span_h + verb_ctx_proj(soft_attention(verb_ptr_logits, encoder_hidden)).

    Utilisé par les têtes avec input=SPAN_H_ROLE (role_coarse, role_oblique).
    Le detach() sur verb_ptr_logits est implicite (les gradients ne se propagent pas).
    """
    enabled:        bool
    mode:           str = "soft_attention"   # "soft_attention" | "hard_pointer"
    source:         str = "verb_ptr_logits"  # tête qui fournit les logits de pointeur
    projection_dim: int = 512                # dim de la projection linéaire


@dataclass
class ContextModulesConfig:
    """Modules de contexte partagés entre plusieurs têtes."""
    verb_ctx: Optional[VerbCtxConfig] = None


# ─────────────────────────────────────────────────────────────
#  Architecture — backbone & span encoder
# ─────────────────────────────────────────────────────────────

@dataclass
class BackboneConfig:
    """Encodeur de base (transformeur)."""
    model_id:      str         # ex: "microsoft/deberta-v3-base"
    max_length:    int  = 512
    hidden_size:   int  = 768  # déduit automatiquement du modèle si omis
    freeze_layers: int  = 0    # nombre de couches à geler (0 = rien)


@dataclass
class SpanEncoderConfig:
    """Encodage d'un span à partir des hidden states de l'encodeur."""
    pooling:    PoolingType = PoolingType.MEAN
    hidden_dim: int         = 512   # dim de la représentation de span après projection
    dropout:    float       = 0.1


@dataclass
class ArchitectureConfig:
    """Architecture complète du modèle multitask."""
    backbone:        BackboneConfig
    span_encoder:    SpanEncoderConfig
    context_modules: ContextModulesConfig
    heads:           list[HeadConfig]

    def head(self, name: str) -> HeadConfig:
        """Accès rapide à une tête par son nom."""
        for h in self.heads:
            if h.name == name:
                return h
        raise KeyError(f"Head '{name}' not found")


# ─────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────

@dataclass
class LabelSetConfig:
    """
    Description d'un ensemble de labels.
    source = chemin Python vers le dict (ex: "labels.FINE2ID").
    none_id = id de la classe NONE à exclure des métriques si ignore_none=True.
    """
    source:  str
    none_id: Optional[str] = None


@dataclass
class CandidatesConfig:
    """Configuration de la génération des spans candidats."""
    strategy:       str   = "all_ngrams"  # seul mode supporté actuellement
    max_span_width: int   = 8
    include_svo:    bool  = True          # inclure les spans SVO hors n-grams


@dataclass
class HardNegativesConfig:
    """Configuration du hard negative mining."""
    enabled:        bool  = True
    every_n_epochs: int   = 1
    decay:          float = 0.85
    max_weight:     float = 8.0
    min_weight:     float = 0.3
    boost_fp_boundary:   float = 5.0
    boost_fn_boundary:   float = 2.0
    boost_coarse_err:    float = 2.5
    boost_fine_err:      float = 3.0
    boost_fp_svo:        float = 3.0
    boost_fn_svo:        float = 2.0
    boost_role_coarse:   float = 2.5


@dataclass
class DatasetConfig:
    """Configuration du dataset d'entraînement."""
    format:      str   = "jsonl_spans"    # seul format supporté
    label_sets:  dict[str, LabelSetConfig] = field(default_factory=dict)
    candidates:  CandidatesConfig          = field(default_factory=CandidatesConfig)
    hard_negatives: HardNegativesConfig    = field(default_factory=HardNegativesConfig)
    gold_version: Optional[str]            = None  # override par --gold-version CLI


# ─────────────────────────────────────────────────────────────
#  Curriculum d'entraînement
# ─────────────────────────────────────────────────────────────

@dataclass
class PhaseConfig:
    """
    Phase du curriculum : active un ensemble de têtes dès que la gate est franchie.
    Les têtes avec gate=ALWAYS sont actives dès le départ (phase implicite).
    """
    name:         str
    heads_active: list[str]   # noms des têtes à activer lors de cette phase
    gate:         GateConfig


@dataclass
class DifficultyConfig:
    """Progression par niveaux de difficulté du dataset (easy → full)."""
    names:                  list[str]
    hard_per_gold:          list[int]    # négatifs durs par exemple gold
    soft_factors:           list[float]  # facteur pour les négatifs légers
    max_epochs_per_level:   int   = 6
    min_delta_for_progress: float = 0.0003
    patience:               int   = 5


@dataclass
class RescueConfig:
    """Détection de stagnation / régression boundary → contre-mesures."""
    boundary_window:    int   = 5
    boundary_target:    float = 0.90
    boundary_min_delta: float = 0.003
    bnd_boost_factor:   float = 1.20
    regression_window:  int   = 3
    regression_delta:   float = 0.008


@dataclass
class EarlyStoppingConfig:
    patience:  int   = 5
    min_delta: float = 0.0003


@dataclass
class CurriculumConfig:
    """Orchestration temporelle de l'entraînement : phases, difficultés, rescue."""
    phases:            list[PhaseConfig]
    difficulty_levels: DifficultyConfig
    rescue:            RescueConfig
    early_stopping:    EarlyStoppingConfig


# ─────────────────────────────────────────────────────────────
#  Optimiseur
# ─────────────────────────────────────────────────────────────

@dataclass
class OptimizerConfig:
    lr:                  float = 8e-6
    head_lr_multiplier:  float = 4.0    # LR têtes = lr * multiplier
    warmup_epochs:       int   = 0
    max_grad_norm:       float = 1.0
    layer_lr_decay:      float = 0.9    # layer-wise LR decay
    ema_decay:           float = 0.999
    class_weight_power:  float = 0.5
    label_smoothing:     float = 0.0


# ─────────────────────────────────────────────────────────────
#  Hardware
# ─────────────────────────────────────────────────────────────

@dataclass
class HardwareProfile:
    bs:      int          # batch size
    accum:   int  = 1     # gradient accumulation steps
    workers: int  = 0     # DataLoader workers (0 = main process, pas de multiprocessing)


@dataclass
class HardwareConfig:
    """Profils hardware — sélectionnés automatiquement selon VRAM détectée."""
    default:   HardwareProfile
    h100_80gb: Optional[HardwareProfile] = None
    l40s_48gb: Optional[HardwareProfile] = None
    a100_40gb: Optional[HardwareProfile] = None
    rtx_4090:  Optional[HardwareProfile] = None
    rtx_3090:  Optional[HardwareProfile] = None


# ─────────────────────────────────────────────────────────────
#  Config racine
# ─────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    name_suffix:    str
    max_epochs:     int
    loss_weighting: str  = "fixed"   # "fixed" | "uncertainty" | "gradnorm"
    ner_only_bench: bool = False


@dataclass
class MultiTaskConfig:
    """
    Modèle de haut niveau du système d'entraînement.
    Chaque fichier JSON de config est une instanciation de cette classe.

    Hiérarchie :
        MultiTaskConfig
        ├── run          : RunConfig            — paramètres du run W&B
        ├── architecture : ArchitectureConfig   — modèle + têtes
        ├── dataset      : DatasetConfig        — données + labels
        ├── optimizer    : OptimizerConfig      — LR, EMA, grad norm
        ├── curriculum   : CurriculumConfig     — phases, difficultés, rescue
        └── hardware     : HardwareConfig       — profils GPU

    Usage futur (avec reader) :
        cfg = MultiTaskConfig.from_json("configs/svo-v820-rc1.json")
        model = MultiTaskModel(cfg.architecture)
        trainer = Trainer(model, cfg)
    """
    run:          RunConfig
    architecture: ArchitectureConfig
    dataset:      DatasetConfig
    optimizer:    OptimizerConfig
    curriculum:   CurriculumConfig
    hardware:     HardwareConfig

