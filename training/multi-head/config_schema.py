"""
config_schema.py — Modèle de haut niveau du système d'entraînement multitask pimpmyrag.

Basé sur Pydantic v2 (BaseModel) — trois usages :

  1. Définition des types (enums + modèles)
  2. Génération du JSON Schema standard :
       python3 config_schema.py                        # écrit configs/config_schema.json
       python3 config_schema.py --out path/to/out.json

  3. Validation + désérialisation d'un fichier de config :
       cfg = MultiTaskConfig.from_json_file("configs/svo-v820-rc1.json")
       # → ValidationError explicite si le JSON ne respecte pas le schéma
"""
from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ─────────────────────────────────────────────────────────────
#  Enums
# ─────────────────────────────────────────────────────────────

class HeadType(str, Enum):
    """Type de tête de prédiction."""
    SPAN_BINARY           = "span_binary"
    SPAN_MULTICLASS       = "span_multiclass"
    SPAN_MULTICLASS_MULTI = "span_multiclass_multi"
    TOKEN_POINTER         = "token_pointer"
    SPAN_BILINEAR         = "span_bilinear"


class InputTensor(str, Enum):
    """Tenseur d'entrée d'une tête."""
    SPAN_H         = "span_h"
    SPAN_H_ROLE    = "span_h_role"
    ENCODER_HIDDEN = "encoder_hidden"


class LossType(str, Enum):
    CROSS_ENTROPY = "cross_entropy"
    FOCAL         = "focal"
    BCE           = "bce"


class GateType(str, Enum):
    ALWAYS           = "always"
    METRIC_THRESHOLD = "metric_threshold"
    EPOCH_THRESHOLD  = "epoch_threshold"


class PoolingType(str, Enum):
    MEAN       = "mean"
    MAX        = "max"
    FIRST_LAST = "first_last"


class ClassWeightStrategy(str, Enum):
    AUTO = "auto"
    NONE = "none"


class LossWeightingStrategy(str, Enum):
    FIXED       = "fixed"
    UNCERTAINTY = "uncertainty"
    GRADNORM    = "gradnorm"


# ─────────────────────────────────────────────────────────────
#  Composants de base
# ─────────────────────────────────────────────────────────────

class LossConfig(BaseModel):
    type:  LossType
    gamma: Optional[float] = Field(None, description="Requis si type=focal")

    @model_validator(mode="after")
    def _check_gamma(self) -> LossConfig:
        if self.type == LossType.FOCAL and self.gamma is None:
            raise ValueError("gamma est requis pour loss.type = 'focal'")
        return self


class LambdaRamp(BaseModel):
    """Montée progressive du lambda : start → target sur ramp_epochs epochs."""
    start:       float = Field(gt=0)
    target:      float = Field(gt=0)
    ramp_epochs: int   = Field(ge=1)


LambdaSchedule = Annotated[
    Union[LambdaRamp, float],
    Field(description="Float constant ou {start, target, ramp_epochs}")
]


class GateConfig(BaseModel):
    type:      GateType
    metric:    Optional[str]   = Field(None, description="Ex: 'val/boundary_f1'")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    epoch:     Optional[int]   = Field(None, ge=0)

    @model_validator(mode="after")
    def _check_fields(self) -> GateConfig:
        if self.type == GateType.METRIC_THRESHOLD:
            if self.metric is None or self.threshold is None:
                raise ValueError("metric_threshold requiert metric + threshold")
        if self.type == GateType.EPOCH_THRESHOLD and self.epoch is None:
            raise ValueError("epoch_threshold requiert epoch")
        return self


# ─────────────────────────────────────────────────────────────
#  Architecture — têtes
# ─────────────────────────────────────────────────────────────

class SubHeadConfig(BaseModel):
    name:      str
    label_set: str


class HeadConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name:    str
    type:    HeadType
    input:   InputTensor
    loss:    LossConfig
    lambda_: LambdaSchedule = Field(alias="lambda")
    gate:    GateConfig

    label_set:          Optional[str]               = None
    seq_input:          Optional[InputTensor]        = None
    sub_heads:          Optional[list[SubHeadConfig]] = None

    class_weights:      ClassWeightStrategy = ClassWeightStrategy.AUTO
    class_weight_power: float               = Field(0.5, ge=0.0, le=2.0)
    ignore_none:        bool                = False
    num_layers:         int                 = Field(1, ge=1)
    hidden_dim:         Optional[int]       = Field(None, ge=32)

    @model_validator(mode="after")
    def _check_type_constraints(self) -> HeadConfig:
        if self.type == HeadType.TOKEN_POINTER and self.seq_input is None:
            raise ValueError(f"Tête '{self.name}' (token_pointer) requiert seq_input")
        if self.type == HeadType.SPAN_MULTICLASS_MULTI and not self.sub_heads:
            raise ValueError(f"Tête '{self.name}' (span_multiclass_multi) requiert sub_heads")
        if (self.type not in (HeadType.SPAN_MULTICLASS_MULTI, HeadType.SPAN_BILINEAR)
                and self.label_set is None):
            raise ValueError(f"Tête '{self.name}' requiert label_set")
        return self


# ─────────────────────────────────────────────────────────────
#  Architecture — modules contextuels
# ─────────────────────────────────────────────────────────────

class VerbCtxConfig(BaseModel):
    """span_h_role = span_h + verb_ctx_proj(softmax(verb_ptr_logits.detach()) @ encoder_hidden)"""
    enabled:        bool
    mode:           str = Field("soft_attention", pattern="^(soft_attention|hard_pointer)$")
    source:         str = "verb_ptr_logits"
    projection_dim: int = Field(512, ge=64)


class ContextModulesConfig(BaseModel):
    verb_ctx: Optional[VerbCtxConfig] = None


class BackboneConfig(BaseModel):
    model_id:      str
    max_length:    int = Field(512, ge=64, le=4096)
    hidden_size:   int = Field(768, ge=128)
    freeze_layers: int = Field(0, ge=0)


class SpanEncoderConfig(BaseModel):
    pooling:    PoolingType = PoolingType.MEAN
    hidden_dim: int         = Field(512, ge=64)
    dropout:    float       = Field(0.1, ge=0.0, le=0.9)


class ArchitectureConfig(BaseModel):
    backbone:        BackboneConfig
    span_encoder:    SpanEncoderConfig
    context_modules: ContextModulesConfig
    heads:           list[HeadConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_head_names_unique(self) -> ArchitectureConfig:
        names = [h.name for h in self.heads]
        if len(names) != len(set(names)):
            dups = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Noms de têtes dupliqués : {dups}")
        return self

    @model_validator(mode="after")
    def _check_span_h_role_needs_verb_ctx(self) -> ArchitectureConfig:
        needs_role = any(h.input == InputTensor.SPAN_H_ROLE for h in self.heads)
        has_verb_ctx = (self.context_modules.verb_ctx is not None
                        and self.context_modules.verb_ctx.enabled)
        if needs_role and not has_verb_ctx:
            raise ValueError("input=span_h_role utilisé mais context_modules.verb_ctx non activé")
        return self

    def head(self, name: str) -> HeadConfig:
        for h in self.heads:
            if h.name == name:
                return h
        raise KeyError(f"Head '{name}' not found")


# ─────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────

class LabelSetConfig(BaseModel):
    source:  str = Field(description="Ex: 'labels.FINE2ID'")
    none_id: Optional[str] = None


class CandidatesConfig(BaseModel):
    strategy:       str  = Field("all_ngrams", pattern="^all_ngrams$")
    max_span_width: int  = Field(8, ge=1, le=32)
    include_svo:    bool = True


class HardNegativesConfig(BaseModel):
    enabled:           bool  = True
    every_n_epochs:    int   = Field(1, ge=1)
    decay:             float = Field(0.85, gt=0.0, le=1.0)
    max_weight:        float = Field(8.0, gt=0.0)
    min_weight:        float = Field(0.3, gt=0.0)
    boost_fp_boundary: float = Field(5.0, gt=0.0)
    boost_fn_boundary: float = Field(2.0, gt=0.0)
    boost_coarse_err:  float = Field(2.5, gt=0.0)
    boost_fine_err:    float = Field(3.0, gt=0.0)
    boost_fp_svo:      float = Field(3.0, gt=0.0)
    boost_fn_svo:      float = Field(2.0, gt=0.0)
    boost_role_coarse: float = Field(2.5, gt=0.0)


class DatasetConfig(BaseModel):
    format:         str                       = "jsonl_spans"
    label_sets:     dict[str, LabelSetConfig] = Field(default_factory=dict)
    candidates:     CandidatesConfig          = Field(default_factory=CandidatesConfig)
    hard_negatives: HardNegativesConfig       = Field(default_factory=HardNegativesConfig)
    gold_version:   Optional[str]             = None


# ─────────────────────────────────────────────────────────────
#  Curriculum
# ─────────────────────────────────────────────────────────────

class PhaseConfig(BaseModel):
    name:         str
    heads_active: list[str] = Field(min_length=1)
    gate:         GateConfig


class DifficultyConfig(BaseModel):
    names:                  list[str]
    hard_per_gold:          list[int]
    soft_factors:           list[float]
    max_epochs_per_level:   int   = Field(6,      ge=1)
    min_delta_for_progress: float = Field(0.0003, ge=0.0)
    patience:               int   = Field(5,      ge=1)

    @model_validator(mode="after")
    def _check_lengths(self) -> DifficultyConfig:
        n = len(self.names)
        if len(self.hard_per_gold) != n or len(self.soft_factors) != n:
            raise ValueError("names, hard_per_gold et soft_factors doivent avoir la même longueur")
        return self


class RescueConfig(BaseModel):
    boundary_window:    int   = Field(5,     ge=2)
    boundary_target:    float = Field(0.90,  ge=0.0, le=1.0)
    boundary_min_delta: float = Field(0.003, ge=0.0)
    bnd_boost_factor:   float = Field(1.20,  gt=1.0)
    regression_window:  int   = Field(3,     ge=2)
    regression_delta:   float = Field(0.008, ge=0.0)


class EarlyStoppingConfig(BaseModel):
    patience:  int   = Field(5,      ge=1)
    min_delta: float = Field(0.0003, ge=0.0)


class CurriculumConfig(BaseModel):
    phases:            list[PhaseConfig]
    difficulty_levels: DifficultyConfig
    rescue:            RescueConfig        = Field(default_factory=RescueConfig)
    early_stopping:    EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)


# ─────────────────────────────────────────────────────────────
#  Optimiseur
# ─────────────────────────────────────────────────────────────

class OptimizerConfig(BaseModel):
    lr:                 float = Field(8e-6,  gt=0.0)
    head_lr_multiplier: float = Field(4.0,   gt=0.0)
    warmup_epochs:      int   = Field(0,     ge=0)
    max_grad_norm:      float = Field(1.0,   gt=0.0)
    layer_lr_decay:     float = Field(0.9,   gt=0.0, le=1.0)
    ema_decay:          float = Field(0.999, gt=0.0, lt=1.0)
    class_weight_power: float = Field(0.5,   ge=0.0, le=2.0)
    label_smoothing:    float = Field(0.0,   ge=0.0, lt=1.0)


# ─────────────────────────────────────────────────────────────
#  Hardware
# ─────────────────────────────────────────────────────────────

class HardwareProfile(BaseModel):
    bs:      int = Field(ge=1)
    accum:   int = Field(1, ge=1)
    workers: int = Field(0, ge=0)


class HardwareConfig(BaseModel):
    default:   HardwareProfile
    h100_80gb: Optional[HardwareProfile] = None
    l40s_48gb: Optional[HardwareProfile] = None
    a100_40gb: Optional[HardwareProfile] = None
    rtx_4090:  Optional[HardwareProfile] = None
    rtx_3090:  Optional[HardwareProfile] = None


# ─────────────────────────────────────────────────────────────
#  Config racine
# ─────────────────────────────────────────────────────────────

class RunConfig(BaseModel):
    name_suffix:    str
    max_epochs:     int                   = Field(ge=1)
    loss_weighting: LossWeightingStrategy = LossWeightingStrategy.FIXED
    ner_only_bench: bool                  = False


class MultiTaskConfig(BaseModel):
    """
    Modèle de haut niveau — chaque fichier JSON de config est une instanciation de cette classe.

    Usage :
        cfg = MultiTaskConfig.from_json_file("configs/svo-v820-rc1.json")
        schema = MultiTaskConfig.model_json_schema()
    """
    model_config = ConfigDict(extra="ignore")   # ignore _comment, _version, _schema, _note…

    run:          RunConfig
    architecture: ArchitectureConfig
    dataset:      DatasetConfig      = Field(default_factory=DatasetConfig)
    optimizer:    OptimizerConfig    = Field(default_factory=OptimizerConfig)
    curriculum:   CurriculumConfig
    hardware:     HardwareConfig

    @classmethod
    def from_json_file(cls, path: str | Path) -> MultiTaskConfig:
        """Lit, nettoie les clés _* et valide un fichier JSON de config."""
        raw = Path(path).read_text(encoding="utf-8")
        # Supprime les clés de documentation (_comment, _note, _version…)
        raw = re.sub(r'"_[^"]*"\s*:\s*(?:"[^"]*"|\[[^\]]*\]|[^,}\n]+),?\n?', "", raw)
        data = json.loads(raw)
        return cls.model_validate(data)

    @classmethod
    def save_schema(cls, path: str | Path | None = None) -> dict:
        """Génère le JSON Schema et l'écrit sur disque si path fourni."""
        schema = cls.model_json_schema()
        if path:
            Path(path).write_text(
                json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"✅ JSON Schema écrit dans {path}")
        return schema


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Génère ou valide le schéma de config MultiTask")
    parser.add_argument("--out",      default="configs/config_schema.json",
                        help="Fichier de sortie du JSON Schema")
    parser.add_argument("--validate", default=None,
                        help="Valide un fichier JSON de config (ex: configs/svo-v819-rc2.json)")
    args = parser.parse_args()

    if args.validate:
        try:
            cfg = MultiTaskConfig.from_json_file(args.validate)
            print(f"✅ {args.validate} — valide")
            print(f"   backbone         = {cfg.architecture.backbone.model_id}")
            print(f"   heads            = {[h.name for h in cfg.architecture.heads]}")
            print(f"   max_epochs       = {cfg.run.max_epochs}")
        except Exception as e:
            print(f"❌ Erreur de validation :\n{e}")
            raise SystemExit(1)
    else:
        MultiTaskConfig.save_schema(args.out)
