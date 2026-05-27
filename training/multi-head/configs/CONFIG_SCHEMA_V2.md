# Schéma de Configuration v2 — pimpmyrag NER multitask

## Vue d'ensemble

La config v2 est **complètement déclarative** : l'architecture du modèle, les têtes, les données
et le curriculum sont entièrement définis dans le JSON. Python ne fait que lire et appliquer.

---

## Types de base

### `HeadType` (enum)

| Valeur | Description |
|--------|-------------|
| `span_binary` | Classification binaire sur représentation de span (boundary, svo_boundary) |
| `span_multiclass` | Classification N classes sur span (coarse, fine, role_coarse, voice…) |
| `span_multiclass_multi` | Plusieurs têtes multiclass partageant le même input (morpho : gender+number+person) |
| `token_pointer` | Pointeur span→token dans la séquence encodeur (verb_ptr) |
| `span_bilinear` | Compatibilité bilinéaire entre paires de spans (compat) |

### `InputTensor` (enum)

| Valeur | Description |
|--------|-------------|
| `span_h` | Représentation de span brute (pooling sur tokens du span) |
| `span_h_role` | `span_h + verb_ctx_proj(soft_attn_verb)` — conditionné par le verbe gouverneur |
| `encoder_hidden` | Séquence complète de l'encodeur (pour token_pointer) |

### `LossConfig` (objet)

```json
{ "type": "focal",        "gamma": 1.5 }
{ "type": "cross_entropy" }
{ "type": "bce" }
```

### `LambdaSchedule` (scalar ou objet)

```json
0.55                                              // constant
{ "start": 0.05, "target": 0.75, "ramp_epochs": 10 }  // warmup linéaire
```

### `GateConfig` (objet)

```json
{ "type": "always" }
{ "type": "metric_threshold", "metric": "val/boundary_f1", "threshold": 0.92 }
{ "type": "epoch_threshold",  "epoch": 5 }
```

---

## Section `architecture`

```json
"architecture": {
  "backbone": {
    "model_id":      string,    // ex: "microsoft/deberta-v3-base"
    "max_length":    int,       // 512
    "hidden_size":   int,       // 768 (auto si omis)
    "freeze_layers": int        // 0 = rien freeze
  },
  "span_encoder": {
    "pooling":    "mean" | "max" | "first_last",
    "hidden_dim": int,          // 512
    "dropout":    float         // 0.1
  },
  "context_modules": {
    "verb_ctx": {
      "enabled":        bool,
      "mode":           "soft_attention" | "hard_pointer",
      "source":         "verb_ptr_logits",  // quelle tête fournit les logits
      "projection_dim": int                 // 512
    }
  },
  "heads": [ HeadDefinition, ... ]
}
```

### `HeadDefinition` complète

```json
{
  "name":      string,        // identifiant unique, ex: "boundary"
  "type":      HeadType,
  "input":     InputTensor,   // ou liste pour multi-input
  "label_set": string,        // clé dans dataset.label_sets (ou liste pour multi)
  "loss":      LossConfig,
  "lambda":    LambdaSchedule,
  "gate":      GateConfig,

  // Optionnel selon le type :
  "num_layers": int,          // couches MLP (défaut 1)
  "hidden_dim": int,          // dim interne MLP (défaut = span_hidden_dim)
  "class_weights": "auto" | "none",
  "class_weight_power": float,
  "focal_role_gamma": float,  // pour loss focal spécifique à cette tête

  // Pour span_multiclass_multi uniquement :
  "sub_heads": [
    { "name": "gender", "label_set": "gender" },
    { "name": "number", "label_set": "number" },
    { "name": "person", "label_set": "person" }
  ],

  // Pour token_pointer uniquement :
  "seq_input": "encoder_hidden"
}
```

---

## Section `dataset`

```json
"dataset": {
  "format": "jsonl_spans",        // seul format supporté actuellement
  "gold_version": string,         // ex: "v8.20" (override par --gold-version)

  "splits": {
    "train": "data/train_{gold_version}.jsonl",
    "val":   "data/val_{gold_version}.jsonl",
    "test":  "data/test_{gold_version}.jsonl"
  },

  "label_sets": {
    // clé → source dans labels.py
    "ner_boundary":     { "source": "labels.BOUNDARY_LABELS" },
    "ner_coarse":       { "source": "labels.COARSE2ID",         "none_id": "COARSE_NONE_ID" },
    "ner_fine":         { "source": "labels.FINE2ID",           "none_id": "FINE_NONE_ID" },
    "svo_boundary":     { "source": "labels.SVO_BOUNDARY" },
    "svo_role_coarse":  { "source": "labels.ROLE_COARSE2ID",    "none_id": "ROLE_COARSE_NONE_ID" },
    "svo_role_oblique": { "source": "labels.ROLE_OBLIQUE2ID",   "none_id": "ROLE_OBLIQUE_NONE_ID" },
    "svo_voice":        { "source": "labels.VOICE2ID",          "none_id": "VOICE_NONE_ID" },
    "certainty":        { "source": "labels.CERTAINTY2ID",      "none_id": "CERTAINTY_NONE_ID" },
    "gender":           { "source": "labels.GENDER2ID",         "none_id": "GENDER_NONE_ID" },
    "number":           { "source": "labels.NUMBER2ID",         "none_id": "NUMBER_NONE_ID" },
    "person":           { "source": "labels.PERSON2ID",         "none_id": "PERSON_NONE_ID" },
    "svo_syn":          { "source": "labels.SYN2ID",            "none_id": "SYN_NONE_ID" }
  },

  "candidates": {
    "strategy":      "all_ngrams",    // génère tous les n-grams candidats
    "max_span_width": 8,
    "include_svo":   true             // inclut spans annotés SVO même hors n-grams
  },

  "hard_negatives": {
    "enabled":        bool,
    "every_n_epochs": int,
    "decay":          float,
    "max_weight":     float,
    "min_weight":     float,
    "boosts": {
      "fp_boundary":   float,
      "fn_boundary":   float,
      "coarse_err":    float,
      "fine_err":      float,
      "fp_svo":        float,
      "fn_svo":        float,
      "role_coarse":   float
    }
  }
}
```

---

## Section `curriculum` (remplace `boundary_first` + `svo_cascade`)

```json
"curriculum": {
  "phases": [
    {
      "name": "phase1_ner_svo_bnd",
      "heads_active": ["boundary", "svo_boundary", "role_coarse", "compat"],
      "gate": { "type": "always" }
    },
    {
      "name": "phase2_roles",
      "heads_active": ["role_coarse", "voice", "certainty", "verb_ptr"],
      "gate": { "type": "metric_threshold", "metric": "val/svo_boundary_f1", "threshold": 0.80 }
    },
    {
      "name": "phase3_ner_cls",
      "heads_active": ["coarse", "fine"],
      "gate": { "type": "metric_threshold", "metric": "val/boundary_f1", "threshold": 0.92 }
    },
    {
      "name": "phase4_oblique_morpho",
      "heads_active": ["role_oblique", "morpho"],
      "gate": { "type": "metric_threshold", "metric": "val/role_coarse_f1", "threshold": 0.40 }
    }
  ],
  "rescue": {
    "boundary_window":    5,
    "boundary_target":    0.90,
    "boundary_min_delta": 0.003,
    "bnd_boost_factor":   1.20
  },
  "difficulty_levels": {
    "names":                   ["easy", "easy+", "medium", "medium+", "hard", "full"],
    "hard_per_gold":           [2, 2, 3, 4, 5, 6],
    "soft_factors":            [1.0, 1.25, 1.5, 2.0, 2.0, 2.0],
    "max_epochs_per_level":    6,
    "min_delta_for_progress":  0.0003,
    "patience":                5
  }
}
```

---

## Exemple complet minimal

```json
{
  "_schema": "v2",
  "run": { "name_suffix": "svo-v820-rc1", "max_epochs": 90 },

  "architecture": {
    "backbone":     { "model_id": "microsoft/deberta-v3-base", "max_length": 512 },
    "span_encoder": { "pooling": "mean", "hidden_dim": 512, "dropout": 0.1 },
    "context_modules": {
      "verb_ctx": { "enabled": true, "mode": "soft_attention", "projection_dim": 512 }
    },
    "heads": [
      { "name": "boundary",    "type": "span_binary",     "input": "span_h",      "loss": { "type": "focal", "gamma": 0.5 }, "lambda": 4.0,  "gate": { "type": "always" } },
      { "name": "svo_boundary","type": "span_binary",     "input": "span_h",      "loss": { "type": "focal", "gamma": 0.5 }, "lambda": 0.55, "gate": { "type": "always" } },
      { "name": "coarse",      "type": "span_multiclass", "input": "span_h",      "label_set": "ner_coarse",      "loss": { "type": "cross_entropy" }, "lambda": { "start": 0.05, "target": 0.75, "ramp_epochs": 10 }, "gate": { "type": "metric_threshold", "metric": "val/boundary_f1", "threshold": 0.92 } },
      { "name": "fine",        "type": "span_multiclass", "input": "span_h",      "label_set": "ner_fine",        "loss": { "type": "focal", "gamma": 1.5 }, "lambda": { "start": 0.15, "target": 1.8, "ramp_epochs": 10 }, "gate": { "type": "metric_threshold", "metric": "val/boundary_f1", "threshold": 0.92 } },
      { "name": "role_coarse", "type": "span_multiclass", "input": "span_h_role", "label_set": "svo_role_coarse", "loss": { "type": "cross_entropy" }, "lambda": 0.55, "gate": { "type": "always" } },
      { "name": "role_oblique","type": "span_multiclass", "input": "span_h_role", "label_set": "svo_role_oblique","loss": { "type": "cross_entropy" }, "lambda": 0.20, "gate": { "type": "metric_threshold", "metric": "val/role_coarse_f1", "threshold": 0.40 } },
      { "name": "verb_ptr",    "type": "token_pointer",   "input": "span_h", "seq_input": "encoder_hidden", "loss": { "type": "cross_entropy" }, "lambda": 0.60, "gate": { "type": "metric_threshold", "metric": "val/svo_boundary_f1", "threshold": 0.80 } },
      { "name": "voice",       "type": "span_multiclass", "input": "span_h",      "label_set": "svo_voice",       "loss": { "type": "cross_entropy" }, "lambda": 0.15, "gate": { "type": "metric_threshold", "metric": "val/svo_boundary_f1", "threshold": 0.80 } },
      { "name": "certainty",   "type": "span_multiclass", "input": "span_h",      "label_set": "certainty",       "loss": { "type": "cross_entropy" }, "lambda": 0.30, "gate": { "type": "metric_threshold", "metric": "val/svo_boundary_f1", "threshold": 0.80 } },
      { "name": "morpho",      "type": "span_multiclass_multi", "input": "span_h", "loss": { "type": "cross_entropy" }, "lambda": 0.10, "gate": { "type": "metric_threshold", "metric": "val/role_coarse_f1", "threshold": 0.40 },
        "sub_heads": [
          { "name": "gender", "label_set": "gender" },
          { "name": "number", "label_set": "number" },
          { "name": "person", "label_set": "person" }
        ]
      },
      { "name": "compat",      "type": "span_bilinear",   "input": "span_h",                                      "loss": { "type": "cross_entropy" }, "lambda": 0.2,  "gate": { "type": "always" } }
    ]
  },

  "dataset": {
    "format": "jsonl_spans",
    "candidates": { "strategy": "all_ngrams", "max_span_width": 8 }
  },

  "optimizer": { "lr": 8e-6, "head_lr_multiplier": 4.0, "layer_lr_decay": 0.9, "ema_decay": 0.999 },

  "curriculum": {
    "phases": [
      { "name": "phase1", "heads_active": ["boundary", "svo_boundary", "role_coarse", "compat"], "gate": { "type": "always" } },
      { "name": "phase2", "heads_active": ["voice", "certainty", "verb_ptr"], "gate": { "type": "metric_threshold", "metric": "val/svo_boundary_f1", "threshold": 0.80 } },
      { "name": "phase3", "heads_active": ["coarse", "fine"], "gate": { "type": "metric_threshold", "metric": "val/boundary_f1", "threshold": 0.92 } },
      { "name": "phase4", "heads_active": ["role_oblique", "morpho"], "gate": { "type": "metric_threshold", "metric": "val/role_coarse_f1", "threshold": 0.40 } }
    ]
  },

  "hardware": {
    "h100_80gb": { "bs": 160, "accum": 1, "workers": 8 },
    "rtx_4090":  { "bs": 80,  "accum": 1, "workers": 8 },
    "default":   { "bs": 16,  "accum": 2, "workers": 0 }
  }
}
```

