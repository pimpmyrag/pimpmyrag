#!/usr/bin/env python3
"""
export_vec_full.py
~~~~~~~~~~~~~~~~~~
Export vectorisé du SpanMultiTaskModel vers ONNX — 12 têtes v4.

Interface ONNX :
  Inputs:
    input_ids        [B, L]  int64
    attention_mask   [B, L]  int64
    span_starts      [N]     int64
    span_ends        [N]     int64
    span_batch_ids   [N]     int64

   Outputs (NER):
    boundary_logits      [N, 2]
    coarse_logits        [N, 10]
    fine_logits          [N, 38]

  Outputs (SVO / syntaxe v4):
    svo_boundary_logits  [N, 2]
    syn_logits           [N, 3]   verb_trigger | pron_subj | pron_obj
    role_logits          [N, 7]   SUBJECT | OBJECT | OBLIQUE | OBLIQUE_AGENT | OBLIQUE_CAUSE | APPOS | NONE
    voice_logits         [N, 2]   active | passive
    certainty_logits     [N, 3]   certain | modal | denied
    gender_logits        [N, 3]
    number_logits        [N, 2]
    person_logits        [N, 3]
    verb_ptr_logits      [N, L]
"""
import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from multitask_model import SpanMultiTaskModel
from labels import (
    COARSE_LABELS, FINE_LABELS, NUM_FINE,
    SYN_LABELS, ROLE_LABELS, VOICE_LABELS, CERTAINTY_LABELS,
    GENDER_LABELS, NUMBER_LABELS, PERSON_LABELS,
)


# ──────────────────────────────────────────────────────────
#  Chargement avec tokenizer étendu
# ──────────────────────────────────────────────────────────

def _infer_dims_from_state(state: dict) -> dict:
    """Infère les dimensions des têtes depuis le state_dict du checkpoint."""
    dims = {}
    head_keys = {
        "num_coarse": "coarse_head.bias",
        "num_fine":   "fine_head.bias",
        "num_syn":    "syn_head.bias",
        "num_role":   "role_head.bias",
        "num_voice":  "voice_head.bias",
        "num_certainty": "certainty_head.bias",
        "num_gender": "gender_head.bias",
        "num_number": "number_head.bias",
        "num_person": "person_head.bias",
    }
    for dim_name, key in head_keys.items():
        if key in state:
            dims[dim_name] = state[key].shape[0]
    return dims


def load_model_with_extended_tokenizer(checkpoint_path: str, model_name: str, tokenizer_path: str = None):
    """
    Crée un SpanMultiTaskModel et charge le checkpoint.
    Infère les dimensions des têtes depuis le checkpoint pour gérer
    les checkpoints entraînés avec d'anciennes versions de labels.py.
    Si tokenizer_path est fourni, étend les embeddings au vocab_size du tokenizer
    AVANT de charger les poids — évite le RuntimeError 'size mismatch'.
    """
    # ── 1. Charger le state_dict pour inférer les dimensions ──
    print(f"📦 Chargement checkpoint : {checkpoint_path}")
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    dims = _infer_dims_from_state(state)

    num_coarse = dims.get("num_coarse", len(COARSE_LABELS))
    print(f"   Dimensions inférées depuis checkpoint : {dims}")

    # ── 2. Créer le modèle avec les bonnes dimensions ──
    print(f"📦 Chargement modèle de base : {model_name}")
    model = SpanMultiTaskModel(model_name=model_name, num_coarse=num_coarse).float()

    # Si les dims des têtes dans le checkpoint diffèrent du labels.py actuel,
    # recréer les Linear avec les bonnes tailles avant load_state_dict.
    span_hidden_dim = state["boundary_head.bias"].shape[0]  # toujours 2, on prend le weight
    span_hidden_dim = state["boundary_head.weight"].shape[1]  # dim d'entrée
    dim_overrides = {
        "fine_head":      ("num_fine",      NUM_FINE),
        "gender_head":    ("num_gender",    None),
        "number_head":    ("num_number",    None),
        "person_head":    ("num_person",    None),
        "syn_head":       ("num_syn",       None),
        "role_head":      ("num_role",      None),
        "voice_head":     ("num_voice",     None),
        "certainty_head": ("num_certainty", None),
    }
    for head_name, (dim_key, _) in dim_overrides.items():
        if dim_key in dims:
            ckpt_size = dims[dim_key]
            current_head = getattr(model, head_name)
            if current_head.out_features != ckpt_size:
                print(f"   ⚠️  {head_name}: {current_head.out_features} → {ckpt_size} (adapté au checkpoint)")
                setattr(model, head_name, nn.Linear(span_hidden_dim, ckpt_size))

    # Recréer le coarse_fine_mask avec les dims du checkpoint
    if "coarse_fine_mask" in state:
        ckpt_mask_shape = state["coarse_fine_mask"].shape
        current_mask_shape = model.coarse_fine_mask.shape
        if ckpt_mask_shape != current_mask_shape:
            print(f"   ⚠️  coarse_fine_mask: {current_mask_shape} → {ckpt_mask_shape} (adapté au checkpoint)")
            model.coarse_fine_mask = nn.Parameter(torch.zeros(ckpt_mask_shape), requires_grad=False)

    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        extended_vocab = len(tokenizer)
        current_vocab  = model.encoder.embeddings.word_embeddings.weight.size(0)
        if extended_vocab > current_vocab:
            print(f"   Extension vocab : {current_vocab} → {extended_vocab}")
            old_emb = model.encoder.embeddings.word_embeddings
            new_emb = nn.Embedding(extended_vocab, old_emb.embedding_dim)
            with torch.no_grad():
                new_emb.weight[:current_vocab] = old_emb.weight
                new_emb.weight[current_vocab:] = old_emb.weight.mean(dim=0)
            model.encoder.embeddings.word_embeddings = new_emb
            model.encoder.config.vocab_size = extended_vocab
        else:
            print(f"   Vocab tokenizer ({extended_vocab}) ≤ vocab modèle ({current_vocab}) — pas d'extension")

    model.load_state_dict(state, strict=True)
    model.eval()
    print("✅ Checkpoint chargé")
    return model


# ──────────────────────────────────────────────────────────
#  Wrapper 100% tensoriel — aucune boucle Python, aucun .item()
# ──────────────────────────────────────────────────────────

class OnnxSpanWrapperFull(nn.Module):
    """
    Wrapper ONNX pour SpanMultiTaskModel v4.
    Exporte les 12 têtes (NER + SVO/syn/role/voice/certainty + morpho + verb_ptr).
    Entièrement vectorisé : compatible axes dynamiques ONNX.
    """

    def __init__(self, inner: SpanMultiTaskModel):
        super().__init__()
        self.encoder           = inner.encoder
        self.width_emb         = inner.width_emb
        self.span_mlp          = inner.span_mlp
        # Têtes NER
        self.boundary_head     = inner.boundary_head
        self.coarse_head       = inner.coarse_head
        self.fine_head         = inner.fine_head
        # Têtes SVO / syntaxe v4
        self.svo_boundary_head = inner.svo_boundary_head
        self.syn_head          = inner.syn_head        # verb_trigger / pron_subj / pron_obj
        self.role_head         = inner.role_head       # SUBJECT / OBJECT / OBLIQUE / ...
        self.voice_head        = inner.voice_head      # active / passive
        self.certainty_head    = inner.certainty_head  # certain / modal / denied
        # Morpho
        self.gender_head       = inner.gender_head
        self.number_head       = inner.number_head
        self.person_head       = inner.person_head
        # Verb pointer
        self.verb_ptr_query    = inner.verb_ptr_query
        self.verb_ptr_key      = inner.verb_ptr_key
        self.max_width_bucket  = inner.max_width_bucket

    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, L]  int64
        attention_mask: torch.Tensor,   # [B, L]  int64
        span_starts:    torch.Tensor,   # [N]     int64
        span_ends:      torch.Tensor,   # [N]     int64
        span_batch_ids: torch.Tensor,   # [N]     int64
    ):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state                                         # [B, L, H]

        start_vecs = hidden[span_batch_ids, span_starts]           # [N, H]
        end_vecs   = hidden[span_batch_ids, span_ends]             # [N, H]

        lengths = (span_ends - span_starts + 1).float().clamp(min=1).unsqueeze(1)
        zeros   = torch.zeros(hidden.size(0), 1, hidden.size(2),
                              device=hidden.device, dtype=hidden.dtype)
        prefix  = torch.cat([zeros, torch.cumsum(hidden, dim=1)], dim=1)  # [B, L+1, H]
        mean_vecs = (
            prefix[span_batch_ids, span_ends + 1] -
            prefix[span_batch_ids, span_starts]
        ) / lengths

        widths = (span_ends - span_starts + 1).clamp(min=1, max=self.max_width_bucket - 1)
        w_embs = self.width_emb(widths)

        span_reps = torch.cat([start_vecs, end_vecs, mean_vecs, w_embs], dim=-1)
        span_h    = self.span_mlp(span_reps)

        return (
            self.boundary_head(span_h),       # [N, 2]
            self.coarse_head(span_h),         # [N, 10]
            self.fine_head(span_h),           # [N, 38]
            self.svo_boundary_head(span_h),   # [N, 2]
            self.syn_head(span_h),            # [N, 3]
            self.role_head(span_h),           # [N, 7]
            self.voice_head(span_h),          # [N, 2]
            self.certainty_head(span_h),      # [N, 3]
            self.gender_head(span_h),         # [N, 3]
            self.number_head(span_h),         # [N, 2]
            self.person_head(span_h),         # [N, 3]
            torch.bmm(
                self.verb_ptr_key(hidden[span_batch_ids]),
                self.verb_ptr_query(span_h).unsqueeze(-1)
            ).squeeze(-1),                    # [N, L]
        )


# ──────────────────────────────────────────────────────────
#  Noms ONNX + axes dynamiques
# ──────────────────────────────────────────────────────────

OUTPUT_NAMES = [
    "boundary_logits",
    "coarse_logits",
    "fine_logits",
    "svo_boundary_logits",
    "syn_logits",
    "role_logits",
    "voice_logits",
    "certainty_logits",
    "gender_logits",
    "number_logits",
    "person_logits",
    "verb_ptr_logits",
]

DYNAMIC_AXES = {
    "input_ids":       {0: "batch",     1: "seq_len"},
    "attention_mask":  {0: "batch",     1: "seq_len"},
    "span_starts":     {0: "num_spans"},
    "span_ends":       {0: "num_spans"},
    "span_batch_ids":  {0: "num_spans"},
    **{name: {0: "num_spans"} for name in OUTPUT_NAMES},
    "verb_ptr_logits": {0: "num_spans", 1: "seq_len"},
}


def main():
    ap = argparse.ArgumentParser(description="Export ONNX — 12 têtes NER+SVO v4 vectorisé")
    ap.add_argument("--checkpoint",     required=True,  help="Chemin vers best_model_multitask.pt")
    ap.add_argument("--output",         required=True,  help="Chemin de sortie .onnx")
    ap.add_argument("--model-name",     default="microsoft/deberta-v3-base")
    ap.add_argument("--tokenizer-path", default=None,   help="Chemin vers tokenizer étendu (si vocab > 128k)")
    ap.add_argument("--opset",          type=int, default=17)
    ap.add_argument("--seq-len",        type=int, default=128)
    ap.add_argument("--num-spans",      type=int, default=64)
    args = ap.parse_args()

    inner = load_model_with_extended_tokenizer(
        args.checkpoint, args.model_name, args.tokenizer_path
    )
    model = OnnxSpanWrapperFull(inner).eval()

    B, L, N = 1, args.seq_len, args.num_spans
    dummy_ids  = torch.zeros(B, L, dtype=torch.long)
    dummy_mask = torch.ones(B, L,  dtype=torch.long)
    dummy_ss   = (torch.arange(N) % (L - 4) + 2).long()
    dummy_se   = (dummy_ss + 2).clamp(max=L - 2)
    dummy_bid  = torch.zeros(N, dtype=torch.long)

    with torch.no_grad():
        outs = model(dummy_ids, dummy_mask, dummy_ss, dummy_se, dummy_bid)
    shapes = "  ".join(f"{n}:{tuple(t.shape)}" for n, t in zip(OUTPUT_NAMES, outs))
    print(f"✅ Forward OK  {shapes}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"⚙️  Export ONNX opset={args.opset} → {args.output}")

    import logging
    logging.disable(logging.WARNING)

    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask, dummy_ss, dummy_se, dummy_bid),
        args.output,
        dynamo=False,
        opset_version=args.opset,
        input_names=["input_ids", "attention_mask", "span_starts", "span_ends", "span_batch_ids"],
        output_names=OUTPUT_NAMES,
        dynamic_axes=DYNAMIC_AXES,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(args.output) / 1e6
    try:
        import onnx
        m = onnx.load(args.output)
        print(f"✅ Export terminé — {size_mb:.1f} Mo")
        print(f"   Inputs  : {[i.name for i in m.graph.input]}")
        print(f"   Outputs : {[o.name for o in m.graph.output]}")
    except ImportError:
        print(f"✅ Export terminé — {size_mb:.1f} Mo")

    import json
    meta = {
        "coarse_labels":    COARSE_LABELS,
        "fine_labels":      FINE_LABELS,
        "syn_labels":       list(SYN_LABELS),
        "role_labels":      list(ROLE_LABELS),
        "voice_labels":     list(VOICE_LABELS),
        "certainty_labels": list(CERTAINTY_LABELS),
        "gender_labels":    list(GENDER_LABELS),
        "number_labels":    list(NUMBER_LABELS),
        "person_labels":    list(PERSON_LABELS),
    }
    meta_path = Path(args.output).with_suffix(".labels.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"   Labels JSON : {meta_path}")


if __name__ == "__main__":
    main()

