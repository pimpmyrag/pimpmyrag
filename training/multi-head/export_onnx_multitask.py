#!/usr/bin/env python3
"""
Export du SpanMultiTaskModel vers ONNX avec des entrées plates (sans structure Python).

Interface ONNX exportée :
  Inputs:
    - input_ids           [batch, seq_len]  int64
    - attention_mask      [batch, seq_len]  int64
    - span_starts         [N]               int64   (indices tok_start de chaque span)
    - span_ends           [N]               int64   (indices tok_end de chaque span)
    - span_batch_ids      [N]               int64   (batch index pour chaque span)

  Outputs (NER) :
    - boundary_logits     [N, 2]            float32
    - coarse_logits       [N, 9]            float32
    - fine_logits         [N, 32]           float32

  Outputs (SVO / syntaxe v4) :
    - svo_boundary_logits [N, 2]            float32   détecte les spans verb_trigger/pron
    - syn_logits          [N, 3]            float32   verb_trigger | pron_subj | pron_obj
    - role_logits         [N, 7]            float32   SUBJECT | OBJECT | OBLIQUE | OBLIQUE_AGENT | OBLIQUE_CAUSE | APPOS | NONE
    - voice_logits        [N, 2]            float32   active / passive (verb_trigger uniquement)
    - certainty_logits    [N, 3]            float32   certain | modal | denied (verb_trigger uniquement)
    - gender_logits       [N, 3]            float32   M / F / N
    - number_logits       [N, 2]            float32   SG / PL
    - person_logits       [N, 3]            float32   1 / 2 / 3
    - verb_ptr_logits     [N, seq_len]      float32   pointer vers verbe gouverneur (arguments uniquement)
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

from labels import (
    NUM_FINE, NUM_SYN, NUM_ROLE, NUM_VOICE, NUM_CERTAINTY,
    NUM_GENDER, NUM_NUMBER, NUM_PERSON,
    COARSE_LABELS, FINE_LABELS, SYN_LABELS, ROLE_LABELS,
    build_coarse_to_fine_mask,
    # compat
    NUM_SVO,
)
from multitask_model import SpanMultiTaskModel


class OnnxSpanMultiTaskWrapper(nn.Module):
    """
    Wrapper autour de SpanMultiTaskModel pour export ONNX.
    Remplace les spans Python par des tenseurs plats.
    Exporte 12 sorties : boundary/coarse/fine (NER) + svo_boundary/syn/role/voice/certainty/gender/number/person/verb_ptr.
    """

    def __init__(self, inner: SpanMultiTaskModel):
        super().__init__()
        self.encoder          = inner.encoder
        self.width_emb        = inner.width_emb
        self.span_mlp         = inner.span_mlp
        # Têtes NER
        self.boundary_head    = inner.boundary_head
        self.coarse_head      = inner.coarse_head
        self.fine_head        = inner.fine_head
        # Têtes SVO / syntaxe v4
        self.svo_boundary_head = inner.svo_boundary_head
        self.syn_head          = inner.syn_head
        self.role_head         = inner.role_head
        self.voice_head        = inner.voice_head
        self.certainty_head    = inner.certainty_head
        self.gender_head       = inner.gender_head
        self.number_head       = inner.number_head
        self.person_head       = inner.person_head
        # Verb pointer: bilinéaire query/key
        self.verb_ptr_query    = inner.verb_ptr_query
        self.verb_ptr_key      = inner.verb_ptr_key

        self.max_width_bucket = inner.max_width_bucket

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
        span_starts: torch.Tensor,     # [N]
        span_ends: torch.Tensor,       # [N]
        span_batch_ids: torch.Tensor,  # [N]
    ):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # [B, L, H]

        N = span_starts.size(0)
        B, L, H = hidden.shape
        reps = []

        for i in range(N):
            b = int(span_batch_ids[i].item())
            l = int(span_starts[i].item())
            r = int(span_ends[i].item())
            hs = hidden[b]  # [L, H]

            start_vec = hs[l]
            end_vec   = hs[r]
            mean_vec  = hs[l : r + 1].mean(dim=0)

            width = min(r - l + 1, self.max_width_bucket - 1)
            w_emb = self.width_emb(torch.tensor(width, device=hidden.device))

            reps.append(torch.cat([start_vec, end_vec, mean_vec, w_emb]))

        if not reps:
            span_h = torch.zeros((0, self.span_mlp[-2].out_features), device=hidden.device)
        else:
            span_reps = torch.stack(reps)  # [N, span_input_dim]
            span_h = self.span_mlp(span_reps)

        # Verb pointer : [N, L] — logits bilinéaires span_query · token_key
        # ptr_queries : [N, 64]
        # ptr_keys    : [B, L, 64]  →  gather par batch index → [N, L, 64]
        # verb_ptr_logits : [N, L]
        ptr_queries = self.verb_ptr_query(span_h)          # [N, 64]
        ptr_keys    = self.verb_ptr_key(hidden)            # [B, L, 64]

        if span_h.size(0) > 0:
            # Gather les keys du bon batch pour chaque span
            gathered_keys = ptr_keys[span_batch_ids]       # [N, L, 64]
            # Produit matriciel bilinéaire : gathered_keys @ ptr_queries^T
            verb_ptr_logits = torch.bmm(
                gathered_keys,                             # [N, L, 64]
                ptr_queries.unsqueeze(-1)                  # [N, 64, 1]
            ).squeeze(-1)                                  # [N, L]
        else:
            verb_ptr_logits = torch.zeros((0, L), device=hidden.device)

        return (
            self.boundary_head(span_h),      # [N, 2]
            self.coarse_head(span_h),         # [N, 9]
            self.fine_head(span_h),           # [N, 32]
            self.svo_boundary_head(span_h),   # [N, 2]
            self.syn_head(span_h),            # [N, 3]  verb_trigger/pron_subj/pron_obj
            self.role_head(span_h),           # [N, 7]  SUBJECT/OBJECT/OBLIQUE...
            self.voice_head(span_h),          # [N, 2]  active/passive
            self.certainty_head(span_h),      # [N, 3]  certain/modal/denied
            self.gender_head(span_h),         # [N, 3]
            self.number_head(span_h),         # [N, 2]
            self.person_head(span_h),         # [N, 3]
            verb_ptr_logits,                  # [N, L]
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Chemin vers best_model_multitask.pt")
    ap.add_argument("--model-name", default="microsoft/mdeberta-v3-base")
    ap.add_argument("--output", default="best_model_multitask.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--max-spans", type=int, default=64, help="Nb de spans factices pour le trace")
    args = ap.parse_args()

    device = "cpu"
    print(f"Chargement du checkpoint : {args.checkpoint}")
    inner = SpanMultiTaskModel(model_name=args.model_name).to(device).float()
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    inner.load_state_dict(state)
    inner.eval()

    model = OnnxSpanMultiTaskWrapper(inner).eval()

    B, L, N = 1, args.seq_len, args.max_spans
    dummy_input_ids      = torch.zeros(B, L, dtype=torch.long)
    dummy_attention_mask = torch.ones(B, L, dtype=torch.long)
    dummy_span_starts    = torch.arange(N, dtype=torch.long) % (L - 2) + 1
    dummy_span_ends      = (dummy_span_starts + 1).clamp(max=L - 2)
    dummy_span_batch_ids = torch.zeros(N, dtype=torch.long)

    out_path = Path(args.output)
    print(f"Export ONNX → {out_path}  (opset={args.opset})")

    OUTPUT_NAMES = [
        "boundary_logits", "coarse_logits", "fine_logits",
        "svo_boundary_logits", "syn_logits", "role_logits",
        "voice_logits", "certainty_logits",
        "gender_logits", "number_logits", "person_logits",
        "verb_ptr_logits",
    ]
    dynamic_axes = {
        "input_ids":           {0: "batch", 1: "seq_len"},
        "attention_mask":      {0: "batch", 1: "seq_len"},
        "span_starts":         {0: "num_spans"},
        "span_ends":           {0: "num_spans"},
        "span_batch_ids":      {0: "num_spans"},
    }
    for name in OUTPUT_NAMES:
        if name == "verb_ptr_logits":
            dynamic_axes[name] = {0: "num_spans", 1: "seq_len"}
        else:
            dynamic_axes[name] = {0: "num_spans"}

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_span_starts, dummy_span_ends, dummy_span_batch_ids),
        str(out_path),
        opset_version=args.opset,
        input_names=["input_ids", "attention_mask", "span_starts", "span_ends", "span_batch_ids"],
        output_names=OUTPUT_NAMES,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    print(f"✅ Exporté : {out_path}")
    print(f"   Coarse labels ({len(COARSE_LABELS)}) : {COARSE_LABELS}")
    print(f"   Fine labels   ({len(FINE_LABELS)}) : {FINE_LABELS[:5]}…")
    print(f"   Syn labels    ({len(SYN_LABELS)}) : {SYN_LABELS}")
    print(f"   Role labels   ({len(ROLE_LABELS)}) : {ROLE_LABELS}")


if __name__ == "__main__":
    main()

