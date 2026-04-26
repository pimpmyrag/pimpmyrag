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

  Outputs (SVO / syntaxe) :
    - svo_boundary_logits [N, 2]            float32   détecte les spans verbe/pronom
    - svo_logits          [N, 6]            float32   rôle SVO
    - voice_logits        [N, 2]            float32   ACTIVE / PASSIVE
    - gender_logits       [N, 3]            float32   Masc / Fem / NONE
    - number_logits       [N, 3]            float32   Sing / Plur / NONE
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

from labels import (
    NUM_FINE, NUM_SVO, NUM_VOICE, NUM_GENDER, NUM_NUMBER,
    COARSE_LABELS, FINE_LABELS, SVO_LABELS,
    build_coarse_to_fine_mask,
)
from multitask_model import SpanMultiTaskModel


class OnnxSpanMultiTaskWrapper(nn.Module):
    """
    Wrapper autour de SpanMultiTaskModel pour export ONNX.
    Remplace les spans Python par des tenseurs plats.
    Exporte les 8 sorties : boundary/coarse/fine (NER) + svo_boundary/svo/voice/gender/number.
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
        # Têtes SVO / syntaxe
        self.svo_boundary_head = inner.svo_boundary_head
        self.svo_head          = inner.svo_head
        self.voice_head        = inner.voice_head
        self.gender_head       = inner.gender_head
        self.number_head       = inner.number_head

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

        return (
            self.boundary_head(span_h),      # [N, 2]
            self.coarse_head(span_h),         # [N, 9]
            self.fine_head(span_h),           # [N, 32]
            self.svo_boundary_head(span_h),   # [N, 2]
            self.svo_head(span_h),            # [N, 6]
            self.voice_head(span_h),          # [N, 2]
            self.gender_head(span_h),         # [N, 3]
            self.number_head(span_h),         # [N, 3]
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
        "svo_boundary_logits", "svo_logits", "voice_logits",
        "gender_logits", "number_logits",
    ]
    dynamic_axes = {
        "input_ids":           {0: "batch", 1: "seq_len"},
        "attention_mask":      {0: "batch", 1: "seq_len"},
        "span_starts":         {0: "num_spans"},
        "span_ends":           {0: "num_spans"},
        "span_batch_ids":      {0: "num_spans"},
    }
    for name in OUTPUT_NAMES:
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
    print(f"   SVO labels    ({len(SVO_LABELS)}) : {SVO_LABELS}")


if __name__ == "__main__":
    main()

