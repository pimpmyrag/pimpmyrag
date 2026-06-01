#!/usr/bin/env python3
"""
Export vectorisé du SpanMultiTaskModel vers ONNX.
Remplace la boucle Python (qui force le constant folding) par des ops tensorielles pures.

Sorties ONNX :
  - boundary_logits  [N, 2]       : entité / non-entité
  - coarse_logits    [N, num_coarse] : famille d'entité
  - fine_logits      [N, NUM_FINE]   : type fin — DÉJÀ MASQUÉ par coarse (même cascade que le training)
    → le consommateur peut directement faire argmax sans recalculer le masque
"""
import argparse, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from multitask_model import SpanMultiTaskModel


class OnnxSpanWrapperVec(nn.Module):
    """Wrapper 100% tensoriel — pas de boucle Python ni de .item() — pour ONNX.

    Implémente la même cascade NER que SpanMultiTaskModel.forward() :
      1. boundary_logits  depuis span_h
      2. span_h_coarse = span_h * sigmoid(boundary_logits[:,1:2])   [boundary gate]
      3. coarse_logits  depuis span_h_coarse
      4. fine_logits    depuis span_h_coarse, biaisé par log(softmax(coarse) @ coarse_fine_mask)

    Garantit zéro mismatch entre le training et la production ONNX.
    """

    def __init__(self, inner: SpanMultiTaskModel):
        super().__init__()
        self.encoder          = inner.encoder
        self.width_emb        = inner.width_emb
        self.span_mlp         = inner.span_mlp
        self.boundary_head    = inner.boundary_head
        self.coarse_head      = inner.coarse_head
        self.fine_head        = inner.fine_head
        self.max_width_bucket = inner.max_width_bucket
        # Masque coarse→fine : [num_coarse, NUM_FINE] bool
        self.register_buffer("coarse_fine_mask", inner.coarse_fine_mask.float())

    def forward(
        self,
        input_ids:       torch.Tensor,   # [B, L]  int64
        attention_mask:  torch.Tensor,   # [B, L]  int64
        span_starts:     torch.Tensor,   # [N]     int64
        span_ends:       torch.Tensor,   # [N]     int64
        span_batch_ids:  torch.Tensor,   # [N]     int64
    ):
        # ── encodage ─────────────────────────────────────────────────
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # [B, L, H]

        # ── start / end vectors ──────────────────────────────────────
        start_vecs = hidden[span_batch_ids, span_starts]   # [N, H]
        end_vecs   = hidden[span_batch_ids, span_ends]     # [N, H]

        # ── mean vector (masque tensoriel) ───────────────────────────
        L = hidden.size(1)
        positions = torch.arange(L, device=hidden.device)   # [L]
        mask = (
            (positions.unsqueeze(0) >= span_starts.unsqueeze(1)) &
            (positions.unsqueeze(0) <= span_ends.unsqueeze(1))
        ).float()  # [N, L]

        span_hidden = hidden[span_batch_ids]
        lengths = (span_ends - span_starts + 1).float().clamp(min=1).unsqueeze(1)
        mean_vecs = (span_hidden * mask.unsqueeze(-1)).sum(dim=1) / lengths  # [N, H]

        # ── width embedding ──────────────────────────────────────────
        widths = (span_ends - span_starts + 1).clamp(min=1, max=self.max_width_bucket - 1)
        w_embs = self.width_emb(widths)   # [N, W]

        # ── MLP ──────────────────────────────────────────────────────
        span_reps = torch.cat([start_vecs, end_vecs, mean_vecs, w_embs], dim=-1)  # [N, D]
        span_h    = self.span_mlp(span_reps)                                        # [N, H']

        # ── Cascade NER : boundary → coarse (gated) → fine (soft-masked) ────
        #
        # Strictement identique à SpanMultiTaskModel.forward() :
        #   1. boundary sur span_h complet
        #   2. span_h_coarse = span_h * sigmoid(boundary[:,1:2])
        #      → gradient coarse/fine atténué sur spans négatifs (cohérence train)
        #   3. fine = fine_raw + log(softmax(coarse) @ coarse_fine_mask)
        #      → sortie ONNX directement argmax-able, pas besoin de post-traitement

        boundary_logits = self.boundary_head(span_h)                        # [N, 2]

        bnd_gate      = torch.sigmoid(boundary_logits[:, 1:2])              # [N, 1]
        span_h_coarse = span_h * bnd_gate                                   # [N, H']

        coarse_logits  = self.coarse_head(span_h_coarse)                    # [N, C]
        fine_logits_raw = self.fine_head(span_h_coarse)                     # [N, F]

        coarse_probs   = F.softmax(coarse_logits, dim=-1)                   # [N, C]
        coarse_gate_f  = coarse_probs @ self.coarse_fine_mask               # [N, F]
        fine_logits    = fine_logits_raw + torch.log(coarse_gate_f.clamp(min=1e-9))

        return (
            boundary_logits,   # [N, 2]
            coarse_logits,     # [N, num_coarse]
            fine_logits,       # [N, NUM_FINE]  ← déjà masqué par coarse
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",  required=True)
    ap.add_argument("--output",      required=True)
    ap.add_argument("--model-name",  default="microsoft/deberta-v3-base")
    ap.add_argument("--opset",       type=int, default=17)
    args = ap.parse_args()

    print(f"📦 Chargement : {args.checkpoint}")
    inner = SpanMultiTaskModel(model_name=args.model_name).float()
    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    inner.load_state_dict(state)
    inner.eval()

    model = OnnxSpanWrapperVec(inner).eval()

    B, L, N = 1, 128, 64
    dummy_ids  = torch.zeros(B, L, dtype=torch.long)
    dummy_mask = torch.ones(B, L,  dtype=torch.long)
    dummy_ss   = (torch.arange(N) % (L - 4) + 2).long()
    dummy_se   = (dummy_ss + 2).clamp(max=L - 2)
    dummy_bid  = torch.zeros(N, dtype=torch.long)

    # Vérification forward avant export
    with torch.no_grad():
        b, c, f = model(dummy_ids, dummy_mask, dummy_ss, dummy_se, dummy_bid)
    print(f"✅ Forward OK  boundary:{tuple(b.shape)} coarse:{tuple(c.shape)} fine:{tuple(f.shape)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"⚙️  Export ONNX opset={args.opset} → {args.output}")

    import logging; logging.disable(logging.WARNING)
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask, dummy_ss, dummy_se, dummy_bid),
        args.output,
        dynamo=False,
        opset_version=args.opset,
        input_names=["input_ids", "attention_mask", "span_starts", "span_ends", "span_batch_ids"],
        output_names=["boundary_logits", "coarse_logits", "fine_logits"],
        dynamic_axes={
            "input_ids":       {0: "batch", 1: "seq_len"},
            "attention_mask":  {0: "batch", 1: "seq_len"},
            "span_starts":     {0: "num_spans"},
            "span_ends":       {0: "num_spans"},
            "span_batch_ids":  {0: "num_spans"},
            "boundary_logits": {0: "num_spans"},
            "coarse_logits":   {0: "num_spans"},
            "fine_logits":     {0: "num_spans"},
        },
        do_constant_folding=True,
    )

    import os, onnx
    size_mb = os.path.getsize(args.output) / 1e6
    m = onnx.load(args.output)
    ins  = [i.name for i in m.graph.input]
    outs = [o.name for o in m.graph.output]
    print(f"✅ Export terminé — {size_mb:.1f} Mo")
    print(f"   Inputs  : {ins}")
    print(f"   Outputs : {outs}")


if __name__ == "__main__":
    main()

