#!/usr/bin/env python3
"""
Export vectorisé du SpanMultiTaskModel vers ONNX.
Remplace la boucle Python (qui force le constant folding) par des ops tensorielles pures.
"""
import argparse, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from multitask_model import SpanMultiTaskModel


class OnnxSpanWrapperVec(nn.Module):
    """Wrapper 100% tensoriel — pas de boucle Python ni de .item() — pour ONNX."""

    def __init__(self, inner: SpanMultiTaskModel):
        super().__init__()
        self.encoder        = inner.encoder
        self.width_emb      = inner.width_emb
        self.span_mlp       = inner.span_mlp
        self.boundary_head  = inner.boundary_head
        self.coarse_head    = inner.coarse_head
        self.fine_head      = inner.fine_head
        self.max_width_bucket = inner.max_width_bucket

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
        # hidden[span_batch_ids, span_starts] → [N, H]
        start_vecs = hidden[span_batch_ids, span_starts]   # [N, H]
        end_vecs   = hidden[span_batch_ids, span_ends]     # [N, H]

        # ── mean vector (masque tensoriel) ───────────────────────────
        L = hidden.size(1)
        positions = torch.arange(L, device=hidden.device)   # [L]
        # mask[n, l] = 1 si span_starts[n] <= l <= span_ends[n]
        mask = (
            (positions.unsqueeze(0) >= span_starts.unsqueeze(1)) &
            (positions.unsqueeze(0) <= span_ends.unsqueeze(1))
        ).float()  # [N, L]

        # hidden des exemples correspondants : [N, L, H]
        span_hidden = hidden[span_batch_ids]
        # somme pondérée / longueur
        lengths = (span_ends - span_starts + 1).float().clamp(min=1).unsqueeze(1)  # [N, 1]
        mean_vecs = (span_hidden * mask.unsqueeze(-1)).sum(dim=1) / lengths          # [N, H]

        # ── width embedding ──────────────────────────────────────────
        widths = (span_ends - span_starts + 1).clamp(min=1, max=self.max_width_bucket - 1)
        w_embs = self.width_emb(widths)   # [N, W]

        # ── MLP + têtes ──────────────────────────────────────────────
        span_reps = torch.cat([start_vecs, end_vecs, mean_vecs, w_embs], dim=-1)  # [N, D]
        span_h    = self.span_mlp(span_reps)                                        # [N, H']

        return (
            self.boundary_head(span_h),   # [N, 2]
            self.coarse_head(span_h),     # [N, 9]
            self.fine_head(span_h),       # [N, 32]
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

