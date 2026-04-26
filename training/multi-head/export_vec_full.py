#!/usr/bin/env python3
"""
export_vec_full.py
~~~~~~~~~~~~~~~~~~
Export vectorisé du SpanMultiTaskModel vers ONNX — toutes les 8 têtes.

Interface ONNX :
  Inputs:
    input_ids        [B, L]  int64
    attention_mask   [B, L]  int64
    span_starts      [N]     int64
    span_ends        [N]     int64
    span_batch_ids   [N]     int64

  Outputs (NER):
    boundary_logits      [N, 2]
    coarse_logits        [N, 9]
    fine_logits          [N, 32]

  Outputs (SVO / syntaxe):
    svo_boundary_logits  [N, 2]
    svo_logits           [N, 6]
    voice_logits         [N, 2]
    gender_logits        [N, 3]
    number_logits        [N, 3]
"""
import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from multitask_model import SpanMultiTaskModel
from labels import COARSE_LABELS, FINE_LABELS, SVO_LABELS, VOICE_LABELS, GENDER_LABELS, NUMBER_LABELS


# ──────────────────────────────────────────────────────────
#  Wrapper 100% tensoriel — aucune boucle Python, aucun .item()
# ──────────────────────────────────────────────────────────

class OnnxSpanWrapperFull(nn.Module):
    """
    Wrapper ONNX pour SpanMultiTaskModel.
    Exporte les 8 têtes (NER + SVO + voice + gender + number).
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
        # Têtes SVO / syntaxe
        self.svo_boundary_head = inner.svo_boundary_head
        self.svo_head          = inner.svo_head
        self.voice_head        = inner.voice_head
        self.gender_head       = inner.gender_head
        self.number_head       = inner.number_head

        self.max_width_bucket  = inner.max_width_bucket

    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, L]  int64
        attention_mask: torch.Tensor,   # [B, L]  int64
        span_starts:    torch.Tensor,   # [N]     int64
        span_ends:      torch.Tensor,   # [N]     int64
        span_batch_ids: torch.Tensor,   # [N]     int64
    ):
        # ── Encodage backbone ────────────────────────────────────────
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state                         # [B, L, H]

        # ── Vecteurs start / end ─────────────────────────────────────
        start_vecs = hidden[span_batch_ids, span_starts]   # [N, H]
        end_vecs   = hidden[span_batch_ids, span_ends]     # [N, H]

        # ── Vecteur moyen via prefix sum ─────────────────────────────
        # IMPORTANT : évite le tenseur intermédiaire [N, L, H] qui explose
        # la mémoire (ex. N=5000, L=24, H=768 → 350 MB de données temporaires).
        #
        # Idée : prefix[b, i] = sum(hidden[b, 0 .. i-1])
        #   mean(hidden[b, s:e+1]) = (prefix[b, e+1] - prefix[b, s]) / (e-s+1)
        #
        # Complexité mémoire : O(B·L·H) au lieu de O(N·L·H)  ← critique pour N >> B
        lengths = (span_ends - span_starts + 1).float().clamp(min=1).unsqueeze(1)  # [N, 1]
        zeros  = torch.zeros(hidden.size(0), 1, hidden.size(2),
                             device=hidden.device, dtype=hidden.dtype)              # [B, 1, H]
        prefix = torch.cumsum(hidden, dim=1)                                        # [B, L, H]
        prefix = torch.cat([zeros, prefix], dim=1)                                  # [B, L+1, H]
        mean_vecs = (
            prefix[span_batch_ids, span_ends + 1] -   # [N, H]  cumulée jusqu'à end incl.
            prefix[span_batch_ids, span_starts]        # [N, H]  cumulée avant start
        ) / lengths                                    # [N, H]

        # ── Width embedding ──────────────────────────────────────────
        widths = (span_ends - span_starts + 1).clamp(min=1, max=self.max_width_bucket - 1)
        w_embs = self.width_emb(widths)                                  # [N, W]

        # ── MLP partagé ──────────────────────────────────────────────
        span_reps = torch.cat([start_vecs, end_vecs, mean_vecs, w_embs], dim=-1)  # [N, D]
        span_h    = self.span_mlp(span_reps)                                        # [N, H']

        # ── 8 sorties ────────────────────────────────────────────────
        return (
            self.boundary_head(span_h),       # [N, 2]
            self.coarse_head(span_h),          # [N, 9]
            self.fine_head(span_h),            # [N, 32]
            self.svo_boundary_head(span_h),    # [N, 2]
            self.svo_head(span_h),             # [N, 6]
            self.voice_head(span_h),           # [N, 2]
            self.gender_head(span_h),          # [N, 3]
            self.number_head(span_h),          # [N, 3]
        )


# ──────────────────────────────────────────────────────────
#  Export
# ──────────────────────────────────────────────────────────

OUTPUT_NAMES = [
    "boundary_logits",
    "coarse_logits",
    "fine_logits",
    "svo_boundary_logits",
    "svo_logits",
    "voice_logits",
    "gender_logits",
    "number_logits",
]

DYNAMIC_AXES = {
    "input_ids":           {0: "batch",     1: "seq_len"},
    "attention_mask":      {0: "batch",     1: "seq_len"},
    "span_starts":         {0: "num_spans"},
    "span_ends":           {0: "num_spans"},
    "span_batch_ids":      {0: "num_spans"},
    **{name: {0: "num_spans"} for name in OUTPUT_NAMES},
}


def main():
    ap = argparse.ArgumentParser(description="Export ONNX — 8 têtes NER+SVO vectorisé")
    ap.add_argument("--checkpoint",  required=True,  help="Chemin vers checkpoint_best_multitask.pt")
    ap.add_argument("--output",      required=True,  help="Chemin de sortie .onnx")
    ap.add_argument("--model-name",  default="microsoft/deberta-v3-base")
    ap.add_argument("--opset",       type=int, default=17)
    ap.add_argument("--seq-len",     type=int, default=128, help="Longueur séquence pour le trace dummy")
    ap.add_argument("--num-spans",   type=int, default=64,  help="Nb spans dummy pour le trace")
    args = ap.parse_args()

    # ── Chargement modèle ────────────────────────────────────────────
    print(f"📦 Chargement : {args.checkpoint}")
    inner = SpanMultiTaskModel(model_name=args.model_name).float()
    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    inner.load_state_dict(state)
    inner.eval()

    model = OnnxSpanWrapperFull(inner).eval()

    # ── Inputs dummy ────────────────────────────────────────────────
    B, L, N = 1, args.seq_len, args.num_spans
    dummy_ids  = torch.zeros(B, L, dtype=torch.long)
    dummy_mask = torch.ones(B, L,  dtype=torch.long)
    dummy_ss   = (torch.arange(N) % (L - 4) + 2).long()
    dummy_se   = (dummy_ss + 2).clamp(max=L - 2)
    dummy_bid  = torch.zeros(N, dtype=torch.long)

    # ── Vérification forward ─────────────────────────────────────────
    with torch.no_grad():
        outs = model(dummy_ids, dummy_mask, dummy_ss, dummy_se, dummy_bid)
    shapes = "  ".join(f"{n}:{tuple(t.shape)}" for n, t in zip(OUTPUT_NAMES, outs))
    print(f"✅ Forward OK  {shapes}")

    # ── Export ───────────────────────────────────────────────────────
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

    # ── Résumé ───────────────────────────────────────────────────────
    size_mb = os.path.getsize(args.output) / 1e6
    try:
        import onnx
        m = onnx.load(args.output)
        ins  = [i.name for i in m.graph.input]
        outs_names = [o.name for o in m.graph.output]
        print(f"✅ Export terminé — {size_mb:.1f} Mo")
        print(f"   Inputs       : {ins}")
        print(f"   Outputs      : {outs_names}")
    except ImportError:
        print(f"✅ Export terminé — {size_mb:.1f} Mo  (onnx non installé, vérif skippée)")

    # ── Métadonnées labels (utile pour Kotlin) ────────────────────────
    meta = {
        "coarse_labels":  COARSE_LABELS,
        "fine_labels":    FINE_LABELS,
        "svo_labels":     list(SVO_LABELS),
        "voice_labels":   list(VOICE_LABELS),
        "gender_labels":  list(GENDER_LABELS),
        "number_labels":  list(NUMBER_LABELS),
    }
    import json
    meta_path = Path(args.output).with_suffix(".labels.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"   Labels JSON  : {meta_path}")


if __name__ == "__main__":
    main()

