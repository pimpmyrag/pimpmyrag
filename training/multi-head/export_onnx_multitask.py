#!/usr/bin/env python3
"""
Export du SpanMultiTaskModel vers ONNX avec des entrées plates (sans structure Python).

Interface ONNX exportée :
  Inputs:
    - input_ids                [batch, seq_len]  int64
    - attention_mask           [batch, seq_len]  int64
    - span_starts              [N]               int64   (indices tok_start de chaque span)
    - span_ends                [N]               int64   (indices tok_end de chaque span)
    - span_batch_ids           [N]               int64   (batch index pour chaque span)

  Outputs (NER) :
    - boundary_logits          [N, 2]            float32
    - coarse_logits            [N, 10]           float32
    - fine_logits              [N, 38]           float32  (masqué par coarse pour l'inférence)

  Outputs (SVO / syntaxe v4) :
    - svo_boundary_logits      [N, 2]            float32   détecte les spans verb_trigger/pron
    - syn_logits               [N, 3]            float32   verb_trigger | pron_subj | pron_obj
    - role_coarse_logits       [N, 5]            float32   SUBJ | OBJ | OBLIQ | APPOS | OTHER
    - role_oblique_logits      [N, 10]           float32   sous-types OBLIQUE (gatés par P(OBLIQ))
    - voice_logits             [N, 2]            float32   active / passive (verb_trigger uniquement)
    - certainty_logits         [N, 3]            float32   certain | modal | denied (verb_trigger uniquement)
    - gender_logits            [N, 2]            float32   M / F
    - number_logits            [N, 2]            float32   SG / PL
    - person_logits            [N, 3]            float32   1 / 2 / 3
    - verb_ptr_logits          [N, seq_len]      float32   pointer vers verbe gouverneur (arguments uniquement)
    - verb_family_logits       [N, 12]           float32   famille sémantique du verbe (verb_trigger uniquement)
    - verb_polarity_logits     [N, 3]            float32   NEGATIVE | NEUTRAL | POSITIVE
    - verb_aspect_logits       [N, 2]            float32   DURATIF | PONCTUEL
    - verb_source_logits       [N, 3]            float32   DIRECT | HYPOTHETICAL | REPORTED
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from labels import (
    NUM_FINE, NUM_SYN, NUM_VOICE, NUM_CERTAINTY,
    NUM_ROLE_COARSE, NUM_ROLE_OBLIQUE,
    NUM_GENDER, NUM_NUMBER, NUM_PERSON,
    COARSE_LABELS, FINE_LABELS, SYN_LABELS,
    ROLE_COARSE_LABELS, ROLE_OBLIQUE_LABELS,
    NUM_VERB_FAMILY, NUM_VERB_POLARITY, NUM_VERB_ASPECT, NUM_VERB_SOURCE,
    VERB_FAMILY_LABELS, VERB_POLARITY_LABELS, VERB_ASPECT_LABELS, VERB_SOURCE_LABELS,
    build_coarse_to_fine_mask,
)
from multitask_model import SpanMultiTaskModel


class OnnxSpanMultiTaskWrapper(nn.Module):
    """
    Wrapper autour de SpanMultiTaskModel pour export ONNX.
    Exporte 17 sorties : boundary/coarse/fine (NER) +
    svo_boundary/syn/role_coarse/role_oblique/voice/certainty/gender/number/person/verb_ptr +
    verb_family/verb_polarity/verb_aspect/verb_source.
    """

    def __init__(self, inner: SpanMultiTaskModel):
        super().__init__()
        self.encoder               = inner.encoder
        self.width_emb             = inner.width_emb
        self.span_mlp              = inner.span_mlp
        # Têtes NER
        self.boundary_head         = inner.boundary_head
        self.coarse_head           = inner.coarse_head
        self.fine_head             = inner.fine_head
        # Têtes SVO / syntaxe v4
        self.svo_boundary_head     = inner.svo_boundary_head
        self.syn_head              = inner.syn_head
        self.role_coarse_head      = inner.role_coarse_head
        self.ner_fine_to_oblique   = inner.ner_fine_to_oblique
        self.role_oblique_head     = inner.role_oblique_head
        self.voice_head            = inner.voice_head
        self.certainty_head        = inner.certainty_head
        self.gender_head           = inner.gender_head
        self.number_head           = inner.number_head
        self.person_head           = inner.person_head
        # Verb pointer: bilinéaire query/key
        self.verb_ptr_query        = inner.verb_ptr_query
        self.verb_ptr_key          = inner.verb_ptr_key
        # VerbFam heads
        self.verb_family_mlp       = inner.verb_family_mlp
        self.verb_family_head      = inner.verb_family_head
        self.verb_polarity_head    = inner.verb_polarity_head
        self.verb_aspect_head      = inner.verb_aspect_head
        self.verb_source_head      = inner.verb_source_head

        self.max_width_bucket = inner.max_width_bucket

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
        span_starts: torch.Tensor,     # [N]
        span_ends: torch.Tensor,       # [N]
        span_batch_ids: torch.Tensor,  # [N]
    ):
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state  # [B, L, H]

        # ── Span representation vectorisée ─────────────────────────────
        start_vecs = hidden[span_batch_ids, span_starts]  # [N, H]
        end_vecs   = hidden[span_batch_ids, span_ends]    # [N, H]

        lengths = (span_ends - span_starts + 1).float().clamp(min=1).unsqueeze(-1)

        zeros = torch.zeros(
            hidden.size(0),
            1,
            hidden.size(2),
            device=hidden.device,
            dtype=hidden.dtype,
        )

        prefix = torch.cat([zeros, torch.cumsum(hidden, dim=1)], dim=1)  # [B, L+1, H]

        mean_vecs = (
            prefix[span_batch_ids, span_ends + 1]
            - prefix[span_batch_ids, span_starts]
        ) / lengths

        widths = (span_ends - span_starts + 1).clamp(
            min=1,
            max=self.max_width_bucket - 1,
        )

        w_embs = self.width_emb(widths)

        span_reps = torch.cat(
            [start_vecs, end_vecs, mean_vecs, w_embs],
            dim=-1,
        )

        span_h = self.span_mlp(span_reps)

        # ── NER ────────────────────────────────────────────────────────
        boundary_logits = self.boundary_head(span_h)
        coarse_logits   = self.coarse_head(span_h)
        fine_logits_raw = self.fine_head(span_h)

        # ── Role coarse + oblique fine ─────────────────────────────────
        rc_logits = self.role_coarse_head(span_h)

        fine_probs = torch.softmax(fine_logits_raw.detach(), dim=-1)
        oblique_h = span_h + self.ner_fine_to_oblique(fine_probs)

        rc_probs = torch.softmax(rc_logits.detach(), dim=-1)
        rc_log_gate = torch.log(rc_probs[:, 2:3].clamp(min=1e-9))

        role_oblique_logits = self.role_oblique_head(oblique_h) + rc_log_gate

        # ── Verb pointer : [N, L] ──────────────────────────────────────
        ptr_queries = self.verb_ptr_query(span_h)      # [N, D]
        ptr_keys    = self.verb_ptr_key(hidden)        # [B, L, D]

        gathered_keys = ptr_keys[span_batch_ids]       # [N, L, D]

        verb_ptr_logits = torch.bmm(
            gathered_keys,
            ptr_queries.unsqueeze(-1),
        ).squeeze(-1)                                  # [N, L]

        # ── Verb semantic heads ────────────────────────────────────────
        span_h_vf = self.verb_family_mlp(span_h)

        verb_family_logits   = self.verb_family_head(span_h_vf)
        verb_polarity_logits = self.verb_polarity_head(span_h_vf)
        verb_aspect_logits   = self.verb_aspect_head(span_h_vf)
        verb_source_logits   = self.verb_source_head(span_h_vf)

        return (
            boundary_logits,
            coarse_logits,
            fine_logits_raw,

            self.svo_boundary_head(span_h),
            self.syn_head(span_h),

            rc_logits,
            role_oblique_logits,

            self.voice_head(span_h),
            self.certainty_head(span_h),
            self.gender_head(span_h),
            self.number_head(span_h),
            self.person_head(span_h),

            verb_ptr_logits,

            verb_family_logits,
            verb_polarity_logits,
            verb_aspect_logits,
            verb_source_logits,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Chemin vers best_model_multitask.pt")
    ap.add_argument("--model-name", default="microsoft/deberta-v3-base")
    ap.add_argument("--output", default="best_model_multitask.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--max-spans", type=int, default=64, help="Nb de spans factices pour le trace")
    args = ap.parse_args()

    device = "cpu"
    print(f"Chargement du checkpoint : {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    num_coarse = state["coarse_head.weight"].shape[0]
    print(f"   num_coarse={num_coarse} (détecté depuis checkpoint)")
    inner = SpanMultiTaskModel(model_name=args.model_name, num_coarse=num_coarse).to(device).float()
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
        "svo_boundary_logits", "syn_logits",
        "role_coarse_logits", "role_oblique_logits",
        "voice_logits", "certainty_logits",
        "gender_logits", "number_logits", "person_logits",
        "verb_ptr_logits",
        "verb_family_logits", "verb_polarity_logits",
        "verb_aspect_logits", "verb_source_logits",
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
        dynamo=False,
    )
    print(f"✅ Exporté : {out_path}")
    print(f"   Coarse labels        ({len(COARSE_LABELS)})  : {COARSE_LABELS}")
    print(f"   Fine labels          ({len(FINE_LABELS)})  : {FINE_LABELS[:5]}…")
    print(f"   Syn labels           ({len(SYN_LABELS)})   : {SYN_LABELS}")
    print(f"   Role coarse labels   ({len(ROLE_COARSE_LABELS)})   : {ROLE_COARSE_LABELS}")
    print(f"   Role oblique labels  ({len(ROLE_OBLIQUE_LABELS)})  : {ROLE_OBLIQUE_LABELS}")
    print(f"   Verb family labels   ({len(VERB_FAMILY_LABELS)})  : {VERB_FAMILY_LABELS}")
    print(f"   Verb polarity labels ({len(VERB_POLARITY_LABELS)})   : {VERB_POLARITY_LABELS}")
    print(f"   Verb aspect labels   ({len(VERB_ASPECT_LABELS)})   : {VERB_ASPECT_LABELS}")
    print(f"   Verb source labels   ({len(VERB_SOURCE_LABELS)})   : {VERB_SOURCE_LABELS}")


if __name__ == "__main__":
    main()

