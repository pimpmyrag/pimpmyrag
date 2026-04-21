# multitask_model.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from labels import NUM_FINE, NUM_SVO, NUM_VOICE, build_coarse_to_fine_mask


from labels import FINE_LABELS


class SpanMultiTaskModel(nn.Module):
    def __init__(
            self,
            model_name: str,
            num_coarse: int = 9,
            width_emb_dim: int = 32,
            span_hidden_dim: int = 512,
            max_width_bucket: int = 16,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.max_width_bucket = max_width_bucket
        self.width_emb = nn.Embedding(max_width_bucket, width_emb_dim)

        span_input_dim = hidden_size * 3 + width_emb_dim

        self.span_mlp = nn.Sequential(
            nn.Linear(span_input_dim, span_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(span_hidden_dim, span_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Heads NER
        self.boundary_head = nn.Linear(span_hidden_dim, 2)
        self.coarse_head = nn.Linear(span_hidden_dim, num_coarse)
        self.fine_head = nn.Linear(span_hidden_dim, NUM_FINE)

        # Head SVO : svo_verb / svo_subject / svo_object / svo_iobj / pron_subj / pron_obj
        self.svo_head = nn.Linear(span_hidden_dim, NUM_SVO)
        # Head svo_boundary : détecte les spans verbe/pronom (indépendant du boundary NER)
        self.svo_boundary_head = nn.Linear(span_hidden_dim, 2)
        # Head voice : ACTIVE / PASSIVE  (prédite sur les svo_verb uniquement)
        self.voice_head = nn.Linear(span_hidden_dim, NUM_VOICE)

        # Coarse → fine mask
        self.register_buffer("coarse_fine_mask", build_coarse_to_fine_mask())

    def _bucket_width(self, width: int) -> int:
        return min(width, self.max_width_bucket - 1)

    def _build_span_representations(self, hidden_states, spans):
        reps = []
        span_indices = []
        device = hidden_states.device

        for b_idx, sample_spans in enumerate(spans):
            hs = hidden_states[b_idx]
            for sp in sample_spans:
                l = sp["tok_start"]
                r = sp["tok_end"]
                if l < 0 or r >= hs.size(0) or l > r:
                    continue

                start = hs[l]
                end = hs[r]
                mean = hs[l : r + 1].mean(dim=0)

                width = self._bucket_width(r - l + 1)
                w_emb = self.width_emb(torch.tensor(width, device=device))

                reps.append(torch.cat([start, end, mean, w_emb], dim=-1))
                span_indices.append(len(reps) - 1)

        if not reps:
            return (
                torch.zeros((0, self.span_mlp[0].in_features), device=device),
                torch.empty((0,), dtype=torch.long, device=device),
            )

        return torch.stack(reps), torch.arange(len(reps), device=device)

    def forward(self, batch):
        enc = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        hidden = enc.last_hidden_state

        span_reps, span_indices = self._build_span_representations(hidden, batch["spans"])
        span_h = self.span_mlp(span_reps)

        return {
            "span_reps": span_h,
            "span_indices": span_indices,
            "boundary_logits": self.boundary_head(span_h),
            "coarse_logits": self.coarse_head(span_h),
            "fine_logits": self.fine_head(span_h),
            "svo_boundary_logits": self.svo_boundary_head(span_h),
            "svo_logits": self.svo_head(span_h),
            "voice_logits": self.voice_head(span_h),
        }

    def compute_loss(
            self,
            outputs,
            boundary_labels,
            coarse_labels,
            fine_labels,
            svo_boundary_labels,
            svo_labels,
            voice_labels,
            sample_weights,
            boundary_class_weights=None,
            coarse_class_weights=None,
            fine_class_weights=None,
            lambda_boundary=1.0,
            lambda_coarse=1.0,
            lambda_fine=1.2,
            lambda_svo_boundary=1.0,
            lambda_svo=1.0,
            lambda_voice=0.5,
            lambda_compat=0.0,
            focal_gamma=0.0,
    ):
        """
        Loss multi-têtes :
          boundary / coarse / fine  : NER (inchangé)
          svo                       : rôle SVO sur spans silver (positive only)
          voice                     : ACTIVE/PASSIVE sur svo_verb uniquement
        """
        device = outputs["boundary_logits"].device

        b_logits          = outputs["boundary_logits"]
        c_logits          = outputs["coarse_logits"]
        f_logits          = outputs["fine_logits"]
        svo_b_logits      = outputs["svo_boundary_logits"]
        svo_logits        = outputs["svo_logits"]
        voice_logits      = outputs["voice_logits"]

        boundary_labels      = boundary_labels.to(device=device, dtype=torch.long)
        coarse_labels        = coarse_labels.to(device=device, dtype=torch.long)
        fine_labels          = fine_labels.to(device=device, dtype=torch.long)
        svo_boundary_labels  = svo_boundary_labels.to(device=device, dtype=torch.long)
        svo_labels           = svo_labels.to(device=device, dtype=torch.long)
        voice_labels         = voice_labels.to(device=device, dtype=torch.long)
        sample_weights       = sample_weights.to(device=device, dtype=torch.float32)

        if boundary_class_weights is not None:
            boundary_class_weights = boundary_class_weights.to(device=device, dtype=torch.float32)
        if coarse_class_weights is not None:
            coarse_class_weights = coarse_class_weights.to(device=device, dtype=torch.float32)

        # ── 1) Boundary ────────────────────────────────────────────
        loss_b_per_span = F.cross_entropy(b_logits, boundary_labels,
                                           weight=boundary_class_weights, reduction="none")
        if focal_gamma > 0.0:
            b_probs = F.softmax(b_logits.detach(), dim=-1)
            p_t = b_probs.gather(1, boundary_labels.unsqueeze(1)).squeeze(1)
            loss_b_per_span = loss_b_per_span * (1.0 - p_t) ** focal_gamma
        loss_b = (loss_b_per_span * sample_weights).mean()

        # ── 2) Coarse (positive-only, comme fine) ─────────────────
        # On n'entraîne le coarse que sur les spans boundary=1
        # pour éviter que les FP boundary polluent la précision coarse
        pos_mask_coarse = (boundary_labels == 1)
        if pos_mask_coarse.any():
            loss_c_per_span = F.cross_entropy(
                c_logits[pos_mask_coarse],
                coarse_labels[pos_mask_coarse],
                weight=coarse_class_weights,
                reduction="none"
            )
            loss_c = (loss_c_per_span * sample_weights[pos_mask_coarse]).mean()
        else:
            loss_c = torch.tensor(0.0, device=device)

        # ── 3) Fine (spans NER positifs avec un vrai label fine) ──────
        pos_mask = (boundary_labels == 1) & (fine_labels < f_logits.size(-1))
        if pos_mask.any():
            f_logits_pos = f_logits[pos_mask]
            f_labels_pos = fine_labels[pos_mask]
            loss_f = (F.cross_entropy(f_logits_pos, f_labels_pos, reduction="none")
                      * sample_weights[pos_mask]).mean()
        else:
            loss_f = torch.tensor(0.0, device=device)

        # ── 4) SVO boundary (verbes + pronoms) ────────────────────
        loss_svo_b = (F.cross_entropy(svo_b_logits, svo_boundary_labels, reduction="none")
                      * sample_weights).mean()

        # ── 5) SVO (spans silver avec rôle SVO uniquement) ─────────
        # SVO_NONE_ID = NUM_SVO (sentinel pour les spans non-SVO)
        svo_mask = (svo_labels < svo_logits.size(-1))
        if svo_mask.any():
            loss_svo = (F.cross_entropy(svo_logits[svo_mask], svo_labels[svo_mask], reduction="none")
                        * sample_weights[svo_mask]).mean()
        else:
            loss_svo = torch.tensor(0.0, device=device)

        # ── 5) Voice (svo_verb uniquement) ─────────────────────────
        # VOICE_NONE_ID = NUM_VOICE (sentinel pour les spans non-verb)
        voice_mask = (voice_labels < voice_logits.size(-1))
        if voice_mask.any():
            loss_voice = (F.cross_entropy(voice_logits[voice_mask], voice_labels[voice_mask], reduction="none")
                          * sample_weights[voice_mask]).mean()
        else:
            loss_voice = torch.tensor(0.0, device=device)

        # ── Total ──────────────────────────────────────────────────
        total_loss = (
            lambda_boundary     * loss_b
            + lambda_coarse     * loss_c
            + lambda_fine       * loss_f
            + lambda_svo_boundary * loss_svo_b
            + lambda_svo        * loss_svo
            + lambda_voice      * loss_voice
        )

        return {
            "loss":                 total_loss,
            "loss_boundary":        loss_b.detach(),
            "loss_coarse":          loss_c.detach(),
            "loss_fine":            loss_f.detach(),
            "loss_svo_boundary":    loss_svo_b.detach(),
            "loss_svo":             loss_svo.detach(),
            "loss_voice":           loss_voice.detach(),
            "num_positive_spans":   int(pos_mask.sum().item()),
            "num_svo_spans":        int(svo_mask.sum().item()),
        }

