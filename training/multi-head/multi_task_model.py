# multitask_model.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from labels import NUM_FINE, build_coarse_to_fine_mask


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

        # Heads
        self.boundary_head = nn.Linear(span_hidden_dim, 2)
        self.coarse_head = nn.Linear(span_hidden_dim, num_coarse)
        self.fine_head = nn.Linear(span_hidden_dim, NUM_FINE)

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
        }

    def compute_loss(
            self,
            outputs,
            boundary_labels,
            coarse_labels,
            fine_labels,
            sample_weights,
            boundary_class_weights=None,
            coarse_class_weights=None,
            fine_class_weights=None,   # gardé pour compat API, non utilisé ici
            lambda_boundary=1.0,
            lambda_coarse=1.0,
            lambda_fine=1.2,
            lambda_compat=0.0,         # gardé pour compat API, non utilisé ici
            focal_gamma=0.0,           # 0.0 = CE standard, 2.0 = focal loss sur boundary
    ):
        """
        Loss adaptée à l'architecture :
          - boundary : binaire, loss sur tous les spans
          - coarse   : 6 familles + NONE, loss sur tous les spans
          - fine     : 22 labels positifs uniquement, loss sur les spans positifs seulement

        Args:
            outputs: dict contenant
                - "boundary_logits": [N, 2]
                - "coarse_logits":   [N, 7]
                - "fine_logits":     [N, 22]

            boundary_labels: [N]  0/1
            coarse_labels:   [N]  0..6
            fine_labels:     [N]  0..21 pour les spans positifs
                                   (la valeur des spans négatifs est ignorée)
            sample_weights:  [N]  poids par span

        Retourne:
            dict avec:
                - loss
                - loss_boundary
                - loss_coarse
                - loss_fine
                - num_positive_spans
        """
        device = outputs["boundary_logits"].device

        b_logits = outputs["boundary_logits"]
        c_logits = outputs["coarse_logits"]
        f_logits = outputs["fine_logits"]

        boundary_labels = boundary_labels.to(device=device, dtype=torch.long)
        coarse_labels = coarse_labels.to(device=device, dtype=torch.long)
        fine_labels = fine_labels.to(device=device, dtype=torch.long)
        sample_weights = sample_weights.to(device=device, dtype=torch.float32)

        if boundary_class_weights is not None:
            boundary_class_weights = boundary_class_weights.to(device=device, dtype=torch.float32)

        if coarse_class_weights is not None:
            coarse_class_weights = coarse_class_weights.to(device=device, dtype=torch.float32)

        # ─────────────────────────────────────────────
        # 1) Boundary loss : focal loss si focal_gamma > 0
        # ─────────────────────────────────────────────
        loss_b_per_span = F.cross_entropy(
            b_logits,
            boundary_labels,
            weight=boundary_class_weights,
            reduction="none",
        )

        if focal_gamma > 0.0:
            # Focal loss : down-pondère les exemples bien classés
            b_probs = F.softmax(b_logits.detach(), dim=-1)
            p_t = b_probs.gather(1, boundary_labels.unsqueeze(1)).squeeze(1)
            focal_factor = (1.0 - p_t) ** focal_gamma
            loss_b_per_span = loss_b_per_span * focal_factor

        loss_b = (loss_b_per_span * sample_weights).mean()

        # ─────────────────────────────────────────────
        # 2) Coarse loss : tous les spans
        # ─────────────────────────────────────────────
        loss_c_per_span = F.cross_entropy(
            c_logits,
            coarse_labels,
            weight=coarse_class_weights,
            reduction="none",
        )

        loss_c = (loss_c_per_span * sample_weights).mean()

        # ─────────────────────────────────────────────
        # 3) Fine loss : POSITIVE ONLY
        #    On ne calcule la loss fine que sur les spans gold positifs.
        # ─────────────────────────────────────────────
        pos_mask = (boundary_labels == 1)

        if pos_mask.any():
            f_logits_pos = f_logits[pos_mask]      # [N_pos, 22]
            f_labels_pos = fine_labels[pos_mask]   # [N_pos]

            # sanity-check : les labels positifs doivent être dans [0, 21]
            if torch.any(f_labels_pos < 0) or torch.any(f_labels_pos >= f_logits.size(-1)):
                bad_vals = f_labels_pos[
                    (f_labels_pos < 0) | (f_labels_pos >= f_logits.size(-1))
                    ].detach().cpu().tolist()

                raise ValueError(
                    f"fine_labels positifs hors bornes pour fine_head: "
                    f"valeurs invalides={bad_vals[:20]} "
                    f"(fine_head attend des labels dans [0, {f_logits.size(-1)-1}])"
                )

            loss_f_per_span = F.cross_entropy(
                f_logits_pos,
                f_labels_pos,
                reduction="none",
            )

            # On peut appliquer les sample_weights aussi aux positifs
            pos_weights = sample_weights[pos_mask]
            loss_f = (loss_f_per_span * pos_weights).mean()
        else:
            # batch sans spans positifs
            loss_f = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────
        # 4) Loss totale
        # ─────────────────────────────────────────────
        total_loss = (
                lambda_boundary * loss_b
                + lambda_coarse * loss_c
                + lambda_fine * loss_f
        )

        return {
            "loss": total_loss,
            "loss_boundary": loss_b.detach(),
            "loss_coarse": loss_c.detach(),
            "loss_fine": loss_f.detach(),
            "num_positive_spans": int(pos_mask.sum().item()),
        }
