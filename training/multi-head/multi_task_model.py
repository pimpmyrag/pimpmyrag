# multitask_model.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from labels import (
    NUM_FINE, NUM_SYN, NUM_ROLE, NUM_VOICE, NUM_CERTAINTY,
    NUM_GENDER, NUM_NUMBER, NUM_PERSON,
    SYN_NONE_ID, ROLE_NONE_ID, VOICE_NONE_ID, CERTAINTY_NONE_ID,
    COARSE_NONE_ID,
    build_coarse_to_fine_mask,
    # compat
    NUM_SVO,
)
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
        self.boundary_head     = nn.Linear(span_hidden_dim, 2)
        self.coarse_head       = nn.Linear(span_hidden_dim, num_coarse)
        self.fine_head         = nn.Linear(span_hidden_dim, NUM_FINE)

        # Heads syntaxiques v4
        self.svo_boundary_head = nn.Linear(span_hidden_dim, 2)        # détecte verb_trigger/pron
        self.syn_head          = nn.Linear(span_hidden_dim, NUM_SYN)  # verb_trigger/pron_subj/pron_obj
        self.role_head         = nn.Linear(span_hidden_dim, NUM_ROLE) # rôle SVO sur NER + pronoms
        self.voice_head        = nn.Linear(span_hidden_dim, NUM_VOICE)       # active/passive sur verb_trigger
        self.certainty_head    = nn.Linear(span_hidden_dim, NUM_CERTAINTY)   # certain/modal/denied

        # Morpho
        self.gender_head  = nn.Linear(span_hidden_dim, NUM_GENDER)
        self.number_head  = nn.Linear(span_hidden_dim, NUM_NUMBER)
        self.person_head  = nn.Linear(span_hidden_dim, NUM_PERSON)

        # Verb pointer
        _ptr_dim = 64
        self.verb_ptr_query = nn.Linear(span_hidden_dim, _ptr_dim, bias=False)
        self.verb_ptr_key   = nn.Linear(hidden_size,     _ptr_dim, bias=False)

        self.register_buffer("coarse_fine_mask", build_coarse_to_fine_mask())

    def _bucket_width(self, width: int) -> int:
        return min(width, self.max_width_bucket - 1)

    def _build_span_representations(self, hidden_states, spans):
        reps = []
        span_indices = []
        span_batch_indices = []   # ← index batch pour chaque span (utile pour le pointer)
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
                span_batch_indices.append(b_idx)

        if not reps:
            return (
                torch.zeros((0, self.span_mlp[0].in_features), device=device),
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
            )

        return (
            torch.stack(reps),
            torch.arange(len(reps), device=device),
            torch.tensor(span_batch_indices, dtype=torch.long, device=device),
        )

    def forward(self, batch):
        enc = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        hidden = enc.last_hidden_state  # [B, seq, H]

        span_reps, span_indices, span_batch_idx = self._build_span_representations(hidden, batch["spans"])
        span_h = self.span_mlp(span_reps)

        # ── Verb pointer : attention bilinéaire span_query · token_key ──────────
        # ptr_queries : [N, 64]
        # ptr_keys    : [B, seq, 64]  →  gather par batch index → [N, seq, 64]
        # verb_ptr_logits : [N, seq]
        ptr_queries = self.verb_ptr_query(span_h)          # [N, 64]
        ptr_keys    = self.verb_ptr_key(hidden)             # [B, seq, 64]
        if span_h.size(0) > 0:
            gathered_keys   = ptr_keys[span_batch_idx]     # [N, seq, 64]
            verb_ptr_logits = torch.bmm(
                gathered_keys,                             # [N, seq, 64]
                ptr_queries.unsqueeze(-1)                  # [N, 64,  1]
            ).squeeze(-1)                                  # [N, seq]
        else:
            verb_ptr_logits = torch.zeros(
                (0, hidden.size(1)), device=hidden.device
            )

        return {
            "span_reps":           span_h,
            "span_indices":        span_indices,
            "boundary_logits":     self.boundary_head(span_h),
            "coarse_logits":       self.coarse_head(span_h),
            "fine_logits":         self.fine_head(span_h),
            "svo_boundary_logits": self.svo_boundary_head(span_h),
            "syn_logits":          self.syn_head(span_h),
            "role_logits":         self.role_head(span_h),
            "voice_logits":        self.voice_head(span_h),
            "certainty_logits":    self.certainty_head(span_h),
            "gender_logits":       self.gender_head(span_h),
            "number_logits":       self.number_head(span_h),
            "person_logits":       self.person_head(span_h),
            "verb_ptr_logits":     verb_ptr_logits,
            # compat alias
            "svo_logits":          self.syn_head(span_h),
        }

    def compute_loss(
            self,
            outputs,
            boundary_labels,
            coarse_labels,
            fine_labels,
            svo_boundary_labels,
            syn_labels,
            role_labels,
            voice_labels,
            certainty_labels,
            gender_labels,
            number_labels,
            person_labels,
            gov_verb_labels,
            sample_weights,
            boundary_class_weights=None,
            coarse_class_weights=None,
            fine_class_weights=None,
            lambda_boundary=1.0,
            lambda_coarse=1.0,
            lambda_fine=1.2,
            lambda_svo_boundary=0.7,
            lambda_svo=0.5,        # syn type (verb_trigger / pron)
            lambda_role=0.6,       # rôle SVO
            lambda_voice=0.5,
            lambda_certainty=0.4,
            lambda_morpho=0.3,
            lambda_verb_ptr=0.5,
            lambda_compat=0.0,
            focal_gamma=0.0,
    ):
        device = outputs["boundary_logits"].device

        b_logits       = outputs["boundary_logits"]
        c_logits       = outputs["coarse_logits"]
        f_logits       = outputs["fine_logits"]
        svo_b_logits   = outputs["svo_boundary_logits"]
        syn_logits     = outputs["syn_logits"]
        role_logits    = outputs["role_logits"]
        voice_logits   = outputs["voice_logits"]
        cert_logits    = outputs["certainty_logits"]
        g_logits       = outputs["gender_logits"]
        n_logits       = outputs["number_logits"]
        p_logits       = outputs["person_logits"]
        vptr_logits    = outputs["verb_ptr_logits"]

        boundary_labels     = boundary_labels.to(device=device, dtype=torch.long)
        coarse_labels       = coarse_labels.to(device=device, dtype=torch.long)
        fine_labels         = fine_labels.to(device=device, dtype=torch.long)
        svo_boundary_labels = svo_boundary_labels.to(device=device, dtype=torch.long)
        syn_labels          = syn_labels.to(device=device, dtype=torch.long)
        role_labels         = role_labels.to(device=device, dtype=torch.long)
        voice_labels        = voice_labels.to(device=device, dtype=torch.long)
        certainty_labels    = certainty_labels.to(device=device, dtype=torch.long)
        gender_labels       = gender_labels.to(device=device, dtype=torch.long)
        number_labels       = number_labels.to(device=device, dtype=torch.long)
        person_labels       = person_labels.to(device=device, dtype=torch.long)
        gov_verb_labels     = gov_verb_labels.to(device=device, dtype=torch.long)
        sample_weights      = sample_weights.to(device=device, dtype=torch.float32)

        if boundary_class_weights is not None:
            boundary_class_weights = boundary_class_weights.to(device=device, dtype=torch.float32)
        if coarse_class_weights is not None:
            coarse_class_weights = coarse_class_weights.to(device=device, dtype=torch.float32)
        if fine_class_weights is not None:
            fine_class_weights = fine_class_weights.to(device=device, dtype=torch.float32)

        # ── 1) NER Boundary ────────────────────────────────────────────────
        loss_b_per = F.cross_entropy(b_logits, boundary_labels,
                                     weight=boundary_class_weights, reduction="none")
        if focal_gamma > 0.0:
            p_t = F.softmax(b_logits.detach(), dim=-1).gather(1, boundary_labels.unsqueeze(1)).squeeze(1)
            loss_b_per = loss_b_per * (1.0 - p_t) ** focal_gamma
        loss_b = (loss_b_per * sample_weights).mean()

        # ── 2) Coarse (hybride : positifs plein poids + négatifs poids cappé) ────
        # Problème v8.1 : HN mining booste FP_BOUNDARY à ×3.5-5×.
        # Si on garde tous les négatifs (coarse=NONE) avec leur poids boost,
        # NONE domine le gradient → coarse collapse (F1 → 0 epoch 3-4).
        # Si on supprime totalement NONE (positive-only), l'encodeur perd
        # le gradient "non-entité" → fine/ABSTRACT/ORG régressent de 7-12%.
        # Solution hybride : positifs poids normal + négatifs cappés à 1.0
        # avec facteur 0.2 → NONE contribue ~20% vs positifs, sans dominer.
        coarse_pos_mask = (boundary_labels == 1)
        coarse_neg_mask = (boundary_labels == 0)
        _coarse_w = coarse_class_weights if coarse_class_weights is not None else None

        if coarse_pos_mask.any():
            loss_c_pos = (F.cross_entropy(c_logits[coarse_pos_mask], coarse_labels[coarse_pos_mask],
                                          weight=_coarse_w, reduction="none")
                          * sample_weights[coarse_pos_mask]).mean()
        else:
            loss_c_pos = torch.tensor(0.0, device=device)

        if coarse_neg_mask.any():
            # Poids cappé à 1.0 : empêche le HN boost (×3.5-5×) de dominer NONE
            neg_w_capped = sample_weights[coarse_neg_mask].clamp(max=1.0)
            loss_c_none = (F.cross_entropy(c_logits[coarse_neg_mask], coarse_labels[coarse_neg_mask],
                                           reduction="none")
                           * neg_w_capped).mean()
            loss_c = loss_c_pos + 0.2 * loss_c_none
        else:
            loss_c = loss_c_pos

        # ── 3) Fine (NER positifs) ─────────────────────────────────────────
        pos_mask = (boundary_labels == 1) & (fine_labels < f_logits.size(-1))
        if pos_mask.any():
            _fine_w = fine_class_weights if fine_class_weights is not None else None
            loss_f = (F.cross_entropy(f_logits[pos_mask], fine_labels[pos_mask],
                                      weight=_fine_w, reduction="none")
                      * sample_weights[pos_mask]).mean()
        else:
            loss_f = torch.tensor(0.0, device=device)

        # ── 4) Syntactic boundary (verb_trigger / pron) ────────────────────
        loss_svo_b = (F.cross_entropy(svo_b_logits, svo_boundary_labels, reduction="none")
                      * sample_weights).mean()

        # ── 5) Syn type (verb_trigger=0 / pron_subj=1 / pron_obj=2) ───────
        syn_mask = (syn_labels < syn_logits.size(-1))
        if syn_mask.any():
            loss_syn = (F.cross_entropy(syn_logits[syn_mask], syn_labels[syn_mask], reduction="none")
                        * sample_weights[syn_mask]).mean()
        else:
            loss_syn = torch.tensor(0.0, device=device)

        # ── 6) Role (sur NER spans + pronoms qui ont un rôle != NONE) ─────
        role_mask = (role_labels < ROLE_NONE_ID)   # exclut NONE (=6)
        if role_mask.any():
            loss_role = (F.cross_entropy(role_logits[role_mask], role_labels[role_mask], reduction="none")
                         * sample_weights[role_mask]).mean()
        else:
            loss_role = torch.tensor(0.0, device=device)

        # ── 7) Voice (sur verb_trigger uniquement) ─────────────────────────
        voice_mask = (voice_labels < voice_logits.size(-1))
        if voice_mask.any():
            loss_voice = (F.cross_entropy(voice_logits[voice_mask], voice_labels[voice_mask], reduction="none")
                          * sample_weights[voice_mask]).mean()
        else:
            loss_voice = torch.tensor(0.0, device=device)

        # ── 8) Certainty (sur verb_trigger uniquement) ─────────────────────
        cert_mask = (certainty_labels < cert_logits.size(-1))
        if cert_mask.any():
            loss_cert = (F.cross_entropy(cert_logits[cert_mask], certainty_labels[cert_mask], reduction="none")
                         * sample_weights[cert_mask]).mean()
        else:
            loss_cert = torch.tensor(0.0, device=device)

        # ── 9) Morpho : gender + number + person ───────────────────────────
        # Supervisés sur NER spans + syntactic spans qui ont les champs annotés
        gender_mask = (gender_labels < g_logits.size(-1))
        number_mask = (number_labels < n_logits.size(-1))
        person_mask = (person_labels < p_logits.size(-1))
        loss_gender = (F.cross_entropy(g_logits[gender_mask], gender_labels[gender_mask], reduction="none")
                       * sample_weights[gender_mask]).mean() if gender_mask.any() else torch.tensor(0.0, device=device)
        loss_number = (F.cross_entropy(n_logits[number_mask], number_labels[number_mask], reduction="none")
                       * sample_weights[number_mask]).mean() if number_mask.any() else torch.tensor(0.0, device=device)
        loss_person = (F.cross_entropy(p_logits[person_mask], person_labels[person_mask], reduction="none")
                       * sample_weights[person_mask]).mean() if person_mask.any() else torch.tensor(0.0, device=device)

        # ── 10) Verb pointer (spans avec gov_verb_labels != -1) ────────────
        seq_len  = vptr_logits.size(1)
        ptr_mask = (gov_verb_labels >= 0) & (gov_verb_labels < seq_len) & (role_labels < ROLE_NONE_ID)
        if ptr_mask.any() and vptr_logits.size(0) > 0:
            loss_verb_ptr = (F.cross_entropy(vptr_logits[ptr_mask], gov_verb_labels[ptr_mask], reduction="none")
                             * sample_weights[ptr_mask]).mean()
        else:
            loss_verb_ptr = torch.tensor(0.0, device=device)

        # ── 11) Compat : cohérence inter-têtes pour les eventlets ──────────
        #
        # A) role → boundary :
        #    Un span participant (role != NONE) est forcément un span NER (boundary=1).
        #    Si boundary manque ce span, l'eventlet est cassé.
        role_active_mask = (role_labels < ROLE_NONE_ID)
        if lambda_compat > 0.0 and role_active_mask.any():
            forced_boundary = torch.ones(
                role_active_mask.sum(), device=device, dtype=torch.long
            )
            loss_compat_rb = (
                F.cross_entropy(b_logits[role_active_mask], forced_boundary, reduction="none")
                * sample_weights[role_active_mask]
            ).mean()
        else:
            loss_compat_rb = torch.tensor(0.0, device=device)

        # B) boundary ↔ coarse soft alignment :
        #    P(boundary=1) doit s'aligner avec P(coarse≠NONE).
        #    On utilise boundary comme superviseur stable (detach) pour guider coarse.
        if lambda_compat > 0.0 and b_logits.size(0) > 0:
            p_boundary_pos  = torch.softmax(b_logits.detach(), dim=-1)[:, 1]          # [N]
            p_coarse_entity = 1.0 - torch.softmax(c_logits, dim=-1)[:, COARSE_NONE_ID]  # [N]
            loss_compat_bc  = F.mse_loss(p_coarse_entity, p_boundary_pos)
        else:
            loss_compat_bc = torch.tensor(0.0, device=device)

        loss_compat = loss_compat_rb + loss_compat_bc

        # ── Total ──────────────────────────────────────────────────────────
        total_loss = (
            lambda_boundary       * loss_b
            + lambda_coarse       * loss_c
            + lambda_fine         * loss_f
            + lambda_svo_boundary * loss_svo_b
            + lambda_svo          * loss_syn
            + lambda_role         * loss_role
            + lambda_voice        * loss_voice
            + lambda_certainty    * loss_cert
            + lambda_morpho       * (loss_gender + loss_number + loss_person)
            + lambda_verb_ptr     * loss_verb_ptr
            + lambda_compat       * loss_compat
        )

        return {
            "loss":                 total_loss,
            "loss_boundary":        loss_b.detach(),
            "loss_coarse":          loss_c.detach(),
            "loss_fine":            loss_f.detach(),
            "loss_svo_boundary":    loss_svo_b.detach(),
            "loss_syn":             loss_syn.detach(),
            "loss_role":            loss_role.detach(),
            "loss_voice":           loss_voice.detach(),
            "loss_certainty":       loss_cert.detach(),
            "loss_compat":          loss_compat.detach(),
            "num_positive_spans":   int(pos_mask.sum().item()),
            "num_syn_spans":        int(syn_mask.sum().item()),
            "num_role_spans":       int(role_mask.sum().item()),
        }

