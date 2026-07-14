# multitask_model.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from labels import (
    NUM_FINE, NUM_SYN, NUM_VOICE, NUM_CERTAINTY,
    NUM_ROLE_COARSE, ROLE_COARSE_NONE_ID, ROLE_COARSE_OTHER_ID,
    NUM_GENDER, NUM_NUMBER, NUM_PERSON,
    SYN_NONE_ID, VOICE_NONE_ID, CERTAINTY_NONE_ID,
    COARSE_NONE_ID,
    build_coarse_to_fine_mask,
    ROLE_DERIVED_SUBJ_IDS, ROLE_DERIVED_OBJ_IDS,
    ROLE_DERIVED_OBLIQ_IDS, ROLE_DERIVED_APPOS_IDS,
    NUM_SVO,
    # Semantic role (remplace role_oblique)
    NUM_SEMANTIC_ROLE, SEMANTIC_ROLE_NONE_ID, SEMANTIC_ROLE_SKIP_ID,
    # VerbFam
    NUM_VERB_FAMILY, VERB_FAMILY_NONE_ID,
    NUM_VERB_FAMILY_FINE, VERB_FAMILY_FINE_NONE_ID,
    NUM_VERB_POLARITY, VERB_POLARITY_NONE_ID,
    NUM_VERB_ASPECT, VERB_ASPECT_NONE_ID,
    NUM_VERB_SOURCE, VERB_SOURCE_NONE_ID,
    VERB_FAMILY_FINE_MASK,
    # Compat legacy (role_head gardé pour chargement de checkpoints anciens)
    NUM_ROLE, ROLE_NONE_ID,
)
from labels import FINE_LABELS
from heads import build_all_heads


def _agg_group_logit(logits: torch.Tensor, ids: list) -> torch.Tensor:
    """Agrège les logits d'un groupe d'IDs via logsumexp (probabilistiquement stable)."""
    idx = torch.tensor(ids, device=logits.device, dtype=torch.long)
    return torch.logsumexp(logits.index_select(-1, idx), dim=-1)


class SpanMultiTaskModel(nn.Module):
    def __init__(
            self,
            model_name: str,
            num_coarse: int = 9,
            width_emb_dim: int = 32,
            span_hidden_dim: int = 512,
            max_width_bucket: int = 16,
            dropout: float = 0.1,
            detach_ner_classifier_backbone: bool = False,
            ner_coarse_backbone_grad_scale: float = 1,
            ner_fine_backbone_grad_scale: float = 1,
            boundary_aux_from_ner: bool = False,
            boundary_aux_scale: float = 1.0,
    ):
        super().__init__()

        self.detach_ner_classifier_backbone = detach_ner_classifier_backbone
        # Soft detach / gradient scaling pour éviter que coarse/fine ne contredisent trop boundary.
        # 1.0 = gradient complet vers encoder/span_mlp ; 0.0 = detach complet.
        self.ner_coarse_backbone_grad_scale = float(ner_coarse_backbone_grad_scale)
        self.ner_fine_backbone_grad_scale = float(ner_fine_backbone_grad_scale)
        self.boundary_aux_from_ner = boundary_aux_from_ner
        self.boundary_aux_scale = float(boundary_aux_scale)

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
        # Evidence NER → boundary : coarse/fine peuvent corriger boundary sans
        # renvoyer de gradient vers leurs têtes ni vers le backbone partagé.
        self.boundary_ner_evidence_head = nn.Linear(num_coarse + NUM_FINE, 2, bias=False)
        nn.init.zeros_(self.boundary_ner_evidence_head.weight)
        num_fine = NUM_FINE  # utilisé pour ner_fine_to_oblique

        # Heads syntaxiques v4
        self.svo_boundary_head = nn.Linear(span_hidden_dim, 2)        # détecte verb_trigger/pron
        self.syn_head          = nn.Linear(span_hidden_dim, NUM_SYN)  # verb_trigger/pron_subj/pron_obj
        self.role_coarse_head  = nn.Linear(span_hidden_dim, NUM_ROLE_COARSE) # rôle SVO coarse SUBJ/OBJ/OBLIQ/APPOS
        # Biais positionnel pour role_coarse : SUBJ ← avant le verbe, OBJ ← après.
        # span_h (features NER) ne contient pas l'info relative span/verbe → SUBJ collapse.
        # rel_pos = (span_tok_start - verb_tok_start) / 50 ∈ [-1,1] → projeté en [N, hidden].
        # Training : position GOLD (gov_verb_labels du batch). Inférence : verb_ptr prédit.
        # bias=False : le signal positionnel est déjà directionnel, pas besoin de biais.
        self.role_pos_proj = nn.Linear(1, span_hidden_dim, bias=False)
        # ── Semantic role head (remplace role_oblique + role) ──────────────────
        # 19 labels : AGENT/PATIENT/CONTENT/SOURCE/LOCATION/TEMPORAL/CAUSE/PURPOSE/MEASURE/
        #             BENEFICIARY/COMITATIVE/ADVERSARY/DOMAIN/INSTRUMENT/PART_OF/MEMBER_OF/OWNER/IDENTITY/NONE
        # Conditionné sur role_coarse (même cascade que l'ancien role_oblique_head).
        # Supervisé sur TOUS les spans NER (pas seulement OBLIQ) sauf SEMANTIC_ROLE_SKIP_ID.
        self.semantic_role_head = nn.Linear(span_hidden_dim, NUM_SEMANTIC_ROLE)
        # Legacy : role_head (12 labels) conservé pour compatibilité chargement checkpoints anciens
        # lambda_role=0.0 → aucun gradient, pas d'impact sur l'entraînement
        self.role_head = nn.Linear(span_hidden_dim, NUM_ROLE)
        num_fine = NUM_FINE  # gardé pour compatibilité
        self.voice_head        = nn.Linear(span_hidden_dim, NUM_VOICE)
        self.certainty_head    = nn.Linear(span_hidden_dim, NUM_CERTAINTY)

        # Morpho
        self.gender_head  = nn.Linear(span_hidden_dim, NUM_GENDER)
        self.number_head  = nn.Linear(span_hidden_dim, NUM_NUMBER)
        self.person_head  = nn.Linear(span_hidden_dim, NUM_PERSON)

        # ── VerbFam heads (verb_trigger uniquement) ───────────────────────────
        # ARCHITECTURE : MLP dédié sur span_h.detach() (features post-NER MLP, 512d).
        # Même source que voice_head / certainty_head → cohérent.
        # .detach() : pas de compétition gradient avec NER.
        # Avant (v8.18) : span_reps.detach() (features brutes pre-MLP, ~1060d)
        # → DeBERTa ne pouvait pas adapter ses représentations au signal verbfam
        # → collapse vers 2-3 classes dominant toujours.
        _vf_dim = 256
        self.verb_family_mlp = nn.Sequential(
            nn.Linear(span_hidden_dim, _vf_dim),
            nn.LayerNorm(_vf_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(_vf_dim, _vf_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # verb_family (12) → soft mask → verb_family_fine (38)
        # verb_polarity (3), verb_aspect (2), verb_source (3) : indépendants
        self.verb_family_head      = nn.Linear(_vf_dim, NUM_VERB_FAMILY)
        self.verb_family_fine_head = nn.Linear(_vf_dim, NUM_VERB_FAMILY_FINE)
        self.verb_polarity_head    = nn.Linear(_vf_dim, NUM_VERB_POLARITY)
        self.verb_aspect_head      = nn.Linear(_vf_dim, NUM_VERB_ASPECT)
        self.verb_source_head      = nn.Linear(_vf_dim, NUM_VERB_SOURCE)
        # Masque coarse→fine verbfam (comme NER coarse→fine)
        self.register_buffer("verb_family_fine_mask", VERB_FAMILY_FINE_MASK)

        # Verb pointer
        _ptr_dim = 64
        self.verb_ptr_query = nn.Linear(span_hidden_dim, _ptr_dim, bias=False)
        self.verb_ptr_key   = nn.Linear(hidden_size,     _ptr_dim, bias=False)

        # NOTE : verb_ctx_proj / rel_pos_proj / verb_ctx_gate supprimés — causaient des collapses.
        # Réintroduire uniquement après stabilisation de role_coarse > 0.25 (cf. svo-isolated diag).

        # SVO→NER cascade : injecte le score SVO du span conteneur dans la repr NER
        # Permet aux têtes NER de savoir si ce span est dans une zone argumentale SVO.
        self.svo_context_proj = nn.Linear(1, span_hidden_dim)

        self.register_buffer("coarse_fine_mask", build_coarse_to_fine_mask())

        # ── Registre des têtes (un fichier par tête, cf. package `heads/`) ────
        # Les couches nn.Module ci-dessus restent la source de vérité pour le
        # state_dict (compat checkpoints) ; chaque Head encapsule juste la
        # logique (forward applicatif / loss / métriques / dump JSONL).
        self.heads = build_all_heads(self)

    def _bucket_width(self, width: int) -> int:
        return min(width, self.max_width_bucket - 1)

    def _soft_detach(self, x: torch.Tensor, grad_scale: float) -> torch.Tensor:
        """
        Soft detach / gradient scaling.

        Forward:
            retourne exactement x.

        Backward:
            le gradient vers x est multiplié par grad_scale.

        grad_scale = 1.0 -> aucun detach
        grad_scale = 0.0 -> detach complet
        grad_scale = 0.25 -> coarse/fine ne renvoient que 25% du gradient
                             vers span_mlp / encoder
        """
        if grad_scale >= 1.0:
            return x
        if grad_scale <= 0.0:
            return x.detach()
        return x.detach() + grad_scale * (x - x.detach())

    def _build_span_representations(self, hidden_states, spans):
        reps = []
        span_indices = []
        span_batch_indices = []   # ← index batch pour chaque span (utile pour le pointer)
        span_positions = []       # ← (tok_start, tok_end) pour la matrice de containment SVO→NER
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
                span_positions.append((l, r))

        if not reps:
            return (
                torch.zeros((0, self.span_mlp[0].in_features), device=device),
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0, 2), dtype=torch.long, device=device),
            )

        return (
            torch.stack(reps),
            torch.arange(len(reps), device=device),
            torch.tensor(span_batch_indices, dtype=torch.long, device=device),
            torch.tensor(span_positions,     dtype=torch.long, device=device),  # [N, 2]
        )

    def forward(self, batch):
        enc = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        hidden = enc.last_hidden_state  # [B, seq, H]

        span_reps, span_indices, span_batch_idx, span_positions = \
            self._build_span_representations(hidden, batch["spans"])
        span_h = self.span_mlp(span_reps)

        # ── SVO→NER cascade : SVO boundary d'abord ───────────────────────────────
        # Pour chaque span, calcule le score SVO max des spans qui le CONTIENNENT
        # (même phrase). Ce signal aide les têtes NER à savoir si le span est dans
        # une zone argumentale, sans concurrence de gradient (SVO heads gardent span_h brut).
        svo_boundary_logits = self.heads["svo_boundary"].forward({"span_h": span_h})["svo_boundary_logits"]

        N = span_h.size(0)
        if N > 0:
            # detach : le gradient NER ne doit pas remonter à travers la cascade SVO
            svo_probs = torch.softmax(svo_boundary_logits.detach(), dim=-1)[:, 1]  # [N]
            starts = span_positions[:, 0]   # [N]
            ends   = span_positions[:, 1]   # [N]
            # Gate qualité : n'injecte le contexte SVO que si il y a un signal fort
            # (max prob > 0.6) — évite d'injecter du bruit en début de training
            svo_max_conf = svo_probs.max().item()
            if svo_max_conf > 0.6:
                # contains[i, j] = True si span j CONTIENT le span i (même phrase)
                same_sent  = (span_batch_idx.unsqueeze(1) == span_batch_idx.unsqueeze(0))
                j_contains_i = (
                    (starts.unsqueeze(1) <= starts.unsqueeze(0)) &
                    (ends.unsqueeze(1)   >= ends.unsqueeze(0))   &
                    same_sent &
                    ~torch.eye(N, dtype=torch.bool, device=span_h.device)
                )
                svo_probs_expanded = svo_probs.unsqueeze(0).expand(N, N)
                containing_svo = (svo_probs_expanded * j_contains_i.float()).max(dim=1).values
                svo_ctx = self.svo_context_proj(containing_svo.unsqueeze(-1))
                span_h_ner = span_h + svo_ctx
            else:
                # SVO pas encore fiable → pas d'injection (évite le bruit early training)
                span_h_ner = span_h
        else:
            span_h_ner = span_h

        # ── Verb pointer : attention bilinéaire span_query · token_key ──────────
        verb_ptr_logits = self.heads["verb_ptr"].forward({
            "span_h": span_h, "hidden": hidden, "span_batch_idx": span_batch_idx,
        })["verb_ptr_logits"]

        # NOTE : verb conditioning sur role_coarse retiré — toutes les tentatives
        # (verb_ctx seul, + rel_pos random, + rel_pos gold) ont causé des collapses.
        # Sans verb conditioning, role_coarse atteignait 0.231 (svo-v819, mai 26).
        # À réintroduire uniquement après stabilisation de role_coarse > 0.25.
        span_h_role = span_h

        role_logits = self.heads["role"].forward({"span_h_role": span_h_role})["role_logits"]
        # Pré-calcul rc_logits pour la cascade role_coarse → role_oblique
        # Même pattern que verbfam_fine : softmax(role_coarse.detach())[:, OBLIQ] comme gate
        rc_logits = self.heads["role_coarse"].forward({"span_h_role": span_h_role})["role_coarse_logits"]

        # ── NER heads ───────────────────────────────────────────────────────────
        # Boundary reste la tête primaire sur span_h_ner complet.
        # Coarse/fine voient les mêmes features au forward, mais leur gradient vers
        # encoder/span_mlp est réduit via _soft_detach. Cela évite que coarse/fine
        # reprennent tout le contrôle de la représentation et fassent régresser boundary.
        boundary_logits_base = self.heads["boundary"].forward({"span_h_ner": span_h_ner})["boundary_logits_base"]

        if self.detach_ner_classifier_backbone:
            # Compat historique : detach complet pour coarse/fine.
            span_h_coarse = span_h_ner.detach()
            span_h_fine   = span_h_ner.detach()
        else:
            span_h_coarse = self._soft_detach(
                span_h_ner,
                self.ner_coarse_backbone_grad_scale,
            )
            span_h_fine = self._soft_detach(
                span_h_ner,
                self.ner_fine_backbone_grad_scale,
            )

        coarse_logits   = self.heads["coarse"].forward({"span_h_coarse": span_h_coarse})["coarse_logits"]
        fine_logits_raw = self.heads["fine"].forward({"span_h_fine": span_h_fine})["fine_logits"]

        if self.boundary_aux_from_ner and N > 0:
            ner_evidence = torch.cat(
                [
                    F.softmax(coarse_logits.detach(), dim=-1),
                    F.softmax(fine_logits_raw.detach(), dim=-1),
                ],
                dim=-1,
            )
            boundary_ner_evidence_logits = self.boundary_ner_evidence_head(ner_evidence)
            boundary_logits = boundary_logits_base + self.boundary_aux_scale * boundary_ner_evidence_logits
        else:
            boundary_ner_evidence_logits = torch.zeros_like(boundary_logits_base)
            boundary_logits = boundary_logits_base

        # Soft coarse→fine mask pour les MÉTRIQUES et l'INFÉRENCE uniquement.
        # detach() → fine ne pollue pas le gradient coarse.
        # fine_logits_raw utilisé pour la LOSS → gradient stable.
        if N > 0:
            coarse_probs_det = F.softmax(coarse_logits.detach(), dim=-1)          # [N, C]
            coarse_gate_f    = coarse_probs_det @ self.coarse_fine_mask.float()   # [N, F]
            fine_logits_masked = fine_logits_raw + torch.log(coarse_gate_f.clamp(min=1e-9))
        else:
            fine_logits_masked = fine_logits_raw

        # ── Coarse dérivée depuis role_head (logsumexp par groupe) ───────────
        # Comparaison directe avec role_coarse_head native pour diagnostic.
        # Ordre : 0=SUBJ, 1=OBJ, 2=OBLIQ, 3=APPOS (aligné sur ROLE_COARSE_LABELS[:4])
        if role_logits.size(0) > 0:
            _subj  = _agg_group_logit(role_logits, ROLE_DERIVED_SUBJ_IDS)
            _obj   = _agg_group_logit(role_logits, ROLE_DERIVED_OBJ_IDS)
            _obliq = _agg_group_logit(role_logits, ROLE_DERIVED_OBLIQ_IDS)
            _appos = _agg_group_logit(role_logits, ROLE_DERIVED_APPOS_IDS)
            role_coarse_from_role_logits = torch.stack([_subj, _obj, _obliq, _appos], dim=-1)
        else:
            role_coarse_from_role_logits = torch.zeros((0, 4), device=span_h.device)

        # ── VerbFam projection dédiée (span_h post-NER MLP, avec gradient) ───────────
        # span_h SANS detach() : même pattern que voice/certainty_head(span_h).
        # Le gradient verbfam remonte à travers verb_family_mlp → span_mlp → DeBERTa,
        # permettant à l'encodeur de s'adapter aux propriétés sémantiques du verbe.
        # Avant : span_h.detach() → DeBERTa figé → verbfam F1 stagnant (observé run lq2bhpko).
        # Voice/certainty marchent sans detach car signal morphologique fort dès les poids DeBERTa
        # pré-entraînés ; verbfam nécessite une adaptation active (12 classes sémantiques).
        _span_h_vf = self.verb_family_mlp(span_h)  # [N, 256]
        vf_features = {"span_h_vf": _span_h_vf}
        verb_family_logits = self.heads["verb_family"].forward(vf_features)["verb_family_logits"]
        verb_family_fine_logits_raw = self.heads["verb_family_fine"].forward(vf_features)["verb_family_fine_logits_raw"]

        morpho_out = self.heads["morpho"].forward({"span_h": span_h})
        svo_out = self.heads["svo"].forward({"span_h": span_h})

        return {
            "span_reps":           span_h,
            "span_indices":        span_indices,
            # ── NER heads ─────────────────────────────────────────────────────
            # boundary : span_h_ner complet ; coarse/fine : soft-detach configurable.
            "boundary_logits":     boundary_logits,
            "boundary_logits_base": boundary_logits_base,
            "boundary_ner_evidence_logits": boundary_ner_evidence_logits,
            "coarse_logits":       coarse_logits,
            "fine_logits":         fine_logits_raw,       # pour la LOSS
            "fine_logits_masked":  fine_logits_masked,    # pour les MÉTRIQUES et l'INFÉRENCE
            "ner_coarse_backbone_grad_scale": torch.tensor(
                self.ner_coarse_backbone_grad_scale,
                device=span_h.device,
            ),
            "ner_fine_backbone_grad_scale": torch.tensor(
                self.ner_fine_backbone_grad_scale,
                device=span_h.device,
            ),
            # ── SVO heads : span_h brut ────────────────────────────────────────
            "svo_boundary_logits":  svo_boundary_logits,  # déjà calculé plus haut, pas de double appel
            "syn_logits":           svo_out["syn_logits"],
            # role_coarse conditionné sur le verbe (soft-attention verb_ptr)
            "role_coarse_logits":   rc_logits,
            "role_logits":          role_logits,                        # legacy (lambda=0)
            "role_coarse_from_role_logits": role_coarse_from_role_logits,  # coarse drive (diagnostic)
            # Semantic role head : remplace role_oblique_head
            # Supervisé sur TOUS les spans NER (AGENT/PATIENT/etc.) ; skip si SEMANTIC_ROLE_SKIP_ID
            "semantic_role_logits": self.heads["semantic_role"].forward({"span_h_role": span_h_role})["semantic_role_logits"],
            "voice_logits":        self.heads["voice"].forward({"span_h": span_h})["voice_logits"],
            "certainty_logits":    self.heads["certainty"].forward({"span_h": span_h})["certainty_logits"],
            "gender_logits":       morpho_out["gender_logits"],
            "number_logits":       morpho_out["number_logits"],
            "person_logits":       morpho_out["person_logits"],
            "verb_ptr_logits":     verb_ptr_logits,
            # ── VerbFam (verb_trigger uniquement) ─────────────────────────────
            # span_reps.detach() → verb_family_mlp dédié (256 dims, 2 GELU)
            # Features DeBERTa BRUTES (avant span_mlp NER) + projection non-linéaire propre.
            # Aucune compétition gradient avec NER.
            "verb_family_logits":          verb_family_logits,
            "verb_family_fine_logits_raw": verb_family_fine_logits_raw,
            # verb_family_fine masqué par verb_family (soft mask)
            "verb_family_fine_logits": (
                verb_family_fine_logits_raw +
                torch.log(
                    (F.softmax(verb_family_logits.detach(), dim=-1)
                     @ self.verb_family_fine_mask.float()).clamp(min=1e-9)
                )
                if N > 0 else verb_family_fine_logits_raw
            ),
            "verb_polarity_logits": self.heads["verb_polarity"].forward(vf_features)["verb_polarity_logits"],
            "verb_aspect_logits":   self.heads["verb_aspect"].forward(vf_features)["verb_aspect_logits"],
            "verb_source_logits":   self.heads["verb_source"].forward(vf_features)["verb_source_logits"],
            # compat alias
            "svo_logits":          svo_out["svo_logits"],
        }

    def compute_loss(
            self,
            outputs,
            boundary_labels,
            coarse_labels,
            fine_labels,
            svo_boundary_labels,
            syn_labels,
            role_coarse_labels,
            semantic_role_labels,
            role_labels,
            voice_labels,
            certainty_labels,
            gender_labels,
            number_labels,
            person_labels,
            gov_verb_labels,
            sample_weights,
            # verbfam labels (verb_trigger spans uniquement)
            verb_family_labels=None,
            verb_family_fine_labels=None,
            verb_polarity_labels=None,
            verb_aspect_labels=None,
            verb_source_labels=None,
            boundary_class_weights=None,
            coarse_class_weights=None,
            fine_class_weights=None,
            certainty_class_weights=None,
            oblique_class_weights=None,       # deprecated — conservé pour compat appels existants
            semantic_role_class_weights=None,
            role_coarse_class_weights=None,
            # class weights verbfam (None = pas de pondération)
            verb_family_class_weights=None,
            verb_polarity_class_weights=None,
            verb_aspect_class_weights=None,
            verb_source_class_weights=None,
            lambda_boundary=1.0,
            lambda_coarse=1.0,
            lambda_fine=1.2,
            lambda_svo_boundary=0.7,
            lambda_svo=0.5,
            lambda_role_coarse=0.1,
            lambda_semantic_role=0.3,  # remplace lambda_role_oblique
            lambda_role=0.0,           # legacy (désactivé)
            lambda_voice=0.5,
            lambda_certainty=0.4,
            lambda_morpho=0.3,
            lambda_verb_ptr=0.5,
            lambda_compat=0.0,
            # verbfam lambdas (désactivés par défaut — activés via config)
            lambda_verb_family=0.0,
            lambda_verb_family_fine=0.0,
            lambda_verb_polarity=0.0,
            lambda_verb_aspect=0.0,
            lambda_verb_source=0.0,
            focal_gamma=0.0,
            focal_coarse_gamma=0.0,
            focal_fine_gamma=0.0,
            focal_role_gamma=0.0,
            ignore_coarse_none=False,
            weighting=None,
    ):
        device = outputs["boundary_logits"].device

        boundary_labels     = boundary_labels.to(device=device, dtype=torch.long)
        coarse_labels       = coarse_labels.to(device=device, dtype=torch.long)
        fine_labels         = fine_labels.to(device=device, dtype=torch.long)
        svo_boundary_labels = svo_boundary_labels.to(device=device, dtype=torch.long)
        syn_labels          = syn_labels.to(device=device, dtype=torch.long)
        role_coarse_labels    = role_coarse_labels.to(device=device, dtype=torch.long)
        semantic_role_labels  = semantic_role_labels.to(device=device, dtype=torch.long)
        role_labels           = role_labels.to(device=device, dtype=torch.long)
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
        if certainty_class_weights is not None:
            certainty_class_weights = certainty_class_weights.to(device=device, dtype=torch.float32)
        if oblique_class_weights is not None:
            oblique_class_weights = oblique_class_weights.to(device=device, dtype=torch.float32)

        # ── Labels regroupés — passés tels quels à chaque tête (Head.compute_loss) ──
        labels = {
            "boundary_labels":          boundary_labels,
            "coarse_labels":            coarse_labels,
            "fine_labels":              fine_labels,
            "svo_boundary_labels":      svo_boundary_labels,
            "syn_labels":               syn_labels,
            "role_coarse_labels":       role_coarse_labels,
            "semantic_role_labels":     semantic_role_labels,
            "role_labels":              role_labels,
            "voice_labels":             voice_labels,
            "certainty_labels":         certainty_labels,
            "gender_labels":            gender_labels,
            "number_labels":            number_labels,
            "person_labels":            person_labels,
            "gov_verb_labels":          gov_verb_labels,
            "verb_family_labels":       verb_family_labels,
            "verb_family_fine_labels":  verb_family_fine_labels,
            "verb_polarity_labels":     verb_polarity_labels,
            "verb_aspect_labels":       verb_aspect_labels,
            "verb_source_labels":       verb_source_labels,
        }

        # ── Loss RAW par tête (chaque Head encapsule sa propre logique) ─────────
        loss_b = self.heads["boundary"].compute_loss(
            outputs, labels, sample_weights,
            class_weights=boundary_class_weights, focal_gamma=focal_gamma,
        )
        loss_c = self.heads["coarse"].compute_loss(
            outputs, labels, sample_weights,
            class_weights=coarse_class_weights, focal_coarse_gamma=focal_coarse_gamma,
            ignore_coarse_none=ignore_coarse_none,
        )
        loss_f = self.heads["fine"].compute_loss(
            outputs, labels, sample_weights,
            class_weights=fine_class_weights, focal_fine_gamma=focal_fine_gamma,
        )
        loss_svo_b = self.heads["svo_boundary"].compute_loss(outputs, labels, sample_weights)
        loss_syn = self.heads["svo"].compute_loss(outputs, labels, sample_weights)
        loss_role_coarse = self.heads["role_coarse"].compute_loss(
            outputs, labels, sample_weights, class_weights=role_coarse_class_weights,
        )
        loss_role = self.heads["role"].compute_loss(outputs, labels, sample_weights)
        loss_semantic_role = self.heads["semantic_role"].compute_loss(
            outputs, labels, sample_weights, class_weights=semantic_role_class_weights,
        )
        loss_voice = self.heads["voice"].compute_loss(outputs, labels, sample_weights)
        loss_cert = self.heads["certainty"].compute_loss(
            outputs, labels, sample_weights, class_weights=certainty_class_weights,
        )
        loss_morpho = self.heads["morpho"].compute_loss(outputs, labels, sample_weights)
        loss_verb_ptr = self.heads["verb_ptr"].compute_loss(outputs, labels, sample_weights)
        loss_compat = self.heads["compat"].compute_loss(
            outputs, labels, sample_weights, lambda_compat=lambda_compat,
        )
        loss_verb_family = self.heads["verb_family"].compute_loss(
            outputs, labels, sample_weights, class_weights=verb_family_class_weights,
        )
        loss_verb_family_fine = self.heads["verb_family_fine"].compute_loss(outputs, labels, sample_weights)
        loss_verb_polarity = self.heads["verb_polarity"].compute_loss(
            outputs, labels, sample_weights, class_weights=verb_polarity_class_weights,
        )
        loss_verb_aspect = self.heads["verb_aspect"].compute_loss(
            outputs, labels, sample_weights, class_weights=verb_aspect_class_weights,
        )
        loss_verb_source = self.heads["verb_source"].compute_loss(
            outputs, labels, sample_weights, class_weights=verb_source_class_weights,
        )

        # ── Diagnostics de comptage (conservés pour compat logs/monitoring) ─────
        f_logits = outputs["fine_logits"]
        syn_logits = outputs["syn_logits"]
        semantic_role_logits = outputs["semantic_role_logits"]
        pos_mask = (boundary_labels == 1) & (fine_labels >= 0) & (fine_labels < f_logits.size(-1))
        syn_mask = (syn_labels >= 0) & (syn_labels < syn_logits.size(-1))
        sr_mask = (
            (semantic_role_labels >= 0)
            & (semantic_role_labels < semantic_role_logits.size(-1))
            & (semantic_role_labels != SEMANTIC_ROLE_SKIP_ID)
        )
        vt_mask = (svo_boundary_labels == 1)

        # ── Raw losses per task (for dynamic weighting) ─────────────────
        raw_losses = {
            "boundary":          loss_b,
            "coarse":            loss_c,
            "fine":              loss_f,
            "svo_boundary":      loss_svo_b,
            "svo":               loss_syn,
            "role_coarse":       loss_role_coarse,
            "semantic_role":     loss_semantic_role,
            "role":              loss_role,
            "voice":             loss_voice,
            "certainty":         loss_cert,
            "morpho":            loss_morpho,
            "verb_ptr":          loss_verb_ptr,
            "compat":            loss_compat,
            "verb_family":       loss_verb_family,
            "verb_family_fine":  loss_verb_family_fine,
            "verb_polarity":     loss_verb_polarity,
            "verb_aspect":       loss_verb_aspect,
            "verb_source":       loss_verb_source,
        }


        # ── Total (dynamic or fixed weighting) ────────────────────────────
        if weighting is not None:
            ramp_lambdas = {
                "boundary":         lambda_boundary,
                "coarse":           lambda_coarse,
                "fine":             lambda_fine,
                "svo_boundary":     lambda_svo_boundary,
                "svo":              lambda_svo,
                "role_coarse":      lambda_role_coarse,
                "semantic_role":    lambda_semantic_role,
                "role":             lambda_role,
                "voice":            lambda_voice,
                "certainty":        lambda_certainty,
                "morpho":           lambda_morpho,
                "verb_ptr":         lambda_verb_ptr,
                "compat":           lambda_compat,
                "verb_family":      lambda_verb_family,
                "verb_family_fine": lambda_verb_family_fine,
                "verb_polarity":    lambda_verb_polarity,
                "verb_aspect":      lambda_verb_aspect,
                "verb_source":      lambda_verb_source,
            }
            total_loss = weighting.combine(raw_losses, ramp_lambdas)
        else:
            total_loss = (
                lambda_boundary           * loss_b
                + lambda_coarse           * loss_c
                + lambda_fine             * loss_f
                + lambda_svo_boundary     * loss_svo_b
                + lambda_svo              * loss_syn
                + lambda_role_coarse      * loss_role_coarse
                + lambda_semantic_role   * loss_semantic_role
                + lambda_role            * loss_role
                + lambda_voice            * loss_voice
                + lambda_certainty        * loss_cert
                + lambda_morpho           * loss_morpho
                + lambda_verb_ptr         * loss_verb_ptr
                + lambda_compat           * loss_compat
                + lambda_verb_family      * loss_verb_family
                + lambda_verb_family_fine * loss_verb_family_fine
                + lambda_verb_polarity    * loss_verb_polarity
                + lambda_verb_aspect      * loss_verb_aspect
                + lambda_verb_source      * loss_verb_source
            )

        return {
            "loss":                 total_loss,
            "raw_losses":           raw_losses,
            "raw_losses_detached":  {k: v.detach() for k, v in raw_losses.items()},
            "loss_boundary":        loss_b.detach(),
            "loss_coarse":          loss_c.detach(),
            "loss_fine":            loss_f.detach(),
            "loss_svo_boundary":    loss_svo_b.detach(),
            "loss_syn":             loss_syn.detach(),
            "num_positive_spans":   int(pos_mask.sum().item()),
            "num_syn_spans":              int(syn_mask.sum().item()),
            "num_semantic_role_spans":    int(sr_mask.sum().item()),
            "num_vt_spans":               int(vt_mask.sum().item()),
        }
