# multitask_model.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from labels import (
    NUM_FINE, NUM_SYN, NUM_VOICE, NUM_CERTAINTY,
    NUM_ROLE, ROLE_NONE_ID,                          # ← ancienne tête role (12 labels)
    NUM_ROLE_COARSE, ROLE_COARSE_NONE_ID, ROLE_COARSE_OTHER_ID,
    NUM_ROLE_OBLIQUE, ROLE_OBLIQUE_NONE_ID,
    NUM_GENDER, NUM_NUMBER, NUM_PERSON,
    SYN_NONE_ID, VOICE_NONE_ID, CERTAINTY_NONE_ID,
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
        num_fine = NUM_FINE  # utilisé pour ner_fine_to_oblique

        # Heads syntaxiques v4
        self.svo_boundary_head = nn.Linear(span_hidden_dim, 2)        # détecte verb_trigger/pron
        self.syn_head          = nn.Linear(span_hidden_dim, NUM_SYN)  # verb_trigger/pron_subj/pron_obj
        self.role_coarse_head  = nn.Linear(span_hidden_dim, NUM_ROLE_COARSE) # rôle SVO coarse SUBJ/OBJ/OBLIQ/APPOS
        self.role_head         = nn.Linear(span_hidden_dim, NUM_ROLE)        # ancienne tête rôle (12 labels)
        # Tête oblique fine : classifie les sous-types d'oblique (uniquement si role_coarse=OBLIQ)
        # Reçoit span_h + projection du FINE NER (38 labels) — plus discriminant que coarse (7 familles)
        # Ex: hint_time_date/clock/duration → OBLIQUE_TIME ; hint_gpe/loc_generic/fac → OBLIQUE_LOC
        # hint_doctrine/notion/field vs hint_org_name → OBLIQUE_DOMAIN vs OBLIQUE_AGENT
        # Cohérent avec le dataset builder qui infère OBLIQUE_TIME/LOC depuis les labels NER fine.
        # Timing OK : L_ROLE_OBLIQUE_NOW=0 jusqu'à ep26 (ramp role_progress) → bruit early epochs sans coût.
        self.ner_fine_to_oblique = nn.Linear(num_fine, span_hidden_dim, bias=False)
        self.role_oblique_head = nn.Linear(span_hidden_dim, NUM_ROLE_OBLIQUE)
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

        # NOTE : verb_ctx_proj / rel_pos_proj / verb_ctx_gate supprimés — causaient des collapses.
        # Réintroduire uniquement après stabilisation de role_coarse > 0.25 (cf. svo-isolated diag).

        # SVO→NER cascade : injecte le score SVO du span conteneur dans la repr NER
        # Permet aux têtes NER de savoir si ce span est dans une zone argumentale SVO.
        self.svo_context_proj = nn.Linear(1, span_hidden_dim)

        self.register_buffer("coarse_fine_mask", build_coarse_to_fine_mask())

    def _bucket_width(self, width: int) -> int:
        return min(width, self.max_width_bucket - 1)

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
        svo_boundary_logits = self.svo_boundary_head(span_h)  # [N, 2]

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

            # ── Verb context pour role conditioning ────────────────────────────
            # Soft-attention sur l'encoder via verb_ptr_logits (détaché pour ne pas
            # perturber le gradient du pointer). Early training : attention quasi-uniforme
            # → contexte ≈ moyenne de la phrase. Late training : se concentre sur le verbe.
            # Hard pointer: hidden du token verbe predit - O(N*H) vs O(N*seq*H)
            # gathered_hidden [N,512,768] ~ 6GB avec BS=80 sur RTX 4090 -> OOM
            # NOTE : verb conditioning sur role_coarse retiré — toutes les tentatives
            # (verb_ctx seul, + rel_pos random, + rel_pos gold) ont causé des collapses.
            # Sans verb conditioning, role_coarse atteignait 0.231 (svo-v819, mai 26).
            # À réintroduire uniquement après stabilisation de role_coarse > 0.25.
            span_h_role = span_h
        else:
            verb_ptr_logits = torch.zeros(
                (0, hidden.size(1)), device=hidden.device
            )
            span_h_role = span_h

        return {
            "span_reps":           span_h,
            "span_indices":        span_indices,
            # NER heads : span_h_ner (enrichi du contexte SVO des spans conteneurs)
            "boundary_logits":     self.boundary_head(span_h_ner),
            "coarse_logits":       self.coarse_head(span_h_ner),
            "fine_logits":         self.fine_head(span_h_ner),
            # SVO heads : span_h brut (pas de dépendance circulaire)
            "svo_boundary_logits":  self.svo_boundary_head(span_h),
            "syn_logits":           self.syn_head(span_h),
            # role_coarse conditionné sur le verbe (soft-attention verb_ptr)
            "role_coarse_logits":   self.role_coarse_head(span_h_role),
            "role_logits":          self.role_head(span_h_role),           # ancienne tête (12 labels)
            # Tête oblique fine : span_h_role enrichi du signal NER fine détaché (38 labels)
            # Cohérent avec dataset builder : OBLIQUE_TIME/LOC inférés depuis labels NER fine
            "role_oblique_logits": self.role_oblique_head(
                span_h_role + self.ner_fine_to_oblique(
                    torch.softmax(self.fine_head(span_h_ner).detach(), dim=-1)
                )
            ),
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
            role_coarse_labels,
            role_oblique_labels,
            role_labels,                               # ← ancienne tête (12 labels)
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
            certainty_class_weights=None,
            oblique_class_weights=None,
            role_coarse_class_weights=None,
            lambda_boundary=1.0,
            lambda_coarse=1.0,
            lambda_fine=1.2,
            lambda_svo_boundary=0.7,
            lambda_svo=0.5,
            lambda_role_coarse=0.1,
            lambda_role_oblique=0.15,
            lambda_role=0.0,                           # ← ancienne tête (défaut 0 = désactivée)
            lambda_voice=0.5,
            lambda_certainty=0.4,
            lambda_morpho=0.3,
            lambda_verb_ptr=0.5,
            lambda_compat=0.0,
            focal_gamma=0.0,
            focal_coarse_gamma=0.0, # Focal loss sur tête coarse — positifs seulement (≠NONE)
            focal_fine_gamma=0.0,   # Focal loss sur tête fine (séparé de boundary)
            focal_role_gamma=0.0,   # kept for API compat, unused
            ignore_coarse_none=False,  # Si True, exclut spans NONE de la loss coarse (positifs only)
            weighting=None,  # Dynamic loss weighting module (UncertaintyWeighting / GradNormWeighting)
    ):
        device = outputs["boundary_logits"].device

        b_logits       = outputs["boundary_logits"]
        c_logits       = outputs["coarse_logits"]
        f_logits       = outputs["fine_logits"]
        svo_b_logits   = outputs["svo_boundary_logits"]
        syn_logits          = outputs["syn_logits"]
        role_coarse_logits  = outputs["role_coarse_logits"]
        role_logits         = outputs["role_logits"]          # ancienne tête
        role_oblique_logits = outputs["role_oblique_logits"]
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
        role_coarse_labels  = role_coarse_labels.to(device=device, dtype=torch.long)
        role_oblique_labels = role_oblique_labels.to(device=device, dtype=torch.long)
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

        # ── 1) NER Boundary ────────────────────────────────────────────────
        loss_b_per = F.cross_entropy(b_logits, boundary_labels,
                                     weight=boundary_class_weights, reduction="none")
        if focal_gamma > 0.0:
            p_t = F.softmax(b_logits.detach(), dim=-1).gather(1, boundary_labels.unsqueeze(1)).squeeze(1)
            loss_b_per = loss_b_per * (1.0 - p_t) ** focal_gamma
        loss_b = (loss_b_per * sample_weights).mean()

        # ── 2) Coarse ─────────────────────────────────────────────────────────────
        # focal_coarse_gamma appliqué UNIQUEMENT sur spans positifs (coarse≠NONE)
        # pour down-weighter PERSON/LOC déjà bien appris et up-weighter OBJECT/EVENT.
        # Le focal global coarse (incluant NONE) dégradait les perfs → évité ici.
        # ignore_coarse_none=True : loss coarse calculée uniquement sur spans positifs (boundary=1).
        # Raisonnement : boundary head gère déjà la détection entité/non-entité.
        # La tête coarse n'a pas besoin d'apprendre à prédire NONE → signal redondant + bruit.
        if ignore_coarse_none:
            pos_mask_c = (coarse_labels != COARSE_NONE_ID)
            if pos_mask_c.any():
                loss_c_per_pos = F.cross_entropy(
                    c_logits[pos_mask_c], coarse_labels[pos_mask_c],
                    weight=coarse_class_weights, reduction="none")
                if focal_coarse_gamma > 0.0:
                    p_t_c = F.softmax(c_logits[pos_mask_c].detach(), dim=-1).gather(
                        1, coarse_labels[pos_mask_c].unsqueeze(1)).squeeze(1)
                    loss_c_per_pos = loss_c_per_pos * (1.0 - p_t_c) ** focal_coarse_gamma
                loss_c = (loss_c_per_pos * sample_weights[pos_mask_c]).mean()
            else:
                loss_c = torch.tensor(0.0, device=device)
        else:
            loss_c_per = F.cross_entropy(c_logits, coarse_labels,
                                         weight=coarse_class_weights, reduction="none")
            if focal_coarse_gamma > 0.0:
                pos_coarse_mask = (coarse_labels != COARSE_NONE_ID)
                if pos_coarse_mask.any():
                    p_t_c = F.softmax(c_logits[pos_coarse_mask].detach(), dim=-1).gather(
                        1, coarse_labels[pos_coarse_mask].unsqueeze(1)).squeeze(1)
                    loss_c_per = loss_c_per.clone()
                    loss_c_per[pos_coarse_mask] = loss_c_per[pos_coarse_mask] * (1.0 - p_t_c) ** focal_coarse_gamma
            loss_c = (loss_c_per * sample_weights).mean()

        # ── 3) Fine (NER positifs) ─────────────────────────────────────────
        # Focal loss optionnel sur fine : down-weight les classes faciles (person_name,
        # location) qui sont déjà bien apprises, up-weight les classes rares/difficiles
        # (hint_state/doctrine/notion) où le modèle hésite encore.
        # focal_fine_gamma=1.5 : (1-p_correct)^1.5 × CE
        pos_mask = (boundary_labels == 1) & (fine_labels >= 0) & (fine_labels < f_logits.size(-1))
        if pos_mask.any():
            _fine_w = fine_class_weights if fine_class_weights is not None else None
            loss_f_per = F.cross_entropy(f_logits[pos_mask], fine_labels[pos_mask],
                                         weight=_fine_w, reduction="none")
            if focal_fine_gamma > 0.0:
                p_t_f = F.softmax(f_logits[pos_mask].detach(), dim=-1).gather(
                    1, fine_labels[pos_mask].unsqueeze(1)).squeeze(1)
                loss_f_per = loss_f_per * (1.0 - p_t_f) ** focal_fine_gamma
            loss_f = (loss_f_per * sample_weights[pos_mask]).mean()
        else:
            loss_f = torch.tensor(0.0, device=device)

        # ── 4) Syntactic boundary (verb_trigger / pron) ────────────────────
        loss_svo_b = (F.cross_entropy(svo_b_logits, svo_boundary_labels, reduction="none")
                      * sample_weights).mean()

        # ── 5) Syn type (verb_trigger=0 / pron_subj=1 / pron_obj=2) ───────
        syn_mask = (syn_labels >= 0) & (syn_labels < syn_logits.size(-1))
        if syn_mask.any():
            loss_syn = (F.cross_entropy(syn_logits[syn_mask], syn_labels[syn_mask], reduction="none")
                        * sample_weights[syn_mask]).mean()
        else:
            loss_syn = torch.tensor(0.0, device=device)

        # ── 6) Role coarse (SUBJ/OBJ/OBLIQ/APPOS/OTHER) ─────────────────────
        # OTHER (ID=4) est dans le softmax pour la cascade inférence, mais EXCLU
        # de la loss — le gradient vient seulement des 4 vrais rôles.
        # Le softmax apprend implicitement OTHER = "pas SUBJ/OBJ/OBLIQ/APPOS".
        rc_mask = (role_coarse_labels >= 0) & (role_coarse_labels < role_coarse_logits.size(-1)) & (role_coarse_labels != ROLE_COARSE_OTHER_ID)
        if rc_mask.any():
            _rc_w = role_coarse_class_weights.to(device) if role_coarse_class_weights is not None else None
            loss_role_coarse = (F.cross_entropy(role_coarse_logits[rc_mask], role_coarse_labels[rc_mask],
                                                weight=_rc_w, reduction="none")
                                * sample_weights[rc_mask]).mean()
        else:
            loss_role_coarse = torch.tensor(0.0, device=device)

        # ── 6b) Role fin unifié — ancienne tête (12 labels, SUBJ/OBJ/OBLIQUE/OBLIQUE_*) ──
        # Masque : spans avec un vrai rôle != NONE (NONE_ID=6 est AU MILIEU, pas en fin !)
        # < ROLE_NONE_ID exclut incorrectement les labels 7-11 (OBLIQUE_ADVERSARY/BENEFICIARY/etc.)
        role_mask = (role_labels >= 0) & (role_labels != ROLE_NONE_ID)
        if role_mask.any():
            loss_role = (F.cross_entropy(role_logits[role_mask], role_labels[role_mask], reduction="none")
                         * sample_weights[role_mask]).mean()
        else:
            loss_role = torch.tensor(0.0, device=device)

        # ── 6c) Role oblique fin (maské aux spans OBLIQ, avec CWP) ────────────
        # Supervisé uniquement sur spans où role_oblique_label < ROLE_OBLIQUE_NONE_ID
        ro_mask = (role_oblique_labels >= 0) & (role_oblique_labels < role_oblique_logits.size(-1))
        if ro_mask.any():
            _obl_w = oblique_class_weights if oblique_class_weights is not None else None
            loss_role_oblique = (F.cross_entropy(role_oblique_logits[ro_mask], role_oblique_labels[ro_mask],
                                                 weight=_obl_w, reduction="none")
                                 * sample_weights[ro_mask]).mean()
        else:
            loss_role_oblique = torch.tensor(0.0, device=device)

        # ── 7) Voice (sur verb_trigger uniquement) ─────────────────────────
        voice_mask = (voice_labels >= 0) & (voice_labels < voice_logits.size(-1))
        if voice_mask.any():
            loss_voice = (F.cross_entropy(voice_logits[voice_mask], voice_labels[voice_mask], reduction="none")
                          * sample_weights[voice_mask]).mean()
        else:
            loss_voice = torch.tensor(0.0, device=device)

        # ── 8) Certainty (sur verb_trigger uniquement) ─────────────────────
        cert_mask = (certainty_labels >= 0) & (certainty_labels < cert_logits.size(-1))
        if cert_mask.any():
            loss_cert = (F.cross_entropy(cert_logits[cert_mask], certainty_labels[cert_mask],
                                         weight=certainty_class_weights, reduction="none")
                         * sample_weights[cert_mask]).mean()
        else:
            loss_cert = torch.tensor(0.0, device=device)

        # ── 9) Morpho : gender + number + person ───────────────────────────
        # Supervisés sur NER spans + syntactic spans qui ont les champs annotés
        gender_mask = (gender_labels >= 0) & (gender_labels < g_logits.size(-1))
        number_mask = (number_labels >= 0) & (number_labels < n_logits.size(-1))
        person_mask = (person_labels >= 0) & (person_labels < p_logits.size(-1))
        loss_gender = (F.cross_entropy(g_logits[gender_mask], gender_labels[gender_mask], reduction="none")
                       * sample_weights[gender_mask]).mean() if gender_mask.any() else torch.tensor(0.0, device=device)
        loss_number = (F.cross_entropy(n_logits[number_mask], number_labels[number_mask], reduction="none")
                       * sample_weights[number_mask]).mean() if number_mask.any() else torch.tensor(0.0, device=device)
        loss_person = (F.cross_entropy(p_logits[person_mask], person_labels[person_mask], reduction="none")
                       * sample_weights[person_mask]).mean() if person_mask.any() else torch.tensor(0.0, device=device)

        # ── 10) Verb pointer — fires sur spans avec un rôle coarse annoté ────
        seq_len  = vptr_logits.size(1)
        ptr_mask = (gov_verb_labels >= 0) & (gov_verb_labels < seq_len) & (role_coarse_labels >= 0) & (role_coarse_labels < ROLE_COARSE_NONE_ID)
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
        role_active_mask = (role_coarse_labels >= 0) & (role_coarse_labels < ROLE_COARSE_NONE_ID)
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

        # ── Raw losses per task (for dynamic weighting) ─────────────────
        raw_losses = {
            "boundary":     loss_b,
            "coarse":       loss_c,
            "fine":         loss_f,
            "svo_boundary": loss_svo_b,
            "svo":          loss_syn,
            "role_coarse":  loss_role_coarse,
            "role_oblique": loss_role_oblique,
            "role":         loss_role,                 # ancienne tête
            "voice":        loss_voice,
            "certainty":    loss_cert,
            "morpho":       loss_gender + loss_number + loss_person,
            "verb_ptr":     loss_verb_ptr,
            "compat":       loss_compat,
        }

        # ── Total (dynamic or fixed weighting) ────────────────────────────
        if weighting is not None:
            ramp_lambdas = {
                "boundary":     lambda_boundary,
                "coarse":       lambda_coarse,
                "fine":         lambda_fine,
                "svo_boundary": lambda_svo_boundary,
                "svo":          lambda_svo,
                "role_coarse":  lambda_role_coarse,
                "role_oblique": lambda_role_oblique,
                "role":         lambda_role,
                "voice":        lambda_voice,
                "certainty":    lambda_certainty,
                "morpho":       lambda_morpho,
                "verb_ptr":     lambda_verb_ptr,
                "compat":       lambda_compat,
            }
            total_loss = weighting.combine(raw_losses, ramp_lambdas)
        else:
            total_loss = (
                lambda_boundary       * loss_b
                + lambda_coarse       * loss_c
                + lambda_fine         * loss_f
                + lambda_svo_boundary * loss_svo_b
                + lambda_svo          * loss_syn
                + lambda_role_coarse  * loss_role_coarse
                + lambda_role_oblique * loss_role_oblique
                + lambda_role         * loss_role       # ancienne tête
                + lambda_voice        * loss_voice
                + lambda_certainty    * loss_cert
                + lambda_morpho       * (loss_gender + loss_number + loss_person)
                + lambda_verb_ptr     * loss_verb_ptr
                + lambda_compat       * loss_compat
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
            "num_syn_spans":        int(syn_mask.sum().item()),
            "num_oblique_spans":    int(ro_mask.sum().item()),
        }



