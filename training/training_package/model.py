import torch
from torch import nn
from transformers import AutoModel

# ── Masque structurel coarse → fine-grained ───────────────────────────────────
# Indices des labels fine-grained (cf. LABELS dans dataset.py) autorisés
# pour chaque famille coarse (NerCoarseType : PER=0, LOC=1, ORG=2, TIME=3, EVENT=4, OBJECT=5).
# Utilisé pour contraindre les logits à l'inférence ET pendant l'entraînement.
# Total : 4+4+1+3+2+8 = 22 labels couverts.
COARSE_TO_FINE: dict[int, list[int]] = {
    0: [0, 1, 2, 3],              # PER   → person_name, person_role, norp, group_role
    1: [5, 6, 7, 12],             # LOC   → gpe, fac_name, loc_generic, infra
    2: [4],                        # ORG   → org_name
    3: [18, 19, 20],               # TIME  → time_date, time_clock, time_duration
    4: [16, 17],                   # EVENT → event_nominal, event_named
    5: [8, 9, 10, 11, 13, 14, 15, 21],  # OBJECT→ weapon, vehicle, substance, food, tool, object_generic, object_name, quantity
}

class SpanClassifier(nn.Module):
    def __init__(self, model_name, num_labels, num_coarse=6, coarse_embed_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.num_labels = num_labels
        # create classifier in same dtype as encoder parameters by default
        encoder_dtype = None
        for p in self.encoder.parameters():
            encoder_dtype = p.dtype
            break
        # default to float32 if unknown
        if encoder_dtype is None:
            encoder_dtype = torch.float32
        # coarse NER family embedding — 128 dims (14% du signal, vs 4% avec 32 dims)
        self.coarse_embed = nn.Embedding(num_coarse, coarse_embed_dim).to(dtype=encoder_dtype)
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size + coarse_embed_dim, num_labels
        ).to(dtype=encoder_dtype)

    def forward(self, batch):
        outputs = self.encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        last_hidden = outputs.last_hidden_state

        if last_hidden.dtype != self.classifier.weight.dtype:
            last_hidden = last_hidden.to(self.classifier.weight.dtype)

        span_vecs  = []
        coarse_ids = []
        for i, spans in enumerate(batch['spans']):
            for sp in spans:
                start, end = sp.get('start'), sp.get('end')
                if start is None or end is None:
                    continue
                start = max(0, int(start))
                end   = max(start + 1, int(end))
                if start >= last_hidden.size(1):
                    continue
                end = min(end, last_hidden.size(1))
                vec = last_hidden[i, start:end, :].mean(dim=0)
                span_vecs.append(vec)
                coarse_ids.append(int(sp.get('coarse_id', 5)))  # default OBJECT

        if len(span_vecs) == 0:
            return torch.empty((0, self.classifier.out_features), device=last_hidden.device, dtype=self.classifier.weight.dtype)

        span_vecs_t = torch.stack(span_vecs)
        coarse_t    = torch.tensor(coarse_ids, dtype=torch.long, device=last_hidden.device)
        coarse_vecs = self.coarse_embed(coarse_t).to(dtype=span_vecs_t.dtype)

        combined = torch.cat([span_vecs_t, coarse_vecs], dim=-1)  # (N, D + 128)
        logits   = self.classifier(combined)                       # (N, 22)

        # ── Masquage structurel coarse → fine ────────────────────────────────
        # TRAINING  : soft mask -5.0 → pénalise sans bloquer.
        #   exp(-5) ≈ 0.007, gradient fini → évite loss explosive (~1e9)
        #   causée par coarse noise qui peut masquer le gold label.
        # INFÉRENCE : hard mask -1e9 → contrainte stricte inter-famille.
        mask_value = -5.0 if self.training else -1e9
        mask = torch.full_like(logits, mask_value)
        for n, cid in enumerate(coarse_ids):
            allowed = COARSE_TO_FINE.get(cid, list(range(self.num_labels)))
            mask[n, allowed] = 0.0
        return logits + mask
