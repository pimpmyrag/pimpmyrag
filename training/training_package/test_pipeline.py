"""Test rapide : tokenizer local + SpanClassifier + forward pass avec masquage coarse."""
import os
import torch
from transformers import AutoTokenizer
from model import SpanClassifier, COARSE_TO_FINE

TOKENIZER_PATH = os.environ.get('NER_TOKENIZER_PATH', 'debertav3-ner/tokenizer_from_hf')
MODEL_NAME     = 'microsoft/deberta-v3-base'

print('1. Chargement tokenizer local...')
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)
print(f'   OK: {tok.__class__.__name__}')

print('2. Chargement SpanClassifier (coarse_embed_dim=128)...')
model = SpanClassifier(MODEL_NAME, num_labels=22, num_coarse=6, coarse_embed_dim=128).float()
print('   OK')

print('3. Vérification COARSE_TO_FINE couvre bien les 22 labels...')
all_covered = sorted(idx for idxs in COARSE_TO_FINE.values() for idx in idxs)
assert all_covered == list(range(22)), f'Manquants : {set(range(22)) - set(all_covered)}'
print('   OK — 22 labels couverts')

print('4. Forward pass avec masquage coarse=LOC (famille 1 → labels [5,6,7,12])...')
enc = tok(
    ['Paris est une belle ville.'],
    padding=True, truncation=True, max_length=64,
    return_attention_mask=True, return_offsets_mapping=False
)
batch = {
    'input_ids':      torch.tensor(enc['input_ids']),
    'attention_mask': torch.tensor(enc['attention_mask']),
    'spans': [[{'start': 1, 'end': 2, 'coarse_id': 1}]],
}
model.eval()
with torch.no_grad():
    logits = model(batch)

print(f'   logits shape : {logits.shape}  (attendu: torch.Size([1, 22]))')
assert logits.shape == torch.Size([1, 22])

# Seuls les labels LOC doivent être non -inf
loc_labels = COARSE_TO_FINE[1]  # [5, 6, 7, 12]
for i in range(22):
    val = logits[0, i].item()
    if i in loc_labels:
        assert val > -1e8, f'Label {i} devrait être actif pour coarse=LOC'
    else:
        assert val < -1e8, f'Label {i} devrait être masqué pour coarse=LOC'
print(f'   Masquage OK — labels actifs : {loc_labels}')
print(f'   Pred label idx : {logits.argmax(dim=-1).item()} (dans espace LOC contraint)')

print('\nALL OK ✅')

