#!/usr/bin/env python3
"""
Test direct du RoBERTa fine-tuné (model_v5.onnx) sur les phrases de faux positifs ORG.
Affiche les prédictions token par token + les spans reconstruits BIO.
"""
import numpy as np
import json
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL   = "training_output/model_v5.onnx"
TOKDIR  = "training_output/checkpoint-7800"
CONFIG  = "training_output/checkpoint-7800/config.json"

tokenizer = AutoTokenizer.from_pretrained(TOKDIR)
sess      = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
with open(CONFIG) as f:
    id2label = json.load(f)["id2label"]

# ── Phrases à tester ──────────────────────────────────────────────────────────
PHRASES = [
    # Faux positifs ORG signalés
    "Il agit avec prudence et détermination.",
    "La température atteignit -18 degrés.",
    "Le déficit s'élevait à 126 millions.",
    "Aucune arrestation n'eut lieu.",
    "Le résultat fut décevant pour tout le monde.",
    "Il convient de prendre des mesures immédiates.",
    "La situation économique s'est améliorée.",
    "Le taux de chômage atteignit 12 %.",
    "La loi prévoit une peine maximale de vingt ans.",
    # Vrais ORG – doivent être gardés
    "La Société Générale a publié ses résultats.",
    "La BCE a relevé ses taux directeurs.",
    "Microsoft et Google dominent le marché.",
    "L'Assemblée nationale a voté la loi.",
    "Le ministère de l'Économie a publié un rapport.",
    # Vrais PER/LOC/TIME
    "Emmanuel Macron a rencontré Olaf Scholz à Berlin.",
    "La bataille de Verdun eut lieu en 1916.",
]

def decode_spans(tokens, labels):
    """Reconstruit les spans BIO depuis les tokens/labels (ignore subwords)."""
    spans = []
    cur_toks, cur_type = [], None
    for tok, lab in zip(tokens, labels):
        if tok in ("<s>", "</s>", "<pad>"):
            if cur_toks:
                spans.append(("".join(cur_toks).replace("▁", " ").strip(), cur_type))
                cur_toks, cur_type = [], None
            continue
        clean = tok.lstrip("▁")
        if lab.startswith("B-"):
            if cur_toks:
                spans.append(("".join(cur_toks).replace("▁", " ").strip(), cur_type))
            cur_toks = [clean]
            cur_type = lab[2:]
        elif lab.startswith("I-") and cur_type == lab[2:]:
            cur_toks.append(clean)
        else:
            if cur_toks:
                spans.append(("".join(cur_toks).replace("▁", " ").strip(), cur_type))
            cur_toks, cur_type = [], None
    if cur_toks:
        spans.append(("".join(cur_toks).replace("▁", " ").strip(), cur_type))
    return spans

print("=" * 70)
print(f"Modèle : {MODEL}")
print(f"Labels : {list(id2label.values())}")
print("=" * 70)

for phrase in PHRASES:
    enc   = tokenizer(phrase, return_tensors="np")
    feed  = {k: v.astype(np.int64) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    logits = sess.run(["logits"], feed)[0]   # (1, seq, 13)
    preds  = logits.argmax(-1)[0]
    toks   = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    labels = [id2label[str(p)] for p in preds]

    spans = decode_spans(toks, labels)
    org_spans   = [(t, l) for t, l in spans if l == "ORG"]
    other_spans = [(t, l) for t, l in spans if l != "ORG"]

    print(f"\n{'─'*70}")
    print(f"  {phrase}")
    if not spans:
        print("  → (aucune entité)")
    for text, typ in other_spans:
        icon = "✅" if typ in ("PER", "LOC", "TIME", "EVENT") else "ℹ️ "
        print(f"  {icon} [{typ}] {repr(text)}")
    for text, typ in org_spans:
        # Heuristique : vrai ORG si commence par majuscule
        is_fp = text and (text[0].islower() or text[0].isdigit())
        icon  = "🔴 FAUX-POS" if is_fp else "✅ ORG OK  "
        print(f"  {icon} [{typ}] {repr(text)}")

print(f"\n{'='*70}")
print("Terminé.")

