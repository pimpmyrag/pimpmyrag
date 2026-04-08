#!/usr/bin/env python3
"""Test direct RoBERTa - détection ORG sur phrases problématiques."""
import numpy as np, json
import onnxruntime as ort
from transformers import AutoTokenizer

sess      = ort.InferenceSession("training_output/model_v5.onnx", providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained("training_output/checkpoint-7800")
with open("training_output/checkpoint-7800/config.json") as f:
    id2label = json.load(f)["id2label"]

PHRASES = [
    # Faux positifs signalés
    "Il agit avec prudence et détermination.",
    "La température atteignit -18 degrés.",
    "Le déficit s'élevait à 126 millions.",
    "Aucune arrestation n'eut lieu.",
    "Le résultat fut décevant pour tout le monde.",
    "Il convient de prendre des mesures immédiates.",
    "La situation économique s'est améliorée.",
    "Le taux de chômage atteignit 12 %.",
    "La loi prévoit une peine maximale de vingt ans.",
    # Vrais ORG — doivent passer
    "La Société Générale a publié ses résultats.",
    "La BCE a relevé ses taux directeurs.",
    "Microsoft et Google dominent le marché.",
    "L'Assemblée nationale a voté la loi.",
    "Le ministère de l'Économie a publié un rapport.",
]

def predict(text):
    enc    = tokenizer(text, return_tensors="np")
    feed   = {k: v.astype(np.int64) for k, v in enc.items() if k in ("input_ids","attention_mask")}
    logits = sess.run(["logits"], feed)[0][0]          # (seq, 13)
    preds  = logits.argmax(-1)
    toks   = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

    # Reconstruit les spans BIO en texte lisible
    spans, buf, cur = [], [], None
    for tok, pid in zip(toks, preds):
        if tok in ("<s>","</s>","<pad>"): continue
        lab   = id2label[str(pid)]
        word  = tok.lstrip("\u2581")          # supprime le marqueur SentencePiece ▁
        prefix= "\u2581" in tok or tok == word # début de mot ?
        if lab.startswith("B-"):
            if buf: spans.append((" ".join(buf), cur))
            buf, cur = [word], lab[2:]
        elif lab.startswith("I-") and cur == lab[2:]:
            buf.append(word)
        else:
            if buf: spans.append((" ".join(buf), cur))
            buf, cur = [], None
    if buf: spans.append((" ".join(buf), cur))
    return spans

print("="*65)
for phrase in PHRASES:
    spans = predict(phrase)
    org   = [(t,l) for t,l in spans if l=="ORG"]
    other = [(t,l) for t,l in spans if l!="ORG"]
    print(f"\n▶ {phrase}")
    for t,l in other:
        print(f"    [{l:<7}] {t}")
    for t,l in org:
        fp = t and (t[0].islower() or t[0].isdigit())
        print(f"    [{'ORG⚠️ ' if fp else 'ORG✅ '}] {t}")
    if not spans:
        print("    (aucune entité)")
print("\n" + "="*65)

