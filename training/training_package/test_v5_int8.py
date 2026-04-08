#!/usr/bin/env python3
import numpy as np, json, onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("training_output/checkpoint-7800")
sess = ort.InferenceSession("training_output/model_v5_int8.onnx")

with open("training_output/checkpoint-7800/config.json") as f:
    id2label = json.load(f)["id2label"]

print(f"Labels du modele : {id2label}")
print(f"Inputs ONNX     : {[i.name for i in sess.get_inputs()]}")
print("=" * 65)

texts = [
    "Emmanuel Macron est le president de la Republique francaise .",
    "La tour Eiffel se situe a Paris , capitale de la France .",
    "Microsoft a annonce ses resultats financiers pour le premier trimestre 2025 .",
    "Le general de Gaulle a fonde la Cinquieme Republique en 1958 .",
    "L'ouragan Katrina a devaste la Nouvelle-Orleans en aout 2005 .",
    "Apple , Google et Amazon dominent le marche technologique mondial .",
    "Le traite de Versailles a ete signe le 28 juin 1919 .",
]

for text in texts:
    inputs = tokenizer(text, return_tensors="np")
    feed = {
        "input_ids":      inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    logits = sess.run(["logits"], feed)[0]
    preds  = logits.argmax(-1)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    entities = []
    cur_ent, cur_type = [], None
    for tok, pid in zip(tokens, preds):
        lab = id2label[str(pid)]
        if tok in ("<s>", "</s>", "<pad>"):
            continue
        if lab.startswith("B-"):
            if cur_ent:
                entities.append((" ".join(cur_ent).replace("\u2581", "").strip(), cur_type))
            cur_ent = [tok.lstrip("\u2581")]
            cur_type = lab[2:]
        elif lab.startswith("I-") and cur_type == lab[2:]:
            cur_ent.append(tok.lstrip("\u2581"))
        else:
            if cur_ent:
                entities.append((" ".join(cur_ent).replace("\u2581", "").strip(), cur_type))
            cur_ent, cur_type = [], None
    if cur_ent:
        entities.append((" ".join(cur_ent).replace("\u2581", "").strip(), cur_type))

    print(f"\n>> {text}")
    if entities:
        for ent, etype in entities:
            print(f"   {'['+etype+']':<12}  {ent}")
    else:
        print("   (aucune entite detectee)")

print("\n" + "=" * 65)
print("OK - modele v5 INT8 fonctionnel")

