#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test rapide du modèle NER sur des phrases françaises couvrant les 6 labels.
Usage : python test_model.py [--model_dir ./training_output]
"""
import argparse, warnings
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# NE PAS hardcoder l'ordre des labels ici — utiliser model.config.id2label
# (l'ordre dans train_ner.py est B-PER,B-LOC,...,I-PER,... pas B-PER,I-PER,...)

# Phrases de test couvrant tous les labels
TEST_SENTENCES = [
    # PER + ORG
    "Emmanuel Macron a rencontré les dirigeants de l'Union européenne à Bruxelles.",
    # LOC + TIME
    "Le tremblement de terre a frappé la ville de Mexico mardi matin.",
    # EVENT + PER
    "La révolution française de 1789 a conduit à l'exécution de Louis XVI.",
    # OBJECT + ORG
    "Apple a présenté le nouvel iPhone 16 lors de la conférence annuelle.",
    # TIME + LOC
    "En janvier 2024, des inondations ont ravagé le sud de la France.",
    # Phrases mixtes
    "Le président américain Joe Biden a signé un accord de paix avec la Chine en mars.",
    "La tour Eiffel, construite par Gustave Eiffel en 1889, accueille des millions de touristes.",
    "L'attentat du 11 septembre 2001 à New York a changé la politique étrangère des États-Unis.",
    # PER multi-tokens
    "Jean-Pierre Dupont, directeur général de la Banque de France, a démissionné hier.",
    # OBJECT rare
    "Le tableau La Joconde de Léonard de Vinci est exposé au musée du Louvre à Paris.",
]

COLORS = {
    "PER": "\033[94m",     # bleu
    "LOC": "\033[92m",     # vert
    "ORG": "\033[93m",     # jaune
    "TIME": "\033[95m",    # magenta
    "EVENT": "\033[91m",   # rouge
    "OBJECT": "\033[96m",  # cyan
    "O": "\033[0m",
}
RESET = "\033[0m"
BOLD = "\033[1m"

def predict(model, tokenizer, sentence, device):
    id2label = model.config.id2label   # ← ordre réel du modèle
    words = sentence.split()
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt",
                    truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0]
    preds = logits.argmax(-1).cpu().tolist()
    # Aligner sur les mots
    word_ids_list = tokenizer(words, is_split_into_words=True,
                               truncation=True, max_length=256).word_ids()
    word_preds = {}
    for tok_idx, w_id in enumerate(word_ids_list):
        if w_id is None: continue
        if w_id not in word_preds:
            word_preds[w_id] = id2label.get(preds[tok_idx], "O")
    return [word_preds.get(i, "O") for i in range(len(words))]

def print_prediction(sentence, labels):
    words = sentence.split()
    print()
    # Ligne avec couleurs
    colored = []
    for word, lbl in zip(words, labels):
        if lbl == "O":
            colored.append(word)
        else:
            etype = lbl.split("-")[-1]
            color = COLORS.get(etype, "")
            prefix = lbl.split("-")[0]
            colored.append(f"{color}{BOLD}[{word}]{RESET}{color}_{prefix}-{etype}{RESET}")
    print("  " + " ".join(colored))

    # Résumé des entités trouvées
    entities = []
    current_words, current_type = [], None
    for word, lbl in zip(words, labels):
        if lbl == "O":
            if current_words:
                entities.append((" ".join(current_words), current_type))
                current_words, current_type = [], None
        else:
            prefix, etype = lbl.split("-", 1)
            if prefix in ("B", "U"):
                if current_words:
                    entities.append((" ".join(current_words), current_type))
                current_words = [word]
                current_type = etype
                if prefix == "U":
                    entities.append((" ".join(current_words), current_type))
                    current_words, current_type = [], None
            elif prefix in ("I", "L"):
                current_words.append(word)
                if prefix == "L":
                    entities.append((" ".join(current_words), current_type))
                    current_words, current_type = [], None
    if current_words:
        entities.append((" ".join(current_words), current_type))

    if entities:
        ent_str = "  → " + " | ".join(
            f"{COLORS.get(t,'')}{BOLD}{e}{RESET} ({t})" for e, t in entities
        )
        print(ent_str)
    else:
        print("  → (aucune entité détectée)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./training_output")
    parser.add_argument("--sentence", default=None, help="Phrase custom à tester")
    args = parser.parse_args()

    print(f"\n{BOLD}Chargement du modèle depuis {args.model_dir}...{RESET}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    device = torch.device("cpu")
    model.to(device).eval()
    print(f"Modèle chargé — {model.config.num_labels} labels\n")

    print(f"{'─'*70}")
    print(f"  Labels : PER={COLORS['PER']}{BOLD}bleu{RESET}  "
          f"LOC={COLORS['LOC']}{BOLD}vert{RESET}  "
          f"ORG={COLORS['ORG']}{BOLD}jaune{RESET}  "
          f"TIME={COLORS['TIME']}{BOLD}magenta{RESET}  "
          f"EVENT={COLORS['EVENT']}{BOLD}rouge{RESET}  "
          f"OBJECT={COLORS['OBJECT']}{BOLD}cyan{RESET}")
    print(f"{'─'*70}")

    sentences = [args.sentence] if args.sentence else TEST_SENTENCES
    for sent in sentences:
        labels = predict(model, tokenizer, sent, device)
        print(f"\n{BOLD}▶{RESET} {sent}")
        print_prediction(sent, labels)

    print(f"\n{'─'*70}\n")

if __name__ == "__main__":
    main()

