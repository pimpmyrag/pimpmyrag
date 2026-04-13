#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparaison côte à côte des deux modèles NER fine-tunés :
  • XLM-RoBERTa  (training_output/model.onnx)
  • DeBERTa-v3   (../../debertav3-ner/best_model-v2.onnx)

Labels (coarse, BIO) : PER · LOC · OBJECT · ORG · TIME · EVENT

Usage :
  # Phrases de test prédéfinies
  python compare_models.py

  # Phrase à la volée
  python compare_models.py --sentence "Apple a lancé l'iPhone 16 à San Francisco en septembre."

  # Sur un échantillon du JSONL pré-annoté
  python compare_models.py --jsonl ../../scripts/object_event_sentences.preannotated.jsonl --n 20
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# ── Chemins par défaut ───────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
ROBERTA_DIR = ROOT / "training_output"
DEBERTA_DIR      = ROOT / "../../debertav3-ner"
DEBERTA_TOK_DIR  = DEBERTA_DIR / "tokenizer_from_hf"  # tokenizer.json réel (0 bytes à la racine)

# ── Labels ───────────────────────────────────────────────────────────────────
COARSE = ["PER", "LOC", "OBJECT", "ORG", "TIME", "EVENT"]
# id2label pour les deux modèles (même schéma BIO)
ID2LABEL_BIO = {
    0: "O",
    **{i + 1:       f"B-{l}" for i, l in enumerate(COARSE)},
    **{i + 7:       f"I-{l}" for i, l in enumerate(COARSE)},
}
# Fallback : lire depuis config.json si dispo
def _load_id2label(cfg_path: Path) -> dict[int, str]:
    try:
        cfg = json.loads(cfg_path.read_text())
        raw = cfg.get("id2label", {})
        if raw:
            return {int(k): v for k, v in raw.items()}
    except Exception:
        pass
    return ID2LABEL_BIO

# ── Couleurs terminal ────────────────────────────────────────────────────────
_C = {
    "PER":    "\033[94m",  # bleu
    "LOC":    "\033[92m",  # vert
    "ORG":    "\033[93m",  # jaune
    "TIME":   "\033[95m",  # magenta
    "EVENT":  "\033[91m",  # rouge
    "OBJECT": "\033[96m",  # cyan
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"

def _color(label: str, text: str) -> str:
    coarse = label.split("-")[-1] if "-" in label else label
    return _C.get(coarse, "") + text + _RESET


# ═══════════════════════════════════════════════════════════════════════════════
#  MODÈLE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class OnnxNER:
    def __init__(self, name: str, onnx_path: Path, tokenizer_dir: Path):
        self.name = name
        print(f"  Chargement {name}  ({onnx_path.name})…")
        self.sess      = ort.InferenceSession(str(onnx_path),
                                               providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
        self.id2label  = _load_id2label(tokenizer_dir / "config.json")
        inp_names      = {i.name for i in self.sess.get_inputs()}
        self._has_tti  = "token_type_ids" in inp_names

    def predict(self, text: str) -> list[dict]:
        """Retourne une liste de spans {label, text, start, end}."""
        words = text.split()
        enc = self.tokenizer(
            words, is_split_into_words=True,
            return_tensors="np", truncation=True,
            padding=False, max_length=512,
        )
        feed = {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        if self._has_tti:
            feed["token_type_ids"] = (
                enc["token_type_ids"].astype(np.int64)
                if "token_type_ids" in enc
                else np.zeros_like(enc["input_ids"], dtype=np.int64)
            )

        logits    = self.sess.run(["logits"], feed)[0][0]   # (seq, num_labels)
        pred_ids  = logits.argmax(-1)                        # (seq,)
        word_ids  = enc.word_ids(batch_index=0)

        # Aligner sur les mots (premier subword uniquement)
        word_label: dict[int, str] = {}
        for tok_idx, wid in enumerate(word_ids):
            if wid is None or wid in word_label:
                continue
            word_label[wid] = self.id2label.get(int(pred_ids[tok_idx]), "O")

        # Reconstruire le texte caractère par caractère pour les offsets
        spans: list[dict] = []
        char_pos = 0
        current: dict | None = None

        for wid, word in enumerate(words):
            label = word_label.get(wid, "O")
            tag   = label[0] if label != "O" else "O"   # B / I / O
            coarse = label.split("-")[-1] if "-" in label else None

            # Gestion des transitions BIO
            if tag == "B":
                if current:
                    spans.append(current)
                current = {"label": coarse, "start": char_pos,
                           "end": char_pos + len(word),
                           "text": word}
            elif tag == "I" and current and current["label"] == coarse:
                current["end"]  = char_pos + len(word)
                current["text"] += " " + word
            else:
                if current:
                    spans.append(current)
                current = None

            char_pos += len(word) + 1   # +1 espace

        if current:
            spans.append(current)
        return spans


# ═══════════════════════════════════════════════════════════════════════════════
#  AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_spans(spans: list[dict]) -> str:
    if not spans:
        return _DIM + "(aucune entité)" + _RESET
    return "  ".join(
        _color(s["label"], f"[{s['label']}] {s['text']}")
        for s in spans
    )

def _highlight(text: str, spans: list[dict]) -> str:
    """Colorise le texte brut selon les spans prédits."""
    if not spans:
        return text
    result = []
    prev = 0
    for s in sorted(spans, key=lambda x: x["start"]):
        result.append(text[prev:s["start"]])
        result.append(_color(s["label"], _BOLD + text[s["start"]:s["end"]] + _RESET))
        prev = s["end"]
    result.append(text[prev:])
    return "".join(result)


def compare(models: list[OnnxNER], sentences: list[str], gold_spans: list[list] | None = None):
    for si, sentence in enumerate(sentences):
        print(f"\n{'━'*70}")
        print(f"{_BOLD}Phrase {si+1}{_RESET} : {sentence}")
        print()

        all_results = []
        for m in models:
            spans = m.predict(sentence)
            all_results.append(spans)
            print(f"  {_BOLD}{m.name:20s}{_RESET} → {_fmt_spans(spans)}")
            print(f"  {' '*20}   {_highlight(sentence, spans)}")
            print()

        # Afficher les annotations de référence si dispo
        if gold_spans and si < len(gold_spans):
            gold = gold_spans[si]
            if gold:
                print(f"  {_BOLD}{'REF (heurist.)':20s}{_RESET} → ", end="")
                print("  ".join(
                    _color(s["label"].replace("hint_","").upper(),
                           f"[{s['label']}] {s['text']}")
                    for s in gold
                ))
                print()

    print(f"\n{'━'*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_SENTENCES = [
    "Emmanuel Macron a rencontré les dirigeants de l'Union européenne à Bruxelles.",
    "Le tremblement de terre a frappé la ville de Mexico mardi matin.",
    "Apple a présenté le nouvel iPhone 16 lors de la conférence annuelle à San Francisco.",
    "Des soldats ont utilisé des missiles pour détruire le pont stratégique.",
    "La révolution française de 1789 a conduit à l'exécution de Louis XVI.",
    "En janvier 2024, des inondations ont ravagé le sud de la France.",
    "The ship was loaded with weapons and ammunition before leaving the port.",
    "Ahab spotted the white whale from the deck of the Pequod near the Pacific Ocean.",
    "Le vaccin contre la grippe sera disponible dès octobre dans toutes les pharmacies.",
    "Lors du sommet du G7, les dirigeants ont signé un accord sur le climat.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roberta_dir",  default=str(ROBERTA_DIR))
    ap.add_argument("--deberta_dir",  default=str(DEBERTA_DIR))
    ap.add_argument("--deberta_tok",  default=str(DEBERTA_TOK_DIR))
    ap.add_argument("--sentence", default="", help="Phrase unique à tester")
    ap.add_argument("--jsonl",    default="", help="Fichier JSONL pré-annoté à échantillonner")
    ap.add_argument("--n",  type=int, default=20, help="Nb de phrases à tirer du JSONL")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"\n{_BOLD}Chargement des modèles…{_RESET}")

    roberta_onnx = Path(args.roberta_dir) / "model.onnx"
    deberta_onnx = Path(args.deberta_dir) / "best_model-v2.onnx"

    models: list[OnnxNER] = []
    if roberta_onnx.exists():
        models.append(OnnxNER("XLM-RoBERTa", roberta_onnx, Path(args.roberta_dir)))
    else:
        print(f"  [WARN] {roberta_onnx} introuvable — modèle ignoré")

    if deberta_onnx.exists():
        tok_dir = Path(args.deberta_tok)
        models.append(OnnxNER("DeBERTa-v3", deberta_onnx, tok_dir))
    else:
        print(f"  [WARN] {deberta_onnx} introuvable — modèle ignoré")

    if not models:
        sys.exit("[ERROR] Aucun modèle chargé.")

    # ── Choix des phrases ────────────────────────────────────────────────────
    sentences:  list[str]  = []
    gold_spans: list[list] = []

    if args.sentence:
        sentences  = [args.sentence]
        gold_spans = [[]]

    elif args.jsonl and Path(args.jsonl).exists():
        rng   = random.Random(args.seed)
        lines = Path(args.jsonl).read_text("utf-8").splitlines()
        lines = [l for l in lines if l.strip()]
        sample = rng.sample(lines, min(args.n, len(lines)))
        for l in sample:
            obj = json.loads(l)
            sentences.append(obj["text"])
            gold_spans.append(obj.get("spans", []))
        print(f"  {len(sentences)} phrases tirées de {Path(args.jsonl).name}\n")

    else:
        sentences  = DEFAULT_SENTENCES
        gold_spans = [[] for _ in sentences]

    compare(models, sentences, gold_spans)


if __name__ == "__main__":
    main()

