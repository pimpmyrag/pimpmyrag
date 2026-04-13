#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import warnings
import torch
import torch.nn as nn

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ACCELERATE_DISABLE_MIXED_PRECISION"] = "1"

# Dans Docker (Linux) : désactiver MPS (non disponible) et forcer CPU
# Sur Mac natif : laisser PyTorch utiliser Apple Accelerate (Matmul ~8× plus rapide)
_IN_DOCKER = os.path.exists("/.dockerenv")
if _IN_DOCKER:
    os.environ["PYTORCH_MPS_ENABLE"] = "0"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False
else:
    # Fallback CPU pour les ops MPS non supportées (évite les crashs)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


import argparse
import logging
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

# seqeval is preferred for sequence-level metrics (NER); fallback to sklearn-based token-level metrics
try:
    from seqeval.metrics import precision_score, recall_score, f1_score
    from seqeval.scheme import BILOU as _BILOU_SCHEME
    _HAS_SEQEVAL = True
except Exception:
    _HAS_SEQEVAL = False
    _BILOU_SCHEME = None
    precision_score = recall_score = f1_score = None

from collections import Counter
try:
    from sklearn.metrics import f1_score as sk_f1_score, precision_score as sk_precision_score, recall_score as sk_recall_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False



# ------------------------------------------------------------
# LABEL DEFINITIONS
# ------------------------------------------------------------
TYPES = ["PER", "LOC", "OBJECT", "ORG", "TIME", "EVENT"]

LABEL_LIST = (
        ["O"] +
        [f"B-{t}" for t in TYPES] +
        [f"I-{t}" for t in TYPES] +
        [f"L-{t}" for t in TYPES] +
        [f"U-{t}" for t in TYPES]
)

LABEL_TO_ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}



# ------------------------------------------------------------
# READ BILOU (robuste) + helpers pour pieces -> mots
# ------------------------------------------------------------
def read_bilou(path):
    """Lit un fichier BILOU et retourne une liste de dict {tokens: [...], labels: [...]}.
    Le fichier peut utiliser '\t' ou des espaces comme séparateur. Les tokens peuvent être
    des pieces SentencePiece (préfixe '▁').
    """
    docs = []
    tokens, labels = [], []

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")

            if not line.strip():
                if tokens:
                    docs.append({"tokens": tokens, "labels": labels})
                tokens, labels = [], []
                continue

            # Accept both tab-separated and space-separated formats; label is last column
            parts = line.split()
            if len(parts) < 2:
                # malformed line: skip but warn
                print(f"[WARN] Ligne {lineno} ignorée (format inattendu): '{line}'")
                continue

            tok = parts[0]
            lab = parts[-1]
            tokens.append(tok)
            labels.append(lab)

        if tokens:
            docs.append({"tokens": tokens, "labels": labels})

    return docs


def reconstruct_words_from_pieces(pieces):
    """Regroupe des pieces SentencePiece (marque '▁' pour début de mot) en mots.
    Retourne la liste des mots (strings). Si les tokens ne semblent pas être des pieces
    (pas de '▁' trouvé), on renvoie la liste d'origine (considérée déjà mots).
    """
    if not pieces:
        return []
    # heuristique: s'il n'y a aucun token avec préfixe '▁', on suppose que ce sont déjà des mots
    if not any(p.startswith('▁') for p in pieces):
        return pieces.copy()

    words = []
    cur = ''
    for token in pieces:
        if token.startswith('▁'):
            # nouveau mot
            piece = token.lstrip('▁')
            if cur != '':
                words.append(cur)
            cur = piece
        else:
            cur = cur + token
    if cur != '':
        words.append(cur)
    return words


def derive_word_label_from_piece_labels(piece_labels):
    """Heuristique conservative pour déduire un label BILOU pour un mot à partir
    des labels de ses pieces. Règles :
      - si tous 'O' -> 'O'
      - si présence d'un 'U-TYPE' -> utiliser ce 'U-TYPE'
      - si la première non-O est 'B-TYPE' ou 'I-TYPE' ou 'L-TYPE' on renvoie 'U-TYPE'
        (on compacte l'entité interne au mot en unité)
      - fallback -> 'O'
    Cette logique est volontairement simple et doit être adaptée si tes pieces couvrent
    des entités multi-mots: dans ce cas il faudra un alignement plus fin.
    """
    non_o = [l for l in piece_labels if l and l != 'O']
    if not non_o:
        return 'O'
    for l in non_o:
        if l.startswith('U-'):
            return l
    # prendre le type du premier non-O
    first = non_o[0]
    if '-' in first:
        t = first.split('-', 1)[1]
        return f'U-{t}'
    return 'O'


# ------------------------------------------------------------
# TOKENIZE + ALIGN LABELS (XLM-R)
# ------------------------------------------------------------
def build_tokenizer_align(tokenizer):
    """Factory qui retourne une fonction compatible dataset.map()"""

    def tokenize_and_align_labels(example):
        encoded = tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=256
        )

        word_ids = encoded.word_ids()
        labels = []
        prev = None

        for w_id in word_ids:
            if w_id is None:
                labels.append(-100)
            elif w_id != prev:
                labels.append(LABEL_TO_ID[example["labels"][w_id]])
            else:
                labels.append(-100)
            prev = w_id

        encoded["labels"] = labels
        return encoded

    return tokenize_and_align_labels



# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
def align_predictions(preds, labels):
    preds = np.argmax(preds, axis=-1)
    out_preds, out_labels = [], []

    for p_seq, l_seq in zip(preds, labels):
        cp, cl = [], []
        for p_i, l_i in zip(p_seq, l_seq):
            if l_i == -100:
                continue
            cp.append(ID_TO_LABEL[p_i])
            cl.append(ID_TO_LABEL[l_i])
        out_preds.append(cp)
        out_labels.append(cl)

    return out_preds, out_labels


def compute_metrics(eval_pred):
    preds, labels = align_predictions(eval_pred.predictions, eval_pred.label_ids)

    # prefer seqeval for NER-style (span) metrics
    if _HAS_SEQEVAL:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="seqeval")
                p = precision_score(labels, preds, scheme=_BILOU_SCHEME, zero_division=0)
                r = recall_score(labels, preds, scheme=_BILOU_SCHEME, zero_division=0)
                f = f1_score(labels, preds, scheme=_BILOU_SCHEME, zero_division=0)
            # F2 : β=2 → penalise 4× plus les faux négatifs que les faux positifs
            # clé pour un candidat generator (span manquée = définitivement perdue)
            f2 = (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0.0
            return {"precision": p, "recall": r, "f1": f, "f2": f2}
        except Exception as e:
            logging.getLogger("trainer").warning("seqeval compute_metrics failed: %s", e)

    # fallback to sklearn on flattened labels if available
    if _HAS_SKLEARN:
        flat_p = [x for seq in preds for x in seq]
        flat_l = [x for seq in labels for x in seq]
        try:
            p = sk_precision_score(flat_l, flat_p, average="macro", zero_division=0)
            r = sk_recall_score(flat_l, flat_p, average="macro", zero_division=0)
            f = sk_f1_score(flat_l, flat_p, average="macro", zero_division=0)
            return {"precision": p, "recall": r, "f1": f}
        except Exception as e:
            logging.getLogger("trainer").warning("sklearn fallback compute_metrics failed: %s", e)

    return {}


class TrainingInspector(TrainerCallback):
    """Callback qui effectue au bout de chaque epoch une évaluation rapide
    sur un petit échantillon de validation, logge les metrics et détecte
    les effondrements (p.ex. modèle prédit presque uniquement 'O').
    """
    def __init__(self, sample_dataset, tokenizer, id2label, max_samples=200, out_dir=None):
        self.sample = sample_dataset.select(range(min(len(sample_dataset), max_samples)))
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.out_dir = out_dir
        self.history = []  # Pour stocker l'évolution des métriques

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs.get('trainer')
        if trainer is None:
            return
        # run prediction on sample
        preds_output = trainer.predict(self.sample)
        preds, labels = align_predictions(preds_output.predictions, preds_output.label_ids)

        # compute seq-level metrics if available, else fallback to sklearn micro-f1 on flattened labels
        try:
            if _HAS_SEQEVAL:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="seqeval")
                    p = precision_score(labels, preds, scheme=_BILOU_SCHEME, zero_division=0)
                    r = recall_score(labels, preds, scheme=_BILOU_SCHEME, zero_division=0)
                    f = f1_score(labels, preds, scheme=_BILOU_SCHEME, zero_division=0)
            elif _HAS_SKLEARN:
                # flatten
                flat_p = [x for seq in preds for x in seq]
                flat_l = [x for seq in labels for x in seq]
                p = sk_precision_score(flat_l, flat_p, average='macro', zero_division=0)
                r = sk_recall_score(flat_l, flat_p, average='macro', zero_division=0)
                f = sk_f1_score(flat_l, flat_p, average='macro', zero_division=0)
            else:
                p = r = f = None
        except Exception as e:
            trainer.log({"trainer_inspector/error": str(e)})
            return

        # label distribution of predictions
        cnt = Counter([lbl for seq in preds for lbl in seq])
        total_preds = sum(cnt.values()) or 1
        o_ratio = cnt.get('O', 0) / total_preds

        # Matrice de confusion (si sklearn dispo)
        confusion = None
        if _HAS_SKLEARN:
            from sklearn.metrics import confusion_matrix
            flat_p = [x for seq in preds for x in seq]
            flat_l = [x for seq in labels for x in seq]
            try:
                confusion = confusion_matrix(flat_l, flat_p, labels=list(self.id2label.values()))
            except Exception:
                confusion = None

        # Historique
        self.history.append({"epoch": float(state.epoch), "precision": p, "recall": r, "f1": f, "O_ratio": o_ratio})

        # Affichage détaillé
        print("\n[Inspector] Epoch %.2f" % float(state.epoch))
        print("  F1: %.4f | Precision: %.4f | Recall: %.4f | O_ratio: %.3f" % (f, p, r, o_ratio))
        print("  Distribution des classes (préd):", dict(cnt))
        if confusion is not None:
            print("  Matrice de confusion (labels principaux):")
            import numpy as np
            # Affiche seulement les 6 premières classes pour lisibilité
            labels_show = list(self.id2label.values())[:6]
            print("     ", "\t".join(labels_show))
            for i, row in enumerate(confusion[:6, :6]):
                print("  %s\t%s" % (labels_show[i], "\t".join(str(x) for x in row)))
        print("  Historique F1/Precision/Recall:")
        for h in self.history:
            print("    Epoch %.2f: F1=%.4f, Prec=%.4f, Rec=%.4f, O=%.3f" % (h["epoch"], h["f1"], h["precision"], h["recall"], h["O_ratio"]))

        log.info(f"[Inspector] sample_size={len(self.sample)} precision={p} recall={r} f1={f} O_ratio={o_ratio:.3f}")
        trainer.log({
            "inspector/precision": p,
            "inspector/recall": r,
            "inspector/f1": f,
            "inspector/O_ratio": o_ratio,
        })

        # detect collapse (trop de O)
        if o_ratio > 0.95:
            log.warning("[Inspector] High O ratio detected (%.2f) — possible collapse. Dumping examples.", o_ratio)
            if self.out_dir:
                import json
                path = os.path.join(self.out_dir, f"inspector_epoch_{int(state.epoch)}.jsonl")
                with open(path, 'w', encoding='utf-8') as outf:
                    for text, pred_seq, gold_seq in zip(self.sample['tokens'], preds, labels):
                        outf.write(json.dumps({"text": text, "pred": pred_seq, "gold": gold_seq}, ensure_ascii=False) + '\n')


class FrequentEvalCallback(TrainerCallback):
    """Callback to trigger evaluation every `eval_steps` steps so metrics (F1) appear frequently.
    Use with care: calling evaluate() interrupts training then resumes; set eval_steps reasonably.
    """
    def __init__(self, eval_steps: int, min_steps: int = 1):
        self.eval_steps = max(1, int(eval_steps))
        self.min_steps = min_steps

    def on_step_end(self, args, state, control, **kwargs):
        # ensure enough steps have passed
        if state.global_step is None:
            return
        gs = int(state.global_step)
        if gs < self.min_steps:
            return
        if gs % self.eval_steps == 0:
            trainer = kwargs.get('trainer')
            if trainer is None:
                return
            try:
                logging.getLogger('trainer').info(f"[FrequentEval] Triggering evaluation at step {gs}")
                trainer.evaluate()
            except Exception as e:
                logging.getLogger('trainer').warning(f"[FrequentEval] evaluation failed: {e}")



# ------------------------------------------------------------
# WEIGHTED TRAINER — compense le déséquilibre OBJECT/EVENT
# ------------------------------------------------------------
class WeightedTrainer(Trainer):
    """Trainer avec CrossEntropy pondérée.
    Les class_weights (un tenseur par label) sont calculés dynamiquement
    depuis la distribution réelle du dataset d'entraînement, puis clampés
    pour éviter les extrêmes. Le label O reçoit toujours un poids faible
    (0.1) pour ne pas dominer la loss.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # tensor shape [num_labels], ou None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--bilou", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="xlm-roberta-base", help="HF model id (ex: xlm-roberta-base)")
    parser.add_argument("--save_tokenizer_slow", action="store_true", help="save slow tokenizer (SentencePiece) into output_dir")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval_steps", type=int, default=200, help='evaluate every N steps (use with evaluation_strategy=steps)')
    parser.add_argument("--logging_steps", type=int, default=50, help='log every N steps')
    parser.add_argument("--quick_eval", action="store_true", help='use a small quick eval set for frequent evaluation (faster)')
    parser.add_argument("--quick_eval_size", type=int, default=200, help='number of samples to use for quick eval')
    parser.add_argument("--mini_train", action="store_true", help='enable mini-training on 1000 sentences, 1 or 2 epochs, quick_eval, frequent logging')
    parser.add_argument("--mini_eval_ratio", type=float, default=0.3, help='ratio of dataset used for evaluation in mini-training mode (ex: 0.3 for 30 percent)')
    args = parser.parse_args()

    # Mode mini-training : override les paramètres pour un debug rapide
    if getattr(args, "mini_train", False):
        args.epochs = 2
        args.batch_size = 8
        args.lr = 1e-4
        args.eval_steps = 20
        args.logging_steps = 5
        args.quick_eval = True
        args.quick_eval_size = 200
        print(f"[INFO] Mini-training activé : 1000 phrases, 2 epochs, quick_eval ON, logging_steps=5, eval_steps=20, mini_eval_ratio={args.mini_eval_ratio}")

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("trainer")

    MODEL = args.model

    # Load raw dataset
    docs = read_bilou(args.bilou)

    # If the tokens look like SentencePiece pieces (have '▁'), try to reconstruct words
    # and derive word-level BILOU labels (heuristic). This helps alignement avec
    # fast tokenizers qui attendent des mots quand is_split_into_words=True.
    processed_docs = []
    for d in docs:
        toks = d["tokens"]
        labs = d["labels"]
        if any(t.startswith('▁') for t in toks):
            words = reconstruct_words_from_pieces(toks)
            # regrouper piece labels par mot
            word_labels = []
            cur_piece_labels = []
            for token, lab in zip(toks, labs):
                if token.startswith('▁'):
                    if cur_piece_labels:
                        word_labels.append(derive_word_label_from_piece_labels(cur_piece_labels))
                    cur_piece_labels = [lab]
                else:
                    cur_piece_labels.append(lab)
            if cur_piece_labels:
                word_labels.append(derive_word_label_from_piece_labels(cur_piece_labels))

            # as a fallback if lengths mismatch, drop this sentence to avoid crash
            if len(words) != len(word_labels):
                print(f"[WARN] mismatch words/labels lengths ({len(words)}/{len(word_labels)}) – sentence skipped")
                continue
            processed_docs.append({"tokens": words, "labels": word_labels})
        else:
            processed_docs.append(d)

    if not processed_docs:
        raise RuntimeError("Aucune phrase valide après preprocessing des pieces. Vérifie ton fichier BILOU.")

    # ── Oversampling des labels faibles ────────────────────────────────────────
    # OBJECT (947 entités) et EVENT (1670) sont ~4.6× et ~2.6× moins représentés que PER.
    # On duplique ×2 toutes les phrases contenant au moins une de ces entités.
    _rare_types = {"OBJECT", "EVENT"}
    _rare_docs = [
        d for d in processed_docs
        if any(lbl.split("-")[-1] in _rare_types for lbl in d["labels"] if lbl != "O")
    ]
    processed_docs = processed_docs + _rare_docs
    random.seed(42)
    random.shuffle(processed_docs)
    print(f"[INFO] Oversampling : {len(_rare_docs)} phrases OBJECT/EVENT dupliquées → "
          f"{len(processed_docs)} phrases au total")


    # Si mini_train, on prend un sous-échantillon de 1000 phrases max et on split selon mini_eval_ratio
    if getattr(args, "mini_train", False):
        processed_docs = processed_docs[:1000]
        test_size = args.mini_eval_ratio
        ds = Dataset.from_list(processed_docs).train_test_split(test_size=test_size, seed=42)
        print(f"[INFO] Mini-training split : {len(ds['train'])} train / {len(ds['test'])} eval (ratio {test_size})")
        print(f"[DEBUG] Exemples train : {ds['train'][:2]}")
        print(f"[DEBUG] Exemples eval : {ds['test'][:2]}")
    else:
        ds = Dataset.from_list(processed_docs).train_test_split(test_size=0.1, seed=42)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    tokenize_fn = build_tokenizer_align(tokenizer)

    # Tokenize
    train_tok = ds["train"].map(
        tokenize_fn,
        batched=False,
        remove_columns=["tokens", "labels"]
    )

    val_tok = ds["test"].map(
        tokenize_fn,
        batched=False,
        remove_columns=["tokens", "labels"]
    )

    train_tok.set_format("torch")
    val_tok.set_format("torch")

    # Model configuration
    cfg = AutoConfig.from_pretrained(
        MODEL,
        num_labels=len(LABEL_LIST),
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
    )

    model = AutoModelForTokenClassification.from_pretrained(MODEL, config=cfg)
    collator = DataCollatorForTokenClassification(tokenizer)

    # Training arguments
    # En mini_train on désactive les checkpoints intermédiaires (save_strategy="no") pour ne pas
    # remplir le disque avec les optimizer states (~768 MB chacun).
    # eval_strategy="steps" → le Trainer évalue automatiquement et affiche precision/recall/F1.
    is_mini = getattr(args, "mini_train", False)

    # logging_dir est déprécié en transformers v5.x → passer via variable d'environnement
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", os.path.join(args.output_dir, "logs"))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # ── évaluation automatique → F1 apparaît dans les logs ──────────────────
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        # ── sauvegarde : désactivée en mini_train pour économiser l'espace disque ─
        save_strategy="no" if is_mini else "steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        # ── chargement du meilleur modèle uniquement quand on sauvegarde ──────────
        load_best_model_at_end=not is_mini,
        metric_for_best_model="f2",   # F2 : β=2 → prioritise recall (candidat generator)
        greater_is_better=True,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=args.logging_steps,
        fp16=False,
        bf16=False,
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
    )


    # prepare small sample for inspector (use part of val_tok before formatting to torch)
    try:
        sample_ds = ds['test']
    except Exception:
        sample_ds = None

    # Toujours initialiser un jeu d'échantillon pour l'inspector
    inspector = None
    sample_tok = None
    quick_tok = None
    if sample_ds is not None:
        sample_tok = sample_ds.map(tokenize_fn, batched=False, remove_columns=["tokens", "labels"]) 
        if args.quick_eval:
            n = min(args.quick_eval_size, len(sample_tok))
            quick_tok = sample_tok.select(range(n))
        # Toujours créer l'inspector, même si quick_eval n'est pas activé
        inspector = TrainingInspector(sample_tok, tokenizer, ID_TO_LABEL, max_samples=args.quick_eval_size, out_dir=args.output_dir)
    else:
        # Fallback : utiliser val_tok si sample_ds absent
        sample_tok = val_tok
        inspector = TrainingInspector(sample_tok, tokenizer, ID_TO_LABEL, max_samples=args.quick_eval_size, out_dir=args.output_dir)

    # choose eval dataset: quick small one if requested, otherwise full val_tok
    eval_dataset_for_trainer = quick_tok if (args.quick_eval and quick_tok is not None) else val_tok

    # build callbacks list
    callbacks = []
    if inspector is not None:
        callbacks.append(inspector)
    # FrequentEvalCallback supprimé : eval_strategy="steps" dans TrainingArguments gère l'évaluation

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_dataset_for_trainer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    log.info("🚀 Starting training…")

    print("=== DEBUG: Avant trainer.train() ===")
    trainer.train()
    print("=== DEBUG: Après trainer.train() ===")


    # Affichage explicite de l'historique F1/precision/recall du callback inspector
    if inspector is not None and hasattr(inspector, "history") and inspector.history:
        print("\n[Résumé TrainingInspector] Historique F1/Precision/Recall:")
        for h in inspector.history:
            print("  Epoch %.2f: F1=%.4f, Prec=%.4f, Rec=%.4f, O=%.3f" % (h["epoch"], h["f1"], h["precision"], h["recall"], h["O_ratio"]))
    else:
        print("\n=== DEBUG: Début de la section d'évaluation forcée (aucun historique collecté) ===")
        # Forcer une évaluation si rien n'a été collecté
        eval_dataset = ds["test"] if "test" in ds else None
        print(f"[DEBUG] Type eval_dataset: {type(eval_dataset)}, taille: {len(eval_dataset) if eval_dataset is not None else 'None'}")
        if eval_dataset is not None and len(eval_dataset) > 0:
            print("[DEBUG] Exemples eval_dataset:", eval_dataset[:2])
            print("[DEBUG] Début évaluation forcée sur eval_dataset...")
            try:
                f, p, r, o_ratio, cnt = evaluate_ner(model, eval_dataset, LABEL_LIST, tokenizer, args)
                print("[Forcé] F1: %.4f | Precision: %.4f | Recall: %.4f | O_ratio: %.3f" % (f, p, r, o_ratio))
                print("[Forcé] Distribution des classes (pred):", dict(cnt))
                print("=== DEBUG: Fin de l'évaluation forcée, métriques affichées ===")
            except Exception as e:
                print("[Erreur] Exception lors de l'évaluation forcée:", e)
        else:
            print("[Erreur] Dataset d'évaluation vide, aucune métrique calculable.")
        print("=== DEBUG: Fin de la section d'évaluation forcée ===")

    log.info("✅ Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    log.info("✅ Done.")

    # Optionnel : sauvegarder explicitement le tokenizer slow (SentencePiece) dans output_dir
    if getattr(args, "save_tokenizer_slow", False):
        try:
            save_dir = os.path.join(args.output_dir, "tokenizer-slow")
            tok_slow = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
            tok_slow.save_pretrained(save_dir)
            log.info("✅ Slow tokenizer saved to %s", save_dir)
        except Exception as e:
            log.warning("⚠️  Could not save slow tokenizer (sentencepiece): %s", e)

    print("=== FIN DU SCRIPT ===")


def evaluate_ner(model, eval_dataset, label_list, tokenizer, args):
    """
    Évalue le modèle sur le dataset donné et retourne (f1, precision, recall, o_ratio, counter)
    """
    import numpy as np
    from collections import Counter
    model.eval()
    # On suppose que eval_dataset est un Dataset HuggingFace
    # Tokenisation si nécessaire
    if hasattr(eval_dataset, 'format') and getattr(eval_dataset, 'format', None) != 'torch':
        # On suppose que le dataset est déjà tokenisé
        pass
    # DataLoader avec collate_fn pour padding dynamique
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForTokenClassification
    collator = DataCollatorForTokenClassification(tokenizer)
    loader = DataLoader(eval_dataset, batch_size=16, collate_fn=collator)
    all_preds = []
    all_labels = []
    device = next(model.parameters()).device
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
        preds = np.argmax(logits, axis=-1)
        for p_seq, l_seq in zip(preds, labels):
            cp, cl = [], []
            for p_i, l_i in zip(p_seq, l_seq):
                if l_i == -100:
                    continue
                cp.append(ID_TO_LABEL[p_i])
                cl.append(ID_TO_LABEL[l_i])
            all_preds.append(cp)
            all_labels.append(cl)
    # Calcul des métriques
    try:
        from seqeval.metrics import precision_score, recall_score, f1_score
        from seqeval.scheme import BILOU as _BILOU
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="seqeval")
            p = precision_score(all_labels, all_preds, scheme=_BILOU, zero_division=0)
            r = recall_score(all_labels, all_preds, scheme=_BILOU, zero_division=0)
            f = f1_score(all_labels, all_preds, scheme=_BILOU, zero_division=0)
    except Exception:
        # Fallback sklearn
        from sklearn.metrics import f1_score as sk_f1, precision_score as sk_p, recall_score as sk_r
        flat_p = [x for seq in all_preds for x in seq]
        flat_l = [x for seq in all_labels for x in seq]
        p = sk_p(flat_l, flat_p, average="macro", zero_division=0)
        r = sk_r(flat_l, flat_p, average="macro", zero_division=0)
        f = sk_f1(flat_l, flat_p, average="macro", zero_division=0)
    cnt = Counter([lbl for seq in all_preds for lbl in seq])
    total_preds = sum(cnt.values()) or 1
    o_ratio = cnt.get('O', 0) / total_preds
    return f, p, r, o_ratio, cnt


if __name__ == "__main__":
    main()
