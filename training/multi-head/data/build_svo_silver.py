"""
build_svo_silver.py
====================
Construit un dataset silver SVO + pronoms à partir du dataset NER existant.

Entrée  : train_v2.jsonl / val_v2.jsonl / test_v2.jsonl
          (format : {"id": ..., "text": ..., "spans": [{"label", "start", "end", "text"}, ...]})

Sortie  : train_svo_silver.jsonl  (même format, nouveaux labels svo_* + pron_*)

Labels produits
───────────────
  svo_verb      – verbe principal + ses auxiliaires (span char-level)
  svo_subject   – sujet grammatical (NP, sans relative enchâssée)
  svo_object    – objet direct
  svo_iobj      – objet indirect / oblique
  pron_subj     – pronom sujet   (avec features : person, number, gender)
  pron_obj      – pronom objet

Métadonnées par span
────────────────────
  voice       : "ACTIVE" | "PASSIVE"
  head_lemma  : lemme du mot tête du span
  head_upos   : POS universel du mot tête
  # tous les arguments (nsubj/nobj/obl) :
  gender      : "Masc" | "Fem" | null
  number      : "Sing" | "Plur" | null
  person      : "1" | "2" | "3" | null  (surtout utile pour les pronoms)
  animacy     : "Anim" | "Inan" | null
  definiteness: "Def" | "Ind" | null
  full_np_start/end/text : GN complet (sous-arbre sans relative) conservé en métadonnée
  # pronoms seulement :
  pron_person : "1" | "2" | "3"
  pron_number : "Sing" | "Plur"
  pron_gender : "Masc" | "Fem" | null

Usage
─────
  python build_svo_silver.py [--input train_v2.jsonl] [--output train_svo_silver.jsonl]
                             [--split all|train|val|test] [--gpu]
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import stanza


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

SUBJ_DEPRELS  = {"nsubj", "nsubj:pass", "csubj", "csubj:pass"}
OBJ_DEPRELS   = {"obj", "ccomp", "xcomp"}
IOBJ_DEPRELS  = {"iobj", "obl"}
AUX_DEPRELS   = {"aux", "aux:pass", "cop"}

# Sous-arbres exclus du span NP pour éviter les relatives longues
EXCLUDE_FROM_NP = {"relcl", "acl", "advcl", "ccomp", "xcomp", "parataxis"}

# Dépendances incluses pour former le span "tête NE" (NE multi-tokens)
# Exclut déterminants, adjectifs, numéraux → garde uniquement les composantes lexicales du nom
NE_INCLUDE_DEPRELS = {"flat", "flat:name", "nmod:name", "compound", "goeswith"}
# Dépendances des conjonctions coordonnées → chaque conjoint est un span séparé
CONJ_DEPRELS = {"conj"}

# Pronoms personnels français (liste fermée)
FR_PERS_PRONOUNS = {
    "je", "j", "me", "m", "moi",
    "tu", "te", "t", "toi",
    "il", "elle", "le", "la", "lui", "se", "s", "soi",
    "nous", "vous",
    "ils", "elles", "les", "leur", "eux",
    "y", "en",
}

# Features Stanza → champs normalisés
def _feat(word, key: str) -> str | None:
    feats = word.feats or ""
    for f in feats.split("|"):
        if f.startswith(key + "="):
            return f.split("=", 1)[1]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Représentation interne d'un token
# ─────────────────────────────────────────────────────────────────────────────

class Tok:
    __slots__ = ("idx", "text", "lemma", "upos", "deprel", "head",
                 "char_start", "char_end", "feats")

    def __init__(self, word, sent_offset: int):
        self.idx        = word.id
        self.text       = word.text
        self.lemma      = word.lemma or word.text
        self.upos       = word.upos or "X"
        self.deprel     = word.deprel or "dep"
        self.head       = word.head
        # start_char/end_char peuvent être None pour les tokens MWT décomposés
        self.char_start = sent_offset + word.start_char if word.start_char is not None else -1
        self.char_end   = sent_offset + word.end_char   if word.end_char   is not None else -1
        self.feats      = word.feats or ""


# ─────────────────────────────────────────────────────────────────────────────
# Arbre de dépendances
# ─────────────────────────────────────────────────────────────────────────────

def build_children(tokens: list[Tok]) -> dict[int, list[int]]:
    ch: dict[int, list[int]] = {t.idx: [] for t in tokens}
    ch[0] = []
    for t in tokens:
        ch.setdefault(t.head, []).append(t.idx)
    return ch


def subtree(root: int, children: dict, by_idx: dict,
            exclude: set | None = None) -> list[int]:
    res = [root]
    for c in children.get(root, []):
        if exclude and c in by_idx and by_idx[c].deprel in exclude:
            continue
        res.extend(subtree(c, children, by_idx, exclude))
    return res


def charspan(indices: list[int], by_idx: dict, sent_text: str,
             sent_offset: int) -> tuple[int, int, str]:
    toks = sorted(
        (by_idx[i] for i in indices if i in by_idx),
        key=lambda t: t.char_start,
    )
    if not toks:
        return 0, 0, ""
    cs = toks[0].char_start
    ce = toks[-1].char_end
    return cs, ce, sent_text[cs - sent_offset: ce - sent_offset]


def head_ne_indices(root_idx: int, children: dict, by_idx: dict) -> list[int]:
    """
    Retourne les indices du token tête + ses dépendants NE (flat, compound…),
    sans les déterminants, adjectifs, numéraux, etc.
    Utilisé pour aligner les frontières SVO sur celles du NER (tête nominale NE).
    """
    res = [root_idx]
    for c in children.get(root_idx, []):
        if c in by_idx and by_idx[c].deprel in NE_INCLUDE_DEPRELS:
            res.extend(head_ne_indices(c, children, by_idx))
    return res


def collect_conjoints(root_idx: int, children: dict, by_idx: dict) -> list[int]:
    """
    Retourne les indices des tokens directement liés par conj au root
    (conjonction de coordination : Pierre [et] Paul).
    Chaque conjoint sera émis comme span séparé.
    """
    return [
        c for c in children.get(root_idx, [])
        if c in by_idx and by_idx[c].deprel in CONJ_DEPRELS
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Extraction SVO + pronoms pour une phrase
# ─────────────────────────────────────────────────────────────────────────────

def extract_sentence(sentence, sent_offset: int, orig_text: str,
                     require_obl: bool = False,
                     max_sent_tokens: int = 0) -> list[dict]:
    """
    Retourne une liste de spans SVO + pronoms au format dict :
      {"start", "end", "text", "label", "voice", "head_lemma", "head_upos",
       "pron_person"?, "pron_number"?, "pron_gender"?}
    Si require_obl=True, retourne [] si la phrase ne contient aucun oblique (obl/iobj).
    Si max_sent_tokens > 0, retourne [] si la phrase dépasse ce nombre de tokens Stanza.
    """
    tokens = [Tok(w, sent_offset) for w in sentence.words if w.start_char is not None]
    if not tokens:
        return []

    # Filtre longueur (évite les phrases trop longues pour DeBERTa)
    if max_sent_tokens > 0 and len(tokens) > max_sent_tokens:
        return []

    # Filtre : la phrase doit contenir au moins un oblique
    if require_obl and not any(t.deprel in IOBJ_DEPRELS for t in tokens):
        return []

    sent_text = sentence.text
    by_idx    = {t.idx: t for t in tokens}
    children  = build_children(tokens)

    spans: list[dict] = []

    # ── Verbes candidats (ROOT + enchâssés sémantiquement porteurs) ───────────
    verb_roots = [
        t for t in tokens
        if t.upos in {"VERB", "AUX"}
        and t.deprel in {"root", "xcomp", "ccomp", "advcl", "acl"}
    ]

    for verb in verb_roots:
        # Voix
        ch_deprels = {by_idx[c].deprel for c in children.get(verb.idx, []) if c in by_idx}
        is_passive = "nsubj:pass" in ch_deprels
        voice = "PASSIVE" if is_passive else "ACTIVE"

        # Span verbal (verbe + auxiliaires directs)
        v_indices = [verb.idx] + [
            c for c in children.get(verb.idx, [])
            if c in by_idx and by_idx[c].deprel in AUX_DEPRELS
        ]
        v_cs, v_ce, v_txt = charspan(v_indices, by_idx, sent_text, sent_offset)
        if len(v_txt.strip()) < 2:
            continue

        spans.append({
            "start":      v_cs,
            "end":        v_ce,
            "text":       v_txt,
            "label":      "svo_verb",
            "voice":      voice,
            "head_lemma": verb.lemma,
            "head_upos":  verb.upos,
        })

        # Arguments
        for child_idx in children.get(verb.idx, []):
            if child_idx not in by_idx:
                continue
            child = by_idx[child_idx]

            if child.deprel in SUBJ_DEPRELS:
                arg_label = "svo_subject"
            elif child.deprel in OBJ_DEPRELS:
                arg_label = "svo_object"
            elif child.deprel in IOBJ_DEPRELS:
                arg_label = "svo_iobj"
            else:
                continue

            # Collecter les conjoints (Pierre [et] Paul → deux spans séparés)
            conjoints = [child_idx] + collect_conjoints(child_idx, children, by_idx)

            for cj_idx in conjoints:
                if cj_idx not in by_idx:
                    continue
                cj = by_idx[cj_idx]

                # Span "tête NE" : tête + flat/compound uniquement (aligne avec NER)
                ne_idx = head_ne_indices(cj_idx, children, by_idx)
                cs, ce, txt = charspan(ne_idx, by_idx, sent_text, sent_offset)
                if len(txt.strip()) < 1:
                    continue

                # Filtrer svo_iobj clitiques : déjà capturés comme pron_obj,
                # et causent chevauchement + signal contradictoire au modèle.
                # Un vrai oblique intéressant est un GP nominal (> 2 chars, pas clitique).
                if arg_label == "svo_iobj":
                    txt_lc = txt.strip().lower().rstrip("'")
                    if txt_lc in FR_PERS_PRONOUNS or len(txt.strip()) <= 2:
                        continue

                # Span complet du NP (conservé comme info contexte)
                full_idx = subtree(cj_idx, children, by_idx, exclude=EXCLUDE_FROM_NP)
                fcs, fce, ftxt = charspan(full_idx, by_idx, sent_text, sent_offset)

                # Features morpho du token tête (utiles pour la coréf)
                cj_word = sentence.words[cj.idx - 1]
                spans.append({
                    "start":           cs,    # frontière tête NE (aligne NER)
                    "end":             ce,
                    "text":            txt,
                    "label":           arg_label,
                    "voice":           voice,
                    "head_lemma":      cj.lemma,
                    "head_upos":       cj.upos,
                    # Morphologie (coréf)
                    "gender":          _feat(cj_word, "Gender"),    # Masc | Fem | null
                    "number":          _feat(cj_word, "Number"),    # Sing | Plur | null
                    "person":          _feat(cj_word, "Person"),    # 1 | 2 | 3 | null
                    "animacy":         _feat(cj_word, "Animacy"),   # Anim | Inan | null
                    "definiteness":    _feat(cj_word, "Definite"),  # Def | Ind | null
                    "full_np_start":   fcs,   # GN complet conservé en métadonnée
                    "full_np_end":     fce,
                    "full_np_text":    ftxt,
                })

    # ── Pronoms personnels (avec features morpho) ─────────────────────────────
    for tok in tokens:
        if tok.upos != "PRON":
            continue
        lemma_lc = tok.lemma.lower().rstrip("'")
        if lemma_lc not in FR_PERS_PRONOUNS and tok.text.lower().rstrip("'") not in FR_PERS_PRONOUNS:
            continue

        # Rôle syntaxique du pronom
        if tok.deprel in SUBJ_DEPRELS:
            pron_label = "pron_subj"
        elif tok.deprel in OBJ_DEPRELS | IOBJ_DEPRELS | {"expl", "expl:subj", "expl:pass"}:
            pron_label = "pron_obj"
        else:
            continue  # pronom non argumental (démonstratif, etc.)

        # Features morphologiques depuis Stanza
        person = _feat(sentence.words[tok.idx - 1], "Person")
        number = _feat(sentence.words[tok.idx - 1], "Number")
        gender = _feat(sentence.words[tok.idx - 1], "Gender")

        sp: dict = {
            "start":       tok.char_start,
            "end":         tok.char_end,
            "text":        tok.text,
            "label":       pron_label,
            "head_lemma":  tok.lemma,
            "head_upos":   tok.upos,
            "pron_person": person,
            "pron_number": number,
            "pron_gender": gender,
        }
        # Voix du verbe gouverneur (pour aider la coréf plus tard)
        if tok.head in by_idx:
            gov = by_idx[tok.head]
            gov_ch_deprels = {by_idx[c].deprel for c in children.get(gov.idx, []) if c in by_idx}
            sp["voice"] = "PASSIVE" if "nsubj:pass" in gov_ch_deprels else "ACTIVE"
        else:
            sp["voice"] = "ACTIVE"

        spans.append(sp)

    return spans


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────────────────────────────────────

def process_file(
    input_path: Path,
    output_path: Path,
    nlp,
    batch_size: int = 64,
    max_examples: int = 0,
    resume_from: int = 0,
    require_obl: bool = False,
    max_sent_tokens: int = 0,
    max_sents: int = 0,
):
    """
    Traitement en streaming : lecture ligne par ligne + écriture au fil de l'eau.
    - batch_size      : nb de phrases envoyées à Stanza en une fois (throughput x3-5)
    - max_examples    : arrêter après N exemples produits (0 = illimité)
    - resume_from     : reprendre à partir de la ligne N du fichier source (0 = début)
    - require_obl     : ne conserver que les phrases contenant au moins un oblique (obl/iobj)
    - max_sent_tokens : ignorer les phrases Stanza dépassant N tokens (0 = illimité)
    - max_sents       : ne traiter que les N premières phrases d'un exemple (0 = toutes)
    """
    n_total_in = 0
    out_examples_count = 0
    label_counts: Counter = Counter()

    t_start   = time.time()
    t_last    = t_start
    cnt_last  = 0  # out_examples_count au dernier log

    # Buffer pour le traitement par batch Stanza
    batch_rows: list[dict] = []
    batch_texts: list[str] = []

    def flush_batch(f_out):
        nonlocal out_examples_count
        if not batch_texts:
            return
        # Stanza : traiter tout le batch en une passe
        docs = [nlp(t) for t in batch_texts]  # fallback mono si bulk non dispo
        # Utiliser bulk_process si disponible (Stanza >=1.5)
        try:
            docs = nlp.bulk_process(batch_texts)
        except AttributeError:
            pass  # version ancienne : déjà traité en mono

        for ex, doc in zip(batch_rows, docs):
            text = ex.get("text", "")
            all_new_spans: list[dict] = []
            char_offset = 0
            sents = doc.sentences if max_sents == 0 else doc.sentences[:max_sents]
            for sent in sents:
                new_spans = extract_sentence(sent, char_offset, text,
                                             require_obl=require_obl,
                                             max_sent_tokens=max_sent_tokens)
                all_new_spans.extend(new_spans)
                char_offset += len(sent.text)
                rest = text[char_offset:]
                char_offset += len(rest) - len(rest.lstrip())

            if not all_new_spans:
                continue

            existing_spans = ex.get("spans", [])
            merged = existing_spans + all_new_spans

            for sp in all_new_spans:
                label_counts[sp["label"]] += 1

            out_row = {
                "id":    ex.get("id", f"svo_{out_examples_count}"),
                "text":  text,
                "spans": merged,
            }
            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f_out.flush()
            out_examples_count += 1

        batch_rows.clear()
        batch_texts.clear()

    print(f"[SVO] Streaming depuis {input_path.name}"
          + (f" (resume depuis ligne {resume_from})" if resume_from else "")
          + (f" (max {max_examples} exemples)" if max_examples else ""))

    # Mode append si resume, sinon overwrite
    write_mode = "a" if resume_from > 0 else "w"

    with open(input_path, encoding="utf-8") as f_in, \
         open(output_path, write_mode, encoding="utf-8") as f_out:

        for line_idx, line in enumerate(f_in):
            if line_idx < resume_from:
                continue

            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = ex.get("text", "")
            if not text.strip():
                continue

            n_total_in += 1
            batch_rows.append(ex)
            batch_texts.append(text)

            if len(batch_texts) >= batch_size:
                flush_batch(f_out)
                if (n_total_in % 2000) == 0:
                    now      = time.time()
                    elapsed  = now - t_start
                    delta_t  = now - t_last
                    delta_n  = out_examples_count - cnt_last
                    rate_now = delta_n / delta_t if delta_t > 0 else 0.0
                    rate_avg = out_examples_count / elapsed if elapsed > 0 else 0.0
                    t_last   = now
                    cnt_last = out_examples_count
                    print(f"  [ligne {line_idx + 1}] {n_total_in} lues → {out_examples_count} sélectionnées"
                          f" | {rate_now:.1f} ex/s (inst)  {rate_avg:.1f} ex/s (moy)"
                          f" | {dict(label_counts.most_common(4))}")

            if max_examples > 0 and out_examples_count >= max_examples:
                print(f"[SVO] max_examples={max_examples} atteint — arrêt.")
                break

        # Flush du dernier batch partiel
        flush_batch(f_out)

    elapsed_total = time.time() - t_start
    rate_total    = out_examples_count / elapsed_total if elapsed_total > 0 else 0.0
    print(f"\n[SVO] ✅ {out_examples_count} exemples écrits → {output_path}")
    print(f"[SVO]    (source: {n_total_in} lignes traitées | durée: {elapsed_total:.1f}s | {rate_total:.1f} ex/s)")
    print("\n[SVO] Répartition des nouveaux labels :")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:<20} : {count:>6}")

    return out_examples_count


# ─────────────────────────────────────────────────────────────────────────────
# Entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build SVO silver dataset with Stanza")
    parser.add_argument("--split",      default="all",
                        choices=["all", "train", "val", "test"],
                        help="Quel(s) split(s) traiter")
    parser.add_argument("--data_dir",   default=".",
                        help="Dossier contenant train_v2.jsonl etc.")
    parser.add_argument("--suffix_in",  default="_v2",
                        help="Suffixe des fichiers en entrée (ex: _v2 → train_v2.jsonl)")
    parser.add_argument("--suffix_out", default="_svo_silver",
                        help="Suffixe des fichiers en sortie")
    parser.add_argument("--gpu",        action="store_true",
                        help="Utiliser le GPU pour Stanza")
    parser.add_argument("--batch",      type=int, default=64,
                        help="Taille du batch Stanza (défaut=64, augmenter sur GPU)")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Arrêter après N exemples produits (0=illimité, utile pour tester)")
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Reprendre à partir de la ligne N du fichier source (après un crash)")
    parser.add_argument("--require-obl", action="store_true",
                        help="Ne garder que les phrases contenant au moins un oblique (obl/iobj)")
    parser.add_argument("--max-sent-tokens", type=int, default=0,
                        help="Ignorer les phrases Stanza > N tokens (0=illimité, ex: 128 pour DeBERTa)")
    parser.add_argument("--max-sents", type=int, default=0,
                        help="Ne traiter que les N premières phrases par exemple (0=toutes)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    print("[SVO] Chargement Stanza fr (tokenize, mwt, pos, lemma, depparse)…")
    nlp = stanza.Pipeline(
        lang               = "fr",
        processors         = "tokenize,mwt,pos,lemma,depparse",
        use_gpu            = args.gpu,
        tokenize_no_ssplit = True,   # respecter les délimitations de phrase du dataset
        verbose            = False,
    )
    print("[SVO] Pipeline prête.\n")

    for split in splits:
        in_file  = data_dir / f"{split}{args.suffix_in}.jsonl"
        out_file = data_dir / f"{split}{args.suffix_out}.jsonl"
        if not in_file.exists():
            print(f"[SVO] ⚠️  {in_file} introuvable, skip.")
            continue
        print(f"{'─'*60}")
        print(f"[SVO] Split : {split}")
        process_file(
            in_file, out_file, nlp,
            batch_size=args.batch,
            max_examples=args.max_examples,
            resume_from=args.resume_from,
            require_obl=args.require_obl,
            max_sent_tokens=args.max_sent_tokens,
            max_sents=args.max_sents,
        )
        print()


if __name__ == "__main__":
    main()

