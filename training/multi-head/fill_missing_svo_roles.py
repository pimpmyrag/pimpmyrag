#!/usr/bin/env python3
"""
fill_missing_svo_roles.py — Remplit les rôles SVO manquants sur les spans existants.

MODE FILL-ONLY :
  - NE CRÉE PAS de nouveaux spans
  - Met à jour uniquement les spans NER root SANS svo_role
  - Utilise Stanza depparse + matching de tokens pour assigner SUBJECT/OBJECT/OBLIQUE/APPOS

Usage :
    python3 fill_missing_svo_roles.py \\
        --input  data/train_v8.18.jsonl \\
        --output data/train_v8.18_roles_filled.jsonl \\
        --lang fr

    python3 fill_missing_svo_roles.py --input data/val_v8.18.jsonl \\
        --output data/val_v8.18_roles_filled.jsonl
"""
from __future__ import annotations
import json
import argparse
import time
from pathlib import Path
from collections import Counter

import stanza

# Labels SVO (non-NER, on ne les touche pas)
SVO_LABELS = {"verb_trigger", "pron_subj", "pron_obj"}

# Labels NER dont le sous-type oblique est inféré automatiquement par build_multitask_dataset.py
# → ne jamais leur assigner OBLIQUE_AGENT (ça écraserait l'inférence OBLIQUE_TIME / OBLIQUE_LOC)
NER_TIME_LABELS = {"hint_time_date", "hint_time_clock", "hint_time_duration"}
NER_LOC_LABELS  = {"hint_loc_generic", "hint_gpe", "hint_fac_name", "hint_infra"}
NER_AUTO_OBLIQUE_LABELS = NER_TIME_LABELS | NER_LOC_LABELS

# Dépendances → rôle SVO
DEPREL_SUBJECT = {"nsubj", "nsubj:pass", "csubj", "csubj:pass", "nsubj:outer"}
DEPREL_OBJECT  = {"obj", "iobj", "ccomp", "xcomp", "obj:agent"}
# nmod/nmod:poss exclus : ce sont souvent des génitifs ("de Baudelaire", "du gouvernement")
# qui ne sont PAS des participants SVO directs.
# advmod exclu : adverbes, rarement des entités NER.
DEPREL_OBLIQUE = {
    "obl", "obl:agent", "obl:arg", "obl:mod", "obl:tmod", "obl:lmod",
}
DEPREL_APPOS   = {"appos"}

# Marqueurs causaux → OBLIQUE_CAUSE
CAUSAL_PREPS = {"suite", "grâce", "grace", "raison", "cause", "conséquence", "résultat"}
# Marqueurs adversaires
ADVERSARY_PREPS = {"contre", "face", "encontre"}
# Marqueurs bénéficiaires
BENEFICIARY_PREPS = {"pour", "faveur", "profit", "bénéfice", "benefice"}

def get_case_markers(word, all_words):
    """Retourne les lemmes des marqueurs case/mark attachés à ce mot."""
    return {
        w.lemma.lower()
        for w in all_words
        if w.head == word.id and w.deprel in ("case", "mark")
        and w.lemma is not None
    }

def deprel_to_role(word, all_words, is_passive=False):
    """Mappe un deprel Stanza + contexte → rôle SVO."""
    dep = (word.deprel or "").lower()

    if dep in DEPREL_SUBJECT:
        return "SUBJECT"

    if dep in DEPREL_OBJECT:
        return "OBJECT"

    if dep in DEPREL_APPOS:
        return "APPOS"

    if dep in DEPREL_OBLIQUE:
        markers = get_case_markers(word, all_words)

        # Affiner selon les marqueurs
        if markers & ADVERSARY_PREPS:
            return "OBLIQUE_ADVERSARY"
        if markers & BENEFICIARY_PREPS:
            return "OBLIQUE_BENEFICIARY"
        if markers & CAUSAL_PREPS:
            return "OBLIQUE_CAUSE"
        if {"avec", "côtés"} & markers:
            return "OBLIQUE_COMITATIVE"
        if {"selon", "d'après", "daprès", "suivant"} & markers:
            return "OBLIQUE_SOURCE"
        if {"contre", "face"} & markers:
            return "OBLIQUE_ADVERSARY"
        if {"par"} & markers and is_passive:
            return "OBLIQUE_AGENT"

        return "OBLIQUE"

    return None


def deprel_to_role_extended(word, sent, ner_label: str | None, is_passive: bool,
                            tok_to_span_fn) -> str | None:
    """
    Règles supplémentaires pour les deprels non couverts par deprel_to_role :
      - advmod  : hint_time_* modifiant un VERB/AUX → OBLIQUE
      - flat:name : remonte à la tête réelle et réapplique la règle de base
      - nmod    : si la tête du nmod est dans un span NER avec un rôle → OBLIQUE
                  (ex: "le gouvernement des États-Unis" → États-Unis OBLIQUE)
    """
    dep = (word.deprel or "").lower()
    all_words = sent.words

    # advmod : adverbe temporel modifiant directement un verbe
    if dep == "advmod" and ner_label in NER_TIME_LABELS:
        if word.head and word.head > 0:
            head_idx = word.head - 1
            if 0 <= head_idx < len(all_words):
                if all_words[head_idx].upos in ("VERB", "AUX"):
                    return "OBLIQUE"

    # flat:name : fait partie d'un nom propre multi-token ; remonte à la tête
    if dep == "flat:name":
        if word.head and word.head > 0:
            head_idx = word.head - 1
            if 0 <= head_idx < len(all_words):
                head_word = all_words[head_idx]
                role = deprel_to_role(head_word, all_words, is_passive)
                if role:
                    return role

    # nmod : modificateur nominal — si la tête appartient à un span NER avec rôle
    # → on assigne OBLIQUE (entité associée à un participant)
    if dep in ("nmod", "nmod:poss", "nummod") and word.head and word.head > 0:
        head_idx = word.head - 1
        if 0 <= head_idx < len(all_words):
            head_word = all_words[head_idx]
            head_span = tok_to_span_fn(head_word)
            if head_span and head_span.get("svo_role"):
                # N'applique pas sur hint_norp (adjectifs) ni les labels "purement
                # quantitatifs" qui n'ont pas de valeur participative directe
                if ner_label not in {"hint_norp"}:
                    return "OBLIQUE"

    return None

def find_token_for_span(span_start, span_end, doc):
    """
    Trouve le token Stanza dont l'offset chevauche le span NER.
    Préfère le token dont le chef (head) est un verbe.
    Retourne (word, sentence) ou (None, None).
    """
    candidates = []
    for sent in doc.sentences:
        for word in sent.words:
            ws = word.start_char
            we = word.end_char
            if ws is None or we is None:
                continue
            # Chevauchement
            overlap_start = max(ws, span_start)
            overlap_end   = min(we, span_end)
            if overlap_end > overlap_start:
                span_len = max(1, span_end - span_start)
                tok_len  = max(1, we - ws)
                ratio = (overlap_end - overlap_start) / min(span_len, tok_len)
                if ratio >= 0.5:
                    candidates.append((word, sent, ratio))

    if not candidates:
        return None, None

    # Priorité : mot dont le gouverneur est un verbe (VERB/AUX)
    def priority(item):
        word, sent, ratio = item
        head_is_verb = False
        if word.head and word.head > 0:
            head_pos = word.head - 1
            if 0 <= head_pos < len(sent.words):
                head_word = sent.words[head_pos]
                head_is_verb = (head_word.upos in ("VERB", "AUX"))
        return (head_is_verb, ratio)

    candidates.sort(key=priority, reverse=True)
    return candidates[0][0], candidates[0][1]

def is_passive_sentence(sent):
    """Détecte si la phrase est passive (auxpass ou nsubj:pass)."""
    return any(
        w.deprel in ("aux:pass", "nsubj:pass", "auxpass")
        for w in sent.words
    )

def find_gov_verb_start(word, sent):
    """Retourne le start_char du verbe gouverneur du token."""
    if word.head and word.head > 0:
        head_idx = word.head - 1
        if 0 <= head_idx < len(sent.words):
            head_word = sent.words[head_idx]
            if head_word.upos in ("VERB", "AUX"):
                return head_word.start_char
            # Remonter d'un niveau si nécessaire
            if head_word.head and head_word.head > 0:
                grandpa_idx = head_word.head - 1
                if 0 <= grandpa_idx < len(sent.words):
                    gp = sent.words[grandpa_idx]
                    if gp.upos in ("VERB", "AUX"):
                        return gp.start_char
    return None


def propagate_conj_roles(ner_spans: list, doc) -> int:
    """
    Propagation de coordination : si un span a deprel=conj et que son
    token coordinateur appartient à un span NER qui a déjà un svo_role,
    le span conjoint hérite du même rôle (et gov_verb_start).

    Itère jusqu'à stabilisation (gère les chaînes A→B→C).
    Retourne le nombre de spans mis à jour.
    """
    # Index : (start_char, end_char) → span NER
    span_by_chars: dict[tuple, dict] = {
        (s["start"], s["end"]): s for s in ner_spans
    }

    # Index token → span NER qui le contient (chevauchement ≥ 50 %)
    def token_to_span(word):
        ws, we = word.start_char, word.end_char
        if ws is None or we is None:
            return None
        best, best_ratio = None, 0.0
        for (ss, se), sp in span_by_chars.items():
            ov = min(we, se) - max(ws, ss)
            if ov > 0:
                ratio = ov / min(max(1, we - ws), max(1, se - ss))
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = sp
        return best if best_ratio >= 0.5 else None

    # Pré-calcule le mapping token → span pour tous les tokens du doc
    tok_span: dict[int, dict] = {}   # word.id (global) → span
    global_id = 0
    sent_list = list(doc.sentences)
    for sent in sent_list:
        for w in sent.words:
            sp = token_to_span(w)
            tok_span[global_id] = sp
            global_id += 1

    updated = 0
    # Plusieurs passes pour les chaînes de coord (A→B→C)
    for _pass in range(4):
        changed = 0
        global_id = 0
        for sent in sent_list:
            for w in sent.words:
                sp = tok_span.get(global_id)
                global_id += 1
                if sp is None or sp.get("svo_role"):
                    continue  # déjà un rôle ou pas mappé
                if (w.deprel or "").lower() != "conj":
                    continue
                # Trouver le span de la tête de conjonction
                if not w.head or w.head <= 0:
                    continue
                head_global = global_id - 1 - (w.id - 1) + (w.head - 1)
                # recalcul : on parcourt linéairement pour fiabilité
                head_sp = None
                gid2 = 0
                for s2 in sent_list:
                    for w2 in s2.words:
                        if w2.start_char == w.start_char:
                            # trouve le head par offset dans la même phrase
                            if w.head and w.head > 0 and (w.head - 1) < len(sent.words):
                                hw = sent.words[w.head - 1]
                                head_sp = token_to_span(hw)
                            break
                        gid2 += 1
                    else:
                        continue
                    break

                if head_sp and head_sp.get("svo_role") and head_sp is not sp:
                    sp["svo_role"] = head_sp["svo_role"]
                    # Propager gov_verb_start si absent
                    if not sp.get("gov_verb_start") and head_sp.get("gov_verb_start"):
                        sp["gov_verb_start"] = head_sp["gov_verb_start"]
                    changed += 1
                    updated += 1
        if changed == 0:
            break

    return updated

def process_file(input_path: Path, output_path: Path, nlp, batch_size=100):
    data = []
    with open(input_path) as f:
        for line in f:
            data.append(json.loads(line))

    total = len(data)
    n_spans_updated = 0
    n_gov_added = 0
    role_counts = Counter()
    label_counts = Counter()

    print(f"\n📂 {input_path.name} — {total:,} phrases")
    print(f"   Batch Stanza : {batch_size} phrases/batch")

    ckpt_path = output_path.with_suffix(".ckpt.jsonl")
    # Reprendre depuis checkpoint si existant
    done_ids = set()
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done_ids.add(row.get("id"))
                except Exception:
                    pass
        print(f"   ♻️  Checkpoint : {len(done_ids)} phrases déjà traitées")

    out_f = open(ckpt_path, "a")

    t0 = time.time()
    for batch_start in range(0, total, batch_size):
        batch = data[batch_start:batch_start + batch_size]

        # Filtrer les phrases tout-SVO (pas de spans sans rôle) pour accélérer
        to_parse = []
        for row in batch:
            if row.get("id") in done_ids:
                to_parse.append(None)
                continue
            spans = row.get("spans", [])
            needs_fill = any(
                s.get("label") not in SVO_LABELS
                and not s.get("svo_role")
                for s in spans
            )
            to_parse.append(row if needs_fill else None)

        texts_to_parse = [row["text"] for row in to_parse if row is not None]
        if not texts_to_parse:
            for row in batch:
                if row.get("id") not in done_ids:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            continue

        docs = nlp([stanza.Document([], text=t) for t in texts_to_parse])
        doc_iter = iter(docs)

        for i, row in enumerate(batch):
            if row.get("id") in done_ids:
                continue
            if to_parse[i] is None:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            doc = next(doc_iter)
            spans = row.get("spans", [])
            text = row.get("text", "")

            ner_spans = [s for s in spans if s.get("label") not in SVO_LABELS]

            # Pré-calcul token → span NER (pour deprel_to_role_extended / nmod)
            def tok_to_span(w):
                ws, we = w.start_char, w.end_char
                if ws is None or we is None:
                    return None
                best, best_r = None, 0.0
                for sp in ner_spans:
                    ov = min(we, sp["end"]) - max(ws, sp["start"])
                    if ov > 0:
                        r = ov / min(max(1, we - ws), max(1, sp["end"] - sp["start"]))
                        if r > best_r:
                            best_r, best = r, sp
                return best if best_r >= 0.5 else None

            for span in ner_spans:
                if span.get("svo_role"):
                    continue

                s_start, s_end = span["start"], span["end"]

                # Root check
                is_root = all(
                    not (o["start"] <= s_start and o["end"] >= s_end
                         and (o["start"] < s_start or o["end"] > s_end))
                    for o in ner_spans if o is not span
                )
                if not is_root:
                    continue

                # Chercher le token Stanza correspondant
                word, sent = find_token_for_span(s_start, s_end, doc)
                if word is None or sent is None:
                    continue

                passive = is_passive_sentence(sent)

                # Passe 1 : règles deprel de base
                role = deprel_to_role(word, sent.words, passive)

                # Passe 1b : règles étendues (advmod, flat:name, nmod)
                if not role:
                    role = deprel_to_role_extended(
                        word, sent, span.get("label"), passive, tok_to_span
                    )

                # OBLIQUE_TIME / OBLIQUE_LOC sont inférés automatiquement par
                # build_multitask_dataset.py depuis le label NER — ne pas écraser
                # avec OBLIQUE_AGENT (ex: "Par la suite" « par » déclenchait la règle)
                if role == "OBLIQUE_AGENT" and span.get("label") in NER_AUTO_OBLIQUE_LABELS:
                    role = "OBLIQUE"

                if role:
                    span["svo_role"] = role
                    n_spans_updated += 1
                    role_counts[role] += 1
                    label_counts[span.get("label", "?")] += 1

                    # Ajouter gov_verb_start si absent
                    if not span.get("gov_verb_start") and role not in ("APPOS",):
                        gov = find_gov_verb_start(word, sent)
                        if gov is not None:
                            span["gov_verb_start"] = gov
                            n_gov_added += 1

            # Passe 2 : propagation de coordination (conj)
            n_conj = propagate_conj_roles(ner_spans, doc)
            if n_conj:
                n_spans_updated += n_conj
                role_counts["__conj_propagated__"] = \
                    role_counts.get("__conj_propagated__", 0) + n_conj

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

        elapsed = time.time() - t0
        done = batch_start + len(batch)
        pct = done / total * 100
        speed = done / elapsed * 60
        print(f"   {done:>6}/{total}  ({pct:.0f}%)  {speed:.0f} phrases/min  "
              f"  rôles ajoutés={n_spans_updated}", end="\r")

    out_f.close()
    print(f"\n   ✅ Terminé en {time.time()-t0:.0f}s")
    print(f"   Spans mis à jour : {n_spans_updated:,}")
    print(f"   gov_verb_start ajoutés : {n_gov_added:,}")
    print(f"\n   Rôles :")
    for role, cnt in sorted(role_counts.items(), key=lambda x: -x[1]):
        print(f"     {role:<25} {cnt:>6}")
    print(f"\n   Top labels mis à jour :")
    for label, cnt in sorted(label_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"     {label:<28} {cnt:>5}")

    # Finaliser le fichier output depuis le checkpoint
    print(f"\n   📝 Écriture de {output_path.name}...")
    written = 0
    seen_ids = set()
    with open(output_path, "w") as out:
        # D'abord écrire depuis checkpoint (ordre peut varier)
        # On reconstruit depuis data original en ordre, en utilisant le ckpt comme source
        ckpt_data = {}
        with open(ckpt_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    rid = row.get("id", f"__idx_{written}")
                    ckpt_data[rid] = row
                except Exception:
                    pass

        for row in data:
            rid = row.get("id", "")
            final = ckpt_data.get(rid, row)
            out.write(json.dumps(final, ensure_ascii=False) + "\n")
            written += 1

    print(f"   ✅ {written:,} phrases écrites → {output_path}")
    ckpt_path.unlink(missing_ok=True)
    return n_spans_updated


def main():
    p = argparse.ArgumentParser(description="Fill missing SVO roles via Stanza depparse")
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--lang",   default="fr")
    p.add_argument("--batch",  type=int, default=64)
    args = p.parse_args()

    print(f"🔧 Chargement Stanza ({args.lang})...")
    nlp = stanza.Pipeline(
        args.lang,
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=False,
        use_gpu=False,
        verbose=False,
    )
    print("✅ Stanza prêt")

    total = process_file(
        Path(args.input),
        Path(args.output),
        nlp,
        batch_size=args.batch,
    )
    print(f"\n🎯 Total : {total:,} spans mis à jour")


if __name__ == "__main__":
    main()








