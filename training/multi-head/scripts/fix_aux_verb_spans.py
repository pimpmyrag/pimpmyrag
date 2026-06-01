#!/usr/bin/env python3
"""
fix_aux_verb_spans.py — Étend les verb_trigger spans qui ne couvrent qu'un auxiliaire
pour inclure le verbe sémantique principal (infinitif ou participe passé qui suit).

Approche : Stanza dependency parser
  - token POS=AUX avec deprel='aux' ou 'aux:pass' → head = verbe principal
  - Span étendu : couvre de min(aux.start, head.start) à max(aux.end, head.end)
  - Incluant les tokens intermédiaires (adverbes éventuels)

Usage:
  python3 scripts/fix_aux_verb_spans.py \
      --input  data/train_v8.20.jsonl \
      --output data/train_v8.21.jsonl
  (idem pour val + test)
"""
import argparse
import json
import sys
import re
from pathlib import Path

# ── Chargement Stanza (lazy) ───────────────────────────────────────────────
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        import stanza
        print("⏳ Chargement Stanza fr...", flush=True)
        _nlp = stanza.Pipeline(
            lang="fr",
            processors="tokenize,pos,lemma,depparse",
            tokenize_no_ssplit=True,  # on envoie phrase par phrase
            verbose=False,
        )
        print("✅ Stanza prêt", flush=True)
    return _nlp


# ── Auxiliaires connus (pour filtrage rapide avant parse Stanza) ──────────
AUX_TOKENS = {
    # avoir
    "a", "ai", "as", "avons", "avez", "ont",
    "avais", "avait", "avions", "aviez", "avaient",
    "aurai", "auras", "aura", "aurons", "aurez", "auront",
    "aurais", "aurait", "aurions", "auriez", "auraient",
    "eus", "eut", "eûmes", "eûtes", "eurent",
    "aie", "aies", "ait", "ayons", "ayez", "aient",
    # être
    "est", "suis", "es", "sommes", "êtes", "sont",
    "étais", "était", "étions", "étiez", "étaient",
    "serai", "seras", "sera", "serons", "serez", "seront",
    "serais", "serait", "serions", "seriez", "seraient",
    "fus", "fut", "fûmes", "fûtes", "furent",
    "sois", "soit", "soyons", "soyez", "soient",
    # modaux courants
    "peut", "peuvent", "pouvait", "pouvaient", "pourrait", "pourraient",
    "doit", "doivent", "devait", "devaient", "devrait", "devraient",
    "va", "vont", "allait", "allaient",
    "vient", "viennent",
}


def is_aux_only_span(span_text: str) -> bool:
    """Retourne True si le texte du span est un auxiliaire seul."""
    return span_text.strip().lower() in AUX_TOKENS


def find_char_offset(doc_text: str, stanza_sent, token) -> tuple[int, int]:
    """Retourne (char_start, char_end) d'un token Stanza dans la phrase originale."""
    # Stanza fournit start_char / end_char directement sur les mots
    return token.start_char, token.end_char


def extend_span_stanza(sentence_text: str, span_start: int, span_end: int) -> tuple[int, int] | None:
    """
    Utilise Stanza pour trouver le verbe principal associé à l'auxiliaire
    dans le span [span_start, span_end).
    Retourne (new_start, new_end) ou None si pas d'extension trouvée.
    """
    nlp = get_nlp()
    try:
        doc = nlp(sentence_text)
    except Exception as e:
        return None

    if not doc.sentences:
        return None

    sent = doc.sentences[0]

    # Trouver le(s) token(s) qui se superposent au span
    aux_tokens = []
    for word in sent.words:
        # Chevauchement avec le span (start_char peut être None sur certains tokens Stanza)
        if word.start_char is None or word.end_char is None:
            continue
        if word.start_char < span_end and word.end_char > span_start:
            aux_tokens.append(word)

    if not aux_tokens:
        return None

    # Pour chaque token auxiliaire, chercher le head VERB
    candidate_ranges = []
    for aux_tok in aux_tokens:
        # Vérifier que c'est bien un AUX ou un VERB avec deprel d'auxiliaire
        is_aux_pos = aux_tok.upos in ("AUX", "VERB")
        is_aux_dep = aux_tok.deprel in ("aux", "aux:pass", "cop")

        if not (is_aux_pos and is_aux_dep):
            continue

        # Trouver le head (le verbe principal)
        head_id = aux_tok.head  # 1-based index dans la phrase
        if head_id == 0:
            # root — cas rare, on skip
            continue

        head_word = sent.words[head_id - 1]

        # Vérifier que head a des offsets valides
        if head_word.start_char is None or head_word.end_char is None:
            continue
        if head_word.upos not in ("VERB", "AUX"):
            continue

        verb_form = ""
        if head_word.feats:
            for feat in head_word.feats.split("|"):
                if feat.startswith("VerbForm="):
                    verb_form = feat.split("=")[1]

        # On accepte Part (participe passé) et Inf (infinitif)
        # On accepte aussi sans VerbForm (ex: "déclaré" peut avoir VerbForm=Part)
        if verb_form not in ("Part", "Inf", ""):
            continue

        # Calculer le range étendu : de l'aux jusqu'au head (ou inversement)
        new_start = min(aux_tok.start_char, head_word.start_char)
        new_end   = max(aux_tok.end_char,   head_word.end_char)

        # Sanity check : l'extension doit être raisonnable (< 60 chars)
        if new_end - new_start > 60:
            continue

        candidate_ranges.append((new_start, new_end, head_word.text, verb_form))

    if not candidate_ranges:
        return None

    # Prendre la range qui étend le plus (= inclut le verbe le plus important)
    best = max(candidate_ranges, key=lambda x: x[1] - x[0])
    return best[0], best[1]


def process_file(input_path: str, output_path: str, split: str = "train") -> dict:
    stats = {"total": 0, "verbs": 0, "aux_only": 0, "extended": 0, "skipped": 0}

    sentences = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(json.loads(line))

    stats["total"] = len(sentences)
    print(f"\n📂 {split}: {len(sentences)} phrases", flush=True)

    # Identifier les phrases avec auxiliaires seuls
    to_fix = []
    for i, sent in enumerate(sentences):
        for j, sp in enumerate(sent.get("spans", [])):
            if sp.get("label") == "verb_trigger":
                stats["verbs"] += 1
                if is_aux_only_span(sp.get("text", "")):
                    stats["aux_only"] += 1
                    to_fix.append((i, j))

    print(f"  verb_trigger: {stats['verbs']}, auxiliaires seuls: {stats['aux_only']}", flush=True)

    # Traiter par batch de 100 pour afficher la progression
    BATCH = 100
    for batch_start in range(0, len(to_fix), BATCH):
        batch = to_fix[batch_start:batch_start + BATCH]
        for (sent_idx, span_idx) in batch:
            sent = sentences[sent_idx]
            sp = sent["spans"][span_idx]
            text = sent["text"]
            s_start, s_end = sp["start"], sp["end"]

            result = extend_span_stanza(text, s_start, s_end)
            if result is None:
                stats["skipped"] += 1
                continue

            new_start, new_end = result
            if new_start == s_start and new_end == s_end:
                stats["skipped"] += 1
                continue

            # Mettre à jour le span
            sp["start"] = new_start
            sp["end"]   = new_end
            sp["text"]  = text[new_start:new_end]
            stats["extended"] += 1

        done = min(batch_start + BATCH, len(to_fix))
        print(f"  ⏳ {done}/{len(to_fix)} traités — {stats['extended']} étendus", end="\r", flush=True)

    print(f"\n  ✅ {stats['extended']} spans étendus, {stats['skipped']} inchangés", flush=True)

    # Écrire le fichier de sortie
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sent in sentences:
            f.write(json.dumps(sent, ensure_ascii=False) + "\n")
    print(f"  💾 Écrit → {output_path}", flush=True)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Étend les spans auxiliaires pour inclure le verbe principal")
    parser.add_argument("--input",  required=True, help="Fichier JSONL d'entrée")
    parser.add_argument("--output", required=True, help="Fichier JSONL de sortie")
    parser.add_argument("--split",  default="train", help="Nom du split (pour logs)")
    parser.add_argument("--dry-run", action="store_true", help="Affiche les exemples sans écrire")
    args = parser.parse_args()

    if args.dry_run:
        # Mode dry-run : afficher 20 exemples d'extensions
        print("🔍 Dry-run — 20 premiers exemples d'extension:")
        nlp = get_nlp()
        count = 0
        with open(args.input) as f:
            for line in f:
                sent = json.loads(line.strip())
                for sp in sent.get("spans", []):
                    if sp.get("label") == "verb_trigger" and is_aux_only_span(sp.get("text", "")):
                        result = extend_span_stanza(sent["text"], sp["start"], sp["end"])
                        old_text = sp["text"]
                        if result:
                            new_text = sent["text"][result[0]:result[1]]
                            if new_text != old_text:
                                print(f'  "{old_text}" → "{new_text}"  |  ctx: ...{sent["text"][max(0,sp["start"]-10):sp["end"]+30]}...')
                                count += 1
                        if count >= 20:
                            break
                if count >= 20:
                    break
        return

    stats = process_file(args.input, args.output, args.split)
    print(f"\n📊 Stats finales: {stats}")


if __name__ == "__main__":
    main()

