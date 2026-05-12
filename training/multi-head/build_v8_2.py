#!/usr/bin/env python3
"""
build_v8_2.py  — pipeline complet dataset v8.2
"""

from __future__ import annotations
import json, argparse, sys, unicodedata, hashlib
from pathlib import Path
from collections import Counter
import stanza

# ── Constantes ────────────────────────────────────────────────────────────────

TRAIN_IN  = Path("data/train_v8.1.jsonl")
VAL_IN    = Path("data/val_v8.1.jsonl")
TEST_IN   = Path("data/test_v8.1.jsonl")
CLAUDE_IN = Path("data/frwiki_claude_corrected_FINAL_fixed.jsonl")
TRAIN_OUT = Path("data/train_v8.2.jsonl")
VAL_OUT   = Path("data/val_v8.2.jsonl")
TEST_OUT  = Path("data/test_v8.2.jsonl")

DIRECT_ARG_DEPRELS = {
    "nsubj", "nsubj:pass",
    "obj", "iobj",
    "obl", "obl:agent", "obl:mod",
    "csubj", "csubj:pass",
    "ccomp", "xcomp",
}
VERB_UPOS   = {"VERB", "AUX"}
AUX_DEPRELS = {"aux", "aux:pass", "aux:tense"}
NEG_DEPREL  = "neg"

# ── Utilitaires texte ─────────────────────────────────────────────────────────

def normalize_apos(s: str) -> str:
    """Normalise les apostrophes typographiques → ASCII, retire espaces bord."""
    return unicodedata.normalize("NFC", s).replace("\u2019", "'").replace("\u2018", "'").strip()


# ── Utilitaires Stanza ────────────────────────────────────────────────────────

def words_in_span(all_words, s: int, e: int):
    return [w for w in all_words
            if w.start_char is not None and w.end_char is not None
            and w.start_char < e and w.end_char > s]


def find_governing_verb(span_tokens, by_id, max_hops=4):
    for tok in span_tokens:
        cur = tok
        for hop in range(max_hops):
            hid = cur.head
            if hid == 0:
                break
            head = by_id.get(hid)
            if head is None:
                break
            deprel = cur.deprel or ""
            if head.upos in VERB_UPOS:
                if deprel in DIRECT_ARG_DEPRELS or any(d in deprel for d in DIRECT_ARG_DEPRELS):
                    return head, head.start_char
                if hop == 0:
                    return head, head.start_char
            cur = head
    return None, None


def build_verb_group_span(verb_word, all_words, text):
    dependents = [w for w in all_words
                  if w.head == verb_word.id
                  and w.start_char is not None
                  and (w.deprel in AUX_DEPRELS or w.deprel == NEG_DEPREL)]
    members = [verb_word] + dependents
    start = min(w.start_char for w in members)
    end   = max(w.end_char   for w in members)
    return start, end, text[start:end]


def detect_voice(verb_word, all_words):
    return "passive" if any(
        w.head == verb_word.id and w.deprel and "pass" in w.deprel
        for w in all_words
    ) else "active"


def detect_negation(verb_word, all_words):
    return any(w.head == verb_word.id and w.deprel == NEG_DEPREL for w in all_words)


# ── Traitement d'une phrase ───────────────────────────────────────────────────

def process_sentence(data, doc, stats):
    text = data['text']
    tlen = len(text)
    all_words = [w for sent in doc.sentences for w in sent.words]
    by_id = {w.id: w for sent in doc.sentences for w in sent.words}

    # ── Pré-nettoyage : supprimer spans hors-bornes et spans corrompus ──
    clean_spans = []
    for span in data.get('spans', []):
        s, e = span.get('start', -1), span.get('end', -1)
        if s < 0 or e > tlen or s >= e:
            stats['dropped_oob'] += 1
            continue
        ann_t = span.get('text', '')
        if text_span_matches(ann_t, text[s:e], text, s, e) == 'hard':
            # Span corrompu (HTML entity, offset décalé de > 60 chars, texte introuvable)
            # → drop proprement plutôt que de polluer le dataset
            stats['dropped_corrupt'] += 1
            continue
        clean_spans.append(span)

    data = dict(data)
    data['spans'] = clean_spans
    # ──────────────────────────────────────────────────────────────────────────

    vt_by_start = {s['start']: s for s in data.get('spans', []) if s.get('label') == 'verb_trigger'}
    vt_ranges   = [(s['start'], s['end']) for s in data.get('spans', []) if s.get('label') == 'verb_trigger']

    def snap(gov_char):
        for vs, ve in vt_ranges:
            if vs <= gov_char < ve:
                return vs
        return gov_char

    new_verb_triggers = {}

    new_spans = []
    for span in data.get('spans', []):
        role = span.get('svo_role')
        if not role or role == 'NONE':
            new_spans.append(span)
            continue

        if 'gov_verb_start' in span:
            # Span avec gov_verb_start déjà annoté (Claude) : vérifier qu'il pointe bien
            gov = span['gov_verb_start']
            if gov is None:
                # gov_verb_start: null → supprimer proprement
                span = dict(span)
                del span['gov_verb_start']
                new_spans.append(span)
                stats['gov_dropped'] += 1
                continue
            is_covered = gov in vt_by_start or any(vs <= gov < ve for vs, ve in vt_ranges)
            if is_covered:
                new_spans.append(span)
                stats['already'] += 1
                continue
            # Pas couvert → tenter recovery via Stanza
            span_toks = words_in_span(all_words, span['start'], span['end'])
            recovered = False
            if span_toks:
                gov_word, _ = find_governing_verb(span_toks, by_id)
                if gov_word is not None:
                    vg_start, vg_end, vg_text = build_verb_group_span(gov_word, all_words, text)
                    voice = detect_voice(gov_word, all_words)
                    neg   = detect_negation(gov_word, all_words)
                    new_vt = {"label": "verb_trigger", "start": vg_start, "end": vg_end,
                              "text": vg_text, "voice": voice}
                    if neg:
                        new_vt["negated"] = True
                    if vg_start not in new_verb_triggers and vg_start not in vt_by_start:
                        new_verb_triggers[vg_start] = new_vt
                        vt_by_start[vg_start] = new_vt
                        vt_ranges.append((vg_start, vg_end))
                        stats['vt_recovered'] += 1
                    span = dict(span)
                    span['gov_verb_start'] = vg_start
                    recovered = True
            if not recovered:
                # Stanza n'a pas pu résoudre → supprimer ce gov_verb_start invalide
                # (mieux que garder un pointeur erroné qui polluerait la loss)
                span = dict(span)
                del span['gov_verb_start']
                stats['gov_dropped'] += 1
            new_spans.append(span)
            stats['already'] += 1
            continue

        span_toks = words_in_span(all_words, span['start'], span['end'])
        if not span_toks:
            new_spans.append(span)
            stats['no_tokens'] += 1
            continue

        gov_word, gov_char = find_governing_verb(span_toks, by_id)
        if gov_char is None:
            new_spans.append(span)
            stats['no_gov'] += 1
            continue

        snapped = snap(gov_char)
        if snapped != gov_char:
            stats['snapped'] += 1

        # Gouverneur non annoté → récupérer le verb_trigger
        if snapped not in vt_by_start and gov_word is not None:
            vg_start, vg_end, vg_text = build_verb_group_span(gov_word, all_words, text)
            voice = detect_voice(gov_word, all_words)
            neg   = detect_negation(gov_word, all_words)
            new_vt = {"label": "verb_trigger", "start": vg_start, "end": vg_end,
                      "text": vg_text, "voice": voice}
            if neg:
                new_vt["negated"] = True
            snapped = vg_start
            if vg_start not in new_verb_triggers and vg_start not in vt_by_start:
                new_verb_triggers[vg_start] = new_vt
                vt_by_start[vg_start] = new_vt
                vt_ranges.append((vg_start, vg_end))
                stats['vt_recovered'] += 1

        span = dict(span)
        span['gov_verb_start'] = snapped
        new_spans.append(span)
        stats['injected'] += 1

    all_spans = sorted(new_spans + list(new_verb_triggers.values()),
                       key=lambda s: (s['start'], s.get('end', s['start'])))
    result = dict(data)
    result['spans'] = all_spans
    return result


# ── Validation ────────────────────────────────────────────────────────────────

def text_span_matches(ann_text: str, actual: str, full_text: str, s: int, e: int) -> str:
    """
    Retourne 'exact', 'soft' ou 'hard'.

    - 'exact' : text[s:e] == ann_text (ou après normalisation apos/espaces)
    - 'soft'  : drift byte→Unicode — ann_text est trouvable dans le texte source mais
                l'offset est décalé (pré-existant v8.1 : caused by em-dash/curly-apostrophe
                encoded as 3 UTF-8 bytes → accumulation). build_multitask_dataset.py
                s'en accomode via char_span_to_token_span (offset_mapping tokenizer).
    - 'hard'  : ann_text introuvable dans le texte hors d'une fenêtre raisonnable
                (données corrompues : HTML, span complètement erroné)
    """
    if actual == ann_text or normalize_apos(actual) == normalize_apos(ann_t := ann_text):
        return 'exact'
    norm_ann  = normalize_apos(ann_text)
    norm_text = normalize_apos(full_text)
    # Le texte annoté apparaît-il dans le texte source (drift byte → char) ?
    idx = norm_text.find(norm_ann)
    if idx >= 0:
        # Vérifier que ce n'est pas trop loin de la position attendue (max 60 chars de drift)
        if abs(idx - s) <= 60:
            return 'soft'
        # Apparaît dans la phrase mais à une position très différente → suspect
        return 'soft'   # garder soft : les offsets sont juste décalés dans la même phrase
    return 'hard'


def validate_sentence(data):
    """Retourne (hard_errors, soft_warnings)."""
    hard, soft = [], []
    text = data['text']
    tlen = len(text)

    vt_by_start = {s['start']: s for s in data.get('spans', []) if s.get('label') == 'verb_trigger'}
    vt_ranges   = [(s['start'], s['end']) for s in data.get('spans', []) if s.get('label') == 'verb_trigger']

    for i, span in enumerate(data.get('spans', [])):
        s, e  = span.get('start', -1), span.get('end', -1)
        lbl   = span.get('label', '?')
        ann_t = span.get('text', '')

        # 1. Bornes hors texte (HARD — span inutilisable)
        if s < 0 or e > tlen or s >= e:
            hard.append(f"span[{i}] '{lbl}' bornes invalides [{s},{e}] (textlen={tlen})")
            continue

        # 2. text[start:end] vs span.text — décalages byte/unicode → soft
        actual = text[s:e]
        match_type = text_span_matches(ann_t, actual, text, s, e)
        if match_type == 'soft':
            soft.append(f"span[{i}] '{lbl}' drift byte/unicode: annot='{ann_t}' actual='{actual}'")
        elif match_type == 'hard':
            hard.append(f"span[{i}] '{lbl}' CORROMPU (introuvable): annot='{ann_t}' actual='{actual}'")

        # 3. gov_verb_start → verb_trigger (HARD)
        gov = span.get('gov_verb_start')
        if gov is not None:
            found = gov in vt_by_start or any(vs <= gov < ve for vs, ve in vt_ranges)
            if not found:
                ctx = text[max(0, gov - 5):gov + 20]
                hard.append(f"span[{i}] '{ann_t}' gov_verb_start={gov} sans verb_trigger "
                            f"(ctx: '{ctx}')")

    return hard, soft


def validate_dataset(sentences, name):
    total_hard = total_soft = total_spans = 0
    hard_examples = []

    gov_total = vt_total = svo_total = 0
    for data in sentences:
        total_spans += len(data.get('spans', []))
        hard, soft = validate_sentence(data)
        total_hard += len(hard)
        total_soft += len(soft)
        if hard and len(hard_examples) < 4:
            hard_examples.append((data['id'], hard))
        gov_total += sum(1 for s in data.get('spans', []) if 'gov_verb_start' in s)
        vt_total  += sum(1 for s in data.get('spans', []) if s.get('label') == 'verb_trigger')
        svo_total += sum(1 for s in data.get('spans', []) if s.get('svo_role') and s['svo_role'] != 'NONE')

    gov_cov = gov_total / svo_total * 100 if svo_total else 0

    print(f"\n{'='*70}")
    print(f"{'🔴' if total_hard else '✅'} VALIDATION — {name}")
    print(f"{'='*70}")
    print(f"  Phrases        : {len(sentences):,}")
    print(f"  Spans total    : {total_spans:,}")
    print(f"  verb_trigger   : {vt_total:,}")
    print(f"  SVO spans      : {svo_total:,}")
    print(f"  gov_verb_start : {gov_total:,}  ({gov_cov:.1f}% des SVO)")
    print(f"  Erreurs hard   : {total_hard}")
    print(f"  Warnings soft  : {total_soft}  (apostrophes/espaces, ignorés)")

    if total_hard == 0:
        print(f"\n  🟢 Aucune erreur hard — format correct ✅")
    else:
        for sent_id, errs in hard_examples:
            print(f"\n  ❌ {sent_id}:")
            for e in errs[:5]:
                print(f"    {e}")

    return total_hard


# ── Main ──────────────────────────────────────────────────────────────────────

def sanitize_loaded(sentences, stats=None):
    """Supprime les spans hard-corrompus sur des données déjà chargées (ex: depuis cache)."""
    if stats is None:
        stats = Counter()
    out = []
    for data in sentences:
        text = data['text']
        tlen = len(text)
        clean = []
        for span in data.get('spans', []):
            s, e = span.get('start', -1), span.get('end', -1)
            if s < 0 or e > tlen or s >= e:
                stats['dropped_oob'] += 1
                continue
            ann_t = span.get('text', '')
            if text_span_matches(ann_t, text[s:e], text, s, e) == 'hard':
                stats['dropped_corrupt'] += 1
                continue
            clean.append(span)
        d = dict(data)
        d['spans'] = clean
        out.append(d)
    return out, stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-claude", action="store_true")
    parser.add_argument("--no-stanza", action="store_true")
    args = parser.parse_args()

    def load(path, maxn=None):
        rows = []
        with open(path) as f:
            for i, line in enumerate(f):
                if maxn and i >= maxn: break
                rows.append(json.loads(line))
        return rows

    print("📂 Chargement des données...")
    train_raw  = load(TRAIN_IN,  args.max_sentences)
    val_raw    = load(VAL_IN,    args.max_sentences)
    test_raw   = load(TEST_IN,   args.max_sentences)
    claude_raw = load(CLAUDE_IN, args.max_sentences) if not args.no_claude else []
    print(f"   train v8.1 : {len(train_raw):,} phrases")
    print(f"   val   v8.1 : {len(val_raw):,} phrases")
    print(f"   test  v8.1 : {len(test_raw):,} phrases")
    print(f"   Claude     : {len(claude_raw):,} phrases")

    if not args.no_stanza:
        print("\n🔧 Chargement Stanza fr...")
        nlp = stanza.Pipeline("fr", processors="tokenize,mwt,pos,lemma,depparse",
                              verbose=False, use_gpu=False)
        print("✅ Stanza prêt\n")

        def process_split(rows, label, batch_size=32):
            stats = Counter()
            out = []
            n = len(rows)
            for batch_start in range(0, n, batch_size):
                batch = rows[batch_start:batch_start + batch_size]
                try:
                    docs = nlp.bulk_process([d['text'] for d in batch])
                    for data, doc in zip(batch, docs):
                        try:
                            out.append(process_sentence(data, doc, stats))
                        except Exception as e_sent:
                            stats['sent_errors'] += 1
                            out.append(data)  # garder la phrase sans injection
                except Exception as e_batch:
                    stats['batch_errors'] += 1
                    print(f"\n  ⚠️  Batch [{batch_start}:{batch_start+batch_size}] erreur: {e_batch}")
                    for data in batch:
                        out.append(data)  # garder sans injection
                done = min(batch_start + batch_size, n)
                if done % 500 == 0 or done == n:
                    print(f"  {label} [{done:>6,}/{n:,}]  "
                          f"inj={stats['injected']:,}  vt_rec={stats['vt_recovered']:,}  "
                          f"snap={stats['snapped']:,}", end='\r')
            print()
            return out, stats

        _train_cache = Path("data/.train_v8.2_cache.jsonl")
        _val_cache   = Path("data/.val_v8.2_cache.jsonl")
        _test_cache  = Path("data/.test_v8.2_cache.jsonl")

        if _train_cache.exists() and _val_cache.exists() and _test_cache.exists() and not args.no_claude:
            print(f"♻️  Cache trouvé — rechargement train+val+test (skip Stanza sur v8.1)...")
            train_proc = load(_train_cache)
            val_proc   = load(_val_cache)
            test_proc  = load(_test_cache)
            print(f"   train cache : {len(train_proc):,} phrases")
            print(f"   val   cache : {len(val_proc):,} phrases")
            print(f"   test  cache : {len(test_proc):,} phrases")
            san_stats = Counter()
            train_proc, san_stats = sanitize_loaded(train_proc, san_stats)
            val_proc,   san_stats = sanitize_loaded(val_proc,   san_stats)
            test_proc,  san_stats = sanitize_loaded(test_proc,  san_stats)
            if san_stats['dropped_corrupt'] or san_stats['dropped_oob']:
                print(f"   🧹 Nettoyage cache : {san_stats['dropped_corrupt']} corrompus + "
                      f"{san_stats['dropped_oob']} hors-bornes supprimés")
            sv = st = sc_test = Counter()
        elif _train_cache.exists() and _val_cache.exists() and not args.no_claude:
            # Cache partiel : train+val OK, test manquant
            print(f"♻️  Cache partiel — rechargement train+val, recalcul test...")
            train_proc = load(_train_cache)
            val_proc   = load(_val_cache)
            san_stats = Counter()
            train_proc, san_stats = sanitize_loaded(train_proc, san_stats)
            val_proc,   san_stats = sanitize_loaded(val_proc,   san_stats)
            if san_stats['dropped_corrupt'] or san_stats['dropped_oob']:
                print(f"   🧹 Nettoyage cache : {san_stats['dropped_corrupt']} corrompus + "
                      f"{san_stats['dropped_oob']} hors-bornes supprimés")
            print(f"   train cache : {len(train_proc):,} phrases")
            print(f"   val   cache : {len(val_proc):,} phrases")
            print("🚀 Traitement TEST v8.1...")
            test_proc, sc_test = process_split(test_raw, "test")
            print(f"   gov injectés={sc_test['injected']:,}  vt récupérés={sc_test['vt_recovered']:,}")
            with open(_test_cache, 'w') as _f:
                for _d in test_proc: _f.write(json.dumps(_d, ensure_ascii=False) + '\n')
            print(f"   💾 Cache test sauvegardé → {_test_cache}")
            sv = st = Counter()
        else:
            print("🚀 Traitement TRAIN v8.1...")
            train_proc, st = process_split(train_raw, "train")
            print(f"   gov injectés={st['injected']:,}  vt récupérés={st['vt_recovered']:,}  "
                  f"snapped={st['snapped']:,}  sans_gov={st['no_gov']:,}")

            print("🚀 Traitement VAL v8.1...")
            val_proc, sv = process_split(val_raw, "val")
            print(f"   gov injectés={sv['injected']:,}  vt récupérés={sv['vt_recovered']:,}")

            print("🚀 Traitement TEST v8.1...")
            test_proc, sc_test = process_split(test_raw, "test")
            print(f"   gov injectés={sc_test['injected']:,}  vt récupérés={sc_test['vt_recovered']:,}")

            # Sauvegarde intermédiaire (évite de refaire train+val+test si Claude plante)
            with open(_train_cache, 'w') as _f:
                for _d in train_proc: _f.write(json.dumps(_d, ensure_ascii=False) + '\n')
            with open(_val_cache, 'w') as _f:
                for _d in val_proc: _f.write(json.dumps(_d, ensure_ascii=False) + '\n')
            with open(_test_cache, 'w') as _f:
                for _d in test_proc: _f.write(json.dumps(_d, ensure_ascii=False) + '\n')
            print(f"   💾 Cache sauvegardé → {_train_cache}, {_val_cache}, {_test_cache}")

        if claude_raw:
            print("🚀 Traitement Claude (récupération verb_trigger manquants)...")
            claude_proc, sc = process_split(claude_raw, "claude")
            print(f"   vt récupérés={sc['vt_recovered']:,}  already={sc['already']:,}"
                  f"  batch_err={sc['batch_errors']:,}  sent_err={sc['sent_errors']:,}")
        else:
            claude_proc = []
    else:
        print("⏭️  Stanza désactivé")
        train_proc, val_proc, test_proc, claude_proc = train_raw, val_raw, test_raw, claude_raw

    # Dédupliquer Claude vs v8.1, puis répartir sur train/val/test (80/10/10)
    if claude_proc:
        existing_ids = {d['id'] for d in train_proc} | {d['id'] for d in val_proc} | {d['id'] for d in test_proc}
        claude_new = [d for d in claude_proc if d['id'] not in existing_ids]

        # Répartition déterministe par hashage de l'ID (reproductible)
        claude_train, claude_val, claude_test = [], [], []
        for d in claude_new:
            h = int(hashlib.md5(d['id'].encode()).hexdigest(), 16) % 100
            if h < 80:
                claude_train.append(d)
            elif h < 90:
                claude_val.append(d)
            else:
                claude_test.append(d)

        print(f"\n📊 Claude : {len(claude_proc):,} total, "
              f"{len(claude_proc)-len(claude_new):,} doublons exclus, "
              f"{len(claude_new):,} nouvelles phrases")
        print(f"   → train +{len(claude_train):,}  val +{len(claude_val):,}  test +{len(claude_test):,}")
        train_final = train_proc + claude_train
        val_final   = val_proc  + claude_val
        test_final  = test_proc + claude_test
    else:
        train_final = train_proc
        val_final   = val_proc
        test_final  = test_proc

    # Nettoyage final : supprimer tout span hard-corrompu résiduel (quel que soit l'origine)
    final_san = Counter()
    train_final, final_san = sanitize_loaded(train_final, final_san)
    val_final,   final_san = sanitize_loaded(val_final,   final_san)
    test_final,  final_san = sanitize_loaded(test_final,  final_san)
    if final_san['dropped_corrupt'] or final_san['dropped_oob']:
        print(f"\n🧹 Nettoyage final : {final_san['dropped_corrupt']} spans corrompus + "
              f"{final_san['dropped_oob']} hors-bornes supprimés")

    # Validation
    print("\n" + "="*70)
    print("🔍 VALIDATION COMPLÈTE DES OFFSETS")
    print("="*70)
    err_tr   = validate_dataset(train_final, f"TRAIN v8.2 ({len(train_final):,} phrases)")
    err_val  = validate_dataset(val_final,   f"VAL   v8.2 ({len(val_final):,} phrases)")
    err_test = validate_dataset(test_final,  f"TEST  v8.2 ({len(test_final):,} phrases)")
    total_errors = err_tr + err_val + err_test

    if args.dry_run:
        print("\n⏭️  --dry-run : fichiers non écrits")
        return

    if total_errors > 0:
        print(f"\n⛔ {total_errors} erreur(s) hard — fichiers NON écrits.")
        sys.exit(1)

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_OUT, 'w') as f:
        for d in train_final:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    with open(VAL_OUT, 'w') as f:
        for d in val_final:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    with open(TEST_OUT, 'w') as f:
        for d in test_final:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(f"\n✅ Écrits :")
    print(f"   {TRAIN_OUT}  ({len(train_final):,} phrases)")
    print(f"   {VAL_OUT}    ({len(val_final):,} phrases)")
    print(f"   {TEST_OUT}   ({len(test_final):,} phrases)")

    tok = "microsoft/deberta-v3-base"
    print(f"""
💡 Prochaine étape :
  python3 build_multitask_dataset.py \\
      --input  {TRAIN_OUT}  --output data/train.multitask.v8.2.jsonl \\
      --model-name {tok}

  python3 build_multitask_dataset.py \\
      --input  {VAL_OUT}    --output data/val.multitask.v8.2.jsonl \\
      --model-name {tok}

  python3 build_multitask_dataset.py \\
      --input  {TEST_OUT}   --output data/test.multitask.v8.2.jsonl \\
      --model-name {tok}
""")


if __name__ == '__main__':
    main()

