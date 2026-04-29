#!/usr/bin/env python3
"""
debug_svo_heads.py
~~~~~~~~~~~~~~~~~~
Diagnostic des têtes SVO du modèle ONNX v4.

Objectif : comprendre pour chaque span candidat ce que le modèle prédit réellement
sur les têtes svo_boundary / syn / role — notamment pourquoi les verbes ont des rôles
argumentaux et les arguments ont une boundary quasi-nulle.

Usage :
    python debug_svo_heads.py
"""

import math
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# ── Config ────────────────────────────────────────────────────────────────────

ONNX_PATH   = "/Users/simon_longuet/IdeaProjects/pimpmyrag/models/deberta/fine-tuning-29042026/best_model_multitask_full.onnx"
TOK_PATH    = "/Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head/tokenizer_export_clean/tokenizer.json"
MAX_SEQ_LEN = 128
MAX_SPAN_LEN = 12

# Labels v4
SYN_LABELS       = ["verb_trigger", "pron_subj", "pron_obj"]
ROLE_LABELS      = ["SUBJECT", "OBJECT", "OBLIQUE", "OBLIQUE_AGENT", "OBLIQUE_CAUSE", "APPOS", "NONE"]
BOUNDARY_LABELS  = ["O", "B"]

# Phrases de test
SENTENCES = [
    "Marc Dupont a déclaré que la France soutiendra l'Ukraine.",
    "La Fondation Horizon a publié son bilan annuel pour 2025.",
    "Sarah Konaté supervise les tests sur le terrain depuis le laboratoire.",
    "L'audit réalisé par le cabinet Deloitte a confirmé les résultats.",
    "Ce projet vise à installer des panneaux solaires dans les zones rurales.",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def load_session(path):
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=opts,
                                 providers=["CPUExecutionProvider"])

def encode(tokenizer, text, max_len):
    enc = tokenizer.encode(text)
    ids = enc.ids[:max_len]
    mask = [1] * len(ids)
    pad = max_len - len(ids)
    ids  += [0] * pad
    mask += [0] * pad
    # offsets : (start_char, end_char) par token
    offsets = [o for o in enc.offsets[:max_len]]
    offsets += [(0, 0)] * pad
    return ids, mask, offsets, len(enc.ids[:max_len])

def generate_spans(seq_len, max_span_len, offsets):
    """Génère les spans comme buildCandidates en Kotlin :
    - Démarre sur les tokens dont l'offset char est > (0,0) (hors CLS/SEP)
    - Ignore les spans dont le texte char est vide (offsets (0,0) = token spécial)
    """
    starts, ends = [], []
    for s in range(seq_len):
        # Ignorer les tokens spéciaux (CLS/SEP ont offset (0,0))
        if s < len(offsets) and offsets[s] == (0, 0):
            continue
        for e in range(s, min(s + max_span_len, seq_len)):
            if e < len(offsets) and offsets[e] == (0, 0):
                break
            starts.append(s)
            ends.append(e)
    return starts, ends

def tok_to_char(offsets, tok_idx):
    if tok_idx < len(offsets):
        return offsets[tok_idx][0]
    return -1

def span_text(text, offsets, tok_start, tok_end):
    if tok_start >= len(offsets) or tok_end >= len(offsets):
        return "?"
    cs = offsets[tok_start][0]
    ce = offsets[tok_end][1]
    return text[cs:ce] if cs >= 0 and ce > cs else "?"

# ── Inférence ─────────────────────────────────────────────────────────────────

def run_sentence(session, tokenizer, text):
    ids, mask, offsets, real_len = encode(tokenizer, text, MAX_SEQ_LEN)

    starts, ends = generate_spans(real_len, MAX_SPAN_LEN, offsets)
    n = len(starts)

    inputs = {
        "input_ids":      np.array([ids],  dtype=np.int64),
        "attention_mask": np.array([mask], dtype=np.int64),
        "span_starts":    np.array(starts, dtype=np.int64),
        "span_ends":      np.array(ends,   dtype=np.int64),
        "span_batch_ids": np.zeros(n,      dtype=np.int64),
    }

    outs = session.run(None, inputs)
    out_names = [o.name for o in session.get_outputs()]
    out_map   = dict(zip(out_names, outs))

    bnd_logits  = out_map["boundary_logits"]       # [N, 2]
    svob_logits = out_map["svo_boundary_logits"]   # [N, 2]
    syn_logits  = out_map["syn_logits"]            # [N, 3]
    role_logits = out_map["role_logits"]           # [N, 7]

    results = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        p_bnd  = softmax(bnd_logits[i])[1]
        p_svob = softmax(svob_logits[i])[1]
        syn_p  = softmax(syn_logits[i])
        role_p = softmax(role_logits[i])

        syn_idx  = int(np.argmax(syn_p))
        role_idx = int(np.argmax(role_p))

        results.append({
            "tok_start": s,
            "tok_end":   e,
            "text":      span_text(text, offsets, s, e),
            "p_bnd":     float(p_bnd),
            "p_svob":    float(p_svob),
            "syn":       SYN_LABELS[syn_idx],
            "syn_p":     float(syn_p[syn_idx]),
            "role":      ROLE_LABELS[role_idx],
            "role_p":    float(role_p[role_idx]),
        })

    return results

# ── Analyse ───────────────────────────────────────────────────────────────────

TAU_BND  = 0.50   # seuil svo_boundary
TAU_NER  = 0.50   # seuil ner boundary (simplifié)

def analyse(results, text):
    # Trier par p_svob desc, garder le top-30 par score intéressant
    interesting = sorted(
        [r for r in results if r["p_svob"] >= 0.05 or (r["role"] != "NONE" and r["role_p"] > 0.80)],
        key=lambda r: -max(r["p_svob"], r["p_bnd"] * (r["role_p"] if r["role"] != "NONE" else 0))
    )[:40]

    print(f"\n{'─'*100}")
    print(f"📝  {text}")
    print(f"{'─'*100}")
    print(f"{'SPAN TEXT':<35} {'NER_bnd':>8} {'SVO_bnd':>8} {'SYN':<13} {'ROLE':<16} {'role_p':>7}  flag")
    print(f"{'─'*100}")

    for r in interesting:
        # Flag : est-ce un verbe avec p_svob élevée (attendu) ?
        is_std_verb = r["p_svob"] >= TAU_BND and r["syn"] == "verb_trigger"
        # Flag : argument avec p_svob faible mais rôle fort
        is_forced_arg = r["p_svob"] < TAU_BND and r["role"] != "NONE" and r["role_p"] > 0.85
        # Flag : problème — verbe avec rôle != NONE
        is_confused_verb = r["p_svob"] >= TAU_BND and r["syn"] == "verb_trigger" and r["role"] != "NONE"
        # Flag : rôle fort mais syn inattendu
        is_syn_mismatch = r["role"] not in ("NONE",) and r["syn"] == "verb_trigger" and r["p_svob"] < 0.1

        flag = ""
        if is_std_verb and not is_confused_verb:  flag = "✅ verb"
        elif is_confused_verb:                     flag = "⚠️  verb+role CONFUSED"
        elif is_forced_arg:                        flag = "🔵 forced arg"
        elif is_syn_mismatch:                      flag = "🔴 syn≠role"

        span = r["text"][:34]
        print(f"{span:<35} {r['p_bnd']:>8.4f} {r['p_svob']:>8.4f} {r['syn']:<13} {r['role']:<16} {r['role_p']:>7.4f}  {flag}")

    # Stats globales
    all_svob_triggered = [r for r in results if r["p_svob"] >= TAU_BND]
    verbs_confused     = [r for r in all_svob_triggered if r["role"] != "NONE"]
    forced_args        = [r for r in results if r["p_svob"] < TAU_BND and r["role"] != "NONE" and r["role_p"] > 0.85]

    print(f"\n  📊 Stats : spans avec p_svob≥{TAU_BND}: {len(all_svob_triggered)}"
          f"  dont verbes_rôle_confus: {len(verbs_confused)}"
          f"  | args_forcés_probables: {len(forced_args)}")

    # Distribution des rôles sur les spans avec p_svob élevée
    if all_svob_triggered:
        from collections import Counter
        role_dist = Counter(r["role"] for r in all_svob_triggered)
        print(f"  📊 Distribution rôles (p_svob≥{TAU_BND}) : {dict(role_dist.most_common())}")

    # Distribution des synLabel sur les args forcés
    if forced_args:
        from collections import Counter
        syn_dist = Counter(r["syn"] for r in forced_args)
        print(f"  📊 Distribution syn (args forcés)         : {dict(syn_dist.most_common())}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🔍 Chargement tokenizer…")
    tokenizer = Tokenizer.from_file(TOK_PATH)

    print("🔍 Chargement session ONNX…")
    session = load_session(ONNX_PATH)
    print(f"   Outputs : {[o.name for o in session.get_outputs()]}")

    for text in SENTENCES:
        results = run_sentence(session, tokenizer, text)
        analyse(results, text)

    # ── Test supplémentaire : regarder la distribution de p_svob sur TOUS les spans
    # d'une phrase pour voir si le modèle est bien calibré
    print(f"\n\n{'═'*100}")
    print("📊 DISTRIBUTION p_svob sur tous les spans d'une phrase de référence :")
    text_ref = "Marc Dupont a déclaré que la France soutiendra l'Ukraine."
    results = run_sentence(session, tokenizer, text_ref)
    buckets = [0]*11
    for r in results:
        idx = min(int(r["p_svob"] * 10), 10)
        buckets[idx] += 1
    total = len(results)
    for i, cnt in enumerate(buckets):
        lo = i * 0.1
        hi = lo + 0.1
        bar = "█" * int(cnt / max(total, 1) * 80)
        print(f"  [{lo:.1f}-{hi:.1f}[  {cnt:4d} spans  {bar}")
    print(f"  Total : {total} spans candidats")

    print(f"\n{'═'*100}")
    print("📊 TOP-10 spans par p_svob :")
    top10 = sorted(results, key=lambda r: -r["p_svob"])[:10]
    for r in top10:
        print(f"  p_svob={r['p_svob']:.4f}  p_bnd={r['p_bnd']:.4f}  syn={r['syn']:<13}  role={r['role']:<16}  '{r['text']}'")

    print(f"\n{'═'*100}")
    print("📊 TOP-10 spans par role_p (role != NONE) :")
    top10_role = sorted([r for r in results if r["role"] != "NONE"], key=lambda r: -r["role_p"])[:10]
    for r in top10_role:
        print(f"  role_p={r['role_p']:.4f}  p_svob={r['p_svob']:.4f}  syn={r['syn']:<13}  role={r['role']:<16}  '{r['text']}'")

if __name__ == "__main__":
    main()

