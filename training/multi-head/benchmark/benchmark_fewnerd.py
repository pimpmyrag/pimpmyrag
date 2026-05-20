#!/usr/bin/env python3
"""
benchmark_fewnerd.py — Benchmark sur Few-NERD (anglais, 66 fine types).
Taxonomie la plus proche de la nôtre. Test zero-shot cross-lingue.
"""
import argparse, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import torch

MULTI_HEAD_DIR = "/Users/simon_longuet/IdeaProjects/pimpmyrag/training/multi-head"
sys.path.insert(0, MULTI_HEAD_DIR)
from test_model_sentences_v3 import load_model_and_tokenizer, predict_texts_batch

# ── Few-NERD fine labels (67, idx 0 = O) ──────────────────────────────────
# Mapping Few-NERD fine → notre fine label le plus proche
FEWNERD_TO_OURS = {
    # person-*
    "person-actor":        "hint_person_name",
    "person-artist/author":"hint_person_name",
    "person-athlete":      "hint_person_name",
    "person-director":     "hint_person_name",
    "person-other":        "hint_person_name",
    "person-politician":   "hint_person_name",
    "person-scholar":      "hint_person_name",
    "person-soldier":      "hint_person_name",
    # location-*
    "location-GPE":                        "hint_gpe",
    "location-bodiesofwater":              "hint_loc_generic",
    "location-island":                     "hint_loc_generic",
    "location-mountain":                   "hint_loc_generic",
    "location-other":                      "hint_loc_generic",
    "location-park":                       "hint_fac_name",
    "location-road/railway/highway/transit":"hint_infra",
    # building-*
    "building-airport":        "hint_fac_name",
    "building-hospital":       "hint_fac_name",
    "building-hotel":          "hint_fac_name",
    "building-library":        "hint_fac_name",
    "building-other":          "hint_fac_name",
    "building-restaurant":     "hint_fac_name",
    "building-sportsfacility": "hint_fac_name",
    "building-theater":        "hint_fac_name",
    # organization-*
    "organization-company":                 "hint_org_name",
    "organization-education":               "hint_inst_name",
    "organization-government/governmentagency":"hint_inst_name",
    "organization-media/newspaper":         "hint_org_name",
    "organization-other":                   "hint_org_name",
    "organization-politicalparty":          "hint_org_name",
    "organization-religion":                "hint_org_name",
    "organization-showorganization":        "hint_org_name",
    "organization-sportsleague":            "hint_org_name",
    "organization-sportsteam":              "hint_org_name",
    # event-*
    "event-attack/battle/war/militaryconflict":"hint_event_named",
    "event-disaster":          "hint_event_named",
    "event-election":          "hint_event_named",
    "event-other":             "hint_event_named",
    "event-protest":           "hint_event_named",
    "event-sportsevent":       "hint_event_named",
    # art-* → work
    "art-broadcastprogram":    "hint_work_of_art",
    "art-film":                "hint_work_of_art",
    "art-music":               "hint_work_of_art",
    "art-other":               "hint_work_of_art",
    "art-painting":            "hint_work_of_art",
    "art-writtenart":          "hint_work_of_art",
    # product-*
    "product-airplane":        "hint_vehicle",
    "product-car":             "hint_vehicle",
    "product-ship":            "hint_vehicle",
    "product-train":           "hint_vehicle",
    "product-food":            "hint_food",
    "product-weapon":          "hint_weapon",
    "product-software":        "hint_object_name",
    "product-game":            "hint_object_name",
    "product-other":           "hint_object_name",
    # other-*
    "other-astronomything":    None,  # pas dans notre taxo
    "other-award":             None,
    "other-biologything":      None,
    "other-chemicalthing":     "hint_substance",
    "other-currency":          "hint_money",
    "other-disease":           "hint_disease",
    "other-educationaldegree": None,
    "other-god":               None,  # mythologie
    "other-language":          "hint_language",
    "other-law":               "hint_law",
    "other-livingthing":       None,
    "other-medical":           "hint_disease",
}

# Reverse : notre fine → coarse (pour comparaison)
OUR_FINE_TO_COARSE = {
    "hint_person_name":"PER","hint_person_role":"PER","hint_norp":"PER","hint_group_role":"PER",
    "hint_gpe":"LOC","hint_fac_name":"LOC","hint_loc_generic":"LOC","hint_infra":"LOC",
    "hint_org_name":"ORG","hint_inst_name":"ORG","hint_inst_role":"ORG",
    "hint_event_named":"EVENT","hint_event_nominal":"EVENT",
    "hint_work_of_art":"WORK","hint_law":"WORK","hint_document":"WORK","hint_work_generic":"WORK",
    "hint_time_date":"TIME","hint_time_clock":"TIME","hint_time_duration":"TIME",
    "hint_disease":"ABSTRACT","hint_language":"ABSTRACT",
    "hint_vehicle":"OBJECT","hint_weapon":"OBJECT","hint_food":"OBJECT",
    "hint_substance":"OBJECT","hint_object_name":"OBJECT",
    "hint_money":"VALUE","hint_measure":"VALUE","hint_count":"VALUE",
    "hint_percentage":"VALUE","hint_rate":"VALUE",
}

def text_overlap(ps,pe,gs,ge):
    return max(0, min(pe,ge)-max(ps,gs))

def match_quality(ps,pe,ptext,gs,ge,gtext):
    if ps==gs and pe==ge: return "exact"
    if ptext.strip()==gtext.strip(): return "text_eq"
    if gtext.strip() in ptext.strip() or ptext.strip() in gtext.strip(): return "partial"
    if text_overlap(ps,pe,gs,ge)>0: return "overlap"
    return None

def f1(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return p, r, 2*p*r/(p+r) if p+r else 0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",default="/Users/simon_longuet/IdeaProjects/pimpmyrag/checkpoint_best_multitask.pt")
    ap.add_argument("--model-name",default="microsoft/deberta-v3-base")
    ap.add_argument("--limit",type=int,default=500)
    ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--tau-boundary",type=float,default=0.70)
    ap.add_argument("--tau-coarse",type=float,default=0.45)
    ap.add_argument("--device",default="cpu")
    a=ap.parse_args()

    print("="*70)
    print("BENCHMARK : pimpmyrag → Few-NERD (EN, 66 fine types) ZERO-SHOT")
    print("="*70)

    from datasets import load_dataset
    ds = load_dataset("DFKI-SLT/few-nerd", "supervised", split="test", verification_mode="no_checks")
    feats = ds.features
    fine_names = feats['fine_ner_tags'].feature.names

    if a.limit > 0: ds = ds.select(range(min(a.limit, len(ds))))
    print(f"\n📦 {len(ds)} phrases (EN)")

    model, tokenizer = load_model_and_tokenizer(a.model_name, a.checkpoint, None, a.device)
    print("   ✅ Modèle chargé")

    # ── Préparer textes + gold spans ───────────────────────────────────────
    texts, gold_all = [], []
    skipped_types = Counter()
    for ex in ds:
        tokens = ex["tokens"]
        fine_tags = ex["fine_ner_tags"]
        text = " ".join(tokens)
        texts.append(text)
        # BIO → char spans
        spans, cur_type, cur_start = [], None, 0
        for i, t in enumerate(fine_tags):
            fn = fine_names[t] if t > 0 else "O"
            our = FEWNERD_TO_OURS.get(fn)
            if t > 0 and fn != "O":
                if cur_type is None:
                    cur_type, cur_start = our, i
                elif our != cur_type:
                    if cur_type is not None:
                        gold_text = " ".join(tokens[cur_start:i])
                        cs = sum(len(tokens[j])+1 for j in range(cur_start))
                        spans.append((cs, cs+len(gold_text), cur_type, gold_text))
                    elif cur_type is None:
                        skipped_types[fn] += 1
                    cur_type, cur_start = our, i
            else:
                if cur_type is not None:
                    gold_text = " ".join(tokens[cur_start:i])
                    cs = sum(len(tokens[j])+1 for j in range(cur_start))
                    spans.append((cs, cs+len(gold_text), cur_type, gold_text))
                cur_type = None
        if cur_type is not None:
            gold_text = " ".join(tokens[cur_start:len(fine_tags)])
            cs = sum(len(tokens[j])+1 for j in range(cur_start))
            spans.append((cs, cs+len(gold_text), cur_type, gold_text))
        # Filtrer les spans None (types Few-NERD sans mapping)
        mapped = [(s,e,t,txt) for s,e,t,txt in spans if t is not None]
        gold_all.append(mapped)

    total_gold = sum(len(g) for g in gold_all)
    print(f"📝 {len(texts)} textes, {total_gold} entités gold (après mapping)")

    # ── Inférence ──────────────────────────────────────────────────────────
    print(f"\n🔮 Inférence (batch={a.batch_size}, τ_bnd={a.tau_boundary})...")
    pred_all = []; t0 = time.time()
    for i in range(0, len(texts), a.batch_size):
        batch = texts[i:i+a.batch_size]
        results = predict_texts_batch(model, tokenizer, batch, a.device,
                                      tau_boundary=a.tau_boundary, tau_coarse=a.tau_coarse,
                                      tau_fine=0.0, topk_coarse=2)
        for res in results:
            preds = [(ent["char_start"], ent["char_end"], ent["fine"], ent["text"])
                     for ent in res["ner"]]
            pred_all.append(preds)
        if (i//a.batch_size) % 50 == 0:
            el = max(.01, time.time()-t0)
            print(f"   {i+len(batch):6d}/{len(texts)} ({(i+len(batch))/el:.0f} s/s)")
    el = max(.01, time.time()-t0)
    total_pred = sum(len(p) for p in pred_all)
    print(f"   ✅ {el:.1f}s ({len(texts)/el:.1f} sent/s) — {total_pred} preds\n")

    # ── RECALL sur gold (par fine label) ───────────────────────────────────
    recall_stats = Counter()
    recall_by_type = defaultdict(lambda: Counter())

    for golds, preds in zip(gold_all, pred_all):
        for gs,ge,gt,gtext in golds:
            best = None
            for ps,pe,pt,ptext in preds:
                m = match_quality(ps,pe,ptext,gs,ge,gtext)
                if m is not None and pt == gt:
                    best = m
                    break
                elif m is not None and OUR_FINE_TO_COARSE.get(pt,"?") == OUR_FINE_TO_COARSE.get(gt,"??"):
                    if best is None: best = "coarse_match"
                elif m is not None and best is None:
                    best = "type_mismatch"
            recall_stats[best or "missed"] += 1
            recall_by_type[gt][best or "missed"] += 1

    found_exact = recall_stats.get("exact",0) + recall_stats.get("text_eq",0)
    found_partial = recall_stats.get("partial",0) + recall_stats.get("overlap",0)
    found_coarse = recall_stats.get("coarse_match",0)
    found_mismatch = recall_stats.get("type_mismatch",0)
    missed = recall_stats.get("missed",0)

    print("═"*70)
    print("RECALL SUR GOLD — « retrouve-t-on ce que Few-NERD annote ? »")
    print("═"*70)
    print(f"  Total gold (mappés)  : {total_gold}")
    print(f"  Match exact fine     : {found_exact:5d}  ({found_exact/max(1,total_gold)*100:.1f}%)")
    print(f"  Match partiel fine   : {found_partial:5d}  ({found_partial/max(1,total_gold)*100:.1f}%)")
    print(f"  Match coarse OK      : {found_coarse:5d}  ({found_coarse/max(1,total_gold)*100:.1f}%)")
    print(f"  Type mismatch        : {found_mismatch:5d}  ({found_mismatch/max(1,total_gold)*100:.1f}%)")
    print(f"  Manqués              : {missed:5d}  ({missed/max(1,total_gold)*100:.1f}%)")
    recall_fine = (found_exact+found_partial)/max(1,total_gold)*100
    recall_coarse = (found_exact+found_partial+found_coarse)/max(1,total_gold)*100
    recall_total = (total_gold-missed)/max(1,total_gold)*100
    print(f"  ─────────────────────────────")
    print(f"  RECALL fine strict   : {recall_fine:.1f}%")
    print(f"  RECALL coarse        : {recall_coarse:.1f}%")
    print(f"  RECALL détection     : {recall_total:.1f}%  (toute détection)")

    # Par type
    print(f"\n  {'Fine label':>25s}  {'Gold':>5s}  {'Exact':>5s}  {'Part':>5s}  {'Coarse':>6s}  {'Miss':>5s}  {'Rec%':>6s}")
    print("  " + "─"*65)
    for t in sorted(recall_by_type.keys(), key=lambda x: -sum(recall_by_type[x].values())):
        c = recall_by_type[t]
        tg = sum(c.values())
        if tg < 3: continue
        ex = c.get("exact",0)+c.get("text_eq",0)
        pa = c.get("partial",0)+c.get("overlap",0)
        co = c.get("coarse_match",0)
        mi = c.get("missed",0)
        rec = (ex+pa)/tg*100 if tg else 0
        print(f"  {(t or '(unmapped)'):>25s}  {tg:5d}  {ex:5d}  {pa:5d}  {co:6d}  {mi:5d}  {rec:5.1f}%")

    # ── Tableau final ──────────────────────────────────────────────────────
    print(f"""
{'═'*70}
RÉSUMÉ — Few-NERD ZERO-SHOT (cross-lingue FR→EN)
{'═'*70}

  Recall fine strict     : {recall_fine:.1f}%  (exact fine label match)
  Recall coarse          : {recall_coarse:.1f}%  (bon coarse, fine peut varier)
  Recall détection       : {recall_total:.1f}%  (entité détectée, tout type)
  Entités manquées       : {missed}/{total_gold}

⚠  Notre modèle est entraîné sur du FRANÇAIS et testé ici sur de l'ANGLAIS.
   DeBERTa-v3-base est multilingue mais pas spécifiquement optimisé pour l'EN.
   Le recall mesure la capacité à détecter les mêmes entités qu'un benchmark
   académique reconnu, avec une taxonomie fine quasi-identique.
""")

if __name__=="__main__":
    main()

