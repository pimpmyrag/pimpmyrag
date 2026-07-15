#!/usr/bin/env python3
"""
monitor_run.py — Monitoring W&B run pimpmyrag-ner
==================================================
Usage :
    python3 monitor_run.py                    # dernier run actif (ou plus récent)
    python3 monitor_run.py --run <run_id>     # run spécifique
    python3 monitor_run.py --compare N        # compare les N derniers runs (boundary/epoch)
    python3 monitor_run.py --watch 60         # rafraîchit toutes les 60s
    python3 monitor_run.py --epochs 10        # afficher seulement les 10 dernières epochs
"""
import os, sys, time, argparse
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "training/multi-head/.secrets.env"))

import wandb

PROJECT = "pimpmyrag-pimpmyrag/pimpmyrag-ner"

# Seuils SVO trigger
SVO_TRIGGER_BND    = 0.77
SVO_TRIGGER_COARSE = 0.87

def get_api():
    return wandb.Api(api_key=os.environ["WANDB_API_KEY"])

def find_run(api, run_id=None):
    runs = api.runs(PROJECT, order="-created_at")
    if run_id:
        for r in runs:
            if r.id == run_id or run_id in r.name:
                return r
        print(f"❌ Run '{run_id}' introuvable")
        sys.exit(1)
    for r in runs:
        if r.state == "running":
            return r
    return next(iter(runs), None)

def trend_icon(d):
    if d is None:    return "   "
    if d >  0.005:   return "📈 "
    if d < -0.002:   return "📉 "
    return "➡️  "

def fmt(v, decimals=3):
    if v != v or v is None: return "  —  "
    return f"{v:.{decimals}f}"

def get_hist(r):
    hist = r.history(samples=500, pandas=True)
    if hist.empty or "epoch" not in hist.columns:
        return None
    hist = hist.dropna(subset=["epoch"]).sort_values("epoch")
    all_numeric = [c for c in hist.columns if hist[c].dtype.kind in "fi"]
    by_ep = hist.groupby("epoch").agg({k: "max" for k in all_numeric if k != "epoch"})
    return by_ep

def print_run_detail(r, max_epochs=None):
    state_icon = {"running": "🟢", "crashed": "💀", "finished": "✅"}.get(r.state, "❓")
    print(f"\n{'═'*80}")
    print(f"  {state_icon}  {r.name}")
    print(f"     ID: {r.id}  |  State: {r.state}  |  Created: {r.created_at}")
    print(f"{'═'*80}")

    by_ep = get_hist(r)
    if by_ep is None:
        print("  ⚠️  Pas encore de données d'epochs.")
        return

    if max_epochs:
        by_ep = by_ep.tail(max_epochs)

    def col(k):
        return by_ep[k] if k in by_ep.columns else None

    def val(row, k):
        v = row.get(k, float("nan")) if hasattr(row, "get") else float("nan")
        return float(v) if v == v else float("nan")

    # ── NER core ──────────────────────────────────────────────────────────────
    print(f"\n  📊 NER core")
    print(f"    {'ep':>3}  {'boundary':>8}  {'Δbnd':>6}  {'coarse':>7}  {'fine':>6}  {'concrete':>8}  {'abstract':>8}  {'trend'}")
    print(f"    {'─'*72}")
    prev_bnd = None
    for ep, row in by_ep.iterrows():
        bnd      = val(row, "val/boundary_f1")
        coarse   = val(row, "val/coarse_f1")
        fine     = val(row, "val/fine_f1")
        concrete = val(row, "val/fine_concrete_f1")
        abstract = val(row, "val/fine_abstract_f1")
        d = (bnd - prev_bnd) if (prev_bnd is not None and bnd == bnd) else None
        delta = f"{d:+.3f}" if d is not None else "   — "
        trn   = trend_icon(d)
        print(f"    {int(ep):>3}  {fmt(bnd):>8}  {delta:>6}  {fmt(coarse):>7}  {fmt(fine):>6}  {fmt(concrete):>8}  {fmt(abstract):>8}  {trn}")
        prev_bnd = bnd if bnd == bnd else prev_bnd

    # ── Coarse par classe ──────────────────────────────────────────────────────
    coarse_classes = ["PER", "LOC", "ORG", "TIME", "EVENT", "VALUE", "WORK", "OBJECT", "ABSTRACT"]
    coarse_class_keys = [f"val/coarse_f1_{c}" for c in coarse_classes]
    if any(k in by_ep.columns for k in coarse_class_keys):
        present = [(c, k) for c, k in zip(coarse_classes, coarse_class_keys) if k in by_ep.columns]
        print(f"\n  🏷️  Coarse par classe (f1)")
        header = "  ".join(f"{c:>7}" for c, _ in present)
        print(f"    {'ep':>3}  {header}")
        print(f"    {'─'*max(40, 6+9*len(present))}")
        for ep, row in by_ep.iterrows():
            vals = "  ".join(f"{fmt(val(row, k)):>7}" for _, k in present)
            print(f"    {int(ep):>3}  {vals}")

    # ── Fine par famille coarse — tableau imbriqué ────────────────────────────
    FAMILIES = {
        "PER":      ["hint_person_name", "hint_person_role", "hint_norp", "hint_group_role"],
        "LOC":      ["hint_gpe", "hint_fac_name", "hint_loc_generic", "hint_infra"],
        "ORG":      ["hint_org_name", "hint_inst_name", "hint_inst_role"],
        "TIME":     ["hint_time_date", "hint_time_clock", "hint_time_duration"],
        "EVENT":    ["hint_event_nominal", "hint_event_named"],
        "OBJECT":   ["hint_weapon", "hint_vehicle", "hint_substance", "hint_food",
                     "hint_tool", "hint_object_generic", "hint_object_name"],
        "VALUE":    ["hint_measure", "hint_percentage", "hint_count", "hint_money", "hint_rate"],
        "WORK":     ["hint_work_of_art", "hint_law", "hint_document", "hint_work_generic"],
        "ABSTRACT": ["hint_disease", "hint_language", "hint_doctrine", "hint_state",
                     "hint_notion", "hint_field"],
    }
    family_has_data = any(
        f"val/fine_f1_{lbl}" in by_ep.columns
        for fines in FAMILIES.values() for lbl in fines
    )
    if family_has_data:
        epochs = list(by_ep.index)
        n_ep   = len(epochs)
        W_LBL  = 20   # largeur colonne label
        W_EP   = 6    # largeur colonne epoch
        # En-tête
        ep_header = "  ".join(f"ep{int(e):>2}" for e in epochs)
        sep_line  = "─" * (4 + W_LBL + 2 + n_ep * (W_EP + 2))
        print(f"\n  📋 Fine par famille coarse (f1 val)")
        print(f"    {'':>{W_LBL}}  {ep_header}")
        for coarse_name, fine_labels in FAMILIES.items():
            # Filtrer les labels présents avec au moins une valeur > 0
            present = []
            for lbl in fine_labels:
                key = f"val/fine_f1_{lbl}"
                if key not in by_ep.columns:
                    continue
                series = [val(by_ep.iloc[i], key) for i in range(n_ep)]
                present.append((lbl, series))
            if not present:
                continue
            # Ligne coarse (coarse f1 par epoch)
            coarse_series = [val(by_ep.iloc[i], f"val/coarse_f1_{coarse_name}") for i in range(n_ep)]
            coarse_row = "  ".join(f"{fmt(v):>{W_EP}}" for v in coarse_series)
            print(f"    {sep_line}")
            print(f"    {('▸ ' + coarse_name):<{W_LBL}}  {coarse_row}")
            # Lignes fine labels (toujours affichées, — si jamais vu)
            for lbl, series in present:
                short = lbl.replace("hint_", "")
                fine_row = "  ".join(
                    f"{'  —  ':>{W_EP}}" if (v != v or v < 0.001) else f"{fmt(v):>{W_EP}}"
                    for v in series
                )
                print(f"      {'  ' + short:<{W_LBL-2}}  {fine_row}")
        print(f"    {sep_line}")

    # ── SVO boundary ──────────────────────────────────────────────────────────
    svo_bnd_keys = ["val/svo_boundary_f1", "val/svo_bnd_f1_non_verb",
                    "val/svo_bnd_precision_non_verb", "val/svo_bnd_recall_non_verb"]
    svo_bnd_present = [k for k in svo_bnd_keys if k in by_ep.columns]
    if svo_bnd_present:
        print(f"\n  🔍 SVO boundary")
        hdrs = ["svo_bnd", "non_verb_f1", "non_verb_prec", "non_verb_rec"]
        header = "  ".join(f"{h:>12}" for h, k in zip(hdrs, svo_bnd_keys) if k in by_ep.columns)
        print(f"    {'ep':>3}  {header}")
        sep = '─' * max(30, 6+14*len(svo_bnd_present))
        print(f"    {sep}")
        for ep, row in by_ep.iterrows():
            vals = "  ".join(f"{fmt(val(row, k)):>12}" for k in svo_bnd_present)
            print(f"    {int(ep):>3}  {vals}")

    # ── role_coarse ───────────────────────────────────────────────────────────
    rc_keys = {
        "total": "val/role_coarse_f1",
        "SUBJ":  "val/role_coarse_f1_SUBJ",
        "OBJ":   "val/role_coarse_f1_OBJ",
        "OBLIQ": "val/role_coarse_f1_OBLIQ",
        "APPOS": "val/role_coarse_f1_APPOS",
    }
    rc_rec_keys = {
        "SUBJ":  "val/role_coarse_recall_SUBJ",
        "OBJ":   "val/role_coarse_recall_OBJ",
        "OBLIQ": "val/role_coarse_recall_OBLIQ",
        "APPOS": "val/role_coarse_recall_APPOS",
    }
    if any(v in by_ep.columns for v in rc_keys.values()):
        print(f"\n  🎭 role_coarse  (f1 | recall en italique)")
        print(f"    {'ep':>3}  {'total':>7}  {'SUBJ_f1':>8} {'SUBJ_rc':>8}  {'OBJ_f1':>7} {'OBJ_rc':>7}  {'OBLIQ_f1':>8} {'OBLIQ_rc':>8}  {'APPOS_f1':>8} {'APPOS_rc':>8}")
        print(f"    {'─'*100}")
        for ep, row in by_ep.iterrows():
            total  = fmt(val(row, rc_keys["total"]))
            s_f1   = fmt(val(row, rc_keys["SUBJ"]))
            s_rc   = fmt(val(row, rc_rec_keys["SUBJ"]))
            o_f1   = fmt(val(row, rc_keys["OBJ"]))
            o_rc   = fmt(val(row, rc_rec_keys["OBJ"]))
            ob_f1  = fmt(val(row, rc_keys["OBLIQ"]))
            ob_rc  = fmt(val(row, rc_rec_keys["OBLIQ"]))
            ap_f1  = fmt(val(row, rc_keys["APPOS"]))
            ap_rc  = fmt(val(row, rc_rec_keys["APPOS"]))
            print(f"    {int(ep):>3}  {total:>7}  {s_f1:>8} {s_rc:>8}  {o_f1:>7} {o_rc:>7}  {ob_f1:>8} {ob_rc:>8}  {ap_f1:>8} {ap_rc:>8}")

    # ── semantic_role ─────────────────────────────────────────────────────────
    sr_core = ["AGENT", "PATIENT", "CONTENT", "CAUSE", "LOCATION", "TEMPORAL",
               "BENEFICIARY", "COMITATIVE", "ADVERSARY", "DOMAIN",
               "INSTRUMENT", "MEASURE", "SOURCE", "PURPOSE",
               "PART_OF", "MEMBER_OF", "OWNER", "IDENTITY"]
    sr_f1_keys   = [f"val/semantic_role_f1_{c}" for c in sr_core]
    sr_present   = [(c, k) for c, k in zip(sr_core, sr_f1_keys) if k in by_ep.columns]
    sr_total_key = "val/semantic_role_f1"
    sr_casc_key  = "val/semantic_role_cascaded_f1"
    if sr_present or sr_total_key in by_ep.columns:
        short_names = [c[:10] for c, _ in sr_present]
        print(f"\n  🌀 semantic_role (f1 val — tous spans supervisés)")
        casc_hdr = f"  {'cascaded':>9}" if sr_casc_key in by_ep.columns else ""
        per_lbl_hdr = "  ".join(f"{n:>10}" for n in short_names)
        print(f"    {'ep':>3}  {'total':>7}{casc_hdr}  {per_lbl_hdr}")
        sep = '─' * max(30, 6 + 9 + (11 if sr_casc_key in by_ep.columns else 0) + 12 * len(sr_present))
        print(f"    {sep}")
        for ep, row in by_ep.iterrows():
            total  = fmt(val(row, sr_total_key))
            casc   = (f"  {fmt(val(row, sr_casc_key)):>9}" if sr_casc_key in by_ep.columns else "")
            vals   = "  ".join(f"{fmt(val(row, k)):>10}" for _, k in sr_present)
            print(f"    {int(ep):>3}  {total:>7}{casc}  {vals}")

    # ── Morpho + Certainty + Verb pointer ─────────────────────────────────────
    morpho_keys = [
        ("voice",    "val/voice_f1"),
        ("certainty","val/certainty_f1"),
        ("gender",   "val/gender_f1"),
        ("number",   "val/number_f1"),
        ("person",   "val/person_f1"),
        ("verb_ptr", "val/verb_ptr_acc"),
    ]
    morpho_present = [(n, k) for n, k in morpho_keys if k in by_ep.columns]
    if morpho_present:
        header = "  ".join(f"{n:>9}" for n, _ in morpho_present)
        print(f"\n  🔡 Morpho / Voice / Verb-ptr")
        print(f"    {'ep':>3}  {header}")
        sep = '─' * max(30, 6+11*len(morpho_present))
        print(f"    {sep}")
        for ep, row in by_ep.iterrows():
            vals = "  ".join(f"{fmt(val(row, k)):>9}" for _, k in morpho_present)
            print(f"    {int(ep):>3}  {vals}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_keys = [k for k in ["train/loss", "val/loss"] if k in by_ep.columns]
    if loss_keys:
        print(f"\n  📉 Loss")
        print(f"    {'ep':>3}  " + "  ".join(f"{k.split('/')[-1]:>10}" for k in loss_keys))
        print(f"    {'─'*35}")
        for ep, row in by_ep.iterrows():
            vals = [fmt(val(row, k)) for k in loss_keys]
            print(f"    {int(ep):>3}  " + "  ".join(f"{v:>10}" for v in vals))

    # ── SVO trigger status ─────────────────────────────────────────────────────
    last = by_ep.iloc[-1] if not by_ep.empty else None
    if last is not None:
        bnd_v      = val(last, "val/boundary_f1")
        coarse_v   = val(last, "val/coarse_f1")
        rc_v       = val(last, "val/role_coarse_f1")
        rc_subj    = val(last, "val/role_coarse_f1_SUBJ")
        rc_obj     = val(last, "val/role_coarse_f1_OBJ")
        rc_obliq   = val(last, "val/role_coarse_f1_OBLIQ")
        svo_bnd_v  = val(last, "val/svo_boundary_f1")
        bnd_ok     = bnd_v    > SVO_TRIGGER_BND    if bnd_v    == bnd_v    else False
        coarse_ok  = coarse_v > SVO_TRIGGER_COARSE if coarse_v == coarse_v else False
        print(f"\n  🎯 État SVO trigger (seuils : bnd>{SVO_TRIGGER_BND} & coarse>{SVO_TRIGGER_COARSE})")
        print(f"     NER boundary  : {fmt(bnd_v):>7}  {'✅' if bnd_ok    else '⏳'} (seuil {SVO_TRIGGER_BND})")
        print(f"     NER coarse    : {fmt(coarse_v):>7}  {'✅' if coarse_ok else '⏳'} (seuil {SVO_TRIGGER_COARSE})")
        print(f"     SVO bnd       : {fmt(svo_bnd_v):>7}")
        print(f"     role_coarse   : {fmt(rc_v):>7}  [SUBJ={fmt(rc_subj)}  OBJ={fmt(rc_obj)}  OBLIQ={fmt(rc_obliq)}]")
        if bnd_ok and coarse_ok:
            print(f"     → 🚀 TRIGGER ACTIF — role_coarse en plein régime")
        else:
            missing = []
            if not bnd_ok:    missing.append(f"bnd ({fmt(bnd_v)} < {SVO_TRIGGER_BND})")
            if not coarse_ok: missing.append(f"coarse ({fmt(coarse_v)} < {SVO_TRIGGER_COARSE})")
            print(f"     → ⏳ en attente de : {', '.join(missing)}")


def print_compare(api, n=6):
    runs = [r for r in api.runs(PROJECT, order="-created_at") if r.state in ("running","finished","crashed")][:n]
    print(f"\n{'═'*115}")
    print(f"  Comparaison des {n} derniers runs — boundary + role_coarse par epoch + peak")
    print(f"{'═'*115}")
    print(f"  {'':1} {'run':55s}  {'ep1':>5} {'ep2':>5} {'ep3':>5} {'ep4':>5} {'ep5':>5} {'ep6':>5} {'ep8':>5} {'peak_bnd':>8} {'peak_rc':>7}")
    print(f"  {'─'*115}")
    for r in runs:
        try:
            hist = r.history(samples=500, pandas=True)
            if hist.empty or "epoch" not in hist.columns: continue
            hist = hist.dropna(subset=["epoch"])
            by_bnd = hist.groupby("epoch")["val/boundary_f1"].max() if "val/boundary_f1" in hist.columns else {}
            by_rc  = hist.groupby("epoch")["val/role_coarse_f1"].max() if "val/role_coarse_f1" in hist.columns else {}
            peak_bnd = max(by_bnd.values, default=float("nan"))
            peak_rc  = max(by_rc.values,  default=float("nan"))
            bnd_vals = [f"{by_bnd.get(ep, float('nan')):.3f}" if by_bnd.get(ep, float('nan'))==by_bnd.get(ep, float('nan')) else "  — " for ep in [1,2,3,4,5,6,8]]
            icon = {"running":"🟢","crashed":"💀","finished":"✅"}.get(r.state,"❓")
            pk_bnd = f"{peak_bnd:.3f}" if peak_bnd == peak_bnd else "  — "
            pk_rc  = f"{peak_rc:.3f}"  if peak_rc  == peak_rc  else "  — "
            print(f"  {icon} {r.name[:55]:55s}  {'  '.join(bnd_vals)}  {pk_bnd:>8} {pk_rc:>7}")
        except: pass


def main():
    parser = argparse.ArgumentParser(description="Monitor W&B run pimpmyrag-ner")
    parser.add_argument("--run",     default=None, help="run ID ou partie du nom")
    parser.add_argument("--all",     action="store_true", help="résumé tous les runs récents")
    parser.add_argument("--compare", type=int, default=0, metavar="N", help="compare les N derniers runs")
    parser.add_argument("--watch",   type=int, default=0, metavar="SEC", help="rafraîchit toutes les SEC secondes")
    parser.add_argument("--epochs",  type=int, default=None, help="afficher seulement les N dernières epochs")
    args = parser.parse_args()

    api = get_api()

    def run_once():
        if args.all or args.compare:
            n = args.compare if args.compare else 8
            print_compare(api, n)
        else:
            r = find_run(api, args.run)
            if not r:
                print("❌ Aucun run trouvé")
                sys.exit(1)
            print_run_detail(r, max_epochs=args.epochs)

    if args.watch:
        print(f"👁️  Watch mode — rafraîchissement toutes les {args.watch}s (Ctrl+C pour arrêter)")
        while True:
            os.system("clear")
            run_once()
            print(f"\n  ⏱  Dernier refresh : {time.strftime('%H:%M:%S')} — prochain dans {args.watch}s")
            time.sleep(args.watch)
    else:
        run_once()

if __name__ == "__main__":
    main()
