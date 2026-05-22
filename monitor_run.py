#!/usr/bin/env python3
"""
monitor_run.py — Monitoring W&B run pimpmyrag-ner
==================================================
Usage :
    python3 monitor_run.py                    # dernier run actif (ou plus récent)
    python3 monitor_run.py --run <run_id>     # run spécifique
    python3 monitor_run.py --all              # tous les runs récents (résumé)
    python3 monitor_run.py --compare N        # compare les N derniers runs
    python3 monitor_run.py --watch 60         # rafraîchit toutes les 60s
"""
import os, sys, time, argparse
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "training/multi-head/.secrets.env"))

import wandb

PROJECT = "pimpmyrag-pimpmyrag/pimpmyrag-ner"

METRIC_GROUPS = {
    "NER core": [
        "val/boundary_f1", "val/coarse_f1", "val/fine_f1",
    ],
    "NER détail — TIME": [
        "val/fine_f1_hint_time_date", "val/fine_recall_hint_time_date",
        "val/fine_f1_hint_time_duration", "val/fine_f1_hint_time_clock",
        "val/coarse_f1_TIME", "val/coarse_recall_TIME",
    ],
    "NER détail — INST": [
        "val/fine_f1_hint_inst_name",  "val/fine_recall_hint_inst_name",
        "val/fine_f1_hint_inst_role",  "val/fine_recall_hint_inst_role",
    ],
    "SVO": [
        "val/svo_f1", "val/role_f1", "val/voice_f1",
    ],
    "Loss": [
        "train/loss", "val/loss",
    ],
}

ALL_KEYS = ["epoch"] + [k for keys in METRIC_GROUPS.values() for k in keys]

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
    # Préférer un run en cours
    for r in runs:
        if r.state == "running":
            return r
    # Sinon le plus récent
    return next(iter(runs), None)

def trend_icon(d):
    if d is None:    return "  "
    if d >  0.005:   return "📈"
    if d < -0.002:   return "📉"
    return "➡️ "

def fmt(v):
    if v != v: return "  — "   # nan
    return f"{v:.3f}"

def print_run_detail(r, max_epochs=None):
    state_icon = {"running": "🟢", "crashed": "💀", "finished": "✅"}.get(r.state, "❓")
    print(f"\n{'═'*80}")
    print(f"  {state_icon}  {r.name}")
    print(f"     ID: {r.id}  |  State: {r.state}  |  Created: {r.created_at}")
    print(f"{'═'*80}")

    # Pas de filtre keys= : évite les retours vides quand certaines clés n'existent pas encore
    hist = r.history(samples=500, pandas=True)
    if hist.empty or "epoch" not in hist.columns:
        print("  ⚠️  Pas encore de données d'epochs.")
        return

    hist = hist.dropna(subset=["epoch"]).sort_values("epoch")
    by_ep = hist.groupby("epoch").agg({k: "max" for k in ALL_KEYS if k != "epoch" and k in hist.columns})
    if max_epochs:
        by_ep = by_ep.tail(max_epochs)

    # ── NER core ──────────────────────────────────────────────────────────────
    print(f"\n  📊 NER core")
    print(f"    {'ep':>3}  {'boundary':>8}  {'Δbnd':>6}  {'coarse':>7}  {'fine':>6}  {'concrete':>8}  {'abstract':>8}  {'trend'}")
    print(f"    {'─'*70}")
    prev_bnd = None
    for ep, row in by_ep.iterrows():
        ep = int(ep)
        bnd      = row.get("val/boundary_f1",    float("nan"))
        coarse   = row.get("val/coarse_f1",      float("nan"))
        fine     = row.get("val/fine_f1",        float("nan"))
        concrete = row.get("val/fine_concrete_f1", float("nan"))
        abstract = row.get("val/fine_abstract_f1", float("nan"))
        d = (bnd - prev_bnd) if (prev_bnd is not None and bnd == bnd) else None
        delta  = f"{d:+.3f}" if d is not None else "  — "
        trn    = trend_icon(d)
        print(f"    {ep:>3}  {fmt(bnd):>8}  {delta:>6}  {fmt(coarse):>7}  {fmt(fine):>6}  {fmt(concrete):>8}  {fmt(abstract):>8}  {trn}")
        prev_bnd = bnd if bnd == bnd else prev_bnd

    # ── TIME ──────────────────────────────────────────────────────────────────
    time_keys = [k for k in ["val/coarse_f1_TIME","val/coarse_recall_TIME",
                              "val/fine_f1_hint_time_date","val/fine_recall_hint_time_date",
                              "val/fine_f1_hint_time_duration","val/fine_f1_hint_time_clock"] if k in by_ep.columns]
    if time_keys:
        print(f"\n  🕐 TIME labels")
        header_cols = ["coarse_f1","coarse_rec","date_f1","date_rec","dur_f1","clk_f1"]
        print(f"    {'ep':>3}  " + "  ".join(f"{h:>8}" for h in header_cols[:len(time_keys)]))
        print(f"    {'─'*60}")
        for ep, row in by_ep.iterrows():
            vals = [fmt(row.get(k, float("nan"))) for k in time_keys]
            print(f"    {int(ep):>3}  " + "  ".join(f"{v:>8}" for v in vals))

    # ── INST ──────────────────────────────────────────────────────────────────
    inst_keys = [k for k in ["val/fine_f1_hint_inst_name","val/fine_recall_hint_inst_name",
                              "val/fine_f1_hint_inst_role","val/fine_recall_hint_inst_role"] if k in by_ep.columns]
    if inst_keys:
        print(f"\n  🏛️  INST labels")
        header_cols = ["name_f1","name_rec","role_f1","role_rec"]
        print(f"    {'ep':>3}  " + "  ".join(f"{h:>8}" for h in header_cols[:len(inst_keys)]))
        print(f"    {'─'*45}")
        for ep, row in by_ep.iterrows():
            vals = [fmt(row.get(k, float("nan"))) for k in inst_keys]
            print(f"    {int(ep):>3}  " + "  ".join(f"{v:>8}" for v in vals))

    # ── SVO ───────────────────────────────────────────────────────────────────
    svo_keys = [k for k in ["val/svo_f1","val/role_f1","val/voice_f1"] if k in by_ep.columns]
    if svo_keys:
        print(f"\n  🔗 SVO")
        print(f"    {'ep':>3}  " + "  ".join(f"{k.split('/')[-1]:>8}" for k in svo_keys))
        print(f"    {'─'*40}")
        for ep, row in by_ep.iterrows():
            vals = [fmt(row.get(k, float("nan"))) for k in svo_keys]
            print(f"    {int(ep):>3}  " + "  ".join(f"{v:>8}" for v in vals))

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_keys = [k for k in ["train/loss","val/loss"] if k in by_ep.columns]
    if loss_keys:
        print(f"\n  📉 Loss")
        print(f"    {'ep':>3}  " + "  ".join(f"{k.split('/')[-1]:>10}" for k in loss_keys))
        print(f"    {'─'*35}")
        for ep, row in by_ep.iterrows():
            vals = [fmt(row.get(k, float("nan"))) for k in loss_keys]
            print(f"    {int(ep):>3}  " + "  ".join(f"{v:>10}" for v in vals))

    # ── Résumé SVO trigger ────────────────────────────────────────────────────
    last = by_ep.iloc[-1] if not by_ep.empty else None
    if last is not None:
        bnd    = last.get("val/boundary_f1", float("nan"))
        coarse = last.get("val/coarse_f1",   float("nan"))
        fine   = last.get("val/fine_f1",     float("nan"))
        print(f"\n  🎯 État SVO trigger (seuils : bnd>0.77 & coarse>0.87)")
        bnd_ok    = bnd    > 0.77  if bnd    == bnd    else False
        coarse_ok = coarse > 0.87  if coarse == coarse else False
        print(f"     boundary : {fmt(bnd):>6}  {'✅' if bnd_ok else '⏳'} (seuil 0.77)")
        print(f"     coarse   : {fmt(coarse):>6}  {'✅' if coarse_ok else '⏳'} (seuil 0.87)")
        if bnd_ok and coarse_ok:
            print(f"     → 🚀 TRIGGER ACTIF")
        else:
            print(f"     → en attente")


def print_compare(api, n=6):
    runs = [r for r in api.runs(PROJECT, order="-created_at") if r.state in ("running","finished","crashed")][:n]
    print(f"\n{'═'*110}")
    print(f"  Comparaison des {n} derniers runs — boundary par epoch + peak")
    print(f"{'═'*110}")
    print(f"  {'':1} {'run':62s}  {'ep1':>5} {'ep2':>5} {'ep3':>5} {'ep4':>5} {'ep5':>5} {'ep6':>5} {'ep8':>5} {'peak':>5}")
    print(f"  {'─'*110}")
    for r in runs:
        try:
            hist = r.history(samples=500, pandas=True)
            if hist.empty or "epoch" not in hist.columns: continue
            hist = hist.dropna(subset=["epoch","val/boundary_f1"]) if "val/boundary_f1" in hist.columns else hist.dropna(subset=["epoch"])
            by_ep = hist.groupby("epoch")["val/boundary_f1"].max()
            peak  = by_ep.max()
            vals  = [f"{by_ep.get(ep, float('nan')):.3f}" if by_ep.get(ep, float('nan'))==by_ep.get(ep, float('nan')) else "  — " for ep in [1,2,3,4,5,6,8]]
            icon  = {"running":"🟢","crashed":"💀","finished":"✅"}.get(r.state,"❓")
            print(f"  {icon} {r.name[:62]:62s}  {'  '.join(vals)}  {peak:.3f}")
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

