#!/usr/bin/env python3
"""Suit l'evolution du training W&B (ASCII only, pas d'emoji)."""
import sys
from dotenv import dotenv_values
import wandb

key = dotenv_values(".secrets.env", encoding='utf-8').get("WANDB_API_KEY")
api = wandb.Api(api_key=key)
runs = list(api.runs("pimpmyrag-ner", order="created_at"))

header = f"{'Run':<22} {'state':<9} {'ep':>3}  {'val_coarse':>10}  {'val_fine':>9}  {'val_boundary':>12}  {'val_loss':>9}  {'score':>8}"
print(header)
print("-" * len(header))

for r in runs:
    hist = list(r.scan_history(page_size=5))
    last = hist[-1] if hist else {}
    ep  = last.get('epoch', '?')
    vcf = last.get('val/coarse_f1',   float('nan'))
    vff = last.get('val/fine_f1',     float('nan'))
    vbf = last.get('val/boundary_f1', float('nan'))
    vl  = last.get('val/loss',        float('nan'))
    sc  = last.get('score',           float('nan'))
    state = "[running]" if r.state == "running" else "[done]   "
    print(f"{r.name:<22} {state} {str(ep):>3}  {vcf:>10.4f}  {vff:>9.4f}  {vbf:>12.4f}  {vl:>9.4f}  {sc:>8.4f}")

# Detail run en cours
running = [r for r in runs if r.state == "running"]
if running:
    r = running[-1]
    print(f"\nRun actif : {r.name}")
    print(f"URL       : {r.url}")
    steps = list(r.scan_history(page_size=30))
    step_rows = [s for s in steps if s.get('train/loss') is not None]
    if step_rows:
        print(f"\nDerniers steps train :")
        for s in step_rows[-5:]:
            print(f"  step={s.get('_step','?'):>4}  loss={s.get('train/loss', float('nan')):.4f}")
    epoch_rows = [s for s in steps if s.get('val/coarse_f1') is not None]
    if epoch_rows:
        last_ep = epoch_rows[-1]
        print(f"\nDernier epoch log  : ep={last_ep.get('epoch','?')}  val_coarse={last_ep.get('val/coarse_f1',float('nan')):.4f}  val_fine={last_ep.get('val/fine_f1',float('nan')):.4f}")
