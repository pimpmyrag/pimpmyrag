#!/usr/bin/env python3
"""Analyse les metriques fine par label depuis W&B."""
from dotenv import dotenv_values
import wandb

key = dotenv_values(".secrets.env", encoding='utf-8').get("WANDB_API_KEY")
api = wandb.Api(api_key=key)
runs = list(api.runs("pimpmyrag-ner", order="created_at"))

# Dernier run avec metriques fines valides
best_run, last = None, {}
for r in reversed(runs):
    hist = list(r.scan_history(page_size=5))
    if hist and hist[-1].get('val/fine_f1', 0) > 0.5:
        best_run, last = r, hist[-1]
        break

if not best_run:
    print("Pas de run avec metriques fines > 0.5")
    exit()

ep = last.get('epoch', '?')
print(f"Run: {best_run.name}  ep={ep}")
print(f"  val/fine_f1   = {last.get('val/fine_f1', float('nan')):.4f}")
print(f"  val/coarse_f1 = {last.get('val/coarse_f1', float('nan')):.4f}")
print(f"  val/boundary_f1 = {last.get('val/boundary_f1', float('nan')):.4f}")

# W&B ne logue que les macros — pas de per-label dans les runs actuels
# On compare avec l'evolution historique
print("\nEvolution val/fine_f1 par epoch (tous runs):")
print(f"  {'ep':>4}  {'fine_f1':>9}  {'coarse_f1':>10}  {'score':>8}")
for r in runs:
    hist = list(r.scan_history(page_size=5))
    if not hist:
        continue
    h = hist[-1]
    ep_r = h.get('epoch')
    ff = h.get('val/fine_f1')
    cf = h.get('val/coarse_f1')
    sc = h.get('score')
    if ep_r and ff and ff > 0:
        print(f"  {ep_r:>4}  {ff:>9.4f}  {cf:>10.4f}  {sc:>8.4f}")

