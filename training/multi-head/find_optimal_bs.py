#!/usr/bin/env python3
"""
find_optimal_bs.py — Détecte automatiquement le batch size maximal sans OOM.

Principe : binary search entre bs_min et bs_max en faisant un vrai
forward + backward pass avec le modèle multitask complet (DeBERTa-v3-base +
toutes les têtes). Teste chaque taille 1x (suffisant, l'allocation est
déterministe en BF16).

Usage :
    python3 find_optimal_bs.py --bs-min 64 --bs-max 256 --safety-margin 4
    → Affiche sur stdout une seule ligne : "OPTIMAL_BS=<N>"
    → Peut aussi écrire dans --output-file pour que detect_hw() puisse lire

Cette approche garantit que le BS correspond exactement au matériel réel
(VRAM disponible, model weights, têtes SVO, etc.).
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def make_fake_batch(bs: int, seq_len: int, num_candidates: int, device):
    """Génère un batch synthétique représentatif du dataset multitask."""
    import torch
    from labels import (
        ROLE_COARSE_NONE_ID, ROLE_OBLIQUE_NONE_ID,
        VOICE_NONE_ID, CERTAINTY_NONE_ID,
    )

    N = bs * num_candidates  # total spans

    return {
        "input_ids":       torch.randint(0, 30000, (bs, seq_len), device=device),
        "attention_mask":  torch.ones(bs, seq_len, dtype=torch.long, device=device),
        "spans":           [[{"start": 1, "end": 3, "text": "x"} for _ in range(num_candidates)] for _ in range(bs)],
        "boundary_labels": torch.randint(0, 2, (N,), device=device),
        "coarse_labels":   torch.randint(0, 10, (N,), device=device),
        "fine_labels":     torch.randint(0, 38, (N,), device=device),
        "svo_boundary_labels": torch.zeros(N, dtype=torch.long, device=device),
        "syn_labels":      torch.full((N,), -1, device=device),
        "role_coarse_labels": torch.full((N,), ROLE_COARSE_NONE_ID, device=device),
        "role_oblique_labels": torch.full((N,), ROLE_OBLIQUE_NONE_ID, device=device),
        "voice_labels":    torch.full((N,), VOICE_NONE_ID, device=device),
        "certainty_labels": torch.full((N,), CERTAINTY_NONE_ID, device=device),
        "gender_labels":   torch.full((N,), 2, device=device),
        "number_labels":   torch.full((N,), 2, device=device),
        "person_labels":   torch.full((N,), 3, device=device),
        "gov_verb_labels": torch.full((N,), -1, device=device),
        "sample_weights":  torch.ones(N, device=device),
    }


def try_bs(bs: int, model, optimizer, device, seq_len: int, cands: int) -> bool:
    """
    Essaie un forward+backward avec BS=bs.
    Retourne True si ça tient en VRAM, False si OOM.
    """
    import torch
    torch.cuda.empty_cache()
    gc.collect()
    try:
        batch = make_fake_batch(bs, seq_len, cands, device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model({
                "input_ids":      batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "spans":          batch["spans"],
            })
            loss_dict = model.compute_loss(
                outputs=outputs,
                boundary_labels=batch["boundary_labels"],
                coarse_labels=batch["coarse_labels"],
                fine_labels=batch["fine_labels"],
                svo_boundary_labels=batch["svo_boundary_labels"],
                syn_labels=batch["syn_labels"],
                role_coarse_labels=batch["role_coarse_labels"],
                role_oblique_labels=batch["role_oblique_labels"],
                voice_labels=batch["voice_labels"],
                certainty_labels=batch["certainty_labels"],
                gender_labels=batch["gender_labels"],
                number_labels=batch["number_labels"],
                person_labels=batch["person_labels"],
                gov_verb_labels=batch["gov_verb_labels"],
                sample_weights=batch["sample_weights"],
                lambda_boundary=5.0,
                lambda_coarse=0.75,
                lambda_fine=1.8,
                lambda_svo_boundary=0.35,
                lambda_svo=0.70,
                lambda_role_coarse=0.35,
                lambda_role_oblique=0.25,
                lambda_voice=0.15,
                lambda_certainty=0.30,
                lambda_morpho=0.10,
                lambda_verb_ptr=0.60,
            )
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.zero_grad(set_to_none=True)

        used = torch.cuda.max_memory_allocated(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        pct = used / total * 100
        print(f"  BS={bs:4d} → {used:.2f}/{total:.1f} GB ({pct:.1f}%) ✅", flush=True)
        return True

    except torch.cuda.OutOfMemoryError:
        print(f"  BS={bs:4d} → OOM ❌", flush=True)
        torch.cuda.empty_cache()
        gc.collect()
        return False
    except Exception as e:
        print(f"  BS={bs:4d} → erreur inattendue: {e}", flush=True)
        return False


def find_bs(args) -> int:
    import torch
    from transformers import AutoTokenizer
    from multitask_model import SpanMultiTaskModel

    device = "cuda"
    torch.backends.cudnn.benchmark = False  # reproductible
    torch.cuda.empty_cache()

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🖥️  {gpu_name} — {vram_gb:.1f} GB VRAM", flush=True)
    print(f"🔍 Binary search batch size [{args.bs_min}, {args.bs_max}] (pas de {args.step})", flush=True)

    # Charger le modèle complet (avec toutes les têtes SVO)
    print(f"⏳ Chargement {args.model_name}...", flush=True)
    t0 = time.time()
    model = SpanMultiTaskModel(
        model_name=args.model_name,
        span_hidden_dim=768,
    ).to(device)
    model.train()
    print(f"   Modèle chargé en {time.time()-t0:.1f}s", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-6)

    # Pré-génération tokenizer pour seq_len typique
    # (on utilise seq_len=256 qui correspond aux phrases NER courtes-moyennes)
    seq_len = args.seq_len
    cands = args.candidates_per_sample

    # Binary search
    lo, hi = args.bs_min, args.bs_max
    best = lo  # toujours safe

    # Vérifie d'abord bs_min (ne devrait jamais OOM)
    print(f"\n--- Test bs_min={lo} ---", flush=True)
    if not try_bs(lo, model, optimizer, device, seq_len, cands):
        print(f"⚠️  Même bs_min={lo} OOM — utilisation bs={lo}", flush=True)
        print(f"OPTIMAL_BS={lo}")
        return lo

    # Test bs_max pour voir si on est déjà safe tout en haut
    print(f"\n--- Test bs_max={hi} ---", flush=True)
    if try_bs(hi, model, optimizer, device, seq_len, cands):
        best = hi
    else:
        # Binary search entre lo et hi
        while lo + args.step <= hi:
            mid = (lo + hi) // 2
            # Arrondi au multiple de step le plus proche
            mid = (mid // args.step) * args.step
            if mid <= lo:
                break
            print(f"\n--- Test BS={mid} ---", flush=True)
            if try_bs(mid, model, optimizer, device, seq_len, cands):
                best = mid
                lo = mid
            else:
                hi = mid - args.step

    # Appliquer margin de sécurité
    optimal = max(args.bs_min, best - args.safety_margin)
    # Arrondir au multiple de 4 inférieur (meilleur pour les tensors CUDA)
    optimal = (optimal // 4) * 4
    optimal = max(args.bs_min, optimal)

    used_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"\n✅ Optimal BS = {optimal}  (max safe = {best}, margin = -{args.safety_margin})", flush=True)
    print(f"   VRAM estimée : ~{used_gb:.1f} GB / {total_gb:.1f} GB", flush=True)
    print(f"OPTIMAL_BS={optimal}")
    return optimal


def main():
    p = argparse.ArgumentParser(description="Trouve le batch size optimal pour le GPU courant")
    p.add_argument("--model-name",  default="microsoft/deberta-v3-base")
    p.add_argument("--bs-min",      type=int, default=64)
    p.add_argument("--bs-max",      type=int, default=256)
    p.add_argument("--step",        type=int, default=8,
                   help="Pas du binary search (multiple de 4 recommandé)")
    p.add_argument("--safety-margin", type=int, default=8,
                   help="Soustrait N du max safe pour buffer OOM (défaut 8)")
    p.add_argument("--seq-len",     type=int, default=256,
                   help="Longueur de séquence de test (256 = typique NER)")
    p.add_argument("--candidates-per-sample", type=int, default=30,
                   help="Candidats spans par phrase dans le batch test")
    p.add_argument("--output-file", default=None,
                   help="Si fourni, écrit OPTIMAL_BS=N dans ce fichier JSON")
    p.add_argument("--force",       action="store_true",
                   help="Force re-calcul même si le cache existe")
    args = p.parse_args()

    # Cache : évite de refaire le calcul si déjà fait pour ce GPU + mêmes paramètres
    cache_file = Path("optimal_bs_cache.json") if args.output_file is None else Path(args.output_file)
    if not args.force and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                if (cached.get("gpu") == gpu_name
                        and cached.get("seq_len") == args.seq_len
                        and cached.get("candidates_per_sample") == args.candidates_per_sample):
                    bs = cached["bs"]
                    print(f"📋 Cache hit : {gpu_name} seq={args.seq_len} cands={args.candidates_per_sample} → BS={bs}", flush=True)
                    print(f"OPTIMAL_BS={bs}")
                    return
                elif cached.get("gpu") == gpu_name:
                    print(f"📋 Cache invalidé (seq/cands ont changé) — recompute", flush=True)
        except Exception:
            pass

    optimal = find_bs(args)

    # Écrire en cache
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
        cache_data = {"gpu": gpu_name, "bs": optimal, "model": args.model_name,
                      "seq_len": args.seq_len, "candidates_per_sample": args.candidates_per_sample}
        cache_file.write_text(json.dumps(cache_data, indent=2))
        print(f"💾 Cache écrit → {cache_file}", flush=True)
    except Exception as e:
        print(f"⚠️ Impossible d'écrire le cache : {e}", flush=True)


if __name__ == "__main__":
    main()

