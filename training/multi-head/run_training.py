#!/usr/bin/env python3
"""
run_training.py — Boucle adaptative pimpmyrag NER multitask.
Lit ses hyperparamètres dans un fichier JSON (--config).
Remplace run_adaptive_training.sh (logique en Python, plus lisible/maintenable).

Usage :
    python3 run_training.py --config configs/bndwarm-oblique.json
    python3 run_training.py --config configs/bndwarm-oblique.json --start-epoch 13 --keep-checkpoint
    python3 run_training.py --config configs/bndwarm-oblique.json --ner-only-bench
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────
#  Chargement config
# ─────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    raw = Path(path).read_text(encoding="utf-8")
    # Supprimer les clés _comment sans casser le JSON
    import re as _re
    raw = _re.sub(r'"_comment"\s*:\s*"[^"]*",?\n?', '', raw)
    return json.loads(raw)


# ─────────────────────────────────────────────────────
#  Détection hardware
# ─────────────────────────────────────────────────────

def detect_hw(cfg: dict) -> dict:
    hw_cfg = cfg["hardware"]
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no cuda")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🖥️  GPU : {gpu_name} ({vram_gb:.1f} GB VRAM)")

        if vram_gb >= 70:
            hw = {"device": "cuda", "gpu_name": gpu_name, **hw_cfg["h100_80gb"]}
            print(f"🔥 H100/A100-80GB détecté → BS={hw['bs']}, workers={hw['workers']}")
        elif vram_gb >= 40:
            hw = {"device": "cuda", "gpu_name": gpu_name, **hw_cfg["a100_40gb"]}
            print(f"⚡ A100-40GB/L40S détecté → BS={hw['bs']}")
        elif vram_gb >= 20:
            hw = {"device": "cuda", "gpu_name": gpu_name, **hw_cfg["rtx_4090"]}
            print(f"💪 RTX 4090/3090 détecté → BS={hw['bs']}, accum={hw['accum']}")
        else:
            hw = {"device": "cuda", "gpu_name": gpu_name, **hw_cfg["default"]}
            print(f"❓ GPU inconnu {vram_gb:.1f}GB → config default BS={hw['bs']}")

        # ── Auto-tuning BS (si AUTO_BS=1 ou si find_optimal_bs.py présent) ──────
        # Lance find_optimal_bs.py en subprocess pour trouver le BS maximal sans OOM.
        # Le résultat est mis en cache dans optimal_bs_cache.json (1 seul calcul par GPU).
        # Désactivable via AUTO_BS=0 (utile pour debug ou tests rapides).
        if os.environ.get("AUTO_BS", "1") != "0" and Path("find_optimal_bs.py").exists():
            bs_static = hw["bs"]
            # Si le cache existe mais contient une valeur < bs_static, il est obsolète → forcer recompute
            cache_path = Path("optimal_bs_cache.json")
            force_recompute = False
            if cache_path.exists():
                try:
                    cached_check = json.loads(cache_path.read_text())
                    if int(cached_check.get("bs", 0)) < bs_static:
                        print(f"⚠️  Cache auto-BS obsolète (bs={cached_check.get('bs')} < static={bs_static}) — recompute forcé")
                        cache_path.unlink()
                        force_recompute = True
                except Exception:
                    pass
            print(f"🔍 Auto-tuning batch size (BS statique={bs_static}, AUTO_BS=1)...")
            find_bs_cmd = [
                sys.executable, "find_optimal_bs.py",
                "--bs-min",   str(bs_static),           # jamais descendre sous la valeur statique
                "--bs-max",   str(int(bs_static * 2.5)),
                "--step",     "4",
                "--safety-margin", "8",
                "--output-file", "optimal_bs_cache.json",
                "--model-name", cfg.get("run", {}).get("model", "microsoft/deberta-v3-base"),
            ]
            if force_recompute:
                find_bs_cmd.append("--force")
            result = subprocess.run(
                find_bs_cmd,
                capture_output=False,  # laisse le stdout/stderr passer (visible dans tee)
                text=True,
            )
            # Lire le résultat depuis le cache
            if cache_path.exists():
                try:
                    cached = json.loads(cache_path.read_text())
                    opt_bs = int(cached["bs"])
                    if opt_bs > bs_static:
                        print(f"✅ Auto-BS : {bs_static} → {opt_bs} (+{opt_bs - bs_static} samples/step)")
                    elif opt_bs == bs_static:
                        print(f"✅ Auto-BS : BS={opt_bs} confirmé (déjà optimal)")
                    else:
                        # Ne devrait plus arriver (bs-min = bs_static), mais garde-fou
                        print(f"⚠️  Auto-BS retourné {opt_bs} < static {bs_static} — BS statique conservé")
                        opt_bs = bs_static
                    hw["bs"] = opt_bs
                except Exception as e:
                    print(f"⚠️  Impossible de lire le cache auto-BS : {e} — BS statique conservé ({bs_static})")
            else:
                print(f"⚠️  Cache auto-BS absent — BS statique conservé ({bs_static})")
        # ─────────────────────────────────────────────────────────────────────────

    except Exception:
        try:
            import torch
            if torch.backends.mps.is_available():
                hw = {"device": "mps", "gpu_name": "MPS", "bs": 24, "accum": 2, "workers": 0}
                print("🍎 Device MPS")
            else:
                raise RuntimeError
        except Exception:
            hw = {"device": "cpu", "gpu_name": "CPU", "bs": 16, "accum": 2, "workers": 0}
            print("💻 Device CPU")
    return hw


# ─────────────────────────────────────────────────────
#  Calcul ramps lambdas — CASCADE SVO-FIRST
# ─────────────────────────────────────────────────────
#
# Architecture en 4 phases metric-driven :
#
#   PHASE 1 (toujours) : NER boundary + SVO boundary à plein régime
#                        + role_coarse démarre en warmup (λ bas, labels gold)
#   PHASE 2 (svo_bnd_f1 > seuil) : role_coarse rampe vers λ plein
#                                   voice + verb_ptr + certainty rampent
#   PHASE 3 (ner_bnd_f1 > seuil) : NER coarse + fine rampent (boundary-first)
#   PHASE 4 (role_coarse_f1 > seuil) : role_oblique + morpho rampent
#
# Pourquoi role_coarse démarre en Phase 1 ?
# Les labels role_coarse viennent du JSONL gold (pas de prédictions modèle).
# Bloquer role_coarse jusqu'à svo_bnd_f1>0.85 gâche des epochs d'apprentissage.
# Le seuil svo_bnd déclenche la montée en puissance, pas le démarrage.

def compute_lambdas(cfg: dict, state: dict) -> dict:
    """Calcule les lambdas effectifs pour l'epoch courante selon la cascade SVO-first."""
    L  = cfg["lambdas"]
    bf = cfg["boundary_first"]
    epoch = state["epoch"]

    # ── NER only bench → tout SVO à zéro ─────────────────────────────────────
    if state.get("ner_only_bench"):
        return {
            "boundary":     state.get("l_boundary", L["boundary"]),
            "coarse":       L["coarse"],
            "fine":         L["fine"],
            "svo_boundary": 0.0, "svo": 0.0,
            "role_coarse":  0.0, "role_oblique": 0.0,
            "voice": 0.0, "certainty": 0.0, "morpho": 0.0,
            "verb_ptr": 0.0, "compat": 0.0,
        }

    # ── PHASE 1 : Fondations (toujours actives) ─────────────────────────────
    # NER boundary et SVO boundary à plein régime dès epoch 1.
    l_boundary     = state.get("l_boundary", L["boundary"])
    l_svo_boundary = L["svo_boundary"]  # PAS rampé — toujours à plein

    # ── PHASE 2 : SVO roles (triggered par svo_bnd_f1) ──────────────────────
    # role_coarse démarre en warmup (Phase 1) avec un λ réduit.
    # Quand svo_bnd_f1 >= seuil, on rampe vers le λ plein (Phase 2).
    sc = cfg.get("svo_cascade", {})
    # Lambda warmup role_coarse (actif dès epoch 1, avant Phase 2)
    rc_warmup = sc.get("role_coarse_warmup_lambda", L["role_coarse"] * 0.15)

    phase2_trigger_ep = state.get("phase2_trigger_epoch")  # epoch où phase 2 a déclenché
    if phase2_trigger_ep is not None:
        ramp_ep = epoch - phase2_trigger_ep
        ramp_total = sc.get("role_ramp_epochs", 8)
        phase2_prog = min(1.0, max(0.0, ramp_ep / ramp_total))
    else:
        phase2_prog = 0.0

    # role_coarse : warmup (Phase 1) → plein régime (Phase 2 ramp)
    l_role_coarse = rc_warmup + (L["role_coarse"] - rc_warmup) * phase2_prog
    l_role_coarse = round(l_role_coarse, 4)

    l_svo         = round(L["svo"]          * phase2_prog, 4)
    l_voice       = round(state.get("l_voice", L["voice"]) * phase2_prog, 4)
    l_certainty   = round(L["certainty"]    * phase2_prog, 4)
    l_verb_ptr    = round(state.get("l_verb_ptr", L["verb_ptr"]) * phase2_prog, 4)

    # ── PHASE 3 : NER classification (triggered par ner_bnd_f1) ──────────────
    # Quand NER boundary est fiable → on commence coarse + fine NER.
    # (identique à l'ancien boundary_first, indépendant de SVO)
    if not bf["enabled"] or state.get("coarse_fine_unlocked"):
        unlock_ep = state.get("coarse_fine_unlock_epoch", 0)
        cf_ramp   = epoch - unlock_ep
        cf_prog   = min(1.0, max(0.0, cf_ramp / bf["unlock_ramp_epochs"])) if bf["enabled"] else 1.0
        l_coarse  = bf["coarse_warmup_lambda"] + (L["coarse"] - bf["coarse_warmup_lambda"]) * cf_prog
        l_fine    = bf["fine_warmup_lambda"]   + (L["fine"]   - bf["fine_warmup_lambda"])   * cf_prog
    else:
        l_coarse = bf["coarse_warmup_lambda"]
        l_fine   = bf["fine_warmup_lambda"]

    # ── PHASE 4 : SVO fine (triggered par role_coarse_f1) ────────────────────
    # Quand role_coarse distingue bien SUBJ/OBJ/OBLIQ → on affine les obliques.
    phase4_trigger_ep = state.get("phase4_trigger_epoch")
    if phase4_trigger_ep is not None:
        ramp_ep = epoch - phase4_trigger_ep
        ramp_total = sc.get("oblique_ramp_epochs", 10)
        phase4_prog = min(1.0, max(0.0, ramp_ep / ramp_total))
    else:
        phase4_prog = 0.0

    l_role_oblique = round(L["role_oblique"] * phase4_prog, 4)
    l_morpho       = round(L["morpho"]       * phase4_prog, 4)

    return {
        "boundary":     l_boundary,
        "coarse":       round(l_coarse, 4),
        "fine":         round(l_fine, 4),
        "svo_boundary": l_svo_boundary,
        "svo":          l_svo,
        "role_coarse":  l_role_coarse,
        "role_oblique": l_role_oblique,
        "voice":        l_voice,
        "certainty":    l_certainty,
        "morpho":       l_morpho,
        "verb_ptr":     l_verb_ptr,
        "compat":       L.get("compat", 0.0),
    }


# ─────────────────────────────────────────────────────
#  Extraction métriques depuis le log epoch
# ─────────────────────────────────────────────────────

def extract_metric(log: str, key: str, prefix: str = "Val") -> float | None:
    pattern = rf"{key}=([\d.]+)"
    for line in reversed(log.splitlines()):
        if prefix in line:
            m = re.search(pattern, line)
            if m:
                return float(m.group(1))
    # fallback global
    matches = re.findall(pattern, log)
    return float(matches[-1]) if matches else None


# ─────────────────────────────────────────────────────
#  Appel train_multi_task.py (1 epoch)
# ─────────────────────────────────────────────────────

def run_epoch(cfg: dict, hw: dict, state: dict, lambdas: dict,
              gold_version: str, log_dir: Path, epoch: int) -> str:
    data_dir    = Path("data")
    suffix      = state.get("run_suffix", "adaptive")
    train_file  = data_dir / f"train.{suffix}.multitask.jsonl"
    val_file    = data_dir / "val.multitask.jsonl"
    test_file   = data_dir / "test.multitask.jsonl"
    f_cfg       = cfg["focal"]
    hn          = cfg["hard_negatives"]
    opt         = cfg["optimizer"]
    run_cfg     = cfg["run"]
    in_warmup   = state.get("in_warmup", False)

    _bool = lambda v: ["", str(v)][bool(v)]

    cmd = [
        "python3", "-u", "train_multi_task.py",
        "--train", str(train_file),
        "--val",   str(val_file),
        "--test",  str(test_file),
        "--model-name", run_cfg.get("model", "microsoft/deberta-v3-base"),
        "--epochs",      str(epoch),
        "--start-epoch", str(epoch),
        "--patience", "0",
        "--batch-size",   str(hw["bs"]),
        "--accum-steps",  str(hw["accum"]),
        "--num-workers",  str(hw["workers"]),
        "--lr",                   str(opt["lr"]),
        "--head-lr-multiplier",   str(opt["head_lr_multiplier"]),
        "--warmup-epochs",        str(opt["warmup_epochs"]),
        "--max-grad-norm",        str(opt["max_grad_norm"]),
        "--layer-lr-decay",       str(opt["layer_lr_decay"]),
        "--ema-decay",            str(opt["ema_decay"]),
        "--class-weight-power",   str(opt["class_weight_power"]),
        "--lambda-boundary",   str(lambdas["boundary"]),
        "--lambda-coarse",     str(lambdas["coarse"]),
        "--lambda-fine",       str(lambdas["fine"]),
        "--lambda-svo-boundary",  str(lambdas["svo_boundary"]),
        "--lambda-svo",           str(lambdas["svo"]),
        "--lambda-role-coarse",   str(lambdas["role_coarse"]),
        "--lambda-role-oblique",  str(lambdas["role_oblique"]),
        "--lambda-voice",         str(lambdas["voice"]),
        "--lambda-certainty",     str(lambdas["certainty"]),
        "--lambda-morpho",        str(lambdas["morpho"]),
        "--lambda-verb-ptr",      str(lambdas["verb_ptr"]),
        "--lambda-compat",        str(lambdas["compat"]),
        "--focal-gamma",          str(f_cfg["boundary_gamma"]),
        "--focal-fine-gamma",     str(f_cfg["fine_gamma"]),
        "--focal-coarse-gamma",   str(f_cfg["coarse_gamma"]),
        "--device", hw["device"],
        "--hn-every", str(hn["every_n_epochs"]),
        "--hn-decay",        str(hn["decay"]),
        "--hn-max-weight",   str(hn["max_weight"]),
        "--hn-min-weight",   str(hn["min_weight"]),
        "--hn-boost-fp",     str(hn["boost_fp"]),
        "--hn-boost-fn",     str(hn["boost_fn"]),
        "--hn-boost-coarse", str(hn["boost_coarse"]),
        "--hn-boost-fine",   str(hn["boost_fine"]),
        "--hn-boost-fp-svo", "0.0" if in_warmup or state.get("ner_only_bench") else str(hn["boost_fp_svo"]),
        "--hn-boost-fn-svo", "0.0" if in_warmup or state.get("ner_only_bench") else str(hn["boost_fn_svo"]),
        "--hn-boost-role-coarse", str(hn["boost_role_coarse"]),
        "--loss-weighting", run_cfg.get("loss_weighting", "fixed"),
        "--ignore-coarse-none",
        "--amp",
        "--wandb-run-name",  state["wandb_run_name"],
        "--wandb-tags",      state["wandb_tags"],
        "--wandb-id-file",   "wandb_run_id.txt",
    ]
    if state.get("ner_only_bench"):
        cmd.append("--ner-only-score")
    if state.get("resume_arg"):
        cmd += ["--resume", state["resume_arg"]]

    log_path = log_dir / f"epoch_{epoch}.log"
    print(f"\n{'═'*50}")
    print(f"  Epoch {epoch}/{run_cfg['max_epochs']}  |  Niveau {state.get('level_name','?')} "
          f"(stagnation={state.get('stagnation',0)}/{cfg['early_stopping']['patience']})")
    print(f"{'═'*50}")
    print(f"🎛️  Lambdas : bnd={lambdas['boundary']}  crs={lambdas['coarse']}  fin={lambdas['fine']}")
    print(f"             svo_b={lambdas['svo_boundary']}  svo={lambdas['svo']}  "
          f"rc={lambdas['role_coarse']}  ro={lambdas['role_oblique']}")
    log_path = log_dir / f"epoch_{epoch}.log"
    # ...existing code (prints lambdas)...
    sys.stdout.flush()

    # Popen + lecture ligne-par-ligne : écrit dans fichier ET stdout
    # Pas de deadlock car on consomme le buffer en continu (vs communicate())
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,  # line-buffered
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
        proc.wait()

    output = log_path.read_text(encoding="utf-8", errors="replace")

    return output


# ─────────────────────────────────────────────────────
#  Rebuild dataset (wrapper autour de build_multitask_dataset.py)
# ─────────────────────────────────────────────────────

def rebuild_dataset(level: int, cfg: dict, gold_version: str, state: dict):
    levels   = cfg["difficulty_levels"]
    hard = levels["hard_per_gold"][level]
    soft = levels["soft_factors"][level]
    name = levels["names"][level]
    suffix = state.get("run_suffix", "adaptive")
    print(f"\n📦 Rebuild dataset niveau {name} (hard={hard}, soft={soft})")

    # Build train adaptatif (hard negatives selon niveau)
    subprocess.run([
        "python3", "build_multitask_dataset.py",
        "--input",         f"data/train_{gold_version}.jsonl",
        "--output",        f"data/train.{suffix}.multitask.jsonl",
        "--hard-per-gold", str(hard),
        "--soft-factor",   str(soft),
    ], check=True)

    # Build val + test (niveau fixe : hard=2, soft=1.0 — évaluation stable)
    for split in ["val", "test"]:
        subprocess.run([
            "python3", "build_multitask_dataset.py",
            "--input",  f"data/{split}_{gold_version}.jsonl",
            "--output", f"data/{split}.multitask.jsonl",
            "--hard-per-gold", "2",
            "--soft-factor",   "1.0",
        ], check=True)


# ─────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Boucle adaptative pimpmyrag NER multitask")
    p.add_argument("--config",         required=True,  help="Chemin vers le fichier JSON de config")
    p.add_argument("--gold-version",   default=None,   help="Override GOLD_VERSION (ex: v8.18)")
    p.add_argument("--start-epoch",    type=int, default=1)
    p.add_argument("--start-level",    type=int, default=0)
    p.add_argument("--keep-checkpoint", action="store_true")
    p.add_argument("--ner-only-bench",  action="store_true")
    p.add_argument("--device",          default=None, help="Forcer cuda/mps/cpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    gold_version = args.gold_version or os.environ.get("GOLD_VERSION", "v8.18")
    print(f"📋 Config : {args.config}")
    print(f"📦 Dataset : {gold_version}")

    # Hardware
    hw = detect_hw(cfg)
    if args.device:
        hw["device"] = args.device

    # ID GPU court pour le nom de run
    gpu_short = hw["gpu_name"].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "").replace(" ", "_")
    from datetime import datetime
    ts = datetime.now().strftime("%m%d-%H%M")
    run_suffix   = cfg["run"].get("name_suffix", "adaptive")
    wandb_run_name = f"{gold_version}-{run_suffix}-deberta-bs{hw['bs']}-{gpu_short}-{ts}"
    wandb_tags     = f"{gold_version},deberta-v3,fp32,adaptive"
    if args.ner_only_bench:
        wandb_run_name += "-neronly"
        wandb_tags     += ",ner-only"

    # Gestion checkpoint de reprise
    resume_arg = ""
    if args.keep_checkpoint and Path("checkpoint_best_multitask.pt").exists():
        import torch
        ckpt = torch.load("checkpoint_best_multitask.pt", map_location="cpu")
        ckpt_fine = ckpt.get("model_state", {})
        # Vérification compatibilité fine_head (38 labels attendus)
        fine_key = next((k for k in ckpt_fine if "fine_head.weight" in k), None)
        if fine_key and ckpt_fine[fine_key].shape[0] != 38:
            print(f"⚠️  Checkpoint incompatible (fine={ckpt_fine[fine_key].shape[0]} ≠ 38) → démarrage froid")
        else:
            resume_arg = "checkpoint_best_multitask.pt"
            print(f"♻️  Reprise depuis checkpoint (score={ckpt.get('best_score', -1):.4f})")
    elif not args.keep_checkpoint:
        for f in ["checkpoint_best_multitask.pt", "checkpoint_last_multitask.pt", "wandb_run_id.txt"]:
            Path(f).unlink(missing_ok=True)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_main = log_dir / "training.log"

    def log(msg):
        print(msg)
        with open(log_main, "a") as f:
            f.write(msg + "\n")

    log(f"🚀 Démarrage training adaptatif — {datetime.now().isoformat()}")

    # Traçabilité : git SHA + config dump
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip() or "?"
    git_msg = subprocess.run(["git", "log", "--oneline", "-1"],
                             capture_output=True, text=True).stdout.strip() or "?"
    log(f"📌 Git : {git_sha}  ({git_msg})")
    log(f"   Config : {args.config}  |  Dataset : {gold_version}  |  GPU : {hw['gpu_name']}")
    log(f"   Lambdas : {json.dumps(cfg['lambdas'], indent=None)}")

    # État adaptatif
    levels = cfg["difficulty_levels"]
    # Compat : ancienne config (svo_trigger) → fields ignorés si svo_cascade présent
    use_svo_cascade = "svo_cascade" in cfg
    state = {
        "epoch":                 args.start_epoch,
        "level":                 args.start_level,
        "level_name":            levels["names"][args.start_level],
        "stagnation":            0,
        "boundary_stagnation":   0,
        "epochs_at_level":       0,
        "best_score":            -1.0,
        "best_boundary":         -1.0,
        "best_svo_bnd":          -1.0,
        "best_role_coarse":      -1.0,
        "boundary_window":       [],
        "rescue_window":         [],
        "rescue_applied":        False,
        "regression_count":      0,
        "coarse_fine_unlocked":  not cfg["boundary_first"]["enabled"],
        "coarse_fine_unlock_epoch": 0,
        # Cascade SVO-first (phases 2 & 4)
        "phase2_trigger_epoch":  None,  # SVO roles → triggered par svo_bnd_f1
        "phase4_trigger_epoch":  None,  # SVO fine  → triggered par role_coarse_f1
        "l_boundary":            cfg["lambdas"]["boundary"],
        "l_voice":               cfg["lambdas"]["voice"],
        "l_verb_ptr":            cfg["lambdas"]["verb_ptr"],
        "ner_only_bench":        args.ner_only_bench,
        "run_suffix":            run_suffix,
        "resume_arg":            resume_arg,
        "wandb_run_name":        wandb_run_name,
        "wandb_tags":            wandb_tags,
    }

    rebuild_dataset(state["level"], cfg, gold_version, state)

    es = cfg["early_stopping"]
    bp = cfg["boundary_plateau"]
    br = cfg["boundary_rescue"]
    bf = cfg["boundary_first"]
    sc = cfg.get("svo_cascade", {})
    max_epochs = cfg["run"]["max_epochs"]

    # Log architecture cascade
    if use_svo_cascade:
        rc_warmup = sc.get("role_coarse_warmup_lambda", cfg["lambdas"]["role_coarse"] * 0.15)
        log(f"🏗️  Architecture : CASCADE SVO-FIRST")
        log(f"   Phase 1 : NER bnd + SVO bnd à plein régime + role_coarse warmup λ={rc_warmup}")
        log(f"   Phase 2 : role_coarse → λ plein + voice + verb_ptr → quand svo_bnd_f1 > {sc.get('role_thr_svo_bnd', '?')}")
        log(f"   Phase 3 : NER coarse + fine → quand ner_bnd > {bf['unlock_threshold']}")
        log(f"   Phase 4 : role_oblique + morpho → quand role_coarse_f1 > {sc.get('oblique_thr_role_crs', '?')}")

    # ─── BOUCLE PRINCIPALE ───────────────────────────────────────────
    while state["epoch"] <= max_epochs:
        epoch = state["epoch"]

        lambdas = compute_lambdas(cfg, state)

        # Log cascade status
        p2 = "🟢" if state["phase2_trigger_epoch"] is not None else "⏳"
        p3 = "🟢" if state["coarse_fine_unlocked"] else "⏳"
        p4 = "🟢" if state["phase4_trigger_epoch"] is not None else "⏳"
        log(f"      Cascade : P1=🟢  P2(roles)={p2}  P3(ner_cls)={p3}  P4(oblique)={p4}")

        # Lancer l'epoch
        output = run_epoch(cfg, hw, state, lambdas, gold_version, log_dir, epoch)

        # Extraire métriques
        val_score    = extract_metric(output, "Score")
        boundary_f1  = extract_metric(output, "Boundary F1")
        coarse_f1    = extract_metric(output, "Coarse")
        voice_f1     = extract_metric(output, "Voice F1")
        svo_bnd_f1   = extract_metric(output, "SVO Bnd F1")
        role_crs_f1  = extract_metric(output, "Role Crs F1")

        if val_score is None:
            log(f"⚠️  Impossible d'extraire le val score ep{epoch} — on continue")
            state["resume_arg"] = "checkpoint_best_multitask.pt" if Path("checkpoint_best_multitask.pt").exists() else ""
            state["epoch"] += 1
            continue

        state["resume_arg"] = "checkpoint_best_multitask.pt" if Path("checkpoint_best_multitask.pt").exists() else ""

        log(f"📊 Ep {epoch} — Score={val_score:.4f}  NER_Bnd={boundary_f1}  Coarse={coarse_f1}  "
            f"SVO_Bnd={svo_bnd_f1}  RoleCrs={role_crs_f1}  Voice={voice_f1}  (best={state['best_score']:.4f})")

        # ── Score stagnation ──────────────────────────────────────────────────
        if val_score > state["best_score"] + es["min_delta"]:
            state["best_score"] = val_score
            state["stagnation"] = 0
            log(f"✅ Amélioration! best_score={val_score:.4f}")
        else:
            state["stagnation"] += 1
            log(f"⏸️  Pas d'amélioration ({state['stagnation']}/{es['patience']})")

        # ── NER Boundary tracking ────────────────────────────────────────────
        if boundary_f1 is not None:
            if boundary_f1 > state["best_boundary"] + es["min_delta"]:
                state["best_boundary"] = boundary_f1

            # Fenêtre glissante boundary
            state["boundary_window"].append(boundary_f1)
            if len(state["boundary_window"]) > bp["window"]:
                state["boundary_window"].pop(0)

            if len(state["boundary_window"]) >= bp["window"]:
                gain = boundary_f1 - state["boundary_window"][0]
                if gain < bp["min_delta"]:
                    state["boundary_stagnation"] += 1
                else:
                    state["boundary_stagnation"] = 0

            # ── PHASE 3 : NER coarse/fine unlock (boundary-first) ───────────
            if cfg["boundary_first"]["enabled"] and not state["coarse_fine_unlocked"]:
                if boundary_f1 >= bf["unlock_threshold"]:
                    state["coarse_fine_unlocked"] = True
                    state["coarse_fine_unlock_epoch"] = epoch
                    log(f"🔓 PHASE 3 — NER Bnd {boundary_f1:.4f} ≥ {bf['unlock_threshold']} "
                        f"→ UNLOCK coarse+fine (ramp {bf['unlock_ramp_epochs']}ep)")

            # ── Rescue regression boundary ──────────────────────────────────
            if not state["rescue_applied"]:
                drop = state["best_boundary"] - boundary_f1
                if drop >= bp["regression_delta"]:
                    state["regression_count"] += 1
                    if state["regression_count"] >= bp["regression_window"]:
                        state["rescue_applied"]  = True
                        state["l_boundary"] = round(state["l_boundary"] * br["bnd_boost"], 4)
                        cfg["lambdas"]["coarse"]  = round(cfg["lambdas"]["coarse"]  * br["factor_coarse"], 4)
                        state["l_verb_ptr"] = round(state["l_verb_ptr"] * br["factor_vptr_voice"], 4)
                        state["l_voice"]    = round(state["l_voice"]    * br["factor_vptr_voice"], 4)
                        log(f"🚨 RESCUE ep{epoch} — bnd chute {bp['regression_window']}ep depuis {state['best_boundary']:.4f}"
                            f" → L_BND={state['l_boundary']} L_COARSE={cfg['lambdas']['coarse']:.4f}")
                        state["regression_count"] = 0
                else:
                    state["regression_count"] = 0

                # NER rescue plateau
                state["rescue_window"].append(boundary_f1)
                if len(state["rescue_window"]) > br["window"]:
                    state["rescue_window"].pop(0)
                if len(state["rescue_window"]) >= br["window"] and not state["rescue_applied"]:
                    gain = boundary_f1 - state["rescue_window"][0]
                    if boundary_f1 < br["target"] and gain < br["min_delta"]:
                        state["rescue_applied"]  = True
                        state["l_boundary"] = round(state["l_boundary"] * br["bnd_boost"], 4)
                        cfg["lambdas"]["coarse"]  = round(cfg["lambdas"]["coarse"]  * br["factor_coarse"], 4)
                        state["l_verb_ptr"] = round(state["l_verb_ptr"] * br["factor_vptr_voice"], 4)
                        state["l_voice"]    = round(state["l_voice"]    * br["factor_vptr_voice"], 4)
                        log(f"🚨 NER RESCUE ep{epoch} — boundary {boundary_f1:.4f}<{br['target']} "
                            f"stagne (+{gain:.4f}/{br['window']}ep)")

        # ── PHASE 2 : SVO roles trigger (svo_bnd_f1 > seuil) ────────────────
        if use_svo_cascade and svo_bnd_f1 is not None:
            if svo_bnd_f1 > state["best_svo_bnd"]:
                state["best_svo_bnd"] = svo_bnd_f1
            if state["phase2_trigger_epoch"] is None:
                thr = sc["role_thr_svo_bnd"]
                if svo_bnd_f1 >= thr:
                    state["phase2_trigger_epoch"] = epoch
                    log(f"🎯 PHASE 2 — SVO Bnd {svo_bnd_f1:.4f} ≥ {thr} → UNLOCK role_coarse + voice + verb_ptr "
                        f"(ramp {sc['role_ramp_epochs']}ep)")

        # ── PHASE 4 : SVO fine trigger (role_coarse_f1 > seuil) ──────────────
        if use_svo_cascade and role_crs_f1 is not None:
            if role_crs_f1 > state["best_role_coarse"]:
                state["best_role_coarse"] = role_crs_f1
            if state["phase4_trigger_epoch"] is None and state["phase2_trigger_epoch"] is not None:
                thr = sc["oblique_thr_role_crs"]
                if role_crs_f1 >= thr:
                    state["phase4_trigger_epoch"] = epoch
                    log(f"🎯 PHASE 4 — Role Crs {role_crs_f1:.4f} ≥ {thr} → UNLOCK role_oblique + morpho "
                        f"(ramp {sc['oblique_ramp_epochs']}ep)")

        # ── Changement de niveau de difficulté ───────────────────────────────
        state["epochs_at_level"] += 1
        needs_level_up = (
            (
                state["stagnation"] >= es["patience"]
                or state["boundary_stagnation"] >= es["patience"]
                or state["epochs_at_level"] >= levels["max_epochs_per_level"]
            )
            and state["level"] < len(levels["names"]) - 1
        )
        if needs_level_up:
            state["level"] += 1
            state["level_name"] = levels["names"][state["level"]]
            state["stagnation"] = 0
            state["boundary_stagnation"] = 0
            state["epochs_at_level"] = 0
            log(f"⬆️  Passage au niveau {state['level_name']} (level {state['level']})")
            rebuild_dataset(state["level"], cfg, gold_version, state)
        elif state["stagnation"] >= es["patience"] and state["level"] >= len(levels["names"]) - 1:
            log(f"🛑 Early stopping — plus aucune amélioration au niveau max")
            break

        state["epoch"] += 1

    log(f"\n✅ Training terminé — best_score={state['best_score']:.4f}  best_boundary={state['best_boundary']:.4f}"
        f"  best_svo_bnd={state['best_svo_bnd']:.4f}  best_role_coarse={state['best_role_coarse']:.4f}")


if __name__ == "__main__":
    main()

