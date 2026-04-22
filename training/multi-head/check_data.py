#!/usr/bin/env python3
"""
check_data.py
=============
Vérification pré-training : valide toutes les sources de données avant de lancer
run_adaptive_training.sh sur GPU (car ça coûte cher de découvrir un bug à epoch 3).

Checks :
  1. Existence des fichiers requis
  2. Format JSONL (JSON valide, champs id/text/spans)
  3. Labels connus (FINE2ID + ALL_SVO_LABELS)
  4. Cohérence offsets : text[start:end] == span["text"]  (quand "text" présent)
  5. Offsets dans les bornes du texte
  6. Métadonnées SVO obligatoires (voice sur svo_verb/svo_subject/svo_object/svo_iobj)
  7. Survie à la tokenisation DeBERTa (char→token span non None)
  8. Doublons d'IDs au sein d'un fichier et entre splits
  9. run_adaptive_training.sh : train_svo_silver.jsonl est-il branché ?
 10. Statistiques par label + ratio SVO vs NER
 11. Longueur texte (alerte si > 512 tokens DeBERTa)

Usage :
    cd training/multi-head
    python check_data.py [--data-dir data] [--tokenizer microsoft/deberta-v3-base]
                         [--max-check 2000] [--no-tokenizer]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# ─── couleurs terminal ────────────────────────────────────────────────────────
RED    = "\033[91m"
YLW    = "\033[93m"
GRN    = "\033[92m"
BLU    = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GRN}✓{RESET} {msg}")
def warn(msg): print(f"  {YLW}⚠{RESET}  {msg}")
def err(msg):  print(f"  {RED}✗{RESET} {msg}")
def info(msg): print(f"  {BLU}ℹ{RESET}  {msg}")
def section(msg): print(f"\n{BOLD}{'─'*60}\n  {msg}\n{'─'*60}{RESET}")


# ─── labels attendus ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
try:
    from labels import (
        FINE2ID, ALL_SVO_LABELS,
        VOICE2ID, GENDER2ID, NUMBER2ID,
    )
    LABELS_AVAILABLE = True
except ImportError:
    warn("labels.py introuvable — vérification des labels désactivée")
    FINE2ID = {}
    ALL_SVO_LABELS = {
        "svo_verb", "svo_subject", "svo_object", "svo_iobj", "pron_subj", "pron_obj"
    }
    VOICE2ID = {"ACTIVE": 0, "PASSIVE": 1}
    GENDER2ID = {"Masc": 0, "Fem": 1, "NONE": 2}
    NUMBER2ID = {"Sing": 0, "Plur": 1, "NONE": 2}
    LABELS_AVAILABLE = False

ALL_KNOWN_LABELS = set(FINE2ID.keys()) | ALL_SVO_LABELS

# Labels SVO pour lesquels `voice` est obligatoire
SVO_VOICE_REQUIRED = {"svo_verb", "svo_subject", "svo_object", "svo_iobj"}


# ─── helpers ─────────────────────────────────────────────────────────────────

def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as e:
                yield lineno, None  # signale l'erreur


def char_span_to_token_span(offsets, start: int, end: int) -> Optional[tuple]:
    tok_start = tok_end = None
    for i, (s, e) in enumerate(offsets):
        if e <= start:
            continue
        if s >= end:
            break
        if tok_start is None:
            tok_start = i
        tok_end = i
    if tok_start is None or tok_end is None:
        return None
    return tok_start, tok_end


# ─── check principal ──────────────────────────────────────────────────────────

def check_file(
    path: Path,
    label: str,
    tokenizer=None,
    max_check: int = 0,
    seen_ids_global: Optional[set] = None,
) -> dict:
    """
    Retourne un dict de stats + erreurs pour le fichier.
    """
    stats = {
        "file": path.name,
        "n_rows": 0,
        "n_spans": 0,
        "n_errors": 0,
        "n_warnings": 0,
        "label_counts": Counter(),
        "unknown_labels": Counter(),
        "offset_errors": 0,
        "tok_drop": 0,
        "tok_checked": 0,
        "long_texts": 0,
        "dup_ids": 0,
        "voice_missing": 0,
    }

    seen_ids_local: set[str] = set()
    errors: list[str] = []
    warnings_list: list[str] = []

    for lineno, row in iter_jsonl(path):
        if row is None:
            errors.append(f"l.{lineno}: JSON invalide")
            stats["n_errors"] += 1
            continue

        stats["n_rows"] += 1
        if max_check > 0 and stats["n_rows"] > max_check:
            break

        # ── champs requis ──────────────────────────────────────────────────
        if "id" not in row:
            errors.append(f"l.{lineno}: champ 'id' manquant")
            stats["n_errors"] += 1
        if "text" not in row:
            errors.append(f"l.{lineno}: champ 'text' manquant")
            stats["n_errors"] += 1
            continue
        if "spans" not in row:
            errors.append(f"l.{lineno}: champ 'spans' manquant")
            stats["n_errors"] += 1

        text = row["text"]
        uid  = str(row.get("id", f"__line_{lineno}__"))

        # ── doublons ──────────────────────────────────────────────────────
        if uid in seen_ids_local:
            stats["dup_ids"] += 1
            warnings_list.append(f"l.{lineno}: id dupliqué '{uid}' dans le fichier")
        seen_ids_local.add(uid)

        if seen_ids_global is not None and uid in seen_ids_global:
            stats["dup_ids"] += 1
            warnings_list.append(f"l.{lineno}: id '{uid}' présent dans un autre split")
        if seen_ids_global is not None:
            seen_ids_global.add(uid)

        # ── tokenisation DeBERTa (longueur) ───────────────────────────────
        offsets = None
        if tokenizer is not None:
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=False,
            )
            offsets = enc["offset_mapping"]
            n_tokens = len(enc["input_ids"])
            if n_tokens > 510:
                stats["long_texts"] += 1
                warnings_list.append(
                    f"l.{lineno}: texte long ({n_tokens} tokens DeBERTa) > 510"
                )

        # ── spans ─────────────────────────────────────────────────────────
        for sp_idx, sp in enumerate(row.get("spans", [])):
            stats["n_spans"] += 1
            lbl    = sp.get("label", "")
            start  = sp.get("start")
            end    = sp.get("end")
            sp_txt = sp.get("text")

            # Label connu ?
            if lbl not in ALL_KNOWN_LABELS:
                stats["unknown_labels"][lbl] += 1
                errors.append(f"l.{lineno} sp[{sp_idx}]: label inconnu '{lbl}'")
                stats["n_errors"] += 1
                continue

            stats["label_counts"][lbl] += 1

            # Offsets dans les bornes
            if start is None or end is None:
                errors.append(f"l.{lineno} sp[{sp_idx}] '{lbl}': start/end manquant")
                stats["n_errors"] += 1
                continue

            if start < 0 or end > len(text) or start >= end:
                errors.append(
                    f"l.{lineno} sp[{sp_idx}] '{lbl}': offset hors bornes "
                    f"start={start} end={end} len(text)={len(text)}"
                )
                stats["offset_errors"] += 1
                stats["n_errors"] += 1
                continue

            # Cohérence text[start:end] == span["text"]
            if sp_txt is not None:
                actual = text[start:end]
                if actual != sp_txt:
                    stats["offset_errors"] += 1
                    errors.append(
                        f"l.{lineno} sp[{sp_idx}] '{lbl}': "
                        f"text[{start}:{end}]={repr(actual)} ≠ span.text={repr(sp_txt)}"
                    )
                    stats["n_errors"] += 1

            # Métadonnées SVO : voice obligatoire sur les spans de rôle
            if lbl in SVO_VOICE_REQUIRED:
                voice = sp.get("voice")
                if voice is None:
                    stats["voice_missing"] += 1
                    warnings_list.append(
                        f"l.{lineno} sp[{sp_idx}] '{lbl}': 'voice' manquant"
                    )
                    stats["n_warnings"] += 1
                elif voice not in VOICE2ID:
                    errors.append(
                        f"l.{lineno} sp[{sp_idx}] '{lbl}': voice inconnue '{voice}'"
                    )
                    stats["n_errors"] += 1

            # Survie à la tokenisation DeBERTa
            if offsets is not None:
                stats["tok_checked"] += 1
                tok_span = char_span_to_token_span(offsets, start, end)
                if tok_span is None:
                    stats["tok_drop"] += 1
                    warnings_list.append(
                        f"l.{lineno} sp[{sp_idx}] '{lbl}' [{start}:{end}] "
                        f"'{sp_txt or text[start:end][:20]}': "
                        f"span perdu à la tokenisation DeBERTa"
                    )
                    stats["n_warnings"] += 1

    stats["errors"]   = errors
    stats["warnings"] = warnings_list
    return stats


def print_stats(s: dict, verbose_errors: int = 20):
    n  = s["n_rows"]
    ne = s["n_errors"]
    nw = s["n_warnings"]

    if ne == 0 and nw == 0:
        ok(f"{n} exemples, {s['n_spans']} spans — aucune erreur")
    elif ne == 0:
        warn(f"{n} exemples, {s['n_spans']} spans — {nw} warning(s), 0 erreur")
    else:
        err(f"{n} exemples, {s['n_spans']} spans — {ne} ERREUR(S), {nw} warning(s)")

    if s["offset_errors"]:
        err(f"  {s['offset_errors']} erreur(s) d'offset")
    if s["tok_drop"]:
        rate = 100 * s["tok_drop"] / max(s["tok_checked"], 1)
        warn(f"  {s['tok_drop']}/{s['tok_checked']} spans perdus à la tokenisation DeBERTa ({rate:.1f}%)")
    if s["long_texts"]:
        warn(f"  {s['long_texts']} texte(s) > 510 tokens DeBERTa (troncature possible)")
    if s["dup_ids"]:
        warn(f"  {s['dup_ids']} id(s) dupliqué(s)")
    if s["voice_missing"]:
        warn(f"  {s['voice_missing']} span(s) SVO sans 'voice'")
    if s["unknown_labels"]:
        err(f"  Labels inconnus : {dict(s['unknown_labels'].most_common(10))}")

    # Répartition labels
    lc = s["label_counts"]
    if lc:
        svo_labels = ALL_SVO_LABELS
        ner_count  = sum(v for k, v in lc.items() if k not in svo_labels)
        svo_count  = sum(v for k, v in lc.items() if k in svo_labels)
        info(f"  NER spans: {ner_count}  |  SVO/pron spans: {svo_count}")

        top = lc.most_common(10)
        info("  Top labels : " + "  ".join(f"{k}={v}" for k, v in top))

    # Erreurs (limitées)
    shown = 0
    for e in s["errors"]:
        if shown >= verbose_errors:
            err(f"  ... et {len(s['errors']) - shown} autre(s) erreur(s) (utiliser --verbose)")
            break
        err(f"  {e}")
        shown += 1

    # Warnings (les 5 premiers)
    for w in s["warnings"][:5]:
        warn(f"  {w}")
    if len(s["warnings"]) > 5:
        warn(f"  ... et {len(s['warnings']) - 5} autre(s) warning(s)")


def check_sh_sources(sh_path: Path, svo_silver_path: Path):
    """Vérifie que train_svo_silver.jsonl est branché dans run_adaptive_training.sh."""
    section("run_adaptive_training.sh — vérification SILVER_SOURCES")
    if not sh_path.exists():
        warn(f"{sh_path} introuvable, vérification ignorée")
        return

    content = sh_path.read_text(encoding="utf-8")
    silver_name = svo_silver_path.name  # ex: train_svo_silver.jsonl

    # chercher une ligne non-commentée qui référence le fichier
    found = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if silver_name in stripped:
            found = True
            ok(f"'{silver_name}' trouvé dans SILVER_SOURCES : {stripped[:80]}")
            break

    if not found:
        err(
            f"'{silver_name}' NON trouvé dans run_adaptive_training.sh !\n"
            f"  Ajoutez dans la section SILVER_SOURCES :\n"
            f"    [ -f \"$DATA/{silver_name}\" ] && "
            f"SILVER_SOURCES=\"$SILVER_SOURCES $DATA/{silver_name}:1.0\""
        )


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vérification données pré-training")
    parser.add_argument("--data-dir",   default="data",          help="Dossier data/")
    parser.add_argument("--tokenizer",  default="microsoft/deberta-v3-base")
    parser.add_argument("--max-check",  type=int, default=2000,
                        help="Nombre max d'exemples à vérifier par fichier (0=tous)")
    parser.add_argument("--no-tokenizer", action="store_true",
                        help="Désactiver la vérification de tokenisation (plus rapide)")
    parser.add_argument("--verbose",    action="store_true",
                        help="Afficher toutes les erreurs sans limite")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    verbose_errors = 9999 if args.verbose else 20

    # ── Chargement tokenizer ──────────────────────────────────────────────────
    tokenizer = None
    if not args.no_tokenizer:
        section("Chargement tokenizer DeBERTa")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            ok(f"Tokenizer '{args.tokenizer}' chargé")
        except Exception as e:
            warn(f"Impossible de charger le tokenizer ({e}) — vérification tok désactivée")

    # ── Fichiers à vérifier ───────────────────────────────────────────────────
    files_config = [
        # (chemin, description, obligatoire)
        (data_dir / "train_v3.jsonl",          "train_v3 (source principale)",    True),
        (data_dir / "val_v3.jsonl",             "val_v3",                          True),
        (data_dir / "test_v3.jsonl",            "test_v3",                         True),
        (data_dir / "train_svo_silver.jsonl",   "train_svo_silver (Stanza)",       False),
        (data_dir / "val_svo_silver.jsonl",     "val_svo_silver (Stanza)",         False),
        (data_dir / "test_svo_silver.jsonl",    "test_svo_silver (Stanza)",        False),
    ]

    # ── Vérification existence ────────────────────────────────────────────────
    section("Existence des fichiers")
    missing_required = False
    for path, desc, required in files_config:
        if path.exists():
            size_mb = path.stat().st_size / 1_048_576
            ok(f"{path.name:<40} {size_mb:6.1f} MB  [{desc}]")
        elif required:
            err(f"{path.name:<40} MANQUANT (requis) [{desc}]")
            missing_required = True
        else:
            warn(f"{path.name:<40} absent (optionnel) [{desc}]")

    if missing_required:
        err("Fichiers requis manquants — arrêt de la vérification")
        sys.exit(1)

    # ── Vérification cross-split (IDs) ────────────────────────────────────────
    # Chaque split a son propre ensemble d'IDs ; on vérifie juste les doublons intra.
    # (V3 et SVO silver peuvent avoir les mêmes IDs : c'est attendu — même exemple enrichi)
    all_stats: list[dict] = []
    total_errors = 0
    total_warnings = 0

    for path, desc, _ in files_config:
        if not path.exists():
            continue

        section(f"Vérification : {path.name}")
        mc = args.max_check
        if mc > 0:
            info(f"(vérification limitée aux {mc} premiers exemples)")

        stats = check_file(
            path,
            label=desc,
            tokenizer=tokenizer,
            max_check=mc,
        )
        print_stats(stats, verbose_errors=verbose_errors)
        all_stats.append(stats)
        total_errors   += stats["n_errors"]
        total_warnings += stats["n_warnings"]

    # ── Doublons IDs entre splits V3 et SVO silver (même split) ──────────────
    section("Cohérence IDs (V3 ↔ SVO silver, même split)")
    for split in ("train", "val", "test"):
        v3_path  = data_dir / f"{split}_v3.jsonl"
        svo_path = data_dir / f"{split}_svo_silver.jsonl"
        if not v3_path.exists() or not svo_path.exists():
            continue

        v3_ids  = {str(json.loads(l)["id"]) for l in open(v3_path, encoding="utf-8")
                   if l.strip() and "id" in json.loads(l)}
        svo_ids = {str(json.loads(l)["id"]) for l in open(svo_path, encoding="utf-8")
                   if l.strip() and "id" in json.loads(l)}

        common = v3_ids & svo_ids
        only_v3  = len(v3_ids) - len(common)
        only_svo = len(svo_ids) - len(common)
        info(
            f"  [{split}]  v3={len(v3_ids)}  svo={len(svo_ids)}  "
            f"en commun={len(common)}  "
            f"only_v3={only_v3}  only_svo={only_svo}"
        )
        if only_svo > 0:
            warn(
                f"  [{split}] {only_svo} IDs dans svo_silver absents de v3 "
                f"(probablement OK si filtrés par require-obl)"
            )
        if common == 0 and len(svo_ids) > 0:
            err(
                f"  [{split}] Aucun ID en commun entre v3 et svo_silver — "
                f"vérifier que les IDs sont cohérents !"
            )
            total_errors += 1

    # ── Vérification run_adaptive_training.sh ────────────────────────────────
    sh_path        = Path(__file__).parent / "run_adaptive_training.sh"
    svo_silvr_path = data_dir / "train_svo_silver.jsonl"
    if svo_silvr_path.exists():
        check_sh_sources(sh_path, svo_silvr_path)
    else:
        section("run_adaptive_training.sh — vérification SILVER_SOURCES")
        warn("train_svo_silver.jsonl absent — vérification du .sh ignorée")

    # ── Résumé global ─────────────────────────────────────────────────────────
    section("RÉSUMÉ")
    total_rows  = sum(s["n_rows"]  for s in all_stats)
    total_spans = sum(s["n_spans"] for s in all_stats)
    print(f"  Fichiers vérifiés : {len(all_stats)}")
    print(f"  Total exemples    : {total_rows}")
    print(f"  Total spans       : {total_spans}")

    if total_errors == 0 and total_warnings == 0:
        print(f"\n  {GRN}{BOLD}✅ Tout est propre — bon training !{RESET}")
    elif total_errors == 0:
        print(f"\n  {YLW}{BOLD}⚠  {total_warnings} warning(s), 0 erreur — probablement OK{RESET}")
    else:
        print(f"\n  {RED}{BOLD}✗  {total_errors} ERREUR(S) et {total_warnings} warning(s) — "
              f"corriger avant le training !{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()

