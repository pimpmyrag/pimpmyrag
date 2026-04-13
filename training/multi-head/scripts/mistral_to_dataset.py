#!/usr/bin/env python3
"""
mistral_to_dataset.py

Wrapper utilitaire :
- lit un fichier jsonl de sorties (Mistral ou autre),
- applique un remapping optionnel des labels (JSON mapping),
- appelle `convert_to_dataset_jsonl.py` pour fixer offsets / tokenization,
- appelle `validate_converted.py` pour vérifier les spans/labels,
- écrit par défaut un fichier `dataset_YYYYMMDD.tmp.jsonl`.

Usage:
  python3 mistral_to_dataset.py --input mistral_output.jsonl [--remap remap.json] [--output dataset_YYYYMMDD.tmp.jsonl] [--tokenizer microsoft/deberta-v3-base] [--max-lines N]

Le fichier remap (optionnel) doit être un JSON simple {"mistral_label": "hint_person_name", ...}
Les valeurs cibles doivent être des labels présents dans `training/multi-head/labels.py` (FINE_LABELS).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any


def load_remap(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise SystemExit(f"Remap file not found: {path}")
    with path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    if not isinstance(data, dict):
        raise SystemExit("Remap file must contain a JSON object mapping source->target label")
    return {str(k): str(v) for k, v in data.items()}


def remap_labels_in_entry(obj: Dict[str, Any], remap: Dict[str, str]):
    """Applique remap sur les labels trouvés dans clés usuelles des spans/candidates."""
    list_keys = ["candidates", "spans", "entities", "annotations"]

    def map_label_value(val):
        if val is None:
            return val
        s = str(val)
        return remap.get(s, s)

    for k in list_keys:
        if k in obj and isinstance(obj[k], list):
            for item in obj[k]:
                if not isinstance(item, dict):
                    continue
                # common label keys
                for label_key in ("label", "fine_label", "type", "fine", "tag"):
                    if label_key in item:
                        item[label_key] = map_label_value(item[label_key])
                # sometimes nested candidate->label fields
                # keep other fields untouched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None, help="Output dataset path (default: dataset_YYYYMMDD.tmp.jsonl)")
    parser.add_argument("--remap", default=None, help="Optional JSON remap file mapping mistral labels to local fine labels")
    parser.add_argument("--tokenizer", default="microsoft/deberta-v3-base")
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--no-coarse", action="store_true", help="Pass --no-coarse to the converter")
    parser.add_argument("--out-bad", default=None, help="Path to write bad examples from validation (JSONL)")

    args = parser.parse_args()
    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    # output default name
    if args.output:
        outp = Path(args.output)
    else:
        date = datetime.now().strftime("%Y%m%d")
        outp = Path(f"dataset_{date}.tmp.jsonl")

    remap = {}
    if args.remap:
        remap = load_remap(Path(args.remap))

    # prepare a temporary preprocessed input where labels have been remapped
    tmp_in = None
    try:
        tmpf = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".jsonl")
        tmp_in = Path(tmpf.name)
        with inp.open("r", encoding="utf-8") as fin:
            for i, raw in enumerate(fin):
                if args.max_lines is not None and i >= args.max_lines:
                    break
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    # keep raw line if invalid json
                    tmpf.write(raw + "\n")
                    continue
                if remap:
                    remap_labels_in_entry(obj, remap)
                tmpf.write(json.dumps(obj, ensure_ascii=False) + "\n")
        tmpf.close()

        # call convert_to_dataset_jsonl.py
        converter = Path(__file__).resolve().parent / "convert_to_dataset_jsonl.py"
        cmd = [sys.executable, str(converter), "--input", str(tmp_in), "--output", str(outp), "--tokenizer", args.tokenizer]
        if args.max_lines is not None:
            cmd.extend(["--max-lines", str(args.max_lines)])
        if args.no_coarse:
            cmd.append("--no-coarse")

        print("Running converter:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # run validator
        validator = Path(__file__).resolve().parent / "validate_converted.py"
        vcmd = [sys.executable, str(validator), "--input", str(outp), "--tokenizer", args.tokenizer]
        if args.out_bad:
            vcmd.extend(["--out-bad", args.out_bad])
        print("Running validator:", " ".join(vcmd))
        subprocess.run(vcmd, check=True)

        print("Done. Output dataset:", outp)
        if args.out_bad:
            print("Bad examples written to:", args.out_bad)

    finally:
        # cleanup temp input
        if tmp_in and tmp_in.exists():
            try:
                tmp_in.unlink()
            except Exception:
                pass


if __name__ == '__main__':
    main()

