#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run conversion and validation inside the project's venv.
# Usage:
#   ./run_convert_and_validate.sh --input <in.jsonl> --out <out.jsonl> [--max-lines N]

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
VENV_PY="$PROJECT_ROOT/../../training/press_fetcher_v2_high_quality_venv/.venv/bin/python3"

usage() {
  cat <<EOF
Usage: $0 --input <in.jsonl> --out <out.jsonl> [--max-lines N]
Runs the conversion script and then validates the produced file.
EOF
}

INPUT=""
OUT=""
MAX_LINES=""
TOKENIZER="microsoft/deberta-v3-base"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --max-lines) MAX_LINES="$2"; shift 2;;
    --tokenizer) TOKENIZER="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$INPUT" || -z "$OUT" ]]; then
  usage
  exit 1
fi

CMD_CONVERT="$VENV_PY training/multi-head/scripts/convert_to_dataset_jsonl.py --input $INPUT --output $OUT --tokenizer $TOKENIZER"
if [[ -n "$MAX_LINES" ]]; then
  CMD_CONVERT="$CMD_CONVERT --max-lines $MAX_LINES"
fi

echo "Running conversion:"
echo "$CMD_CONVERT"
$CMD_CONVERT

echo "Running validation:"
$VENV_PY training/multi-head/scripts/validate_converted.py --input $OUT --tokenizer $TOKENIZER --out-bad ${OUT}.bad_examples.jsonl

echo "Done. Converted file: $OUT"
