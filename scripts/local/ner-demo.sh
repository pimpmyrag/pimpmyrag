#!/usr/bin/env bash
# NER Demo local launcher  — requires Java 21+
# Downloads the ONNX model (~700 MB) automatically on first run.
set -euo pipefail

MODEL_URL="https://github.com/pimpmyrag/pimpmyrag/releases/download/v1.0.0-ner-model/best_model_multitask_full.onnx"
CACHE_DIR="${HOME}/.pimpmyrag"
MODEL_PATH="${CACHE_DIR}/model/best_model_multitask_full.onnx"
PORT="${NER_PORT:-8090}"
JAR="${NER_JAR:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Locate JAR ────────────────────────────────────────────────────────────────
if [[ -z "$JAR" ]]; then
  for f in \
      "$SCRIPT_DIR"/*.jar \
      "$SCRIPT_DIR/../ner-demo/build/libs"/*.jar \
      "$CACHE_DIR"/ner-demo*.jar; do
    [[ -f "$f" && "$f" != *-plain.jar ]] && JAR="$f" && break
  done
fi
if [[ -z "$JAR" || ! -f "$JAR" ]]; then
  echo "ERROR: JAR not found. Download from https://github.com/pimpmyrag/pimpmyrag/releases"
  exit 1
fi

# ── Check Java ────────────────────────────────────────────────────────────────
if ! command -v java &>/dev/null; then
  echo "ERROR: Java 21+ required — https://adoptium.net"
  exit 1
fi

# ── Download model if needed ──────────────────────────────────────────────────
if [[ -z "${NER_MODEL_PATH:-}" && ! -f "$MODEL_PATH" ]]; then
  echo "Downloading ONNX model (~700 MB)..."
  mkdir -p "$(dirname "$MODEL_PATH")"
  curl -fL --retry 3 --progress-bar -o "$MODEL_PATH" "$MODEL_URL"
fi
export NER_MODEL_PATH="${NER_MODEL_PATH:-$MODEL_PATH}"

# ── Locate tokenizer ──────────────────────────────────────────────────────────
if [[ -z "${NER_TOKENIZER_PATH:-}" ]]; then
  for tok in \
      "$SCRIPT_DIR/tokenizer_export_clean" \
      "$SCRIPT_DIR/../training/multi-head/tokenizer_export_clean" \
      "$CACHE_DIR/tokenizer_export_clean"; do
    if [[ -d "$tok" && -f "$tok/tokenizer.json" ]]; then
      export NER_TOKENIZER_PATH="$tok"
      break
    fi
  done
fi
if [[ -z "${NER_TOKENIZER_PATH:-}" ]]; then
  echo "ERROR: tokenizer_export_clean/ not found next to this script."
  exit 1
fi

echo "Starting NER Demo -> http://localhost:${PORT}"
exec java \
  -Xms128m -Xmx512m -XX:+UseG1GC -XX:MaxMetaspaceSize=192m \
  -Dserver.port="$PORT" \
  -DNER_MODEL_PATH="$NER_MODEL_PATH" \
  -DNER_TOKENIZER_PATH="$NER_TOKENIZER_PATH" \
  -jar "$JAR"

