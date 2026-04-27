#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSET_BASE_DIR="${ASSET_BASE_DIR:-$ROOT_DIR/.render-assets}"
MODEL_FILENAME="${MODEL_FILENAME:-best_model_multitask_full.onnx}"
TOKENIZER_DIRNAME="${TOKENIZER_DIRNAME:-tokenizer_export_clean}"

MODEL_PATH_DEFAULT="$ASSET_BASE_DIR/model/$MODEL_FILENAME"
TOKENIZER_BASE_DIR="$ASSET_BASE_DIR/tokenizer"
TOKENIZER_PATH_DEFAULT="$TOKENIZER_BASE_DIR/$TOKENIZER_DIRNAME"

mkdir -p "$ASSET_BASE_DIR/model" "$TOKENIZER_BASE_DIR"

download_file() {
  local url="$1"
  local out="$2"
  echo "[render] download: $url -> $out" >&2
  curl -fL --retry 5 --retry-delay 2 -o "$out" "$url"
}

sha256_check() {
  local file="$1"
  local expected="$2"
  if [[ -z "$expected" ]]; then
    return 0
  fi

  local actual
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$file" | awk '{print $1}')"
  else
    actual="$(shasum -a 256 "$file" | awk '{print $1}')"
  fi

  if [[ "$actual" != "$expected" ]]; then
    echo "[render] checksum mismatch for $file" >&2
    echo "[render] expected: $expected" >&2
    echo "[render] actual  : $actual" >&2
    exit 1
  fi
}

resolve_model() {
  if [[ -n "${NER_MODEL_PATH:-}" ]]; then
    [[ -f "$NER_MODEL_PATH" ]] || { echo "[render] NER_MODEL_PATH not found: $NER_MODEL_PATH" >&2; exit 1; }
    echo "$NER_MODEL_PATH"
    return
  fi

  local model_path="$MODEL_PATH_DEFAULT"
  if [[ ! -f "$model_path" ]]; then
    [[ -n "${MODEL_URL:-}" ]] || { echo "[render] MODEL_URL is required when NER_MODEL_PATH is not set" >&2; exit 1; }
    download_file "$MODEL_URL" "$model_path"
  fi
  sha256_check "$model_path" "${MODEL_SHA256:-}"
  echo "$model_path"
}

extract_tokenizer_archive() {
  local archive="$1"
  local dest="$2"
  rm -rf "$dest"
  mkdir -p "$dest"

  case "$archive" in
    *.zip)
      unzip -oq "$archive" -d "$dest"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "$archive" -C "$dest"
      ;;
    *)
      echo "[render] unsupported tokenizer archive format: $archive" >&2
      exit 1
      ;;
  esac
}

find_tokenizer_dir() {
  local root="$1"
  find "$root" -type f -name tokenizer.json -print -quit | xargs -I{} dirname "{}"
}

resolve_tokenizer() {
  if [[ -n "${NER_TOKENIZER_PATH:-}" ]]; then
    [[ -d "$NER_TOKENIZER_PATH" ]] || { echo "[render] NER_TOKENIZER_PATH not found: $NER_TOKENIZER_PATH" >&2; exit 1; }
    echo "$NER_TOKENIZER_PATH"
    return
  fi

  if [[ -d "$TOKENIZER_PATH_DEFAULT" && -f "$TOKENIZER_PATH_DEFAULT/tokenizer.json" ]]; then
    echo "$TOKENIZER_PATH_DEFAULT"
    return
  fi

  # Fallback : tokenizer versionné dans le dépôt (training/multi-head/tokenizer_export_clean)
  local repo_tokenizer="$ROOT_DIR/training/multi-head/tokenizer_export_clean"
  if [[ -d "$repo_tokenizer" && -f "$repo_tokenizer/tokenizer.json" ]]; then
    echo "[render] tokenizer trouvé dans le repo : $repo_tokenizer" >&2
    echo "$repo_tokenizer"
    return
  fi

  [[ -n "${TOKENIZER_URL:-}" ]] || { echo "[render] TOKENIZER_URL is required when NER_TOKENIZER_PATH is not set" >&2; exit 1; }

  local archive_name
  archive_name="$(basename "$TOKENIZER_URL")"
  local archive_path="$ASSET_BASE_DIR/$archive_name"

  download_file "$TOKENIZER_URL" "$archive_path"
  sha256_check "$archive_path" "${TOKENIZER_SHA256:-}"
  extract_tokenizer_archive "$archive_path" "$TOKENIZER_BASE_DIR"

  local found
  found="$(find_tokenizer_dir "$TOKENIZER_BASE_DIR")"
  [[ -n "$found" ]] || { echo "[render] tokenizer.json not found after extraction" >&2; exit 1; }
  echo "$found"
}

export NER_MODEL_PATH="$(resolve_model)"
export NER_TOKENIZER_PATH="$(resolve_tokenizer)"

echo "[render] NER_MODEL_PATH=$NER_MODEL_PATH"
echo "[render] NER_TOKENIZER_PATH=$NER_TOKENIZER_PATH"

APP_JAR="$(ls "$ROOT_DIR"/ner-demo/build/libs/*.jar | grep -v -- '-plain.jar' | head -n 1)"
[[ -n "$APP_JAR" ]] || { echo "[render] boot jar not found" >&2; exit 1; }

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[render] dry-run enabled, skipping app start"
  exit 0
fi

SERVER_PORT="${PORT:-8080}"
echo "[render] starting on port $SERVER_PORT"
echo "[render] JAR=$APP_JAR"

exec java \
  ${JAVA_TOOL_OPTIONS:--Xms128m -Xmx512m -XX:+UseG1GC -XX:MaxMetaspaceSize=192m -XX:+ExitOnOutOfMemoryError -XX:+HeapDumpOnOutOfMemoryError} \
  -Dserver.port="$SERVER_PORT" \
  -Dspring.profiles.active="${SPRING_PROFILES_ACTIVE:-render}" \
  -jar "$APP_JAR"

