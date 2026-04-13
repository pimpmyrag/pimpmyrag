#!/bin/bash
set -euo pipefail

# Usage: docker run --rm -it <image> -- --train /app/data/train.jsonl --val /app/data/val.jsonl --test /app/data/test.jsonl [other args]
# All args after container name are forwarded to the training script.

if [ "$#" -eq 0 ]; then
  echo "No arguments supplied. Example: -- --train /app/data/train.jsonl --val /app/data/val.jsonl --test /app/data/test.jsonl"
  exit 1
fi

# If the first arg is the separator "--" (from docker compose/run), drop it so Python sees the flags directly
if [ "$1" = "--" ]; then
  shift
fi

# Optional: clear problematic HF model cache to force clean download
rm -rf /root/.cache/huggingface/hub/models--microsoft--deberta-v3-base || true

# EXTRA: explicitly remove any spm.model files that might be corrupted
# (helps tiktoken failing to parse binary spm.model files)
if [ -d "/root/.cache/huggingface" ]; then
  echo "Searching for spm.model files under /root/.cache/huggingface..."
  mapfile -t spm_files < <(find /root/.cache/huggingface -type f -name 'spm.model' || true)
  if [ ${#spm_files[@]} -gt 0 ]; then
    echo "Found ${#spm_files[@]} spm.model file(s). Removing them to force clean re-download..."
    for f in "${spm_files[@]}"; do
      echo "Removing: $f"
      rm -f "$f" || true
    done
  else
    echo "No spm.model files found in HF cache"
  fi
fi

# Ensure sentencepiece is installed (required for slow tokenizer)
python - <<'PY'
import sys, subprocess, os, glob, shutil

# Ensure sentencepiece available
try:
    import sentencepiece as sp
    print('sentencepiece ok:', getattr(sp, '__version__', 'unknown'))
except Exception:
    print('sentencepiece not found, installing...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'sentencepiece'])
    import sentencepiece as sp
    print('sentencepiece installed:', getattr(sp, '__version__', 'unknown'))

from transformers import AutoTokenizer
model_name = 'microsoft/deberta-v3-base'

def try_load():
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=False)
        print('Tokenizer (slow) loaded OK')
        return True
    except Exception as e:
        print('Tokenizer (slow) failed to load: ', e)
        return False

print(f"Attempting to load tokenizer '{model_name}' in slow mode (use_fast=False) (first try)...")
if try_load():
    sys.exit(0)

# If failed, search for spm.model files and provide diagnostics, then remove snapshot dirs and retry once
hf_cache = os.path.expanduser('~/.cache/huggingface')
spm_paths = [p for p in glob.glob(hf_cache + '/**/spm.model', recursive=True)]
if not spm_paths:
    print('No spm.model found in HF cache to diagnose. Will exit with error.')
    sys.exit(2)

print(f'Found {len(spm_paths)} spm.model files. Diagnostics:')
for p in spm_paths:
    try:
        size = os.path.getsize(p)
        print(f"- {p} (size={size} bytes)")
        with open(p, 'rb') as fh:
            head = fh.read(64)
        print('  first bytes:', head[:64])
    except Exception as e:
        print('  failed to read', p, e)

# Remove enclosing snapshot directories (snapshots/<hash>) to force clean re-download
removed = 0
for p in spm_paths:
    # assume structure .../snapshots/<hash>/spm.model
    parts = p.split(os.sep)
    if 'snapshots' in parts:
        idx = parts.index('snapshots')
        snapshot_dir = os.sep.join(parts[:idx+2])
        if os.path.isdir(snapshot_dir):
            print('Removing snapshot dir:', snapshot_dir)
            try:
                shutil.rmtree(snapshot_dir)
                removed += 1
            except Exception as e:
                print('Failed to remove', snapshot_dir, e)
    else:
        # fallback: remove the file only
        try:
            os.remove(p)
            removed += 1
            print('Removed file:', p)
        except Exception as e:
            print('Failed to remove file:', p, e)

print(f'Removed {removed} snapshot dirs/files. Will retry tokenizer load once...')
if try_load():
    print('Tokenizer loaded after cleanup')
    sys.exit(0)

print('\nERROR: tokenizer still fails after cleanup.\n')
print('Suggested actions (on host):')
print(' - ensure you have a clean HF cache: rm -rf ~/.cache/huggingface/hub/models--microsoft--deberta-v3-base')
print(' - or avoid mounting the host HF cache into the container so it downloads fresh files')
print(' - check disk space and network connectivity')

sys.exit(3)
PY

# Find train.py under /app (mounted repo). Support multiple possible layouts.
TRAIN_PATH=""
if [ -f "/app/training/training_package/train.py" ]; then
  TRAIN_PATH="/app/training/training_package/train.py"
elif [ -f "/app/train.py" ]; then
  TRAIN_PATH="/app/train.py"
elif [ -f "/app/training_package/train.py" ]; then
  TRAIN_PATH="/app/training_package/train.py"
else
  # search up to depth 5 for train.py
  TRAIN_PATH=$(find /app -maxdepth 6 -type f -name 'train.py' | head -n 1 || true)
fi

if [ -z "$TRAIN_PATH" ]; then
  echo "Error: train.py not found under /app. Mounted content:" >&2
  ls -la /app || true
  exit 2
fi

echo "Using train.py at: $TRAIN_PATH"

# ---- New argument fixup: if provided --train/--val/--test file paths do not exist inside container,
#     attempt to find matching train*.jsonl / val*.jsonl / test*.jsonl under /app and replace them.
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train)
      if [[ -n "$2" && -f "$2" ]]; then
        ARGS+=("$1" "$2")
      else
        # try to find a train*.jsonl under /app
        FOUND=$(find /app -type f -name 'train*.jsonl' -print -quit || true)
        if [[ -n "$FOUND" ]]; then
          echo "Note: replacing missing train path '$2' with found file: $FOUND"
          ARGS+=("$1" "$FOUND")
        else
          echo "Warning: provided train path '$2' not found and no train*.jsonl located under /app. Passing original value."
          ARGS+=("$1" "$2")
        fi
      fi
      shift 2
      ;;
    --val)
      if [[ -n "$2" && -f "$2" ]]; then
        ARGS+=("$1" "$2")
      else
        FOUND=$(find /app -type f -name 'val*.jsonl' -print -quit || true)
        if [[ -n "$FOUND" ]]; then
          echo "Note: replacing missing val path '$2' with found file: $FOUND"
          ARGS+=("$1" "$FOUND")
        else
          echo "Warning: provided val path '$2' not found and no val*.jsonl located under /app. Passing original value."
          ARGS+=("$1" "$2")
        fi
      fi
      shift 2
      ;;
    --test)
      if [[ -n "$2" && -f "$2" ]]; then
        ARGS+=("$1" "$2")
      else
        FOUND=$(find /app -type f -name 'test*.jsonl' -print -quit || true)
        if [[ -n "$FOUND" ]]; then
          echo "Note: replacing missing test path '$2' with found file: $FOUND"
          ARGS+=("$1" "$FOUND")
        else
          echo "Warning: provided test path '$2' not found and no test*.jsonl located under /app. Passing original value."
          ARGS+=("$1" "$2")
        fi
      fi
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

echo "Running training with args: ${ARGS[*]}"
python "$TRAIN_PATH" "${ARGS[@]}"

# After training, attempt to locate best_model.pt and copy to /app/output
BEST_PATH=""
if [ -f "/app/best_model.pt" ]; then
  BEST_PATH="/app/best_model.pt"
else
  BEST_PATH=$(find /app -type f -name 'best_model.pt' | head -n 1 || true)
fi

if [ -n "$BEST_PATH" ] && [ -f "$BEST_PATH" ]; then
  cp "$BEST_PATH" /app/output/ || true
  echo "Model exported to /app/output/$(basename "$BEST_PATH")"
else
  echo "Warning: best_model.pt not found in /app or subfolders. Check training script output." >&2
fi
