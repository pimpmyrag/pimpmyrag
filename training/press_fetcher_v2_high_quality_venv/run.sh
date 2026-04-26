#!/usr/bin/env bash
set -euo pipefail

# Utilise un venv local (./.venv)
VENV_DIR=".venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Lancement (tu peux modifier target/langs/out)
python collect_news_sentences.py   --config feeds_eu_rss_high_quality_v2.json   --out sentences_50k.jsonl   --target 50000   --langs fr,en,de,it,es,pt   --concurrency 16   --max-articles-per-feed 80   --min-chars 40   --max-chars 300
