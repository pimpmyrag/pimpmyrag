# Presse multilingue RSS — Config V2 (haute qualité)

Ce pack contient :
- une **configuration RSS multilingue** (FR/EN/DE/ES/PT/IT),
- un script de collecte RSS → **phrases JSONL**,
- `requirements.txt`,
- et un launcher `run.sh` qui crée/active un **venv** local (`./.venv`).

## Démarrage rapide (avec venv)
```bash
chmod +x run.sh
./run.sh
```

## Installation manuelle (équivalent)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Vérifier le contenu du ZIP
```bash
unzip -l press_fetcher_v2_high_quality_WITH_SCRIPTS_VENV_FIX.zip
```

## Générer un autre volume / autre sortie
```bash
source .venv/bin/activate
python collect_news_sentences.py   --config feeds_eu_rss_high_quality_v2.json   --out sentences_50k.jsonl   --target 50000   --langs fr,en,de,it,es,pt
```

### Notes
- Mode RSS only : titres/résumés/contenu RSS (pas de scraping full-text).
- `--strict-lang` active un filtrage langue au niveau phrase via `langid`.
