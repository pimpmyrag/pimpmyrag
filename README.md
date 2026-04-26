# PimpMyRAG

[![CI](https://github.com/pimpmyrag/pimpmyrag/actions/workflows/ci.yml/badge.svg)](https://github.com/pimpmyrag/pimpmyrag/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen.svg)](https://opensource.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/pimpmyrag/pimpmyrag/pulls)
[![GitHub Stars](https://img.shields.io/github/stars/pimpmyrag/pimpmyrag?style=social)](https://github.com/pimpmyrag/pimpmyrag/stargazers)
[![Kotlin](https://img.shields.io/badge/Kotlin-2.x-7F52FF?logo=kotlin&logoColor=white)](https://kotlinlang.org)
[![JVM](https://img.shields.io/badge/JVM-17%2B-ED8B00?logo=openjdk&logoColor=white)](https://adoptium.net)

Framework RAG modulaire sur JVM, orienté extraction d'entites/evenements deterministe, avec une stack NER multitete (NER + SVO + voix + morpho) utilisable en Python (training/eval) et en Kotlin ONNX Runtime (inference).

## Documentation par profil

- Externe open-source: `docs/README.opensource.md`
- Ops / production: `docs/README.ops-prod.md`
- Bilingue FR/EN: `docs/README.bilingual.md`

## Architecture

```text
pimpmyrag/
├── rag-model/                  Modeles de donnees partages (UDToken, Entity, ...)
├── rag-engine/                 Interfaces pipeline (Chunker, Embedder, NerExtractor, ...)
├── rag-dsl/                    DSL Kotlin declaratif pour configurer un pipeline RAG
├── rag-dsl-staged/             Variante DSL type-safe par etapes
├── rag-planner/                Compile un RagConfig en ExecutionPlan immuable
├── rag-runner/                 Execute un ExecutionPlan via DAG
├── rag-app/                    Application Spring Boot d'assemblage
├── radar-nli-toolkit/          Classifieur NLI + radar semantique
├── connectors/
│   ├── ner/onnx-ner/           Extracteur ONNX multitete (boundary/coarse/fine + SVO)
│   ├── ud/ms-ud/               Client UD (Stanza/UDPipe via HTTP)
│   ├── embed/infinity/         Embedder via Infinity API
│   ├── embed/onnx-emb/         Embedder ONNX local (DJL)
│   ├── rerank/infinity/        Reranker via Infinity API
│   ├── rerank/onnx-ce/         Cross-encoder ONNX local
│   ├── document-store/mongodb/ Store de documents MongoDB
│   ├── vector/qadrant/         Store vectoriel Qdrant
│   ├── llm/chat-completion/    LLM via API chat-completion
│   └── rag-connectors-stub/    Stubs pour les tests
├── ner-demo/                   Demo UI Spring/Vaadin + MCP pour NER/SVO
├── training/
│   ├── multi-head/             Dataset multitask, training, export ONNX, bench/eval
│   └── training_package/       Ancien package de training NER
└── scripts/                    Evaluation, tests fonctionnels, utilitaires
```

## Prerequis

- JDK 17+
- Gradle 8+
- Python 3.11+ (pour `training/`)
- Docker (services UD, Infinity, Qdrant, MongoDB)

## Build

```zsh
./gradlew build
```

## Lancement rapide (API)

```zsh
docker-compose up -d
./gradlew :rag-app:bootRun
```

API locale: `http://localhost:8080`

Docs module: `rag-app/README.md`

## Demo NER/SVO (UI + MCP)

Le module `ner-demo` expose:

- une UI Vaadin pour tester l'extraction NER/SVO
- un serveur MCP SSE (`/mcp/sse`) pour outils d'analyse

Configuration principale dans `ner-demo/src/main/resources/application.properties`:

- `server.port=8090`
- `ner.model-path=...best_model_multitask_full.onnx`
- `ner.tokenizer-path=.../tokenizer_export_clean`

Lancer la demo:

```zsh
./gradlew :ner-demo:bootRun
```

URLs:

- UI: `http://localhost:8090`
- MCP SSE: `http://localhost:8090/mcp/sse`

## Deploiement Render (demo `ner-demo`)

Le repo inclut `render.yaml` pour deployer automatiquement la demo sur Render.

Build/start sur Render:

- build: `./gradlew :ner-demo:bootJar -x test`
- start: `bash scripts/render/start-ner-demo-render.sh`

Variables d'environnement a definir dans Render:

- `MODEL_URL` : URL du modele ONNX (ex: asset GitHub Release, S3, R2)
- `MODEL_SHA256` : checksum SHA-256 du modele
- `TOKENIZER_URL` : URL d'une archive tokenizer (`.zip` ou `.tar.gz`)
- `TOKENIZER_SHA256` : checksum SHA-256 de l'archive tokenizer
- `PORT` : injecte automatiquement par Render (mappe sur `server.port`)

Variables optionnelles (si artefacts deja presents localement):

- `NER_MODEL_PATH` : chemin local vers le modele ONNX (bypass download)
- `NER_TOKENIZER_PATH` : chemin local vers le dossier tokenizer (bypass download)

Important:

- Le modele ONNX et le tokenizer ne sont pas versionnes dans Git/LFS dans ce repo.
- Ils sont provisionnes au demarrage via `scripts/render/start-ner-demo-render.sh` (download + checksum + extraction tokenizer).
- Au boot, `ner-demo` echoue volontairement si `NER_MODEL_PATH` ou `NER_TOKENIZER_PATH` pointe vers une ressource absente.

Flux recommande avant merge sur `main`:

1. Verifier que `ner-demo` build localement.
2. Committer `render.yaml` + la config Spring (`application.properties`).
3. Push sur la branche reliee a Render (ou `main` selon votre setup) pour declencher le deploiement.

### CI/CD automatise (GitHub Actions -> Render)

Le repo contient deux workflows:

- `.github/workflows/publish-ner-assets.yml` : publie ONNX + tokenizer + `SHA256SUMS.txt` dans une GitHub Release (declenchement manuel)
- `.github/workflows/deploy-render-ner-demo.yml` : declenche le deploy Render automatiquement apres succes de `CI` sur `main`

Secrets/variables GitHub a configurer:

- `RENDER_DEPLOY_HOOK_URL` (secret) : deploy hook Render du service `ner-demo`
- `RENDER_NER_DEMO_URL` (variable, optionnelle) : URL publique pour smoke-check post-deploy

Variables Render a configurer une fois les assets publies:

- `MODEL_URL`
- `MODEL_SHA256`
- `TOKENIZER_URL`
- `TOKENIZER_SHA256`

Resultat: chaque push sur `main` declenche `CI`, puis un deploy Render automatique si `CI` est vert.

## Training multitete (NER + SVO + voice + morpho)

Tout le pipeline est dans `training/multi-head`.

### 1) Installer les deps Python

```zsh
cd training/multi-head
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Construire les datasets multitask

```zsh
python build_multitask_dataset.py \
  --input data/train.jsonl \
  --output data/train_multitask.jsonl \
  --model-name microsoft/deberta-v3-base

python build_multitask_dataset.py \
  --input data/val.jsonl \
  --output data/val_multitask.jsonl \
  --model-name microsoft/deberta-v3-base

python build_multitask_dataset.py \
  --input data/test.jsonl \
  --output data/test_multitask.jsonl \
  --model-name microsoft/deberta-v3-base
```

### 3) Entrainer le modele multitete

```zsh
python train_multi_task.py \
  --train data/train_multitask.jsonl \
  --val data/val_multitask.jsonl \
  --test data/test_multitask.jsonl \
  --model-name microsoft/deberta-v3-base \
  --epochs 8 \
  --batch-size 8 \
  --lr 2e-5 \
  --lambda-boundary 1.0 \
  --lambda-coarse 1.0 \
  --lambda-fine 1.2 \
  --lambda-svo-boundary 1.0 \
  --lambda-svo 1.0 \
  --lambda-voice 0.5 \
  --lambda-morpho 0.3
```

Sorties typiques:

- `checkpoint_last_multitask.pt`
- `checkpoint_best_multitask.pt`
- `best_model_multitask.pt`

### 4) Evaluer le modele sur des phrases

```zsh
python test_model_sentences_v3.py \
  --checkpoint checkpoint_best_multitask.pt \
  --model-name microsoft/deberta-v3-base \
  --topk-coarse 2 \
  --tau-boundary 0.70 \
  --tau-none 0.99 \
  --tau-coarse 0.45 \
  --tau-svo-boundary 0.50 \
  --text "Le secretaire general des Nations Unies est arrive a Paris."
```

## Export ONNX (multitete)

Exporter vers ONNX depuis un checkpoint PyTorch:

```zsh
cd training/multi-head
python export_onnx_multitask.py \
  --checkpoint checkpoint_best_multitask.pt \
  --model-name microsoft/mdeberta-v3-base \
  --output best_model_multitask.onnx \
  --opset 17 \
  --seq-len 128 \
  --max-spans 64
```

Interface ONNX exportee:

- Inputs: `input_ids`, `attention_mask`, `span_starts`, `span_ends`, `span_batch_ids`
- Outputs NER: `boundary_logits`, `coarse_logits`, `fine_logits`
- Outputs SVO: `svo_boundary_logits`, `svo_logits`, `voice_logits`, `gender_logits`, `number_logits`

## Inference Kotlin ONNX (connecteur NER)

Le connecteur `connectors/ner/onnx-ner` charge le modele ONNX multitete via `OnnxMultiHeadEntityExtractor`.

Points importants:

- selection coarse en `top-2` puis meilleur fine par famille (pour augmenter le rappel)
- filtres par seuils (`tauBoundary`, `tauNone`, `tauCoarse`, `minScore`, `tauSvoBoundary`)
- NMS avec conservation des spans imbriques de labels fins differents
- support CoreML optionnel sur macOS (`useCoreMl=true`)

## Tests rapides

Test endpoint NER (API `rag-app` sur `:8080`):

```zsh
python scripts/ner_candidates_test.py
```

## Ressources

- README applicatif: `rag-app/README.md`
- Scripts multitete: `training/multi-head/`
- Extracteur ONNX Kotlin: `connectors/ner/onnx-ner/src/main/kotlin/rag/connectors/ner/onnx/OnnxMultiHeadEntityExtractor.kt`
