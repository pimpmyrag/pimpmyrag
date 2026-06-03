# PimpMyRAG

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-ner--demo-6366f1?style=for-the-badge)](https://ner-demo-ocg3.onrender.com/)
[![Download](https://img.shields.io/badge/📦%20Download-Installers-22c55e?style=for-the-badge)](https://github.com/pimpmyrag/pimpmyrag/releases/latest)
[![CI](https://github.com/pimpmyrag/pimpmyrag/actions/workflows/ci.yml/badge.svg)](https://github.com/pimpmyrag/pimpmyrag/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Kotlin](https://img.shields.io/badge/Kotlin-2.x-7F52FF?logo=kotlin&logoColor=white)](https://kotlinlang.org)
[![JVM](https://img.shields.io/badge/JVM-21-ED8B00?logo=openjdk&logoColor=white)](https://adoptium.net)
[![DeBERTa](https://img.shields.io/badge/Model-DeBERTa--v3-ff6b35?logo=pytorch&logoColor=white)](https://huggingface.co/microsoft/deberta-v3-base)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/pimpmyrag/pimpmyrag/pulls)

---

> 🇫🇷 **Lire en français** | 🇬🇧 [Read in English](#english-version)

---

## 🇫🇷 Version française

Cadre RAG modulaire sur JVM orienté extraction d'entités/événements **déterministe**, avec une stack NER multi-tête (NER + SVO + voix + morpho) utilisable en Python (training/eval) et en Kotlin via ONNX Runtime (inférence).

### 🚀 Démo en ligne

**➡️ [https://ner-demo-ocg3.onrender.com/](https://ner-demo-ocg3.onrender.com/)**

Interface interactive pour tester l'extraction NER/SVO en temps réel :

| Fonctionnalité | Description |
|---|---|
| 🏷️ **NER 38 labels fins** | Extraction d'entités nommées avec taxonomie coarse/fine (`PER`, `LOC`, `ORG`, `TIME`, `EVENT`, `OBJECT`, `VALUE`, `WORK`, `ABSTRACT`) |
| 🔗 **SVO + rôles 12 labels** | Extraction Sujet–Verbe–Objet avec rôles syntaxiques (`SUBJECT`, `OBJECT`, `OBLIQUE_*`, `APPOS`) |
| 🧭 **Verbfam** | Classification sémantique des verbes (`verb_family`, `verb_family_fine`, polarity/aspect/source) |
| 🤖 **LLM Judge** | Évaluation automatique de la qualité NER via un LLM (OpenAI, Mistral, Ollama, GitHub Copilot, Azure…) |
| ⚙️ **Paramètres live** | Ajustement des seuils de détection en temps réel |
| 🌍 **Multilingue** | Interface disponible en FR / EN / DE / ES / IT |

> ⚠️ Le démarrage peut prendre ~30 s (plan Render gratuit avec cold start).

### 📦 Tester localement (sans Docker)

**➡️ [Télécharger les installateurs](https://github.com/pimpmyrag/pimpmyrag/releases/latest)**

| Plateforme | Fichier | Utilisation |
|---|---|---|
| 🐧 **Linux** | `ner-demo-linux-x64.tar.gz` | `tar xzf … && ./NER\ Demo/bin/NER\ Demo` |
| 🍎 **macOS** | `ner-demo-macos.dmg` | Double-clic → glisser dans Applications |
| 🪟 **Windows** | `ner-demo-windows-x64.zip` | Extraire → `NER Demo\NER Demo.exe` |

> **JRE inclus** — Java n'est pas requis.  
> **Premier lancement** : le modèle ONNX (~700 MB) est téléchargé automatiquement dans `~/.pimpmyrag/model/`.  
> **macOS** : si "développeur non identifié" → clic droit → Ouvrir.  
> Puis ouvrir [http://localhost:8090](http://localhost:8090) dans le navigateur.

### Architecture

```text
pimpmyrag/
├── rag-model/           Modèles de données partagés (UDToken, Entity, ...)
├── rag-engine/          Interfaces pipeline (Chunker, Embedder, NerExtractor, ...)
├── rag-dsl/             DSL Kotlin déclaratif pour configurer un pipeline RAG
├── rag-dsl-staged/      Variante DSL type-safe par étapes
├── rag-planner/         Compile un RagConfig en ExecutionPlan immuable
├── rag-runner/          Exécute un ExecutionPlan via DAG
├── rag-app/             Application Spring Boot d'assemblage
├── radar-nli-toolkit/   Classifieur NLI + radar sémantique
├── connectors/
│   ├── ner/onnx-ner/    Extracteur ONNX multi-tête (boundary/coarse/fine + SVO)
│   ├── ud/ms-ud/        Client UD (Stanza/UDPipe via HTTP)
│   ├── embed/           Embedders (Infinity API + ONNX local DJL)
│   ├── rerank/          Rerankers (Infinity API + cross-encoder ONNX)
│   ├── document-store/  MongoDB
│   ├── vector/qdrant/   Store vectoriel Qdrant
│   └── llm/             LLM via API chat-completion
├── ner-demo/            Démo UI Spring/Vaadin + serveur MCP SSE
├── training/multi-head/ Dataset, training, export ONNX, bench/eval
└── scripts/             Eval, tests fonctionnels, utilitaires
```

### Taxonomie NER/SVO

La taxonomie maintenue est générée depuis le code source de vérité `training/multi-head/labels.py` :

| Famille | Labels actifs | Source |
|---|---:|---|
| NER coarse | 10 (`NONE` inclus côté modèle) | `COARSE_LABELS` |
| NER fine | 38 | `FINE_LABELS` |
| Syntax spans | 3 | `SYN_LABELS` |
| Role principal | 12 | `ROLE_LABELS` |
| Verb family / fine | 12 / 38 | `VERB_FAMILY_LABELS`, `VERB_FAMILY_FINE_LABELS` |
| Morphologie/modalité | voice, certainty, gender, number, person | `labels.py` |

Documents générés :

- Taxonomie lisible : [`docs/TAXONOMY.md`](docs/TAXONOMY.md)
- Export machine-readable : [`docs/taxonomy.json`](docs/taxonomy.json)
- JSON Schema : [`docs/taxonomy.schema.json`](docs/taxonomy.schema.json)

Après toute modification de `training/multi-head/labels.py`, régénérer :

```zsh
cd pimpmyrag
source training/multi-head/venv/bin/activate
python3 training/multi-head/export_taxonomy.py
```

### Prérequis

- JDK 21+
- Gradle 8+
- Python 3.11+ (pour `training/`)
- Docker (services UD, Infinity, Qdrant, MongoDB)

### Build

```zsh
./gradlew build
```

### Lancement rapide (API)

```zsh
docker-compose up -d
./gradlew :rag-app:bootRun
```

API locale : `http://localhost:8080`

### Lancer la démo NER localement

```zsh
./gradlew :ner-demo:bootRun
```

- UI : `http://localhost:8090`
- MCP SSE : `http://localhost:8090/mcp/sse`

Variables à définir dans `ner-demo/src/main/resources/application.properties` :

```properties
ner.model-path=.../best_model_multitask_full.onnx
ner.tokenizer-path=.../tokenizer_export_clean
```

### Déploiement Render

Le repo inclut `render.yaml`. Le modèle ONNX (~708 MB) est téléchargé depuis une GitHub Release au moment du build Docker (pas au démarrage — évite les OOM).

Variables d'environnement Render :

| Variable | Description |
|---|---|
| `MODEL_URL` | URL du modèle ONNX (GitHub Release) |
| `MODEL_SHA256` | Checksum SHA-256 optionnel |
| `NER_MODEL_PATH` | Chemin local (bypass download) |

**CI/CD :** chaque push sur `main` → `CI` → deploy Render automatique (nécessite le secret `RENDER_DEPLOY_HOOK_URL` dans les secrets GitHub Actions).

---

## <a name="english-version"></a>🇬🇧 English version

Modular RAG framework on the JVM, built around **deterministic** entity/event extraction, with a multi-head NER stack (NER + SVO + voice + morphology) usable in Python (training/eval) and Kotlin via ONNX Runtime (inference).

### 🚀 Live Demo

**➡️ [https://ner-demo-ocg3.onrender.com/](https://ner-demo-ocg3.onrender.com/)**

Interactive interface to test NER/SVO extraction in real time:

| Feature | Description |
|---|---|
| 🏷️ **NER 38 fine labels** | Named entity extraction with coarse/fine taxonomy (`PER`, `LOC`, `ORG`, `TIME`, `EVENT`, `OBJECT`, `VALUE`, `WORK`, `ABSTRACT`) |
| 🔗 **SVO + 12 role labels** | Subject–Verb–Object extraction with syntactic roles (`SUBJECT`, `OBJECT`, `OBLIQUE_*`, `APPOS`) |
| 🧭 **Verbfam** | Semantic verb classification (`verb_family`, `verb_family_fine`, polarity/aspect/source) |
| 🤖 **LLM Judge** | Automatic NER quality evaluation via LLM (OpenAI, Mistral, Ollama, GitHub Copilot, Azure…) |
| ⚙️ **Live parameters** | Real-time threshold tuning |
| 🌍 **Multilingual UI** | Available in FR / EN / DE / ES / IT |

> ⚠️ First load may take ~30 s (Render free tier cold start).

### 📦 Run locally (no Docker)

**➡️ [Download installers](https://github.com/pimpmyrag/pimpmyrag/releases/latest)**

| Platform | File | How to run |
|---|---|---|
| 🐧 **Linux** | `ner-demo-linux-x64.tar.gz` | `tar xzf … && ./NER\ Demo/bin/NER\ Demo` |
| 🍎 **macOS** | `ner-demo-macos.dmg` | Double-click → drag to Applications |
| 🪟 **Windows** | `ner-demo-windows-x64.zip` | Extract → `NER Demo\NER Demo.exe` |

> **JRE bundled** — Java not required.  
> **First launch**: the ONNX model (~700 MB) is downloaded automatically into `~/.pimpmyrag/model/`.  
> **macOS**: if "unidentified developer" warning → right-click → Open.  
> Then open [http://localhost:8090](http://localhost:8090) in your browser.

### Quick start

```zsh
# Build
./gradlew build

# Run API
docker-compose up -d && ./gradlew :rag-app:bootRun

# Run NER demo locally
./gradlew :ner-demo:bootRun
# → http://localhost:8090
```

### Training (multi-head NER + SVO)

Current taxonomy is generated from `training/multi-head/labels.py` and documented in [`docs/TAXONOMY.md`](docs/TAXONOMY.md). Machine-readable export and schema are available as [`docs/taxonomy.json`](docs/taxonomy.json) and [`docs/taxonomy.schema.json`](docs/taxonomy.schema.json).

```zsh
cd training/multi-head
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build datasets
python build_multitask_dataset.py --input data/train.jsonl --output data/train_multitask.jsonl --model-name microsoft/deberta-v3-base

# Train
python train_multi_task.py --train data/train_multitask.jsonl --val data/val_multitask.jsonl \
  --epochs 8 --batch-size 8 --lr 2e-5

# Export to ONNX
python export_onnx_multitask.py --checkpoint checkpoint_best_multitask.pt \
  --model-name microsoft/mdeberta-v3-base --output best_model_multitask.onnx --opset 17
```

### ONNX interface

- **Inputs:** `input_ids`, `attention_mask`, `span_starts`, `span_ends`, `span_batch_ids`
- **NER outputs:** `boundary_logits`, `coarse_logits`, `fine_logits`
- **SVO outputs:** `svo_boundary_logits`, `svo_logits`, `voice_logits`, `gender_logits`, `number_logits`

### Resources

- App README: `rag-app/README.md`
- Multi-head scripts: `training/multi-head/`
- Kotlin ONNX extractor: `connectors/ner/onnx-ner/src/main/kotlin/rag/connectors/ner/onnx/OnnxMultiHeadEntityExtractor.kt`
- Profile docs: `docs/README.opensource.md` · `docs/README.ops-prod.md` · `docs/README.bilingual.md`
