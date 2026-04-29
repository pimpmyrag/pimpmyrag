# PimpMyRAG - Guide bilingue FR/EN

## FR

### Presentation

PimpMyRAG est un framework RAG modulaire JVM pour l'extraction d'information structuree:

- NER multitete (boundary/coarse/fine) — 32 labels fins, 8 familles coarse
- extraction SVO v4 (sujet/verbe/objet/obliques) avec eventlets structurés
- inference ONNX cote Kotlin

### Architecture SVO v4 (FEATURE PREVIEW)

Le modele v4 utilise **deux signaux indépendants** sur le meme forward pass NER :

| Signal | Head | Role |
|--------|------|------|
| `svo_boundary_head` | détecteur de verbes | `svoBoundaryProb` eleve sur vrais verbes, ~0 sur NP |
| `role_head` | classifieur de roles NP | `roleProb` ~0.99 sur args solides (SUBJECT/OBJECT/OBLIQUE/…) |

**Roles v4** : `NONE` (verbe trigger) · `SUBJECT` · `OBJECT` · `OBLIQUE` · `OBLIQUE_AGENT` · `OBLIQUE_CAUSE` · `APPOS` · `pron_subj` · `pron_obj`

**Eventlets** : apres extraction, les spans sont groupes par verbe gouverneur (`govVerbCharStart`) en evenements structures `verb + subject + obj + iobjs + causes + appositions`.

**`p_confidence`** (score unifie) : `svoBoundaryProb` pour les verbes, `roleProb` pour les args NP.

### MCP — outils de calibration

Config client : `ner-demo/mcp-client-config.json`

| Outil MCP | Description |
|-----------|-------------|
| `getConfig` | Lire les seuils courants |
| `setThreshold` | Modifier un seuil (`tauSvoBoundary` controle les verbes en v4) |
| `analyzeText` | NER+SVO v4 : entities, svoSpans, **eventlets** |
| `evaluateSvoPreview` | Protocole qualite SVO v4 : verbSpans, argumentSpans, eventlets |
| `scanThreshold` | Balayage recall/precision |
| `analyzeBatch` | Stats corpus avec distribution des roles SVO v4 |
| `applyAndAnalyze` | Commit config + verification en un appel |
| `probePerformance` | Benchmark latence/throughput |

### Installation rapide

Prerequis:

- JDK 17+
- Gradle 8+
- Python 3.11+
- Docker

```zsh
./gradlew build
docker-compose up -d
./gradlew :rag-app:bootRun
```

API: `http://localhost:8080`

### Demo

```zsh
./gradlew :ner-demo:bootRun
```

- UI: `http://localhost:8090`
- MCP SSE: `http://localhost:8090/mcp/sse`

### Training multitete

```zsh
cd training/multi-head
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Construire dataset multitask puis entrainer:

```zsh
python build_multitask_dataset.py --input data/train.jsonl --output data/train_multitask.jsonl --model-name microsoft/deberta-v3-base
python train_multi_task.py --train data/train_multitask.jsonl --val data/val_multitask.jsonl --test data/test_multitask.jsonl --model-name microsoft/deberta-v3-base
```

### Export ONNX

```zsh
python export_onnx_multitask.py --checkpoint checkpoint_best_multitask.pt --model-name microsoft/mdeberta-v3-base --output best_model_multitask.onnx --opset 17
```

### Liens

- Hub principal: `README.md`
- Guide open-source: `docs/README.opensource.md`
- Guide ops/prod: `docs/README.ops-prod.md`

### Deploiement Render (demo)

- Spec Render: `render.yaml`
- Build: `./gradlew :ner-demo:bootJar -x test`
- Start: `bash scripts/render/start-ner-demo-render.sh`
- Variables Render recommandees: `MODEL_URL`, `MODEL_SHA256`, `TOKENIZER_URL`, `TOKENIZER_SHA256`
- Le deploiement se declenche apres push Git sur la branche suivie par Render
- Attention: le modele ONNX et le tokenizer ne sont pas versionnes dans Git; ils doivent etre provisionnes avant le boot.

---

## EN

### Overview

PimpMyRAG is a modular JVM RAG framework focused on structured information extraction:

- multi-head NER (boundary/coarse/fine) — 32 fine labels, 8 coarse families
- SVO v4 extraction (subject/verb/object/obliques) with structured eventlets
- Kotlin-side ONNX inference

### SVO v4 Architecture (FEATURE PREVIEW)

The v4 model uses **two independent signals** on the same NER forward pass:

| Signal | Head | Role |
|--------|------|------|
| `svo_boundary_head` | verb trigger detector | `svoBoundaryProb` high on true verbs, ~0 on NPs |
| `role_head` | NP argument role classifier | `roleProb` ~0.99 on solid args (SUBJECT/OBJECT/OBLIQUE/…) |

**V4 roles**: `NONE` (verb trigger) · `SUBJECT` · `OBJECT` · `OBLIQUE` · `OBLIQUE_AGENT` · `OBLIQUE_CAUSE` · `APPOS` · `pron_subj` · `pron_obj`

**Eventlets**: after SVO extraction, spans are grouped by governing verb (`govVerbCharStart`) into structured events `verb + subject + obj + iobjs + causes + appositions`.

**`p_confidence`** (unified score): `svoBoundaryProb` for verbs, `roleProb` for NP args.

### MCP — calibration tools

Client config: `ner-demo/mcp-client-config.json`

| MCP Tool | Description |
|----------|-------------|
| `getConfig` | Read current thresholds |
| `setThreshold` | Update a threshold (`tauSvoBoundary` controls verb detection in v4) |
| `analyzeText` | NER+SVO v4: entities, svoSpans, **eventlets** |
| `evaluateSvoPreview` | Qualitative SVO v4 protocol: verbSpans, argumentSpans, eventlets |
| `scanThreshold` | Recall/precision sweep |
| `analyzeBatch` | Corpus stats with SVO v4 role distribution |
| `applyAndAnalyze` | Commit config + verify in one round-trip |
| `probePerformance` | Latency/throughput benchmark |

### Quick install

Requirements:

- JDK 17+
- Gradle 8+
- Python 3.11+
- Docker

```zsh
./gradlew build
docker-compose up -d
./gradlew :rag-app:bootRun
```

API: `http://localhost:8080`

### Demo

```zsh
./gradlew :ner-demo:bootRun
```

- UI: `http://localhost:8090`
- MCP SSE: `http://localhost:8090/mcp/sse`

### Multi-head training

```zsh
cd training/multi-head
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Build multitask dataset and train:

```zsh
python build_multitask_dataset.py --input data/train.jsonl --output data/train_multitask.jsonl --model-name microsoft/deberta-v3-base
python train_multi_task.py --train data/train_multitask.jsonl --val data/val_multitask.jsonl --test data/test_multitask.jsonl --model-name microsoft/deberta-v3-base
```

### ONNX export

```zsh
python export_onnx_multitask.py --checkpoint checkpoint_best_multitask.pt --model-name microsoft/mdeberta-v3-base --output best_model_multitask.onnx --opset 17
```

### Links

- Main hub: `README.md`
- Open-source guide: `docs/README.opensource.md`
- Ops/prod guide: `docs/README.ops-prod.md`

### Render deployment (demo)

- Render spec: `render.yaml`
- Build: `./gradlew :ner-demo:bootJar -x test`
- Start: `bash scripts/render/start-ner-demo-render.sh`
- Recommended Render env vars: `MODEL_URL`, `MODEL_SHA256`, `TOKENIZER_URL`, `TOKENIZER_SHA256`
- Deployment is triggered by a Git push on the branch tracked by Render
- Note: ONNX model and tokenizer artifacts are not tracked in Git and must be provisioned before startup.

