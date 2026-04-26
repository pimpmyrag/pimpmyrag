# PimpMyRAG - Guide bilingue FR/EN

## FR

### Presentation

PimpMyRAG est un framework RAG modulaire JVM pour l'extraction d'information structuree:

- NER multitete (boundary/coarse/fine)
- extraction SVO (sujet/verbe/objet)
- inference ONNX cote Kotlin

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
- Variables Render a renseigner: `NER_MODEL_PATH`, `NER_TOKENIZER_PATH`
- Le deploiement se declenche apres push Git sur la branche suivie par Render

---

## EN

### Overview

PimpMyRAG is a modular JVM RAG framework focused on structured information extraction:

- multi-head NER (boundary/coarse/fine)
- SVO extraction (subject/verb/object)
- Kotlin-side ONNX inference

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
- Required Render env vars: `NER_MODEL_PATH`, `NER_TOKENIZER_PATH`
- Deployment is triggered by a Git push on the branch tracked by Render

