# PimpMyRAG

[![CI](https://github.com/pimpmyrag/pimpmyrag/actions/workflows/ci.yml/badge.svg)](https://github.com/pimpmyrag/pimpmyrag/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen.svg)](https://opensource.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/pimpmyrag/pimpmyrag/pulls)
[![GitHub Stars](https://img.shields.io/github/stars/pimpmyrag/pimpmyrag?style=social)](https://github.com/pimpmyrag/pimpmyrag/stargazers)
[![Kotlin](https://img.shields.io/badge/Kotlin-2.x-7F52FF?logo=kotlin&logoColor=white)](https://kotlinlang.org)
[![JVM](https://img.shields.io/badge/JVM-17%2B-ED8B00?logo=openjdk&logoColor=white)](https://adoptium.net)

Modular JVM based RAG framework attempt, for deterministic event extraction purpose mainly, but open to other kind of text extractions.

## Architecture

```
pimpmyrag/
├── rag-model/                  Modèles de données partagés (UDToken, Entity, …)
├── rag-engine/                 Interfaces du pipeline (Chunker, Embedder, NerExtractor, …)
├── rag-dsl/                    DSL Kotlin déclaratif pour configurer un pipeline RAG
├── rag-dsl-staged/             Variante DSL type-safe par étapes
├── rag-planner/                Compile un RagConfig en ExecutionPlan immuable
├── rag-runner/                 Exécute un ExecutionPlan via DAG
├── rag-app/                    Application Spring Boot d'assemblage
├── radar-nli-toolkit/          Classifieur NLI + radar sémantique
├── connectors/
│   ├── ner/onnx-ner/           Pipeline NER 2 niveaux (XLM-RoBERTa + DeBERTa-v3)
│   ├── ud/ms-ud/               Client UD (Stanza/UDPipe via HTTP)
│   ├── embed/infinity/         Embedder via Infinity API
│   ├── embed/onnx-emb/         Embedder ONNX local (DJL)
│   ├── rerank/infinity/        Reranker via Infinity API
│   ├── rerank/onnx-ce/         Cross-encoder ONNX local
│   ├── document-store/mongodb/ Store de documents MongoDB
│   ├── vector/qadrant/         Store vectoriel Qdrant
│   ├── llm/chat-completion/    LLM via API chat-completion
│   └── rag-connectors-stub/    Stubs pour les tests
├── training/                   Scripts Python (XLM-RoBERTa NER + DeBERTa SpanClassifier)
└── scripts/                    Évaluation, tests fonctionnels, utilitaires
```

## Prérequis

- JDK 21+
- Gradle 9+
- Python 3.11+ (pour `training/`)
- Docker (services UD, Infinity, Qdrant, MongoDB)

## Build

```zsh
./gradlew build
```

## Lancement (dev)

```zsh
docker-compose up -d          # Démarre MongoDB, Qdrant, Infinity, UD parser
./gradlew :rag-app:bootRun    # API sur http://localhost:8080
```

## Pipeline NER (2 niveaux)

```
Texte → XLM-RoBERTa BIO → coarse (PER/LOC/ORG/TIME/EVENT/OBJECT)
      → mergeNerLabelWithUD → raffinement span + split rôle/nom
      → DeBERTa-v3 SpanClassifier → 22 labels fin-grained
      → EntityCandidate { text, lemma, nerType, nerHint, headDeprel,
                          hopFromTrigger, feats, … }
```

**Résultats courants (POC v0, 11k phrases)** : 87% — cible 15k : ~90%

## Pipeline Eventlet (en cours)

Extraction d'eventlets structurés (≈ 50 types d'événements) :

```
UD tree → VERB root / HINT_EVENT_NOMINAL → trigger
        → EntityCandidate.withHopFrom(trigger.id, sentence.tokens)
        → filtrer hop ≤ 2
        → LR(headDeprel, nerHint, nerType, voice, hop) → rôle arg
        → template matching (embedding centroid) → type d'événement
        → indexation Qdrant avec payload filtrable
```

## Tests NER

```zsh
python scripts/ner_candidates_test.py   # 207 cas, service sur :8080
```

## Training

```zsh
cd training/training_package
python train.py --train train.jsonl --val val.jsonl --test test.jsonl \
                --epochs 10 --coarse-noise 0.20
```

Voir [`training/README.md`](training/README.md) pour le détail complet.
