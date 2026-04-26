# PimpMyRAG - Guide externe open-source

Ce guide est la version "decouverte/contribution" du projet.

## 1) Ce que fait PimpMyRAG

PimpMyRAG est un framework RAG modulaire JVM, focalise sur l'extraction d'information structuree:

- NER multitete (boundary/coarse/fine)
- extraction syntaxique SVO (sujet/verbe/objet)
- connecteurs modulaires (embeddings, rerank, store, LLM)

## 2) Quickstart local

Prerequis:

- JDK 17+
- Gradle 8+
- Python 3.11+
- Docker

Commandes:

```zsh
./gradlew build
docker-compose up -d
./gradlew :rag-app:bootRun
```

API locale: `http://localhost:8080`

## 3) Modules a connaitre

- `rag-model`: modeles partages
- `rag-engine`: interfaces du pipeline
- `rag-app`: service Spring Boot principal
- `connectors/ner/onnx-ner`: inference NER/SVO ONNX
- `training/multi-head`: dataset, entrainement et export ONNX
- `ner-demo`: demo UI + MCP

## 4) Tester rapidement

```zsh
python scripts/ner_candidates_test.py
```

## 5) Contribution

- Ouvrir une issue avant refacto majeur
- Proposer des PR petites et testables
- Inclure une note de perf si changement sur `connectors/ner/onnx-ner`

## 6) Ressources

- README principal: `README.md`
- README application: `rag-app/README.md`
- Guide ops/prod: `docs/README.ops-prod.md`
- Guide bilingue FR/EN: `docs/README.bilingual.md`

