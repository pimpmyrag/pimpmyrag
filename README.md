# PimpMyRAG

[![CI](https://github.com/pimpmyrag/pimpmyrag/actions/workflows/ci.yml/badge.svg)](https://github.com/pimpmyrag/pimpmyrag/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen.svg)](https://opensource.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/pimpmyrag/pimpmyrag/pulls)
[![GitHub Stars](https://img.shields.io/github/stars/pimpmyrag/pimpmyrag?style=social)](https://github.com/pimpmyrag/pimpmyrag/stargazers)
[![Kotlin](https://img.shields.io/badge/Kotlin-2.x-7F52FF?logo=kotlin&logoColor=white)](https://kotlinlang.org)
[![JVM](https://img.shields.io/badge/JVM-21%2B-ED8B00?logo=openjdk&logoColor=white)](https://adoptium.net)

A modular JVM-based framework for RAG + deterministic information extraction.
The current focus is a recall-first NER backbone designed to feed a structured pipeline:

  eventlets -> participants -> coreference -> graph

TL;DR: Build a reproducible extraction backbone first (fast, testable, auditable),
then optionally plug LLMs on top for enrichment or hard cases.

---

## Why not just use an LLM?

LLMs are excellent, but this project targets:
- Determinism & auditability: same input -> same output; easy regression tests.
- Predictable cost & throughput: ONNX models + structured heuristics run locally and scale.
- Extraction-first: robust spans, types, heads, morphology and graph-ready facts.

LLMs are supported as optional connectors, but they are not required for the backbone.

---

## Current Status (POC v0)

### NER backbone (in active consolidation)
Pipeline is stable end-to-end and designed for high recall (downstream will prune/rank).
- Coarse NER (BIO) -> 6 families: PER/LOC/ORG/TIME/EVENT/OBJECT
- UD-aware span refinement + role/name splitting
- Fine-grained SpanClassifier -> 22 hints (roles, names, NORP, GPE, time types, etc.)
- Bench (adversarial): ~87% on ~11k sentences (goal ~90% on ~15k)

### Eventlets (v0 in progress)
Early scaffolding exists; participant typing and coref are the next ML steps.

---

## Architecture

```
pimpmyrag/
??? rag-model/                  Shared data models (UDToken, Entity, ...)
??? rag-engine/                 Pipeline interfaces (Chunker, Embedder, NerExtractor, ...)
??? rag-dsl/                    Kotlin DSL for pipeline configuration
??? rag-dsl-staged/             Type-safe staged DSL variant
??? rag-planner/                Compiles RagConfig into an immutable ExecutionPlan
??? rag-runner/                 Executes ExecutionPlan via DAG
??? rag-app/                    Spring Boot assembly app
??? radar-nli-toolkit/          NLI classifier + semantic radar toolkit
??? connectors/
?   ??? ner/onnx-ner/           2-stage NER (XLM-RoBERTa coarse + DeBERTa-v3 span typing)
?   ??? ud/ms-ud/               UD client (Stanza/UDPipe via HTTP)
?   ??? embed/infinity/         Embedder via Infinity API
?   ??? embed/onnx-emb/         Local ONNX embedder (DJL)
?   ??? rerank/infinity/        Reranker via Infinity API
?   ??? rerank/onnx-ce/         Local ONNX cross-encoder
?   ??? document-store/mongodb/ MongoDB document store
?   ??? vector/qdrant/          Qdrant vector store
?   ??? llm/chat-completion/    Optional LLM connector (chat-completion)
?   ??? rag-connectors-stub/    Stubs for tests
??? training/                   Python training scripts (coarse NER + span classifier)
??? scripts/                    Evaluation, functional tests, utilities
```

---

## Prerequisites

- JDK 21+
- Gradle 9+
- Python 3.11+ (for training/)
- Docker (UD, Infinity, Qdrant, MongoDB)

---

## Build

```bash
./gradlew build
```

---

## Run (dev)

```bash
docker-compose up -d          # MongoDB, Qdrant, Infinity, UD parser
./gradlew :rag-app:bootRun    # API on http://localhost:8080
```

---

## NER Pipeline (2-stage)

```
Text
  -> XLM-RoBERTa BIO tagging
      coarse: PER / LOC / ORG / TIME / EVENT / OBJECT
  -> mergeNerLabelWithUD (UD alignment + trimming + role/name split + expansions)
  -> DeBERTa-v3 SpanClassifier
      22 fine-grained hints
  -> EntityCandidate { text, lemma, nerType, nerHint, headDeprel,
                      hopFromTrigger, feats, ... }
```

Design principle: recall-first extraction to support downstream eventlets and participants.

---

## Eventlet Pipeline (WIP)

Goal: structured eventlets (~50 event types), with trigger + participant roles.

```
UD tree + NER hints
  -> trigger candidates (VERB root / HINT_EVENT_* etc.)
  -> candidate participants via hop distance (<= 2)
  -> role typing model (planned)
  -> event type resolution:
       - template matching (embedding centroid)
       - optional reranking
  -> index to Qdrant with filterable payload
```

---

## Tests

NER candidate tests (requires services on :8080):

```bash
python scripts/ner_candidates_test.py
```

---

## Training

```bash
cd training/training_package
python train.py --train train.jsonl --val val.jsonl --test test.jsonl                 --epochs 10 --coarse-noise 0.20
```

See training/README.md for details.

---

# Timeline & Roadmap

This roadmap is intentionally oriented toward deterministic event extraction:
build a strong NER scaffold, then train the eventlet-specific heads.

## Phase 0 - NOW (v0.x): Consolidate NER backbone (high priority)
Goal: maximize recall while controlling candidate explosion, and stabilize ontology.

Key items:
- Reduce candidate explosion (especially UD-based expansions)
- Add deterministic pruning (top-K overlap clusters, controlled diversity)
- Stabilize span boundaries (dates, de/du/d' chains, codes like AF447)
- Clarify ontology rules:
  - ORG vs GROUP_ROLE policy (or keep both hypotheses with scoring)
  - EVENT vs OBJECT policy for nominal events / documents
- Regression tests + benchmark reporting

Deliverable:
- v0.2: stable NER extraction with clear policies and reproducible tests

## Phase 1 - Participants head (ML): trigger -> role typing
Goal: train a model to type participant roles around a trigger.

Key items:
- Define participant role schema (agent/patient/instrument/location/time/etc.)
- Create training data from templates + weak supervision + manual corrections
- Train a lightweight role classifier (features: hop, deprel, voice, nerHint, nerType, feats)
- Evaluate on eventlet scenarios (recall-first)

Deliverable:
- v0.3: participant typing head + evaluation harness

## Phase 2 - Coreference v0 (ML + rules)
Goal: link mentions across sentences (pronouns + nominal mentions).

Key items:
- Coref features: lemma, gender/number/person, syntactic head, entity types
- Candidate generation: same-head/lemma, pronoun resolution, alias patterns
- Train a pair scorer (mention-pair scoring) + greedy clustering (v0)
- Keep pipeline deterministic and testable

Deliverable:
- v0.4: coref v0 producing mention clusters + links

## Phase 3 - Eventlet pivot templates
Goal: introduce a stable pivot schema so eventlets are easy to serialize, index, query.

Key items:
- Define eventlet JSON contract (trigger, type, participants, evidence spans)
- Template library (~50 event types): constraints + evidence patterns
- Event type resolution improvements (centroid matching + rerank)
- Provenance: evidence spans stored per decision

Deliverable:
- v0.5: eventlet templates + pivot schema + export/import

## Phase 4 - Coref v2 in Neo4j graph
Goal: persist entities/eventlets and coref edges in a graph for analytics and navigation.

Key items:
- Neo4j schema: Entity nodes, Mention nodes, Eventlet nodes
- Coref edges + confidence + provenance
- Graph queries for investigation workflows
- Optional incremental updates

Deliverable:
- v0.6: graph persistence + query examples

## Phase 5 - Manual research API + MCP
Goal: human-in-the-loop exploration and correction, plus tool integration.

Key items:
- Manual search endpoints (filterable evidence, spans, triggers)
- Correction endpoints (feedback -> dataset)
- MCP endpoints for agent/tool integrations (search, inspect, annotate)

Deliverable:
- v0.7: research API + MCP + feedback loop

## Phase 6 - Dedicated UI
Goal: a UI that supports investigation, auditability, and correction.

Key items:
- Timeline view of eventlets
- Participants graph view (coref clusters)
- Evidence explorer (spans, provenance)
- Annotate & retrain workflow

Deliverable:
- v0.8: dedicated UI + annotation feedback loop

---

## Contributing

PRs welcome. Please open an issue for feature requests or design discussions.

---

## Notes (FR)

PimpMyRAG est un framework JVM oriente extraction deterministe.
Le socle actuel est un pipeline NER recall-first enrichi par UD pour produire des candidats robustes,
afin d'alimenter ensuite l'extraction d'eventlets (trigger + participants) puis la coref et un graphe Neo4j.