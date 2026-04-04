# connectors/rerank/onnx-ce

Cross-encoder **ONNX local** pour le reranking — pas de service externe.
Utilise DJL HuggingFace Tokenizer + ONNX Runtime.
Supporte les modèles de type **RERANKER** (binaire) et **NLI** (3 classes).

## Usage

```kotlin
val reranker = OnnxReranker(
    modelPath    = "models/bge-reranker-v2-m3/model.onnx",
    tokenizerDir = "models/bge-reranker-v2-m3",
    modelType    = OnnxModelType.RERANKER,   // ou NLI
    maxLen       = 512
)

val scored: List<ScoredDocument> = reranker.rerank(
    query  = "Qui a signé l'accord ?",
    docs   = retrievedDocs,
    topK   = 10
)
```

## Modes de scoring

| `OnnxModelType` | Score produit |
|---|---|
| `RERANKER` | `sigmoid(logit[1])` — probabilité de pertinence [0,1] |
| `NLI` | `softmax(logits)[2]` — score d'entailment |

## Détection automatique

- `token_type_ids` : inclus uniquement si le modèle les requiert (auto-détecté au chargement)
- Séquences paddées à `maxLen`, troncation sur les deux branches (query + doc)

## Dépendances

- `rag-model`
- `rag-engine`

