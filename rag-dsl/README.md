# rag-dsl

DSL Kotlin pour décrire un pipeline RAG de façon déclarative.

## Exemple

```kotlin
val config = rag {
    ingest  { unit = RagUnitType.SENTENCE }
    chunk   { using = "default"; maxTokens = 128 }
    embed   { model = "bge-m3"; store = "qdrant" }
    retrieve { strategy = "hybrid"; genericTopK = 30; thematicTopK = 10 }
    rerank  { strategy = "xnli"; topK = 10 }
    generate { llm = "mistral" }
}
```

## Configs disponibles

| Config | Description |
|---|---|
| `IngestConfig` | Source + unité + enrichers + filters |
| `ChunkConfig` | Stratégie de chunking + taille max tokens |
| `EmbedConfig` | Modèle + store cible + normalisation |
| `RetrieveConfig` | Stratégie hybride + topK generic/thematic |
| `RetrieveFilterConfig` | Filtres métadonnées sur la retrieval |
| `ReRankConfig` | Stratégie rerank + topK final |
| `GenerateConfig` | LLM cible |

## Dépendances

- `rag-model`
- `rag-engine`

