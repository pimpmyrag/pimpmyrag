# rag-dsl-staged

Variante type-safe du DSL RAG. Chaque étape retourne un type distinct
qui contraint les étapes suivantes **à la compilation** — impossible
de sauter une étape ou de les réordonner.

## Exemple

```kotlin
val plan = rag
    .fromRaw()
    .chunk  { unit = RagUnitType.SENTENCE; maxTokens = 128 }
    .embed  { model = "bge-m3"; store = "qdrant" }
    .retrieve { genericTopK = 30; thematicTopK = 10 }
    .rerank { topK = 10 }
    .generate { llm = "mistral" }
```

## Avantage vs `rag-dsl`

Le compilateur Kotlin refuse un pipeline mal formé :

```kotlin
// ✗ Ne compile pas : embed avant chunk
rag.fromRaw().embed { … }.chunk { … }
```

## Dépendances

- `rag-model`
- `rag-engine`
- `rag-dsl`

