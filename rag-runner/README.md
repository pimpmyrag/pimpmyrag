# rag-runner

Exécuteur de pipeline sous forme de DAG. Prend un `ExecutionPlan`
produit par `rag-planner` et orchestre les étapes via les factories.

## Usage

```kotlin
val runner = PlanRunnerDag(
    chunkerFactory   = chunkerFs,
    embedderFactory  = embedderFs,
    writerFactory    = writerFs,
    readerFactory    = readerFs,
    retrieverFactory = retrieverFs,
    filterFactory    = filterFs,
    rerankerFactory  = rerankerFs,
    generatorFactory = generatorFs,
)

val answer: String = runner.run(
    plan  = plan,
    query = "Qui a signé le traité ?",
    raw   = RagDocument("doc-1", fullText),
)
```

## Extension async

Pour un mode `Flow` (Kotlin coroutines), implémenter `PlanRunnerAsync`
en réutilisant le même `ExecutionPlan` et en composant des `Flow<T>`
à la place des `List<T>`.

## Dépendances

- `rag-model`
- `rag-engine`
- `rag-planner`

