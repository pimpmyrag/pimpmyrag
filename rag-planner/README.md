
# rag-planner (Planner + ExecutionPlan + PlanRunner)

Ce module transforme un `RagConfig` (DSL) en `ExecutionPlan` **immuable**, puis fournit un `PlanRunner` (synchrones) pour exécuter ce plan via les factories.

## Usage (pseudo)
```kotlin
val cfg = rag { /* ... */ }
val plan = Planner().plan(cfg)
val answer = PlanRunner(chunkerFs, embedderFs, writerFs, readerFs, retrieverFs, filterFs, rerankerFs, generatorFs)
    .run(plan, query = "Quel fut son métier ?", raw = RagDocument("doc1", fullText))
```

> Pour un mode asynchrone (Flow), créez `PlanRunnerAsync` qui parcourt les mêmes `Step` mais compose des `Flow`.
