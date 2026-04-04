# connectors/rag-connectors-stub

Implémentations no-op / in-memory pour les tests unitaires et d'intégration.
Aucune dépendance externe (pas de Docker, pas de service réseau requis).

## Factories disponibles

| Factory | `supports(name)` | Comportement |
|---|---|---|
| `DefaultChunkerFactory` | `"default"` | Split sur `.`, produit des `SENTENCE` |
| `StubEmbedderFactory` | `"bge-m3"` | Vecteurs `FloatArray(8)` déterministes par hash |
| `StubVectorWriterFactory` | `"chroma"` | Stocke dans `InMemoryVectorIndex` (singleton) |
| `StubVectorReaderFactory` | `"chroma"` | Cosine similarity sur `InMemoryVectorIndex` |
| `StubDocWriterFactory` | `"mongo"` | Stocke dans `InMemoryDocStore` (singleton) |
| `StubDocReaderFactory` | `"mongo"` | Lookup par id sur `InMemoryDocStore` |
| `StubRerankerFactory` | `"xnli"`, `"pingpong"` | Tri par longueur de texte |
| `StubGeneratorFactory` | `"mistral"` | Renvoie `"Q: … A: <premier doc>"` |
| `StubFilterFactory` | `"language"`, `"length"`, `"cleanup"` | Conserve les docs ≥ 3 caractères |

## Usage dans les tests

```kotlin
val runner = PlanRunnerDag(
    chunkerFactory   = DefaultChunkerFactory(),
    embedderFactory  = StubEmbedderFactory(),
    writerFactory    = StubVectorWriterFactory(),
    readerFactory    = StubVectorReaderFactory(),
    retrieverFactory = StubRetrieverFactory(),   // si besoin
    filterFactory    = StubFilterFactory(),
    rerankerFactory  = StubRerankerFactory(),
    generatorFactory = StubGeneratorFactory(),
)
val answer = runner.run(plan, query = "Qui ?", raw = RagDocument("d1", "Alice signe le traité."))
```

## Stores partagés

`InMemoryVectorIndex` et `InMemoryDocStore` sont des `object` Kotlin (singleton JVM).
Penser à les vider entre les tests si l'isolation est requise :

```kotlin
@BeforeEach fun reset() {
    InMemoryVectorIndex.idToVec.clear()
    InMemoryDocStore.byId.clear()
}
```

## Dépendances

- `rag-model`
- `rag-engine`
- `rag-planner`
- `rag-runner`

