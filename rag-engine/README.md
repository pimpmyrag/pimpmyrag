# rag-engine

Interfaces du pipeline RAG. Définit les **contrats sans implémentation**.
Toute implémentation concrète vit dans `connectors/`.

## Interfaces

```kotlin
interface Chunker            // Découpage de RagDocument en unités
interface Embedder           // Vectorisation en FloatArray
interface NerExtractor       // Extraction d'entités → List<List<Entity>>
interface NerExtractorFromUD // NER consommant la sortie UD directement
interface UDParser           // Parse Universal Dependencies → UDDocument
interface Retriever          // Recherche vectorielle → List<RagDocument>
interface Reranker           // Reranking → List<ScoredDocument>
interface DocumentFilter     // Filtre ingest → List<RagDocument>
interface VectorStoreReader  // Lecture store vectoriel
interface VectorStoreWriter  // Écriture store vectoriel
interface DocumentStoreReader
interface DocumentStoreWriter
```

## Dépendances

- `rag-model` uniquement

