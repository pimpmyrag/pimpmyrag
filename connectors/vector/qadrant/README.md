# connectors/vector/qadrant

`VectorStoreWriter` et `VectorStoreReader` Qdrant.
Utilise le client officiel gRPC `io.qdrant:client`.
Conçu pour stocker les **centroïdes de triggers d'eventlets** et leurs payloads de participants.

## Usage

```kotlin
val client = QdrantClient(
    QdrantGrpcClient.newBuilder("localhost", 6334, false).build()
)

val writer = QdrantVectorStoreWriter(client, collectionName = "eventlets")
writer.write(embeddings, docs)   // upsert avec payload = doc.metadata

val reader = QdrantVectorStoreReader(client, collectionName = "eventlets")

// Recherche sans filtre
val hits: List<ScoredId> = reader.searchIds(queryEmbedding, topK = 20)

// Recherche filtrée par type d'eventlet et langue
val hits = reader.searchIds(
    queryEmbedding,
    topK = 20,
    filter = VectorFilter(mapOf("eventlet_type" to "TRANSFER", "lang" to "fr"))
)
```

## Mapping des métadonnées

Les valeurs de `RagDocument.metadata` sont automatiquement converties en payload Qdrant :

| Type Kotlin | Payload Qdrant |
|---|---|
| `String` | keyword |
| `Int` / `Long` | integer |
| `Boolean` | boolean |
| `Float` / `Double` | range (gte == lte) |
| Autre | keyword (toString) |

## Filtrage

`VectorFilter.metadata` génère des conditions `must` (AND) sur les champs du payload.
Supporte keyword, integer, boolean et range float.

## Dépendances

- `rag-model`
- `rag-engine`

