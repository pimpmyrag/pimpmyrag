# rag-app

Application Spring Boot qui assemble tous les modules en un service HTTP.

## Démarrage

```zsh
# Démarrer les services (MongoDB, Qdrant, Infinity, UD parser)
docker-compose up -d

# Lancer l'application
./gradlew :rag-app:bootRun
# → http://localhost:8080
```

## Endpoints principaux

| Méthode | URL | Description |
|---|---|---|
| `POST` | `/api/classify/extract/candidates` | Extraction NER single |
| `POST` | `/api/classify/extract/candidates/batch` | Extraction NER batch |
| `POST` | `/api/ingest` | Ingestion document |
| `GET`  | `/api/retrieve` | Retrieval RAG |

## Configuration

`src/main/resources/application.yml` — URLs des connecteurs :

```yaml
infinity:
  base-url: http://localhost:7997
qdrant:
  host: localhost
  port: 6333
spring.data.mongodb:
  uri: mongodb://root:password@localhost:27017/admin?authSource=admin
```

## Stack

- Spring Boot 3.3
- Kotlin 1.9
- MongoDB (document store)
- Qdrant (vector store)
- Infinity (embeddings + rerank)
- UD parser (Stanza via HTTP)

## Dépendances Gradle

Tous les modules `rag-*` et `connectors/*` actifs.
