
# infinity-search-spring (module)

Module Spring (autoconfiguration) combinant **Embeddings** et **Rerank** pour un worker **Infinity** déployé sur **RunPod**.

## Auto-configuration
Ajoute le JAR au classpath, puis configure :

```yaml
infinity:
  embedding:
    base-url: "https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1"
    api-key: "${RUNPOD_API_KEY}"
    model-name: "BAAI/bge-m3"
    batch-size: 32
    concurrency: 64
  rerank:
    base-url: "https://<WORKER_HOST>"  # Infinity direct /rerank, sinon https://api.runpod.ai/v2/<ENDPOINT_ID>
    api-key: "${RUNPOD_API_KEY}"
    model-name: "BAAI/bge-reranker-v2-m3"
    mode: infinity  # ou runpod
    concurrency: 64
```

Beans fournis :
- `EmbeddingClient` + `LangChain4jInfinityEmbeddingModel` (impl `EmbeddingModel` LC4J)  
- `RerankClient` + `LangChain4jInfinityScoringModel` (impl `ScoringModel` LC4J)

## Références
- Le worker *infinity-embedding* expose des **embeddings** (OpenAI‑compatible `/v1/embeddings`) et un **reranker** (REST `/rerank` type Cohere, ou via RunPod `/runsync`). citeturn10search59turn10search67turn10search74

```
MODEL_NAMES=BAAI/bge-m3;BAAI/bge-reranker-v2-m3
BACKEND=ctranslate2
DTYPES=auto;auto
RUNPOD_MAX_CONCURRENCY=300
INFINITY_QUEUE_SIZE=48000
```

## Exemples d’usage (injection)
```kotlin
@Autowired lateinit var embModel: dev.langchain4j.model.embedding.EmbeddingModel
@Autowired lateinit var scoringModel: dev.langchain4j.model.scoring.ScoringModel
```

## Tests d’intégration
Voir `src/test/kotlin` — ils utilisent des **variables d’environnement** et se **skip** s’il en manque.
- `RUNPOD_API_KEY` (obligatoire)
- `RUNPOD_EMB_BASE_URL` (ex: https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1)
- `RUNPOD_RERANK_BASE_URL` (Infinity: https://<host> | RunPod: https://api.runpod.ai/v2/<ENDPOINT_ID>)
- `RERANK_MODE` ("infinity" | "runpod", défaut: infinity)
```
