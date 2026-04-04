# connectors/embed/onnx-emb

`Embedder` ONNX entièrement **local** — pas de service externe requis.
Utilise DJL HuggingFace Tokenizer + ONNX Runtime (ORT).
Supporte CPU, CUDA et Core ML (macOS).

## Auto-configuration Spring

Ajouter le JAR au classpath, puis configurer :

```yaml
onnx:
  embedding:
    model-path: "models/bge-m3/model.onnx"
    tokenizer-dir: "models/bge-m3"
    l2-normalize: true
    max-len: 512
    use-gpu: false
    gpu-device-id: 0
    intra-op-threads: 8   # défaut = nombre de cœurs disponibles
    inter-op-threads: 1
```

Bean fourni : `OnnxOrtEmbedder` (impl `Embedder`, `AutoCloseable`)

## Usage manuel

```kotlin
OnnxOrtEmbedder(
    modelPath    = "models/e5-large/model_quantized.onnx",
    tokenizerDir = "models/e5-large",
    maxLen       = 512,
    l2Normalize  = true
).use { embedder ->
    val vecs: List<FloatArray> = embedder.embed(docs)
}
```

## Accélération matérielle

| Environnement | Comportement |
|---|---|
| macOS (Apple Silicon) | Active Core ML automatiquement |
| Linux + GPU | Active CUDA si `use-gpu: true` |
| Fallback | CPU multi-thread ORT (`intra-op-threads`) |

## Dépendances

- `rag-model`
- `rag-engine`

