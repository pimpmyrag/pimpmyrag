package rag.connectors.embed.onnx.config

import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.boot.context.properties.bind.DefaultValue

@ConfigurationProperties(prefix = "onnx.embedding")
data class OnnxEmbeddingConfig(
    @DefaultValue("models/e5-large-mac/quantized/model_quantized.onnx")
    val modelPath: String,
    @DefaultValue("models/e5-large-mac")
    val tokenizerDir: String,
    @DefaultValue("true")
    val l2Normalize: Boolean,
    @DefaultValue("128")
    val maxLen: Int,
    @DefaultValue("false")
    val useGpu: Boolean,
    @DefaultValue("0")
    val gpuDeviceId: Int,
    val intraOpThreads: Int = Runtime.getRuntime().availableProcessors(),
    val interOpThreads: Int = 1
)
