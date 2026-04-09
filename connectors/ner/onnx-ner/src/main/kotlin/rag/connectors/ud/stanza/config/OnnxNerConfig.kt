package rag.connectors.ud.stanza.config

import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.boot.context.properties.bind.DefaultValue

@ConfigurationProperties(prefix = "onnx.ner.ud")
data class OnnxNerConfig(
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
    /** CoreML EP : Apple Neural Engine + GPU (Mac uniquement, ignoré ailleurs). */
    @DefaultValue("false")
    val useCoreMl: Boolean,
    val intraOpThreads: Int = Runtime.getRuntime().availableProcessors(),
    val interOpThreads: Int = 1,
    /** Nombre de phrases UDSentence regroupées en un seul appel ONNX. */
    @DefaultValue("8")
    val sentBatchSize: Int = 8,
)
