package rag.connectors.embed.onnx.config

import org.springframework.boot.autoconfigure.AutoConfiguration
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import rag.connectors.embed.onnx.OnnxOrtEmbedder

@AutoConfiguration
@EnableConfigurationProperties(OnnxEmbeddingConfig::class)
class OnnxEmbeddingAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    fun onnxEmbedder(config: OnnxEmbeddingConfig): OnnxOrtEmbedder {
        return OnnxOrtEmbedder(
            modelPath = config.modelPath,
            tokenizerDir = config.tokenizerDir,
            intraOpThreads = config.intraOpThreads,
            interOpThreads = config.interOpThreads,
            maxLen = config.maxLen,
            l2Normalize = config.l2Normalize,
            useGpu = config.useGpu,
            gpuDeviceId = config.gpuDeviceId
        )
    }
}
