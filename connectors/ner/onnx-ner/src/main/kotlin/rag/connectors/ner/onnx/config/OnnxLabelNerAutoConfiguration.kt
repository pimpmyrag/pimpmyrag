package rag.connectors.ner.onnx.config

import org.springframework.boot.autoconfigure.AutoConfiguration
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import rag.connectors.ner.onnx.OnnxBilouEntityExtractor
import rag.connectors.ud.stanza.config.OnnxLabelNerConfig
import rag.engine.NerExtractor

@AutoConfiguration
@EnableConfigurationProperties(OnnxLabelNerConfig::class)
class OnnxLabelNerAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    fun onnxNer(config: OnnxLabelNerConfig): NerExtractor {
        return OnnxBilouEntityExtractor(
            modelPath = config.modelPath,
            tokenizerDir = config.tokenizerDir,
        )
    }
}
