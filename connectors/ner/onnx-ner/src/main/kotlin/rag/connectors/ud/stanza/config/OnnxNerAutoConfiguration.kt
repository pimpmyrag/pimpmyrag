package rag.connectors.ud.stanza.config

import org.springframework.boot.autoconfigure.AutoConfiguration
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import rag.connectors.ud.stanza.OnnxSpanNerExtractor
import rag.engine.NerExtractorFromUD

@AutoConfiguration
@EnableConfigurationProperties(OnnxNerConfig::class)
class OnnxNerAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    fun onnxNerUd(config: OnnxNerConfig): NerExtractorFromUD {
        return OnnxSpanNerExtractor(
            modelPath = config.modelPath,
            tokenizerDir = config.tokenizerDir,
        )
    }
}
