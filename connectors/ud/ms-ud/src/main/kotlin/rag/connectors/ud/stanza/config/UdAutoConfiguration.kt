package rag.connectors.ud.stanza.config

import org.springframework.boot.autoconfigure.AutoConfiguration
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.core.env.Environment
import org.springframework.web.reactive.function.client.ExchangeStrategies
import org.springframework.web.reactive.function.client.WebClient
import rag.connectors.ud.UdWebClient
import rag.connectors.ud.WebUdParser
import rag.engine.UDParser

/**
 * Autoconfiguration for UD client. Reads properties from Environment:
 * - ud.client.base-url (or ud.client.baseUrl)
 * - ud.client.base-path (or ud.client.basePath)
 * - ud.client.batch-size (or ud.client.batchSize)
 */
@AutoConfiguration
@EnableConfigurationProperties(UdClientProperties::class)
class UdAutoConfiguration(private val env: Environment) {

    @Bean
    @ConditionalOnMissingBean
    fun udBaseUrl(props: UdClientProperties): String {
        val host = props.host
        val port = props.port
        return "http://$host:$port"
    }

    @Bean
    @ConditionalOnMissingBean
    fun webClientBuilder(): WebClient.Builder {
        // increase buffer size if needed
        val strategies = ExchangeStrategies.builder()
            .codecs { cs -> cs.defaultCodecs().maxInMemorySize(16 * 1024 * 1024) }
            .build()
        return WebClient.builder().exchangeStrategies(strategies)
    }

    @Bean
    @ConditionalOnMissingBean
    fun udWebClient(builder: WebClient.Builder, baseUrl: String, props: UdClientProperties): UdWebClient {
        val wc = builder.baseUrl(baseUrl).build()
        return UdWebClient(wc, basePath = props.basePath, batchSize = props.batchSize)
    }

    @Bean
    @ConditionalOnMissingBean
    fun udParser(udWebClient: UdWebClient): UDParser = WebUdParser(udWebClient)
}