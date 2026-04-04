
package com.acme.infinity.config

import com.acme.infinity.lc4j.LangChain4jInfinityScoringModel
import com.acme.infinity.rerank.RerankClient
import io.netty.channel.ChannelOption
import io.netty.handler.timeout.ReadTimeoutHandler
import io.netty.handler.timeout.WriteTimeoutHandler
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.boot.autoconfigure.AutoConfiguration
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.http.HttpHeaders
import org.springframework.http.MediaType
import org.springframework.http.client.reactive.ReactorClientHttpConnector
import org.springframework.web.reactive.function.client.ExchangeStrategies
import org.springframework.web.reactive.function.client.WebClient
import reactor.netty.http.client.HttpClient
import reactor.netty.resources.ConnectionProvider
import java.time.Duration
import java.util.concurrent.TimeUnit

@AutoConfiguration
@EnableConfigurationProperties(RerankProperties::class)
class InfinityClientsAutoConfiguration {


    // --- Rerank beans ---
    @Bean
    @ConditionalOnMissingBean(name = ["infinityRerankHttpClient"])
    fun infinityRerankHttpClient(props: RerankProperties): HttpClient =
        httpClient(props.concurrency, props.connectTimeoutMs, props.requestTimeoutMs)

    @Bean
    @ConditionalOnMissingBean(name = ["infinityRerankWebClient"])
    fun infinityRerankWebClient(@Qualifier("infinityRerankHttpClient") httpClient: HttpClient, props: RerankProperties): WebClient {
        val strategies = ExchangeStrategies.builder()
            .codecs { it.defaultCodecs().maxInMemorySize(32 * 1024 * 1024) }
            .build()
        return WebClient.builder()
            .baseUrl(props.baseUrl.trimEnd('/'))
            .clientConnector(ReactorClientHttpConnector(httpClient))
            .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
            .defaultHeader(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE)
            .defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer ${props.apiKey}")
            .exchangeStrategies(strategies)
            .build()
    }

    @Bean()
    @ConditionalOnMissingBean
    fun infinityRerankClient(@Qualifier("infinityRerankWebClient") webClient: WebClient, props: RerankProperties) =
        RerankClient(webClient, props)

    @Bean("reranker-bge-m3")
    @ConditionalOnMissingBean
    fun langChain4jInfinityScoringModel(client: RerankClient, props: RerankProperties) =
        LangChain4jInfinityScoringModel(client, props)

    private fun httpClient(concurrency: Int, connectMs: Long, readMs: Long): HttpClient {
        val provider = ConnectionProvider.builder("infinity-pool")
            .maxConnections(concurrency * 3)
            .pendingAcquireMaxCount(-1)
            .maxIdleTime(Duration.ofSeconds(30))
            .maxLifeTime(Duration.ofMinutes(10))
            .lifo()
            .build()
        return HttpClient.create(provider)
            .compress(true)
            .followRedirect(true)
            .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, connectMs.toInt())
            .responseTimeout(Duration.ofMillis(readMs))
            .doOnConnected { conn ->
                conn.addHandlerLast(ReadTimeoutHandler(readMs, TimeUnit.MILLISECONDS))
                    .addHandlerLast(WriteTimeoutHandler(readMs, TimeUnit.MILLISECONDS))
            }
    }
}
