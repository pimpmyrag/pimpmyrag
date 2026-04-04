
package com.acme.infinity.config

import com.acme.infinity.client.IInfinityClient
import com.acme.infinity.client.InfinityClient
import com.acme.infinity.lc4j.LangChain4jInfinityEmbeddingModel
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
@EnableConfigurationProperties(InfinityConfig::class)
class InfinityClientsAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean(name = ["infinityHttpClient"])
    fun infinityHttpClient(config: InfinityConfig): HttpClient {
        val provider = ConnectionProvider.builder("infinity-pool")
            .maxConnections(config.concurrency * 3)
            .pendingAcquireMaxCount(-1)
            .maxIdleTime(Duration.ofSeconds(30))
            .maxLifeTime(Duration.ofMinutes(10))
            .lifo()
            .build()

        return HttpClient.create(provider)
            .compress(true)
            .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, config.connectTimeoutMs.toInt())
            .responseTimeout(Duration.ofMillis(config.requestTimeoutMs))
            .doOnConnected { conn ->
                conn.addHandlerLast(ReadTimeoutHandler(config.requestTimeoutMs, TimeUnit.MILLISECONDS))
                    .addHandlerLast(WriteTimeoutHandler(config.requestTimeoutMs, TimeUnit.MILLISECONDS))
            }
    }

    @Bean
    @ConditionalOnMissingBean(name = ["infinityWebClient"])
    fun infinityWebClient(@Qualifier("infinityHttpClient") httpClient: HttpClient, config: InfinityConfig): WebClient {
        val strategies = ExchangeStrategies.builder()
            .codecs { it.defaultCodecs().maxInMemorySize(32 * 1024 * 1024) }
            .build()
        return WebClient.builder()
            .baseUrl(config.baseUrl.trimEnd('/'))
            .clientConnector(ReactorClientHttpConnector(httpClient))
            .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
            .defaultHeader(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE)
            .defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer ${config.apiKey}")
            .exchangeStrategies(strategies)
            .build()
    }

    @Bean
    @ConditionalOnMissingBean
    fun infinityClient(@Qualifier("infinityWebClient") webClient: WebClient, config: InfinityConfig): IInfinityClient {
        return InfinityClient(webClient, config)
    }

    @Bean("infinityLangChain4jEmbedder")
    @ConditionalOnMissingBean
    fun langChain4jInfinityEmbeddingModel(client: IInfinityClient): LangChain4jInfinityEmbeddingModel {
        return LangChain4jInfinityEmbeddingModel(client)
    }
}
