package com.acme.infinity.config

import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.boot.context.properties.bind.DefaultValue

/**
 * Generic configuration properties for the Infinity client.
 * Can be used by Spring, Ktor, or any other framework.
 */
@ConfigurationProperties(prefix = "infinity.embedding")
data class InfinityConfig(
    @DefaultValue("http://localhost:8008/v1/embeddings")
    val baseUrl: String,
    @DefaultValue("")
    val apiKey: String,
    @DefaultValue("BAAI/bge-m3")
    val modelName: String,
    @DefaultValue("32")
    val batchSize: Int,
    @DefaultValue("4")
    val concurrency: Int,
    @DefaultValue("10000")
    val connectTimeoutMs: Long,
    @DefaultValue("30000")
    val requestTimeoutMs: Long
)
