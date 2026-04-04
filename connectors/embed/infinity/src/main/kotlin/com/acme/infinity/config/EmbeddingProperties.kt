
package com.acme.infinity.config

import jakarta.validation.constraints.Max
import jakarta.validation.constraints.Min
import jakarta.validation.constraints.NotBlank
import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.validation.annotation.Validated

@Validated
@ConfigurationProperties(prefix = "infinity.embedding")
data class EmbeddingProperties(
    @field:NotBlank val baseUrl: String,          // e.g. https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1
    @field:NotBlank val apiKey: String,
    val modelName: String = "BAAI/bge-m3",
    @field:Min(1) val batchSize: Int = 32,
    @field:Min(1) @field:Max(512) val concurrency: Int = 64,
    @field:Min(1024) val maxBatchBytes: Int = 800_000,
    @field:Min(1) val maxRetries: Int = 5,
    val initialBackoffMs: Long = 200,
    val maxBackoffMs: Long = 8000,
    val requestTimeoutMs: Long = 1000,
    val connectTimeoutMs: Long = 10_000
)
