
package com.acme.infinity.config

import jakarta.validation.constraints.Max
import jakarta.validation.constraints.Min
import jakarta.validation.constraints.NotBlank
import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.validation.annotation.Validated

@Validated
@ConfigurationProperties(prefix = "infinity.rerank")
data class RerankProperties(
    @field:NotBlank val baseUrl: String,          // Infinity: https://<worker-host>  | RunPod: https://api.runpod.ai/v2/<ENDPOINT_ID>
    @field:NotBlank val apiKey: String,
    val modelName: String = "BAAI/bge-reranker-v2-m3",
    val mode: Mode = Mode.infinity,
    @field:Min(1) @field:Max(512) val concurrency: Int = 64,
    @field:Min(1) val maxRetries: Int = 5,
    val initialBackoffMs: Long = 200,
    val maxBackoffMs: Long = 8000,
    val requestTimeoutMs: Long = 60_000,
    val connectTimeoutMs: Long = 10_000
) {
    enum class Mode { infinity, runpod }
}
