package com.acme.infinity.client

import com.acme.infinity.config.InfinityConfig
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.asFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.toList
import org.springframework.web.reactive.function.client.WebClient
import org.springframework.web.reactive.function.client.awaitBody

/**
 * Spring WebClient-based implementation of the Infinity client.
 */
class InfinityClient(
    private val webClient: WebClient,
    private val config: InfinityConfig
) : IInfinityClient {

    private data class EmbeddingRequest(val model: String, val input: List<String>)
    private data class EmbeddingData(val embedding: FloatArray, val index: Int)
    private data class EmbeddingResponse(val data: List<EmbeddingData>)

    override suspend fun embed(texts: List<String>): List<FloatArray> {
        if (texts.isEmpty()) {
            return emptyList()
        }

        return texts.chunked(config.batchSize)
            .asFlow()
            .map { batch ->
                val request = EmbeddingRequest(model = config.modelName, input = batch)
                val response = webClient.post()
                    .bodyValue(request)
                    .retrieve()
                    .awaitBody<EmbeddingResponse>()
                response.data.sortedBy { it.index }.map { it.embedding }
            }
            .toList()
            .flatten()
    }
}
