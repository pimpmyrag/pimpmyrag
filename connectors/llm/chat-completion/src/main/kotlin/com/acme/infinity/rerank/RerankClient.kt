
package com.acme.infinity.rerank

import com.acme.infinity.config.RerankProperties
import org.springframework.web.reactive.function.client.WebClient
import org.springframework.web.reactive.function.client.bodyToMono
import reactor.core.publisher.Mono
import reactor.util.retry.Retry
import java.time.Duration

class RerankClient(
    private val webClient: WebClient,
    private val props: RerankProperties
) {
    /**
     * Rerank une liste de documents en fonction d'une requête.
     * @param query La requête utilisée pour le reranking.
     * @param docs La liste des documents à reranker.
     * @param topN Le nombre maximal de résultats à retourner (optionnel).
     * @param returnDocuments Si vrai, retourne aussi le texte des documents.
     * @return Une liste de paires (index du document, score de pertinence).
     */
    fun rerank(
        query: String,
        docs: List<String>,
        topN: Int? = null,
        returnDocuments: Boolean = false
    ): Mono<List<Pair<Int, Double>>> {
        return callRerankServer(query, docs, topN, returnDocuments)
    }



    private fun callRerankServer(
        query: String,
        docs: List<String>,
        topN: Int?,
        returnDocuments: Boolean
    ): Mono<List<Pair<Int, Double>>> {
        val request = RerankRequest(
            input = RerankRequestInput(
                action = "rerank",
                payload = RerankPayload(
                    query = query,
                    documents = docs,
                    return_documents = returnDocuments,
                    top_k = topN
                )
            )
        )

        return webClient.post()
            .bodyValue(request)
            .exchangeToMono { response ->
                if (response.statusCode().is2xxSuccessful) {
                    response.bodyToMono<RerankResponse>()
                } else {
                    response.createException().flatMap { Mono.error(it) }
                }
            }
            .retryWhen(
                Retry.backoff(props.maxRetries.toLong(), Duration.ofMillis(props.initialBackoffMs))
                    .maxBackoff(Duration.ofMillis(props.maxBackoffMs))
                    .jitter(0.5)
                    .filter { isRetryable(it) }
            )
            .map { response ->
                response.output.results.map { it.index to it.score }
            }
    }

    private fun isRetryable(t: Throwable): Boolean {
        val msg = t.message?.lowercase() ?: ""
        return listOf("429","too many requests","5xx","500","502","503","504","timeout","timed out","refused","reset","unreachable","connection").any { it in msg }
    }
}
