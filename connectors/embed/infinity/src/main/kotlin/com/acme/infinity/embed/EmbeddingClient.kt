package com.acme.infinity.embed

import com.acme.infinity.config.EmbeddingProperties
import kotlinx.coroutines.reactor.awaitSingle
import kotlinx.coroutines.runBlocking
import org.springframework.web.reactive.function.client.WebClient
import org.springframework.web.reactive.function.client.bodyToMono
import reactor.core.publisher.Flux
import reactor.core.publisher.Mono
import reactor.util.retry.Retry
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration

class EmbeddingClient(
    private val webClient: WebClient,
    private val props: EmbeddingProperties
) {
    fun embedAll(texts: List<String>): Mono<List<FloatArray>> {
        val body = mapOf(
            "input" to mapOf(
                "model" to "intfloat/multilingual-e5-large",
                "input" to listOf("First sentence to embed", "Second sentence to embed")
            )
        )

        println(java.net.InetAddress.getAllByName("api.runpod.ai").toList())
        require(texts.isNotEmpty())
        // Diviser en sous-listes de 1000 pour éviter les gros tris
        val subLists = texts.chunked(1000).take(2)
        return Flux.fromIterable(subLists)
            .flatMapSequential({ subList ->
                val indexed = subList.mapIndexed { i, t -> IndexedText(i, t) }
                val batches = packBatches(indexed, props.batchSize, props.maxBatchBytes)
                Flux.fromIterable(batches)
                    .flatMap({ batch ->
                        val req = RunPodRequest(input = EmbeddingsPayload(model = props.modelName, input = batch.map { it.text }))
                        webClient.post().bodyValue(req)
                            .exchangeToMono { resp ->
                                if (resp.statusCode().is2xxSuccessful) resp.bodyToMono<EmbeddingsResponse>()
                                else resp.createException().flatMap { Mono.error(it) }
                            }.timeout(Duration.ofSeconds(10))
                            .retryWhen(
                                Retry.backoff(props.maxRetries.toLong(), Duration.ofMillis(props.initialBackoffMs))
                                    .maxBackoff(Duration.ofMillis(props.maxBackoffMs))
                                    .jitter(0.5)
                                    .filter { isRetryable(it) }
                            )
                            .map { er ->
                                val vecs = er.output.data.map { it.embedding }
                                require(vecs.size == batch.size) { "Mismatch embeddings=${vecs.size} vs batch=${batch.size}" }
                                batch.mapIndexed { i, ix -> ix.index to vecs[i] }
                            }
                    }, props.concurrency)
                    .flatMapIterable { it }
                    .sort(compareBy { it.first })
                    .map { (_, vec) -> vec.map { it.toFloat() }.toFloatArray() }
                    .collectList()
            }, 1) // Traiter les sous-listes séquentiellement pour éviter surcharge
            .collectList()
            .map { it.flatten() }
    }

    private fun isRetryable(t: Throwable): Boolean {
        val msg = t.message?.lowercase() ?: ""
        return listOf("429","too many requests","5xx","500","502","503","504","timeout","timed out","refused","reset","unreachable","connection").any { it in msg }
    }

    fun testEmbed(): Mono<String> {
        val body = mapOf(
            "input" to mapOf(
                "model" to props.modelName,
                "input" to listOf("Test sentence")
            )
        )
        return webClient.post()
            .bodyValue(body)
            .retrieve()
            .bodyToMono<String>()
            .timeout(Duration.ofSeconds(600))
    }


    fun testEmbedWithJdkClient(): String {
        // Assurez-vous que le jeton est récupéré correctement, ici en dur pour l'exemple
        val bearerToken = System.getenv("RUNPOD_API_KEY") ?: error("RUNPOD_API_KEY env var not set")

        val jsonBody = """
        {
            "input": {
                "model": "${props.modelName}",
                "input": ["Test sentence"]
            }
        }
    """.trimIndent()

        val client = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_2)
            .connectTimeout(Duration.ofSeconds(60))
            .build()

        val request = HttpRequest.newBuilder()
            .uri(java.net.URI.create(props.baseUrl))
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer $bearerToken")
            .timeout(Duration.ofMinutes(2))
            .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
            .build()

        println("Envoi de la requête avec le client HTTP JDK...")
        val response = client.send(request, HttpResponse.BodyHandlers.ofString())
        println("Réponse reçue. Statut : ${response.statusCode()}")

        return response.body()
    }

}
