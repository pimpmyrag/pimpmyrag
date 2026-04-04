
package com.acme.infinity.lc4j

import com.acme.infinity.client.IInfinityClient
import com.acme.infinity.config.EmbeddingProperties
import com.acme.infinity.embed.EmbeddingClient
import kotlinx.coroutines.reactor.awaitSingle
import kotlinx.coroutines.runBlocking
import rag.engine.Embedder
import rag.model.RagDocument

class LangChain4jInfinityEmbeddingModel(
    private val client: IInfinityClient,
) : Embedder {

    override fun embed(documents: List<RagDocument>): List<FloatArray> {
        require(documents.isNotEmpty()) { "textSegments must not be null/empty" }
        val texts = documents.map { it.text }
        return runBlocking {
            client.embed(texts)
        }
    }
}
