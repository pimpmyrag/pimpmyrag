package com.acme.infinity.lc4j

import com.acme.infinity.config.RerankProperties
import com.acme.infinity.rerank.RerankClient
import kotlinx.coroutines.reactor.awaitSingle
import kotlinx.coroutines.runBlocking
import rag.engine.Reranker
import rag.engine.ScoredDocument
import rag.model.RagDocument

class LangChain4jInfinityScoringModel(
    private val client: RerankClient,
    private val props: RerankProperties
) : Reranker {

    override fun rerank(
        query: String,
        docs: List<RagDocument>,
        topK: Int
    ): List<ScoredDocument> {
        require(docs.isNotEmpty()) { "segments must not be null/empty" }
        require(query.isNotBlank()) { "query must not be null/blank" }
        val docsText = docs.map { it.text }
        return runBlocking {
            val pairs = client.rerank(query, docsText).awaitSingle()
            val arr = MutableList(docs.size) { Double.NEGATIVE_INFINITY }
            pairs.forEach { (idx, score) -> if (idx in arr.indices) arr[idx] = score }
            arr.mapIndexed { index, score ->
                ScoredDocument(
                    document = docs[index],
                    score = score
                )
            }
        }
    }
}
