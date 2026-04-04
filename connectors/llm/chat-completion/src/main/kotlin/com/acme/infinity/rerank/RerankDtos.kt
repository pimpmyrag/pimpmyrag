package com.acme.infinity.rerank
data class RerankRequest(
    val input: RerankRequestInput
)

data class RerankRequestInput(
    val action: String, // ex: "rerank"
    val payload: RerankPayload
)

data class RerankPayload(
    val query: String,
    val documents: List<String>,
    val top_k: Int? = null,
    val return_documents: Boolean? = null
)


data class RerankResponse(
    val id: String,
    val output: RerankOutput,
)


data class RerankOutput(
    val results: List<RerankResult>
)

data class RerankResult(
    val index: Int,             // index du document dans la requête
    val score: Double           // score de similarité/relevance
)
