
package com.acme.infinity.embed

//data class EmbeddingRequest(val model: String, val input: List<String>)


//data class EmbeddingResponse(val data: List<EmbeddingItem>)


data class EmbeddingsRequest(
    val input: EmbeddingsRequestInput
)

data class RunPodRequest(
    val input: EmbeddingsPayload
)

data class EmbeddingsRequestInput(
    val action: String = "embeddings", // ex: "embeddings"
    val payload: EmbeddingsPayload
)

data class EmbeddingsPayload(
    val model: String,
    // Liste de textes à encoder
    val input: List<String>
)




data class EmbeddingsResponse(
    val id: String,
    val output: EmbeddingsOutput
)

//@Serializable
//enum class JobStatus {
//    @SerialName("COMPLETED")
//    COMPLETED,
//
//    @SerialName("PENDING")
//    PENDING,
//
//    @SerialName("RUNNING")
//    RUNNING,
//
//    @SerialName("FAILED")
//    FAILED
//}

data class EmbeddingsOutput(
    val data: List<EmbeddingItem>,
    val model: String // ex: "BAAI/bge-m3"
)

data class EmbeddingItem(
    val embedding: List<Double>, // Tableau de floats
    val index: Int
)


