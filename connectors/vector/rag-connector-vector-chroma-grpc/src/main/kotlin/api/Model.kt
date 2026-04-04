
package api

data class VectorEntry(
    val id: String,
    val embedding: FloatArray,
    val document: String? = null,
    val metadata: Map<String, Any>? = null
)

data class VectorData(
    val collection: String,
    val vectors: List<VectorEntry>
)

data class QueryRequest(
    val collection: String,
    val vector: FloatArray,
    val topK: Int = 5,
    val filter: Any? = null
)

data class QueryResult(
    val ids: List<String>,
    val documents: List<String?>,
    val metadatas: List<Map<String, Any>?>,
    val distances: List<Float>
)
