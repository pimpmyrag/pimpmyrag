import io.qdrant.client.PointIdFactory.id
import io.qdrant.client.QdrantClient
import io.qdrant.client.ValueFactory.value
import io.qdrant.client.VectorsFactory.vectors
import io.qdrant.client.grpc.Common
import io.qdrant.client.grpc.Points
import rag.engine.ScoredId
import rag.engine.VectorFilter
import rag.engine.VectorStoreReader
import rag.engine.VectorStoreWriter
import rag.model.RagDocument
import java.util.UUID

class QdrantVectorStoreWriter(
    private val client: QdrantClient,
    private val collectionName: String
) : VectorStoreWriter {
    override fun write(embeddings: List<FloatArray>, docs: List<RagDocument>) {
        val points = embeddings.mapIndexed { idx, embedding ->
            val payload = docs[idx].metadata.mapValues { (_, v) ->
                // Conversion simple: ici, on suppose que v est un String ou un nombre
                when (v) {
                    is String -> value(v)
                    is Long -> value(v)
                    is Double -> value(v)
                    is Boolean -> value(v)
                    else -> value(v.toString())
                }
            }.toMutableMap()
            Points.PointStruct.newBuilder()
                .setId(id(UUID.fromString(docs[idx].id)))
                .setVectors(vectors(embedding.toList()))
                .putAllPayload(payload)
                .build()
        }
        client.upsertAsync(collectionName, points)
    }
}


class QdrantVectorStoreReader(
    private val client: QdrantClient,
    private val collectionName: String
) : VectorStoreReader {
    override fun searchIds(queryEmbedding: FloatArray, topK: Int): List<ScoredId> {
        val searchResult = client.searchAsync(
            Points.SearchPoints.newBuilder()
                .setCollectionName(collectionName)
                .addAllVector(queryEmbedding.toList())
                .setLimit(topK.toLong()).build())
            .get()
        return searchResult.map {
            ScoredId(it.id.uuid, it.score.toDouble())
        }
    }

    override fun searchIds(
        queryEmbedding: FloatArray,
        topK: Int,
        filter: VectorFilter
    ): List<ScoredId> {

        val filterBuilder = Common.Filter.newBuilder()

        filter.metadata.forEach { (key, v) ->
            // On construit soit un FieldCondition avec Match (keyword/integer/boolean/uuid),
            // soit un FieldCondition avec Range pour les doubles/floats.
            val fieldConditionBuilder = Common.FieldCondition.newBuilder().setKey(key)

            when (v) {
                is String -> {
                    val match = Common.Match.newBuilder()
                        .setKeyword(v)
                        .build()
                    fieldConditionBuilder.match = match
                }
                is Int, is Long -> {
                    val match = Common.Match.newBuilder()
                        .setInteger((v as Number).toLong())
                        .build()
                    fieldConditionBuilder.match = match
                }
                is Boolean -> {
                    val match = Common.Match.newBuilder()
                        .setBoolean(v)
                        .build()
                    fieldConditionBuilder.match = match
                }
                is Float, is Double -> {
                    // Pas de match exact float dans Qdrant: on passe par Range (égalité via gte==lte).
                    val d = (v as Number).toDouble()
                    val range = Common.Range.newBuilder()
                        .setGte(d)
                        .setLte(d)
                        .build()
                    fieldConditionBuilder.range = range
                }
                else -> {
                    // Fallback: on tente un match string
                    val match = Common.Match.newBuilder()
                        .setKeyword(v.toString())
                        .build()
                    fieldConditionBuilder.match = match
                }
            }

            val condition = Common.Condition.newBuilder()
                .setField(fieldConditionBuilder.build())
                .build()

            filterBuilder.addMust(condition)
        }

        val searchResult = client.searchAsync(
            Points.SearchPoints.newBuilder()
                .setCollectionName(collectionName)
                .addAllVector(queryEmbedding.toList())
                .setLimit(topK.toLong())
                .setFilter(filterBuilder.build())
                .build()
        ).get()

        return searchResult.map {
            // Selon ton schéma d’ID, adapte ici: .id.uuid, .id.num, etc.
            ScoredId(it.id.uuid, it.score.toDouble())
        }
    }

}
