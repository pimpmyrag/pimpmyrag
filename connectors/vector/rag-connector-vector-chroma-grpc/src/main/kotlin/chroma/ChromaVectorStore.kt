
package chroma

import api.QueryRequest
import api.QueryResult
import api.VectorData
import api.VectorStoreReader
import api.VectorStoreWriter
import com.google.protobuf.ByteString
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ChromaVectorStore(private val client: ChromaGrpcClient) : VectorStoreWriter, VectorStoreReader {
    override suspend fun write(data: VectorData) {
        val request = Chroma.QueryVectorsRequest.newBuilder()
            .addAllVectors(
                data.vectors.map { vector ->
                    Chroma.VectorEmbeddingRecord.newBuilder()
                        .setId(vector.id)
                        .setVector(
                            Chroma.Vector.newBuilder()
                                .setVector(ByteString.copyFrom(vector.embedding.floatArrayToByteArray()))
                                .build()
                        )
                        .build()
                }
            )
            .build()

    }


    override suspend fun delete(collection: String, ids: List<String>) {
        TODO("Not yet implemented")
    }

    override suspend fun query(req: QueryRequest): QueryResult {
        TODO("Not yet implemented")
    }

}


fun FloatArray.floatArrayToByteArray(): ByteArray {
    val buffer = ByteBuffer.allocate(4 * this.size).order(ByteOrder.LITTLE_ENDIAN)
    this.forEach { buffer.putFloat(it) }
    return buffer.array()
}