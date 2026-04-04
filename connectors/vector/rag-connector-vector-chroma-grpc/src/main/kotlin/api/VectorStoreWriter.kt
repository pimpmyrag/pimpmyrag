
package api

import api.VectorData

interface VectorStoreWriter {
    suspend fun write(data: VectorData)
    suspend fun delete(collection: String, ids: List<String>)
}
