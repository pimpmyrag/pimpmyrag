
package api

import api.QueryRequest
import api.QueryResult

interface VectorStoreReader {
    suspend fun query(req: QueryRequest): QueryResult
}
