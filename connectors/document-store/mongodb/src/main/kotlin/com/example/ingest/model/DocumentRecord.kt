
package com.example.ingest.model

import java.time.Instant
import java.util.*

/**
 * Global envelope for a file/document. We use a random UUID as Mongo _id for safety (no size, no collisions across shards),
 * and keep stable business identifiers (docKey) as separate fields if needed.
 */

data class DocumentRecord(
    val _id: String = UUID.randomUUID().toString(),
    val docKey: String? = null,                 // optional stable key (hash of path or external id)
    val originalFilename: String,
    val originalMime: String?,
    val originalMetadata: Map<String, Any?> = emptyMap(),
    val reconstructed: Map<String, Any?> = emptyMap(),
    val stats: Map<String, Any?> = emptyMap(),
    val textHash: String? = null,
    val createdAt: Instant = Instant.now(),
    val updatedAt: Instant = Instant.now()
)
