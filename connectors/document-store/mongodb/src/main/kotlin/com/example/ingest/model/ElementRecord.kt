
package com.example.ingest.model

import java.util.*

/**
 * One Mongo document per Unstructured element.
 * _id uses UUID. We also store (docId, elementIndex) for fast unique indexing and replay.
 */

data class ElementMetadata(
    val pageNumber: Int? = null,
    val pageName: String? = null,
    val filename: String? = null,
    val filetype: String? = null,
    val languages: List<String>? = null,
    val parentId: String? = null,
    val categoryDepth: Int? = null,
    val coordinates: List<List<Double>>? = null
)

data class ElementRecord(
    val _id: String = UUID.randomUUID().toString(),
    val docId: String,
    val elementIndex: Int,            // order in the source list
    val type: String,
    val text: String,
    val metadata: ElementMetadata = ElementMetadata()
)
