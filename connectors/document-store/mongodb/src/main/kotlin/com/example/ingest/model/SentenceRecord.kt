
package com.example.ingest.model

import java.util.*

/**
 * One Mongo document per sentence.
 * _id uses UUID. sentenceId is a monotonic integer per document (unique with docId).
 */

data class SentenceRecord(
    val _id: String = UUID.randomUUID().toString(),
    val docId: String,
    val sentenceId: Int,
    val text: String,
    val spanStart: Int? = null,
    val spanEnd: Int? = null,
    val elementId: String? = null,
    val sectionTitle: String? = null,
    val pageNumber: Int? = null,
    val elementType: String,     // ex: "NarrativeText", "Title", ...
    val isContent: Boolean       // calculé selon le type/source

)
