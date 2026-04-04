package com.example.ingest.mappers

import com.example.ingest.model.DocumentRecord
import com.example.ingest.model.ElementMetadata
import com.example.ingest.model.ElementRecord
import com.example.ingest.model.SentenceRecord
import rag.model.*

object RagDocumentDaoMapper {

    // RagDocument -> List<ElementRecord>
    fun toElementRecords(doc: RagDocument): List<ElementRecord> =
        doc.elements.mapIndexed { idx, el ->
            ElementRecord(
                docId = doc.id,
                elementIndex = idx,
                type = el.type,
                text = el.text,
                metadata = ElementMetadata(
                    pageNumber = el.layout?.pageNumber ?: doc.layout?.pageNumber,
                    pageName = null,
                    filename = doc.metadata["filename"] as? String,
                    filetype = doc.metadata["filetype"] as? String,
                    languages = doc.metadata["languages"] as? List<String>,
                    parentId = el.parentId ?: doc.parentId,
                    categoryDepth = null,
                    coordinates = el.layout?.bbox?.let { listOf(it.map { v -> v.toDouble() }) }
                )
            )
        }


    // RagDocument -> List<SentenceRecord>
    fun toSentenceRecord(doc: RagDocument): SentenceRecord? = when (doc.type) {
        RagUnitType.SENTENCE ->
            SentenceRecord(
                docId = doc.id,
                sentenceId = 0,
                text = doc.text,
                spanStart = doc.span?.start,
                spanEnd = doc.span?.end,
                elementId = doc.id,
                sectionTitle = doc.metadata["sectionTitle"] as? String,
                pageNumber = doc.layout?.pageNumber,
                elementType = doc.type.name,
                isContent = true
            )

        else -> null
    }

    // ElementRecord -> RagElement
    fun fromElementRecord(record: ElementRecord): RagElement =
        RagElement(
            type = record.type,
            text = record.text,
            metadata = mapOf(
                "filename" to record.metadata.filename,
                "filetype" to record.metadata.filetype,
                "languages" to record.metadata.languages
            ).filterValues { it != null },
            layout = record.metadata.pageNumber?.let { Layout(pageNumber = it) },
            parentId = record.metadata.parentId,
            id = record._id
        )

    // SentenceRecord -> RagElement (type "Sentence")
    fun fromSentenceRecord(record: SentenceRecord): RagDocument =
        RagDocument(
            type = RagUnitType.SENTENCE,
            text = record.text,
            span = Span(record.spanStart?:0, record.spanEnd?:0),
            id = record._id,
            metadata = record.sectionTitle?.let { mapOf("sectionTitle" to it) } ?: emptyMap(),
            layout = record.pageNumber?.let { Layout(pageNumber = it) }
        )

    fun toDocumentRecord(doc: RagDocument): DocumentRecord =
        DocumentRecord(
            docKey = doc.id,
            originalFilename = doc.metadata["filename"] as? String ?: doc.source ?: "unknown",
            originalMime = doc.metadata["filetype"] as? String,
            originalMetadata = doc.metadata,
            reconstructed = emptyMap(),
            stats = emptyMap(),
            textHash = null
        )

    fun fromDocumentRecord(record: DocumentRecord): RagDocument =
        RagDocument(
            id = record.docKey ?: record._id,
            text = "", // le texte complet n'est pas stocké dans DocumentRecord
            type = RagUnitType.DOCUMENT,
            metadata = record.originalMetadata,
            elements = emptyList(),
            span = null,
            layout = null,
            source = record.originalFilename,
            parentId = null
        )
}

