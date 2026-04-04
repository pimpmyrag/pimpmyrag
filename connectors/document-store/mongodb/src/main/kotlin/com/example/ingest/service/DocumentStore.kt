
package com.example.ingest.service

import com.example.ingest.mappers.RagDocumentDaoMapper.fromSentenceRecord
import com.example.ingest.mappers.RagDocumentDaoMapper.toSentenceRecord
import com.example.ingest.model.SentenceRecord
import com.example.ingest.repo.MongoProvider
import kotlinx.coroutines.runBlocking
import org.litote.kmongo.and
import org.litote.kmongo.ascending
import org.litote.kmongo.eq
import org.litote.kmongo.gt
import org.litote.kmongo.`in`
import rag.engine.DocumentStoreReader
import rag.engine.DocumentStoreWriter
import rag.model.RagDocument
import rag.model.RagUnitType

class DocumentStore(private val mp: MongoProvider): DocumentStoreReader, DocumentStoreWriter {


    fun loadAllSentencesForEmbeddings(docId: String): List<RagDocument> {
        return runBlocking {
            val filter = and(
                SentenceRecord::docId eq docId,
                SentenceRecord::isContent eq true,
                SentenceRecord::text gt ""
            )
             mp.sentences.find(filter)
                .sort(ascending(SentenceRecord::sentenceId))
                .toList().map { fromSentenceRecord(it)  }
        }

    }

    override fun search(
        unit: RagUnitType,
        ids: List<String>
    ): List<RagDocument> = runBlocking {
        when (unit) {
            RagUnitType.SENTENCE -> {
                val filter = SentenceRecord::docId `in` ids
                mp.sentences.find(filter)
                    .sort(ascending(SentenceRecord::sentenceId))
                    .toList()
                    .map { fromSentenceRecord(it) }
            }
            RagUnitType.DOCUMENT -> {
                val filter = com.example.ingest.model.DocumentRecord::docKey `in` ids
                mp.documents.find(filter)
                    .toList()
                    .map { com.example.ingest.mappers.RagDocumentDaoMapper.fromDocumentRecord(it) }
            }
            else -> {
                // Pour PAGE, PARAGRAPH, etc. : à adapter selon la structure de vos ElementRecord
                val filter = com.example.ingest.model.ElementRecord::docId `in` ids
                mp.elements.find(filter)
                    .toList()
                    .map { com.example.ingest.mappers.RagDocumentDaoMapper.fromElementRecord(it) }
                    .groupBy { it.parentId ?: it.id }
                    .map { (docId, elements) ->
                        RagDocument(
                            id = docId,
                            text = elements.joinToString("\n") { it.text },
                            type = unit,
                            elements = elements
                        )
                    }
            }
        }
    }

    override fun write(docs: List<RagDocument>): Unit = runBlocking {
        mp.sentences.insertMany(docs.mapNotNull { toSentenceRecord(it) })
    }


}
