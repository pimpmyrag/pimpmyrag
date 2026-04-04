package com.example.ingest.repo

import com.example.ingest.model.DocumentRecord
import com.example.ingest.model.ElementRecord
import com.example.ingest.model.SentenceRecord
import com.mongodb.ConnectionString
import com.mongodb.MongoClientSettings
import com.mongodb.client.model.IndexOptions
import org.bson.UuidRepresentation
import org.litote.kmongo.coroutine.CoroutineClient
import org.litote.kmongo.coroutine.CoroutineCollection
import org.litote.kmongo.coroutine.coroutine
import org.litote.kmongo.reactivestreams.KMongo
import org.litote.kmongo.util.KMongoUtil

class MongoProvider(uri: String, dbName: String) {
    val client: CoroutineClient
    val db: org.litote.kmongo.coroutine.CoroutineDatabase

    init {
        val settings = MongoClientSettings.builder()
            .applyConnectionString(ConnectionString(uri))
            .uuidRepresentation(UuidRepresentation.STANDARD)
            .build()
        client = KMongo.createClient(settings).coroutine
        db = client.getDatabase(dbName)
    }

    val documents: CoroutineCollection<DocumentRecord> = db.getCollection("documents")
    val elements: CoroutineCollection<ElementRecord> = db.getCollection("elements")
    val sentences: CoroutineCollection<SentenceRecord> = db.getCollection("sentences")


    suspend fun ensureIndexes() {
        // documents
        documents.createIndex(KMongoUtil.toBson("{docKey: 1}"))
        documents.createIndex(KMongoUtil.toBson("{createdAt: -1}"))

        // elements
        elements.createIndex(
            KMongoUtil.toBson("{docId: 1, elementIndex: 1}"),
            IndexOptions().unique(true)
        )
        elements.createIndex(KMongoUtil.toBson("{docId: 1, type: 1}"))
        elements.createIndex(KMongoUtil.toBson("{docId: 1, 'metadata.pageNumber': 1}"))

        // sentences
        sentences.createIndex(
            KMongoUtil.toBson("{docId: 1, sentenceId: 1}"),
            IndexOptions().unique(true)
        )
        sentences.createIndex(KMongoUtil.toBson("{docId: 1, elementId: 1}"))
        sentences.createIndex(KMongoUtil.toBson("{docId: 1, pageNumber: 1}"))
        sentences.createIndex(KMongoUtil.toBson("{docId: 1, sectionTitle: 1}"))
    }
}