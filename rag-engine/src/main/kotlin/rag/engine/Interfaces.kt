
package rag.engine

import rag.model.*

// ---- Core processing contracts ----
interface Chunker { fun chunk(document: RagDocument, unit: RagUnitType, maxTokens: Int): List<RagDocument> }
interface Embedder { fun embed(documents: List<RagDocument>): List<FloatArray> }
interface NerExtractor {
    fun extractNer(documents: List<RagDocument>): List<List<Entity>>
}



/**
 * Parser abstraction for UD. Implementations may wrap Stanza, spaCy, UDPipe, etc.
 * Input is our internal RagDocument; output is UDDocument suitable for downstream mapping.
 */
interface UDParser { fun parse(documents: List<RagDocument>): List<UDDocument> }

// Extend NerExtractor so an implementation may consume UD results directly.
interface NerExtractorFromUD {
    fun extractNerFromUD(udDocuments: List<UDDocument>): List<List<Entity>>
}
interface Retriever { fun retrieve(query: String, topK: Int): List<RagDocument> }
interface DocumentFilter { fun filter(docs: List<RagDocument>): List<RagDocument> }

data class ScoredId(val id: String, val score: Double)
data class ScoredDocument(val document: RagDocument, val score: Double)
interface Reranker { fun rerank(query: String, docs: List<RagDocument>, topK: Int): List<ScoredDocument> }
interface Classifier { fun rerank(query: String, docs: List<RagDocument>, topK: Int): List<ScoredDocument> }

interface Generator { fun generate(query: String, context: List<RagDocument>): String }

interface DocumentStoreWriter { fun write(docs: List<RagDocument>) }
interface DocumentStoreReader { fun search(unit: RagUnitType, ids: List<String>): List<RagDocument> }

// ---- Vector store IO ----
interface VectorStoreWriter { fun write(embeddings: List<FloatArray>, docs: List<RagDocument>) }
data class VectorFilter(val metadata: Map<String, Any?>)

interface VectorStoreReader {
    fun searchIds(queryEmbedding: FloatArray, topK: Int): List<ScoredId>
    fun searchIds(
        queryEmbedding: FloatArray,
        topK: Int,
        filter: VectorFilter
    ): List<ScoredId>
}
// ---- Factories ----
interface ChunkerFactory { fun supports(name: String): Boolean; fun create(): Chunker }
interface EmbedderFactory { fun supports(name: String): Boolean; fun create(): Embedder }
interface RetrieverFactory { fun supports(name: String): Boolean; fun create(): Retriever }
interface FilterFactory { fun supports(name: String): Boolean; fun create(): DocumentFilter }
interface RerankerFactory { fun supports(name: String): Boolean; fun create(): Reranker }
interface GeneratorFactory { fun supports(name: String): Boolean; fun create(): Generator }
interface VectorStoreWriterFactory { fun supports(name: String): Boolean; fun create(): VectorStoreWriter }
interface VectorStoreReaderFactory { fun supports(name: String): Boolean; fun create(): VectorStoreReader }
interface DocumentStoreWriterFactory { fun supports(name: String): Boolean; fun create(): DocumentStoreWriter }
interface DocumentStoreReaderFactory { fun supports(name: String): Boolean; fun create(): DocumentStoreReader }
