
package rag.connectors.stub

import rag.engine.*
import rag.model.*

class DefaultChunkerFactory : ChunkerFactory {
    override fun supports(name: String) = name == "default"
    override fun create(): Chunker = object : Chunker {
        override fun chunk(document: RagDocument, unit: RagUnitType, maxTokens: Int): List<RagDocument> {
            val parts = document.text.split('.')
            return parts.mapIndexedNotNull { i, s ->
                val t = s.trim(); if (t.isBlank()) null else RagDocument("${document.id}_sent_$i", t, RagUnitType.SENTENCE, document.metadata)
            }
        }
    }
}

class StubEmbedderFactory : EmbedderFactory {
    override fun supports(name: String) = name == "bge-m3"
    override fun create(): Embedder = object : Embedder {
        override fun embed(documents: List<RagDocument>): List<FloatArray> =
            documents.map { d -> FloatArray(8) { i -> ((d.text.hashCode() * (i+1)) % 1000) / 1000f } }
    }
}

object InMemoryVectorIndex { val idToVec = mutableMapOf<String, FloatArray>() }
class StubVectorWriterFactory : VectorStoreWriterFactory {
    override fun supports(name: String) = name == "chroma"
    override fun create(): VectorStoreWriter = object : VectorStoreWriter {
        override fun write(embeddings: List<FloatArray>, docs: List<RagDocument>) {
            docs.zip(embeddings).forEach { (d, v) -> InMemoryVectorIndex.idToVec[d.id] = v }
        }
    }
}
class StubVectorReaderFactory : VectorStoreReaderFactory {
    override fun supports(name: String) = name == "chroma"
    override fun create(): VectorStoreReader = object : VectorStoreReader {
        override fun searchIds(queryEmbedding: FloatArray, topK: Int): List<ScoredId> {
            fun cos(a: FloatArray, b: FloatArray): Double {
                var dot=0.0; var na=0.0; var nb=0.0
                for (i in a.indices) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
                return if (na==0.0||nb==0.0) 0.0 else dot/Math.sqrt(na*nb)
            }
            return InMemoryVectorIndex.idToVec.entries
                .map { ScoredId(it.key, cos(queryEmbedding, it.value)) }
                .sortedByDescending { it.score }
                .take(topK)
        }

        override fun searchIds(
            queryEmbedding: FloatArray,
            topK: Int,
            filter: VectorFilter
        ): List<ScoredId> {
            TODO("Not yet implemented")
        }
    }
}

object InMemoryDocStore { val byId = mutableMapOf<String, RagDocument>() }
class StubDocWriterFactory : DocumentStoreWriterFactory {
    override fun supports(name: String) = name == "mongo"
    override fun create(): DocumentStoreWriter = object : DocumentStoreWriter {
        override fun write(docs: List<RagDocument>) { docs.forEach { doc -> InMemoryDocStore.byId[doc.id] = doc }}
    }
}
class StubDocReaderFactory : DocumentStoreReaderFactory {
    override fun supports(name: String) = name == "mongo"
    override fun create(): DocumentStoreReader = object : DocumentStoreReader {
        override fun search(unit: RagUnitType, ids: List<String>): List<RagDocument> = ids.mapNotNull { InMemoryDocStore.byId[it] }
    }
}

class StubRerankerFactory : RerankerFactory {
    override fun supports(name: String) = name in setOf("xnli", "pingpong")
    override fun create(): Reranker = object : Reranker {
        override fun rerank(query: String, docs: List<RagDocument>, topK: Int): List<ScoredDocument> =
            docs.map { ScoredDocument(it, it.text.length.toDouble()) }.sortedByDescending { it.score }.take(topK)
    }
}

class StubGeneratorFactory : GeneratorFactory {
    override fun supports(name: String) = name == "mistral"
    override fun create(): Generator = object : Generator {
        override fun generate(query: String, context: List<RagDocument>): String = "Q: $query A: ${context.firstOrNull()?.text ?: "(no context)"}"
    }
}

class StubFilterFactory : FilterFactory {
    override fun supports(name: String) = name in setOf("language", "length", "cleanup")
    override fun create(): DocumentFilter = object : DocumentFilter {
        override fun filter(docs: List<RagDocument>): List<RagDocument> = docs.filter { it.text.length >= 3 }
    }
}
