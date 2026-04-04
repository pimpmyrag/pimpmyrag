
package rag.dsl

import rag.model.*

data class IngestConfig(
    var source: String = "",
    var unit: RagUnitType = RagUnitType.SENTENCE,
    var enrichers: MutableList<String> = mutableListOf(),
    val filters: MutableList<String> = mutableListOf()       // ingest-time filters
)

data class DocumentStorageConfig(
    var documentStore: String = "mongo",
    var unit: RagUnitType = RagUnitType.SENTENCE,
    var enrichers: MutableList<String> = mutableListOf(),
    val filters: MutableList<String> = mutableListOf()       // ingest-time filters
)

data class ChunkConfig(
    var using: String = "default",
    var unit: RagUnitType = RagUnitType.SENTENCE,
    var maxTokens: Int = 256
)

data class EmbedConfig(
    var model: String = "",
    var normalize: Boolean = true,
    var store: String = "chroma"
)

data class RetrieveConfig(
    var genericTopK: Int = 30,
    var thematicTopK: Int = 10,
    var strategy: String = "hybrid",
    var store: String = "chroma",
    val filters: MutableList<String> = mutableListOf()       // query-time filters (optional)
)

data class RerankStage(var name: String = "", var using: String = "")

data class RerankConfig(
    var strategy: String = "single",
    var stages: MutableList<RerankStage> = mutableListOf(),
    var topK: Int = 10
)

data class GenerationConfig(
    var llm: String = "",
    var themeBasedPrompt: Boolean = false,
    var maxTokens: Int = 512
)

data class RagConfig(
    val ingest: IngestConfig = IngestConfig(),
    val chunk: ChunkConfig = ChunkConfig(),
    val embed: EmbedConfig = EmbedConfig(),
    val retrieve: RetrieveConfig = RetrieveConfig(),
    val rerank: RerankConfig = RerankConfig(),
    val generate: GenerationConfig = GenerationConfig()
)

fun rag(block: RagConfig.() -> Unit): RagConfig = RagConfig().apply(block)
fun ingest(block: IngestConfig.() -> Unit): IngestConfig = IngestConfig().apply(block)
fun chunk(block: ChunkConfig.() -> Unit): ChunkConfig = ChunkConfig().apply(block)
fun embed(block: EmbedConfig.() -> Unit): EmbedConfig = EmbedConfig().apply(block)
fun retrieve(block: RetrieveConfig.() -> Unit): RetrieveConfig = RetrieveConfig().apply(block)
fun rerank(block: RerankConfig.() -> Unit): RerankConfig = RerankConfig().apply(block)
fun stage(name: String, block: RerankStage.() -> Unit): RerankStage = RerankStage(name).apply(block)
fun generate(block: GenerationConfig.() -> Unit): GenerationConfig = GenerationConfig().apply(block)

fun IngestConfig.enrichWith(vararg names: String) { enrichers.addAll(names) }
fun IngestConfig.filters(vararg names: String) { filters.addAll(names) }
fun RetrieveConfig.filters(vararg names: String) { filters.addAll(names) }
fun MutableList<RerankStage>.stage(name: String, block: RerankStage.() -> Unit) {
    this += RerankStage(name = name).apply(block)
}

//fun main() {
//    rag {
//
//        ingest {
//            unit = RagUnitType.DOCUMENT
//            enrichWith = "wikidata"
//        }
//        embed {
//
//        }
//
//
//    }
//}
