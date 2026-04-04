
package rag.dsl.staged

import rag.model.*
import rag.planner.*

// Opaque references to steps (and optional ports)
data class StepRef(val step: DagStep) {
    fun out(i: Int = 0) = step.outputs[i].name
    fun inn(i: Int = 0) = step.inputs[i].name
    val id get() = step.id
}

class GraphCtx {
    val steps = mutableListOf<DagStep>()
    val edges = mutableListOf<Edge>()
    var tail: StepRef? = null

    fun addStep(s: DagStep): StepRef = StepRef(s).also { steps += s }

    fun link(prev: StepRef, curr: StepRef, outIndex: Int = 0, inIndex: Int = 0) {
        edges += Edge(prev.id, prev.out(outIndex), curr.id, curr.inn(inIndex))
    }

    fun linkFromTail(curr: StepRef) {
        tail?.let { link(it, curr) }
        tail = curr
    }

    fun seedIfNeeded() {
        val first = steps.firstOrNull() ?: return
        if (first.inputs.any { it.type == RawDoc::class }) {
            edges.add(0, Edge("__seed__", PortId("out:raw"), first.id, PortId("in:raw")))
        }
    }
}

// Stage types
class StageStart internal constructor(private val g: GraphCtx) {
    fun fromRaw(): StageRaw = StageRaw(g)
}

class StageRaw internal constructor(private val g: GraphCtx) {
    fun chunk(block: ChunkBlock.() -> Unit): StageChunks {
        val b = ChunkBlock().apply(block)
        val ref = g.addStep(StepChunk(b.id, b.using, b.unit, b.maxTokens))
        g.linkFromTail(ref)
        return StageChunks(g)
    }
}

class StageChunks internal constructor(private val g: GraphCtx) {
    fun embed(block: EmbedBlock.() -> Unit): StageEmbeddings {
        val b = EmbedBlock().apply(block)
        val ref = g.addStep(StepEmbed(b.id, b.model, b.store))
        g.linkFromTail(ref)
        return StageEmbeddings(g)
    }
}

class StageEmbeddings internal constructor(private val g: GraphCtx) {
    fun retrieveIds(block: RetrieveIdsBlock.() -> Unit): StageIds {
        val b = RetrieveIdsBlock().apply(block)
        val ref = g.addStep(StepRetrieveIds(b.id, b.store, b.topK))
        g.linkFromTail(ref)
        return StageIds(g)
    }

    // Hybrid: two retrieve+hydrate branches, then merge
    fun hybrid(block: HybridBlock.() -> Unit): StageDocs {
        val hb = HybridBlock().apply(block)
        val prev = g.tail ?: error("No previous step before hybrid")

        val retA = g.addStep(StepRetrieveIds(hb.genericId, hb.store, hb.genericTopK))
        val hydA = g.addStep(StepHydrate("hyd-${hb.genericId}", hb.hydrate, hb.docGranularity))
        val retB = g.addStep(StepRetrieveIds(hb.themeId, hb.store, hb.thematicTopK))
        val hydB = g.addStep(StepHydrate("hyd-${hb.themeId}", hb.hydrate, hb.docGranularity))
        val merge = g.addStep(StepMergeDocs(hb.mergeId, hb.mergePolicy))

        // Wiring by references (no hard-coded names)
        g.link(prev, retA)
        g.link(prev, retB)
        g.link(retA, hydA)
        g.link(retB, hydB)
        g.link(hydA, merge, inIndex = 0)  // in:left
        g.link(hydB, merge, inIndex = 1)  // in:right

        g.tail = merge
        return StageDocs(g)
    }
}

class StageIds internal constructor(private val g: GraphCtx) {
    fun hydrate(block: HydrateBlock.() -> Unit): StageDocs {
        val b = HydrateBlock().apply(block)
        val ref = g.addStep(StepHydrate(b.id, b.store, b.docGranularity))
        g.linkFromTail(ref)
        return StageDocs(g)
    }
}

class StageDocs internal constructor(private val g: GraphCtx) {
    fun filter(vararg names: String): StageDocs {
        val id = "filters-${g.steps.count { it is StepFilter } + 1}"
        val ref = g.addStep(StepFilter(id, names.toList()))
        g.linkFromTail(ref)
        return this
    }
    fun rerank(block: RerankBlock.() -> Unit): StageScored {
        val b = RerankBlock().apply(block)
        val ref = g.addStep(StepRerank(b.id, b.strategy, b.topK))
        g.linkFromTail(ref)
        return StageScored(g)
    }
}

class StageScored internal constructor(private val g: GraphCtx) {
    fun generate(block: GenerateBlock.() -> Unit): StageEnd {
        val b = GenerateBlock().apply(block)
        val ref = g.addStep(StepGenerate(b.id, b.llm))
        g.linkFromTail(ref)
        return StageEnd(g)
    }
}

class StageEnd internal constructor(private val g: GraphCtx) {
    fun compile(): ExecutionPlanDag {
        g.seedIfNeeded()
        return ExecutionPlanDag(g.steps.toList(), g.edges.toList())
    }
}

// Blocks (user-facing)
class ChunkBlock { var id: String = "chunk-1"; var using: String = "default"; var unit: RagUnitType = RagUnitType.SENTENCE; var maxTokens: Int = 128 }
class EmbedBlock { var id: String = "embed-1"; var model: String = "bge-m3"; var store: String = "chroma" }
class RetrieveIdsBlock { var id: String = "ret-1"; var store: String = "chroma"; var topK: Int = 40 }
class HydrateBlock { var id: String = "hyd-1"; var store: String = "mongo"; var docGranularity: RagUnitType = RagUnitType.DOCUMENT }
class RerankBlock { var id: String = "rr-1"; var strategy: String = "xnli"; var topK: Int = 10 }
class GenerateBlock { var id: String = "gen-1"; var llm: String = "mistral" }

class HybridBlock {
    var store: String = "chroma"
    var hydrate: String = "mongo"
    var docGranularity: RagUnitType = RagUnitType.SENTENCE
    var genericTopK: Int = 80
    var thematicTopK: Int = 40
    var genericId: String = "ret-generic"
    var themeId: String = "ret-theme"
    var mergeId: String = "merge-1"
    var mergePolicy: String = "union"
}

// Top-level entry
fun rag(build: StageStart.() -> StageEnd): ExecutionPlanDag {
    val g = GraphCtx()
    val end = StageStart(g).build()
    return end.compile()
}
