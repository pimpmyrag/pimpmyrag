
package rag.planner

import rag.engine.ScoredDocument
import kotlin.reflect.KClass
import rag.model.*
import rag.engine.ScoredId

// payloads
sealed interface Payload

data class RawDoc(val doc: RagDocument): Payload

data class Chunks(val items: List<RagDocument>): Payload

data class Embeddings(val vectors: List<FloatArray>, val docs: List<RagDocument>): Payload

data class ScoredIdPayload(val items: List<ScoredId>): Payload

data class Docs(val items: List<RagDocument>): Payload

data class ScoredDocs(val items: List<ScoredDocument>): Payload

data class Answer(val text: String): Payload

@JvmInline value class PortId(val value: String)

interface PortSpec { val name: PortId; val type: KClass<out Payload> }

data class InPort(override val name: PortId, override val type: KClass<out Payload>): PortSpec

data class OutPort(override val name: PortId, override val type: KClass<out Payload>): PortSpec

sealed interface DagStep { val id: String; val inputs: List<InPort>; val outputs: List<OutPort> }

data class StepChunk(override val id: String, val chunkerRef: String, val unit: RagUnitType, val maxTokens: Int): DagStep {
    override val inputs = listOf(InPort(PortId("in:raw"), RawDoc::class))
    override val outputs = listOf(OutPort(PortId("out:chunks"), Chunks::class))
}

data class StepEmbed(override val id: String, val embedderRef: String, val writerRef: String): DagStep {
    override val inputs = listOf(InPort(PortId("in:chunks"), Chunks::class))
    override val outputs = listOf(OutPort(PortId("out:emb"), Embeddings::class))
}

data class StepRetrieveIds(override val id: String, val readerRef: String, val topK: Int): DagStep {
    override val inputs = listOf(InPort(PortId("in:emb"), Embeddings::class))
    override val outputs = listOf(OutPort(PortId("out:ids"), ScoredIdPayload::class))
}

data class StepHydrate(override val id: String, val docStoreRef: String, val unit: RagUnitType): DagStep {
    override val inputs = listOf(InPort(PortId("in:ids"), ScoredIdPayload::class))
    override val outputs = listOf(OutPort(PortId("out:docs"), Docs::class))
}

data class StepFilter(override val id: String, val filterRefs: List<String>): DagStep {
    override val inputs = listOf(InPort(PortId("in:docs"), Docs::class))
    override val outputs = listOf(OutPort(PortId("out:docs"), Docs::class))
}

data class StepMergeDocs(override val id: String, val mergePolicy: String = "union"): DagStep {
    override val inputs = listOf(
        InPort(PortId("in:left"), Docs::class),
        InPort(PortId("in:right"), Docs::class)
    )
    override val outputs = listOf(OutPort(PortId("out:docs"), Docs::class))
}

data class StepRerank(override val id: String, val rerankerRef: String, val topK: Int): DagStep {
    override val inputs = listOf(InPort(PortId("in:docs"), Docs::class))
    override val outputs = listOf(OutPort(PortId("out:scored"), ScoredDocs::class))
}

data class StepGenerate(override val id: String, val llmRef: String): DagStep {
    override val inputs = listOf(InPort(PortId("in:scored"), ScoredDocs::class))
    override val outputs = listOf(OutPort(PortId("out:answer"), Answer::class))
}

data class Edge(val fromStep: String, val fromPort: PortId, val toStep: String, val toPort: PortId)

data class ExecutionPlanDag(val steps: List<DagStep>, val edges: List<Edge>)
