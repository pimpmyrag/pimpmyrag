
package rag.runner

import rag.planner.*
import rag.engine.*
import rag.model.*

class DataBus {
    private val bag = mutableMapOf<Pair<String, PortId>, Payload>()
    fun put(stepId: String, port: PortId, payload: Payload) { bag[stepId to port] = payload }
    fun get(stepId: String, port: PortId): Payload? = bag[stepId to port]
    fun getRequired(stepId: String, port: PortId): Payload = get(stepId, port) ?: error("Missing payload on ${stepId}:${port.value}")
}

class PlanRunnerDag(
    private val chunkers: List<ChunkerFactory>,
    private val embedders: List<EmbedderFactory>,
    private val vsWriters: List<VectorStoreWriterFactory>,
    private val vsReaders: List<VectorStoreReaderFactory>,
    private val docReaders: List<DocumentStoreReaderFactory>,
    private val filters: List<FilterFactory>,
    private val rerankers: List<RerankerFactory>,
    private val generators: List<GeneratorFactory>
) {
    private fun <F> resolve(name: String, list: List<F>, pred: (F)->Boolean): F =
        list.firstOrNull { pred(it) } ?: error("No factory supports '$name'")

    fun run(plan: ExecutionPlanDag, query: String, raw: RagDocument): Answer {
        val validator = PlanValidator(); val vr = validator.validate(plan)
        require(vr.ok) { "Invalid plan: ${vr.reason}" }
        val order = validator.topoSort(plan)!!
        val byDst = plan.edges.groupBy { it.toStep }
        val bus = DataBus()
        // seed RawDoc into steps expecting it
        plan.steps.filter { s -> s.inputs.any { it.type == RawDoc::class } }.forEach {
            bus.put("__seed__", PortId("out:raw"), RawDoc(raw))
        }
        for (s in order) {
            val inputs = s.inputs.associate { ip ->
                val incoming = byDst[s.id]?.find { it.toPort == ip.name } ?: error("No incoming edge for ${s.id}:${ip.name}")
                ip.name to bus.getRequired(incoming.fromStep, incoming.fromPort)
            }
            when (s) {
                is StepChunk -> {
                    val cf = resolve(s.chunkerRef, chunkers) { it.supports(s.chunkerRef) }
                    val out = Chunks(cf.create().chunk((inputs[PortId("in:raw")] as RawDoc).doc, s.unit, s.maxTokens))
                    bus.put(s.id, PortId("out:chunks"), out)
                }
                is StepEmbed -> {
                    val ef = resolve(s.embedderRef, embedders) { it.supports(s.embedderRef) }
                    val wf = resolve(s.writerRef, vsWriters) { it.supports(s.writerRef) }
                    val ch = inputs[PortId("in:chunks")] as Chunks
                    val vecs = ef.create().embed(ch.items)
                    wf.create().write(vecs, ch.items)
                    bus.put(s.id, PortId("out:emb"), Embeddings(vecs, ch.items))
                }
                is StepRetrieveIds -> {
                    val rf = resolve(s.readerRef, vsReaders) { it.supports(s.readerRef) }
                    val emb = inputs[PortId("in:emb")] as Embeddings
                    val ids = rf.create().searchIds(emb.vectors.first(), s.topK)
                    bus.put(s.id, PortId("out:ids"), ScoredIdPayload(ids))
                }
                is StepHydrate -> {
                    val df = resolve(s.docStoreRef, docReaders) { it.supports(s.docStoreRef) }
                    val ids = (inputs[PortId("in:ids")] as ScoredIdPayload).items
                    val docs = df.create().search(unit = s.unit, ids.map { it.id })
                    bus.put(s.id, PortId("out:docs"), Docs(docs))
                }
                is StepFilter -> {
                    val inDocs = (inputs[PortId("in:docs")] as Docs).items
                    val filtered = s.filterRefs.fold(inDocs) { acc, ref ->
                        val ff = resolve(ref, filters) { it.supports(ref) }
                        ff.create().filter(acc)
                    }
                    bus.put(s.id, PortId("out:docs"), Docs(filtered))
                }
                is StepMergeDocs -> {
                    val left = (inputs[PortId("in:left")] as Docs).items
                    val right = (inputs[PortId("in:right")] as Docs).items
                    bus.put(s.id, PortId("out:docs"), Docs((left + right).distinctBy { it.id }))
                }
                is StepRerank -> {
                    val rrf = resolve(s.rerankerRef, rerankers) { it.supports(s.rerankerRef) }
                    val docs = (inputs[PortId("in:docs")] as Docs).items
                    val scored = rrf.create().rerank(query, docs, s.topK)
                    bus.put(s.id, PortId("out:scored"), ScoredDocs(scored))
                }
                is StepGenerate -> {
                    val gf = resolve(s.llmRef, generators) { it.supports(s.llmRef) }
                    val scored = (inputs[PortId("in:scored")] as ScoredDocs).items
                    val answer = gf.create().generate(query, scored.map { it.document })
                    bus.put(s.id, PortId("out:answer"), Answer(answer))
                }
            }
        }
        val terminal = plan.steps.filterIsInstance<StepGenerate>().lastOrNull() ?: error("No StepGenerate in plan")
        return bus.getRequired(terminal.id, PortId("out:answer")) as Answer
    }
}
