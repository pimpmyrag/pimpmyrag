package rag.connectors.rerank.onnx
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import rag.engine.Reranker
import rag.engine.ScoredDocument
import rag.model.RagDocument
import java.nio.LongBuffer
import java.nio.file.Paths
import kotlin.collections.get
import kotlin.math.exp

enum class OnnxModelType {
    RERANKER,    // Cross-encoder type reranker (score binaire)
    NLI          // Natural Language Inference (3 classes : entailment, neutral, contradiction)
}

class OnnxReranker(
    modelPath: String,
    tokenizerDir: String,
    private val modelType: OnnxModelType = OnnxModelType.RERANKER,
    intraOpThreads: Int = Runtime.getRuntime().availableProcessors(),
    interOpThreads: Int = 1,
    private val maxLen: Int = 512,
    private val useGpu: Boolean = false,
    private val gpuDeviceId: Int = 0
) : AutoCloseable, Reranker {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val tokenizer: HuggingFaceTokenizer

    private val requiresTokenTypeIds: Boolean

    init {
        val so = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            if (useGpu) {
                try {
                    addCUDA(gpuDeviceId)
                    setMemoryPatternOptimization(true)
                    setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                } catch (e: Exception) {
                    println("⚠️ GPU non disponible, fallback sur CPU: ${e.message}")
                    setIntraOpNumThreads(intraOpThreads)
                    setInterOpNumThreads(interOpThreads)
                    setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL)
                    setMemoryPatternOptimization(true)
                }
            } else {
                setIntraOpNumThreads(intraOpThreads)
                setInterOpNumThreads(interOpThreads)
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL)
                setMemoryPatternOptimization(true)
            }
        }
        // Auto-détection des entrées requises
        session = env.createSession(modelPath, so)
        requiresTokenTypeIds = session.inputNames.contains("token_type_ids")
        tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerDir), mapOf("modelMaxLength" to maxLen.toString()))
    }

    fun rerank(query: String, documents: List<String>): List<Float> {
        if (documents.isEmpty()) return emptyList()

        val pairs = documents.map { listOf(query, it) }
        val encs = pairs.map { pair ->
            tokenizer.encode(pair[0], pair[1], true, true)
        }

        val batch = encs.size
        val seqLen = maxLen

        val ids = Array(batch) { LongArray(seqLen) }
        val attn = Array(batch) { LongArray(seqLen) }
        val tti = Array(batch) { LongArray(seqLen) }

        for (b in 0 until batch) {
            val e = encs[b]
            val srcIds = e.ids.map { it }.toLongArray()
            val srcMask = e.attentionMask.map { it }.toLongArray()
            val srcTti = e.typeIds?.map { it }?.toLongArray() ?: LongArray(srcIds.size)

            val take = minOf(srcIds.size, seqLen)
            System.arraycopy(srcIds, 0, ids[b], 0, take)
            System.arraycopy(srcMask, 0, attn[b], 0, take)
            System.arraycopy(srcTti, 0, tti[b], 0, take)
        }

        val inputIdsT = tensorFrom2D(ids)
        val attnMaskT = tensorFrom2D(attn)
        val tokenTypeT = if (requiresTokenTypeIds) tensorFrom2D(tti) else null

        return try {
            val inputs = buildMap<String, OnnxTensor> {
                put("input_ids", inputIdsT)
                put("attention_mask", attnMaskT)
                if (requiresTokenTypeIds && tokenTypeT != null) {
                    put("token_type_ids", tokenTypeT)
                }
            }

            session.run(inputs).use { outputs ->
                @Suppress("UNCHECKED_CAST")
                val logits = outputs[0].value as Array<FloatArray>

                when (modelType) {
                    OnnxModelType.RERANKER -> {
                        // Reranker binaire : sigmoid sur logit[1]
                        logits.map { scoreArray ->
                            sigmoid(scoreArray[1])
                        }
                    }
                    OnnxModelType.NLI -> {
                        // NLI : softmax puis score d'entailment (classe 2)
                        logits.map { scoreArray ->
                            val probs = softmax(scoreArray)
                            println("DEBUG probs: [0]=${probs[0]}, [1]=${probs[1]}, [2]=${probs[2]}")
                            probs[2] // Score entailment
                        }
                    }
                }
            }
        } finally {
            tokenTypeT?.close()
            attnMaskT.close()
            inputIdsT.close()
        }
    }

    private fun sigmoid(x: Float): Float {
        return 1f / (1f + exp(-x))
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val expScores = logits.map { exp(it.toDouble()) }
        val sumExp = expScores.sum()
        return expScores.map { (it / sumExp).toFloat() }.toFloatArray()
    }

    private fun tensorFrom2D(data: Array<LongArray>): OnnxTensor {
        val b = data.size
        val l = data[0].size
        val totalSize = b * l
        val buf = LongBuffer.allocate(totalSize)
        data.forEach { buf.put(it) }
        buf.flip()
        return OnnxTensor.createTensor(env, buf, longArrayOf(b.toLong(), l.toLong()))
    }

    override fun close() {
        session.close()
        tokenizer.close()
    }

    override fun rerank(
        query: String,
        docs: List<RagDocument>,
        topK: Int
    ): List<ScoredDocument> {
        val docContents = docs.map { it.text }
        val scores = rerank(query, docContents)
        return docs.zip(scores).map { (doc, score) ->
            ScoredDocument(document = doc, score = score.toDouble())
        }.sortedByDescending { it.score }
    }
}