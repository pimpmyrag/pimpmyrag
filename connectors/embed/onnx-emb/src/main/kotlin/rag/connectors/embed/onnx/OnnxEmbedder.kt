package rag.connectors.embed.onnx

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.*
import ai.onnxruntime.providers.CoreMLFlags
import rag.engine.Embedder
import rag.model.RagDocument
import java.nio.LongBuffer
import java.nio.file.Paths
import java.util.EnumSet
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.system.measureTimeMillis

class OnnxOrtEmbedder(
    modelPath: String,
    tokenizerDir: String,
    intraOpThreads: Int = Runtime.getRuntime().availableProcessors(),
    interOpThreads: Int = 1,
    private val maxLen: Int = 512, // Augmenté pour correspondre aux modèles modernes comme bge
    private val l2Normalize: Boolean = true,
    useGpu: Boolean = false,
    gpuDeviceId: Int = 0
) : AutoCloseable, Embedder {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val tokenizer: HuggingFaceTokenizer
    private val hasTokenTypeIds: Boolean

    // Executor pour le pooling parallèle
    private val poolExecutor = Executors.newFixedThreadPool(2)


    init {
        val so = OrtSession.SessionOptions().apply {
            setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            addConfigEntry("session.use_ort_model_bytes_directly", "1")
            addConfigEntry("session.disable_prepacking", "0")
            setMemoryPatternOptimization(true) // Activation globale

            val osName = System.getProperty("os.name").lowercase()
            if (osName.contains("nac")) {
                try {
                    println("🍏 macOS détecté. Tentative d'activation du provider Core ML.")
                    addCoreML(EnumSet.of(CoreMLFlags.ENABLE_ON_SUBGRAPH))
                    setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                    println("✅ Provider Core ML activé.")
                } catch (e: OrtException) {
                    println("⚠️ Échec de l'activation de Core ML, fallback sur CPU: ${e.message}")
                    configureCpu(intraOpThreads, interOpThreads)
                }
            } else if (useGpu) {
                try {
                    println("🚀 Tentative d'activation du provider CUDA pour le GPU.")
                    addCUDA(gpuDeviceId)
                    setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                    println("✅ Provider CUDA activé.")
                } catch (e: Exception) {
                    println("⚠️ GPU non disponible ou CUDA non configuré, fallback sur CPU: ${e.message}")
                    configureCpu(intraOpThreads, interOpThreads)
                }
            } else {
                println("⚙️ Configuration pour une exécution sur CPU.")
                configureCpu(intraOpThreads, interOpThreads)
            }
        }
        session = env.createSession(modelPath, so)
//        println("execution provider are: ${session.executionProviders}")
        tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerDir), mapOf("modelMaxLength" to maxLen.toString()))

        hasTokenTypeIds = session.inputInfo.containsKey("token_type_ids")
    }

    private fun OrtSession.SessionOptions.configureCpu(intraOp: Int, interOp: Int) {
        setIntraOpNumThreads(intraOp)
        setInterOpNumThreads(interOp)
        setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL)
    }

    fun embedTexts(texts: List<String>): List<FloatArray> {
        if (texts.isEmpty()) return emptyList()

        val batchSize = texts.size
        logMemoryUsage("Début embedTexts (batch=$batchSize)")

        val bfore = System.currentTimeMillis()
        val encodings = tokenizer.batchEncode(texts)
        println("encoding took : ${System.currentTimeMillis() - bfore}ms")

        val inputIdsTensor: OnnxTensor
        val attentionMaskTensor: OnnxTensor
        val tokenTypeIdsTensor: OnnxTensor?
        measureTimeMillis {
            inputIdsTensor = createTensorFromEncodings(encodings, batchSize) { it.ids }
            attentionMaskTensor = createTensorFromEncodings(encodings, batchSize) { it.attentionMask }
            tokenTypeIdsTensor = if (hasTokenTypeIds) createTensorFromEncodings(encodings, batchSize) { it.typeIds } else null
        }.also { println("DEBUG: Création des tenseurs en ${it}ms") }


        val inputs = buildMap {
            put("input_ids", inputIdsTensor)
            put("attention_mask", attentionMaskTensor)
            tokenTypeIdsTensor?.let { put("token_type_ids", it) }
        }

        try {
            val lastHiddenState: Array<Array<FloatArray>>
            measureTimeMillis {
                session.run(inputs).use { results ->
                    val outputTensor = results[0] as OnnxTensor
                    @Suppress("UNCHECKED_CAST")
                    lastHiddenState = outputTensor.value as Array<Array<FloatArray>>
                }
            }.also { println("DEBUG: Inférence ONNX en ${it}ms") }

            logMemoryUsage("Après inférence")

            val embeddings: List<FloatArray>
            measureTimeMillis {
                embeddings = meanPoolAndNormalizeParallel(lastHiddenState, attentionMaskTensor, batchSize)
            }.also { println("DEBUG: Pooling et normalisation en ${it}ms") }

            logMemoryUsage("Fin embedTexts")
            return embeddings

        } finally {
            inputIdsTensor.close()
            attentionMaskTensor.close()
            tokenTypeIdsTensor?.close()
        }
    }

    /**
     * Crée un OnnxTensor à partir d'une liste d'encodages en utilisant un seul buffer pour l'efficacité.
     */
    private fun createTensorFromEncodings(
        encodings: Array<ai.djl.huggingface.tokenizers.Encoding>,
        batchSize: Int,
        extractor: (ai.djl.huggingface.tokenizers.Encoding) -> LongArray
    ): OnnxTensor {
        val shape = longArrayOf(batchSize.toLong(), maxLen.toLong())
        val buffer = LongBuffer.allocate(batchSize * maxLen)

        for (encoding in encodings) {
            val data = extractor(encoding)
            val len = min(data.size, maxLen)
            buffer.put(data, 0, len)
            // Pad avec des zéros si la séquence est plus courte que maxLen
            if (len < maxLen) {
                buffer.put(LongArray(maxLen - len) { 0L })
            }
        }
        buffer.flip()
        return OnnxTensor.createTensor(env, buffer, shape)
    }


    private fun meanPoolAndNormalizeParallel(
        lastHiddenState: Array<Array<FloatArray>>,
        attentionMask: OnnxTensor,
        batchSize: Int
    ): List<FloatArray> {
        val hiddenSize = lastHiddenState.getOrNull(0)?.getOrNull(0)?.size ?: 0
        if (hiddenSize == 0) return emptyList()

        val maskData = attentionMask.longBuffer
        val results = arrayOfNulls<FloatArray>(batchSize)
        val futures = (0 until batchSize).map { i ->
            poolExecutor.submit {
                val embedding = FloatArray(hiddenSize)
                var tokenCount = 0
                val sequenceStart = i * maxLen

                for (j in 0 until maxLen) {
                    if (maskData.get(sequenceStart + j) == 1L) {
                        tokenCount++
                        val tokenEmbedding = lastHiddenState[i][j]
                        for (k in 0 until hiddenSize) {
                            embedding[k] += tokenEmbedding[k]
                        }
                    }
                }

                if (tokenCount > 0) {
                    val invTokenCount = 1.0f / tokenCount
                    for (k in 0 until hiddenSize) {
                        embedding[k] *= invTokenCount
                    }
                }

                results[i] = if (l2Normalize) l2Inline(embedding) else embedding
            }
        }

        futures.forEach { it.get() } // Attendre la fin de tous les calculs
        return results.filterNotNull()
    }


    private fun l2Inline(vec: FloatArray): FloatArray {
        var sumSq = 0.0f
        for (v in vec) sumSq += v * v
        val norm = sqrt(sumSq)
        if (norm > 1e-12f) {
            val invNorm = 1.0f / norm
            for (i in vec.indices) {
                vec[i] *= invNorm
            }
        }
        return vec
    }

    override fun close() {
        session.close()
        tokenizer.close()
        poolExecutor.shutdown()
        try {
            if (!poolExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                poolExecutor.shutdownNow()
            }
        } catch (_: InterruptedException) {
            poolExecutor.shutdownNow()
        }
        env.close()
    }

    override fun embed(documents: List<RagDocument>): List<FloatArray> {
        val texts = documents.map { it.text }
        return embedTexts(texts)
    }

    private fun logMemoryUsage(context: String) {
        val runtime = Runtime.getRuntime()
        val usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
        val totalMemory = runtime.totalMemory() / 1024 / 1024
        val maxMemory = runtime.maxMemory() / 1024 / 1024
        println("DEBUG: Mem($context): Used=${usedMemory}MB, Total=${totalMemory}MB, Max=${maxMemory}MB")
    }
}
