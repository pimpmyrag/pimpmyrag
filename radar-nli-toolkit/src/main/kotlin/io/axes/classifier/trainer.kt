// MultiClassEventClassifier.kt
package io.axes.classifier

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.asFlow
import kotlinx.coroutines.flow.flatMapMerge
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.springframework.stereotype.Component
import rag.engine.Embedder
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration
import java.util.Collections
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

data class SoftmaxTrainConfig(
    val learningRate: Double = 0.05,
    val iterations: Int = 2000,
    val weightDecay: Double = 0.0005,
    val seed: Int = 1234,
    val useArcFace: Boolean = true,
    val arcScale: Double = 30.0,     // s
    val arcMarginRad: Double = 0.25, // m (en radians)
    val labelSmoothing: Double = 0.0 // 0..0.1 typiquement
)

class MultiClassSoftmaxClassifier(
    private val inputDim: Int,
    private val categories: List<String>,
    private val config: SoftmaxTrainConfig = SoftmaxTrainConfig()
) {
    private val k = categories.size
    private val catToIndex: Map<String, Int> = categories.withIndex().associate { it.value to it.index }

    // W: [K][D], b: [K]
    private val W: Array<FloatArray> = Array(k) { FloatArray(inputDim) }
    private val b: FloatArray = FloatArray(k)

    init {
        val rnd = java.util.Random(config.seed.toLong())
        val scale = 1.0f / sqrt(inputDim.toFloat())
        for (c in 0 until k) {
            for (i in 0 until inputDim) {
                W[c][i] = ((rnd.nextFloat() - 0.5f) * 2f) * scale
            }
            b[c] = 0f
        }
    }

    fun train(X: List<FloatArray>, yLabels: List<String>) {
        require(X.isNotEmpty()) { "Dataset vide" }
        require(X.size == yLabels.size) { "X et y doivent avoir la même taille" }
        val y = yLabels.map { lab -> catToIndex[lab] ?: error("Label inconnu: $lab") }

        repeat(config.iterations) { iter ->
            val gradW = Array(k) { FloatArray(inputDim) }
            val gradB = FloatArray(k)
            var lossSum = 0.0

            for (n in X.indices) {
                val x = X[n]
                val yIdx = y[n]

                val logits = if (config.useArcFace) arcfaceLogits(x, yIdx) else linearLogits(x)
                val probs = softmax(logits)

                val eps = config.labelSmoothing.coerceIn(0.0, 0.2)
                for (c in 0 until k) {
                    val target = if (c == yIdx) 1.0 - eps else eps / (k - 1).toDouble()
                    val p = probs[c]
                    lossSum += -target * ln((p + 1e-12))

                    // dL/dlogit = p - target
                    val d = (p - target).toFloat()

                    // gradient W et b pour le logit "linéaire"
                    // ArcFace: on approxime la rétroprop en traitant "logit modifié" comme logit
                    // (bonne perf en pratique; version exacte plus lourde)
                    for (i in 0 until inputDim) {
                        gradW[c][i] += d * x[i]
                    }
                    gradB[c] += d
                }
            }

            val invN = 1.0f / X.size.toFloat()

            // Update avec weight decay
            for (c in 0 until k) {
                for (i in 0 until inputDim) {
                    val wd = (config.weightDecay.toFloat() * W[c][i])
                    W[c][i] -= (config.learningRate.toFloat() * (gradW[c][i] * invN + wd))
                }
                b[c] -= (config.learningRate.toFloat() * (gradB[c] * invN))
            }

            if (iter % 200 == 0) {
                val avgLoss = lossSum / X.size.toDouble()
                println("   Iteration $iter: Loss = ${String.format("%.5f", avgLoss)}")
            }
        }
    }

    fun predictProba(x: FloatArray): DoubleArray {
        val logits = if (config.useArcFace) linearLogits(x) else linearLogits(x)
        return softmax(logits)
    }

    fun predictLabel(x: FloatArray): String {
        val probs = predictProba(x)
        var best = 0
        var bestVal = probs[0]
        for (i in 1 until probs.size) {
            if (probs[i] > bestVal) { bestVal = probs[i]; best = i }
        }
        return categories[best]
    }

    private fun linearLogits(x: FloatArray): DoubleArray {
        val out = DoubleArray(k)
        for (c in 0 until k) {
            var s = b[c].toDouble()
            val wc = W[c]
            for (i in 0 until inputDim) s += wc[i].toDouble() * x[i].toDouble()
            out[c] = s
        }
        return out
    }

    // ArcFace: logits = s * cos(theta) sauf pour la classe vraie: s * cos(theta + m)
    // Pré-requis: x L2-normalisé. On normalise aussi W "à la volée".
    private fun arcfaceLogits(x: FloatArray, yIdx: Int): DoubleArray {
        val out = DoubleArray(k)
        val s = config.arcScale
        val m = config.arcMarginRad

        for (c in 0 until k) {
            val cos = cosineWithNormalizedW(x, W[c])
            out[c] = s * cos
        }

        val cosY = out[yIdx] / s
        val sinY = sqrt(max(0.0, 1.0 - cosY * cosY))
        val cosThetaM = cosY * cos(m) - sinY * sin(m)
        out[yIdx] = s * cosThetaM
        return out
    }

    private fun cosineWithNormalizedW(x: FloatArray, w: FloatArray): Double {
        var dot = 0.0
        var wn2 = 0.0
        for (i in 0 until inputDim) {
            dot += x[i].toDouble() * w[i].toDouble()
            wn2 += w[i].toDouble() * w[i].toDouble()
        }
        val wn = sqrt(max(1e-12, wn2))
        return dot / wn
    }

    private fun softmax(logits: DoubleArray): DoubleArray {
        var maxV = logits[0]
        for (i in 1 until logits.size) if (logits[i] > maxV) maxV = logits[i]
        val exps = DoubleArray(logits.size)
        var sum = 0.0
        for (i in logits.indices) {
            val e = exp(logits[i] - maxV)
            exps[i] = e
            sum += e
        }
        for (i in exps.indices) exps[i] /= sum
        return exps
    }
}

// ---------- Utils ----------
fun l2Normalize(v: FloatArray): FloatArray {
    var sum = 0.0
    for (x in v) sum += x * x
    val norm = sqrt(sum).toFloat()
    return if (norm == 0f) v.copyOf() else FloatArray(v.size) { i -> v[i] / norm }
}

fun fmt(x: Double) = String.format("%.3f", x)

data class TrainingExample(val text: String, val label: String)

class MultiClassEventClassifier(
    private val embedder: Embedder,
    private val categories: List<String>,
    private val validationSplit: Double = 0.2,           // NEW: split validation
    private val optimizeThresholds: Boolean = true,      // NEW: seuils par classe
    private val l2NormalizeEmbeddings: Boolean = true    // NEW: L2-normalize
) {
    private val classifiers = ConcurrentHashMap<String, LogisticRegressionClassifier>()


    fun save(path: String) {
        val file = java.io.File(path)
        file.parentFile?.mkdirs()

        val modelData = mapOf(
            "categories" to categories,
            "classifiers" to classifiers.mapValues { (_, clf) ->
                mapOf(
                    "weights" to clf.weights.toList(),
                    "bias" to clf.bias,
                    "mean" to clf.mean.toList(),
                    "std" to clf.std.toList(),
                    "threshold" to clf.threshold
                )
            }
        )

        val mapper = com.fasterxml.jackson.module.kotlin.jacksonObjectMapper()
        file.writeText(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(modelData))
        println("✅ Modèle sauvegardé dans: $path")
    }

    fun getClassifiers() = classifiers.toMap()

    companion object {
        fun load(path: String, embedder: Embedder): MultiClassEventClassifier {
            val file = java.io.File(path)
            val mapper = jacksonObjectMapper()
            val modelData = mapper.readTree(file)

            val categories = modelData["categories"].map { it.asText() }
            val classifier = MultiClassEventClassifier(embedder, categories)

            modelData["classifiers"].fields().forEach { (cat, data) ->
                val weights = data["weights"].map { it.asDouble().toFloat() }.toFloatArray()
                val bias = data["bias"].asDouble().toFloat()
                val mean = data["mean"].map { it.asDouble().toFloat() }.toFloatArray()
                val std = data["std"].map { it.asDouble().toFloat() }.toFloatArray()
                val threshold = data["threshold"].asDouble()

                val clf = LogisticRegressionClassifier(inputDim = weights.size)
                clf.weights = weights
                clf.bias = bias
                clf.mean = mean
                clf.std = std
                clf.threshold = threshold

                classifier.classifiers[cat] = clf
            }

            println("✅ Modèle chargé depuis: $path")
            return classifier
        }
    }

    fun train(examples: List<TrainingExample>) {
        require(examples.isNotEmpty()) { "Dataset vide" }
        val startTime = System.currentTimeMillis()

        println("🚀 Entraînement sur ${examples.size} exemples pour ${categories.size} catégories")

        // ---- Embeddings ----
        println("\n📊 Génération embeddings...")
        val embeddingStart = System.currentTimeMillis()
        val texts = examples.map { it.text }
        // Chunk en groupes de 1000 pour éviter les gros appels
        val textChunks = texts.chunked(128)
        val embeddings = textChunks.flatMapIndexed { index, chunk ->
            println("Iteration embedding ${index + 1}/${textChunks.size} (chunk size=${chunk.size})...")
            embedder.embed(chunk.toRagDocuments()).map { emb ->
                if (l2NormalizeEmbeddings) l2Normalize(emb) else emb
            }
        }
        val yLabels = examples.map { it.label }
        val dim = embeddings.first().size

        val embeddingTime = (System.currentTimeMillis() - embeddingStart) / 1000.0
        println("✅ Embeddings générés en ${String.format("%.1f", embeddingTime)}s (dim=$dim, chunks=${textChunks.size})")

        // ---- Split train/val ----
        val indices = (examples.indices).toMutableList()
        Collections.shuffle(indices, java.util.Random(42)) // reproductible
        val valSize = (examples.size * validationSplit).toInt().coerceAtLeast(1)
        val valIdx = indices.take(valSize)
        val trainIdx = indices.drop(valSize)

        fun <T> subset(list: List<T>, idxs: List<Int>) = idxs.map { list[it] }

        val Xtrain = subset(embeddings, trainIdx)
        val Xval   = subset(embeddings, valIdx)
        val yTrainLabels = subset(yLabels, trainIdx)
        val yValLabels   = subset(yLabels, valIdx)

        println("\n📊 Entraînement classifieurs (one-vs-rest)...")
        val trainStart = System.currentTimeMillis()

        categories.parallelStream().forEach { category ->
            val categoryStart = System.currentTimeMillis()

            val yTrain = yTrainLabels.map { if (it == category) 1 else 0 }
            val yVal   = yValLabels.map   { if (it == category) 1 else 0 }

            val classifier = LogisticRegressionClassifier(
                inputDim = dim,
                // Hyperparams: à ajuster si besoin
                learningRate = 0.02,
                iterations = 2000,
                regularization = 0.01,
                verbose = false,
                patience = 100,          // NEW: early stopping patience
                lrDecay = 0.5,           // NEW: LR scheduler
                lrDecayPatience = 60,    // NEW: si pas d'amélioration, on décroit
                seed = 1234              // NEW: reproductibilité
            )

            classifier.train(Xtrain, yTrain) // intègre class weights + norm

            // Calibrage du seuil par classe (sur validation)
            var threshold = 0.5
            var f1 = 0.0
            if (optimizeThresholds) {
                val scoresVal = Xval.map { classifier.predict(it) }
                threshold = findBestThreshold(scoresVal, yVal)
                f1 = f1At(scoresVal, yVal, threshold)
            }
            classifier.threshold = threshold
            classifiers[category] = classifier

            val categoryTime = (System.currentTimeMillis() - categoryStart) / 1000.0

            // Reporting métriques val
            synchronized(this) {
                val scoresVal = Xval.map { classifier.predict(it) }
                val metrics = metricsAt(scoresVal, yVal, threshold)
                println("   ✅ ${category.padEnd(16)} | " +
                        "P=${fmt(metrics.precision)} R=${fmt(metrics.recall)} F1=${fmt(metrics.f1)} " +
                        "thr=${fmt(threshold)} (${String.format("%.1f", categoryTime)}s)")
            }
        }

        val trainTime = (System.currentTimeMillis() - trainStart) / 1000.0
        println("\n✅ Entraînement terminé en ${String.format("%.1f", trainTime)}s")
        val totalTime = (System.currentTimeMillis() - startTime) / 1000.0
        println("⏱️ Temps total: ${String.format("%.1f", totalTime)}s")
    }

    fun classify(sentence: String): Map<String, Double> {
        var embedding = embedder.embed(listOf(sentence).toRagDocuments()).first()
        if (l2NormalizeEmbeddings) embedding = l2Normalize(embedding)
        return classifiers.mapValues { (_, classifier) ->
            classifier.predict(embedding)
        }
    }

    fun classifyBatch(sentences: List<String>): List<Map<String, Double>> {
        val embs = embedder.embed(sentences.toRagDocuments()).map { emb ->
            if (l2NormalizeEmbeddings) l2Normalize(emb) else emb
        }
        return embs.map { e ->
            classifiers.mapValues { (_, c) -> c.predict(e) }
        }
    }


}

// ---------------- Logistic Regression One-vs-Rest ----------------

class LogisticRegressionClassifier(
    private val inputDim: Int,
    private var learningRate: Double = 0.01,
    private val iterations: Int = 500,
    private val regularization: Double = 0.01,
    private val verbose: Boolean = true,
    private val patience: Int = 50,         // NEW: early stopping
    private val lrDecay: Double = 0.5,      // NEW: multiplicative decay
    private val lrDecayPatience: Int = 30,  // NEW
    seed: Int? = null                       // NEW: reproductibilité
) {
    var weights: FloatArray
    var bias: Float = 0f
    var mean: FloatArray = FloatArray(0)
    var std: FloatArray = FloatArray(0)

    // Seuil recommandé pour cette classe (appris sur validation)
    var threshold: Double = 0.5

    init {
        if (seed != null) {
            // Fixe la seed pour reproductibilité
            val r = java.util.Random(seed.toLong())
            weights = FloatArray(inputDim) { (r.nextFloat() - 0.5f) * 0.02f }
        } else {
            weights = FloatArray(inputDim) { Random.nextFloat() * 0.01f }
        }
    }

    fun train(X: List<FloatArray>, y: List<Int>) {
        require(X.size == y.size) { "X et y doivent avoir la même taille" }
        require(X.isNotEmpty()) { "Dataset vide" }

        // Normalise features (z-score) et conserve mean/std
        val (normalizedX, meanVec, stdVec) = normalize(X)
        this.mean = meanVec
        this.std = stdVec

        // --- Class weights (équilibre pos/neg) ---
        val posCount = y.count { it == 1 }
        val negCount = y.size - posCount
        val wPos = if (posCount > 0) y.size.toFloat() / (2f * posCount) else 1f
        val wNeg = if (negCount > 0) y.size.toFloat() / (2f * negCount) else 1f

        var bestLoss = Double.POSITIVE_INFINITY
        var itSinceBest = 0
        var itSinceLr = 0

        repeat(iterations) { iteration ->
            var totalLoss = 0.0
            val gradWeights = FloatArray(inputDim)
            var gradBias = 0f

            normalizedX.forEachIndexed { idx, x ->
                val p = sigmoid(dotProduct(x, weights) + bias)
                val yi = y[idx]
                val error = (p - yi).toFloat()
                val weight = if (yi == 1) wPos else wNeg

                // gradients
                for (i in gradWeights.indices) {
                    gradWeights[i] += weight * error * x[i]
                }
                gradBias += weight * error

                // weighted logistic loss
                totalLoss += weight * (
                        -yi * ln(p.coerceAtLeast(1e-15)) -
                                (1 - yi) * ln((1 - p).coerceAtLeast(1e-15))
                        )
            }

            val avgLoss = totalLoss / X.size

            // update
            for (i in weights.indices) {
                weights[i] -= (learningRate * (gradWeights[i] / X.size + regularization * weights[i])).toFloat()
            }
            bias -= (learningRate * gradBias / X.size).toFloat()

            // early stopping + LR decay
            if (avgLoss + 1e-8 < bestLoss) {
                bestLoss = avgLoss
                itSinceBest = 0
                itSinceLr = 0
            } else {
                itSinceBest++
                itSinceLr++
            }

            if (itSinceLr >= lrDecayPatience) {
                learningRate *= lrDecay
                itSinceLr = 0
                if (verbose) println("   LR decay -> $learningRate (iter=$iteration)")
            }

            if (verbose && iteration % 100 == 0) {
                println("   Iteration $iteration: Loss = ${String.format("%.5f", avgLoss)}")
            }
            if (itSinceBest >= patience) {
                if (verbose) println("   Early stopping à l'itération $iteration (bestLoss=$bestLoss)")
                return@repeat
            }
        }
    }

    fun predict(x: FloatArray): Double {
        val normalized = normalizeInstance(x, mean, std)
        return sigmoid(dotProduct(normalized, weights) + bias)
    }

    // ---------- Math utils ----------
    private fun sigmoid(z: Float): Double = 1.0 / (1.0 + exp(-z.toDouble()))

    private fun dotProduct(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size)
        var sum = 0f
        for (i in a.indices) sum += a[i] * b[i]
        return sum
    }

    private fun normalize(X: List<FloatArray>): Triple<List<FloatArray>, FloatArray, FloatArray> {
        require(X.isNotEmpty()) { "X vide" }
        val dim = X.first().size

        // Vérification défensive : toutes les dimensions identiques
        for (row in X) require(row.size == dim) { "Dimensions incohérentes" }

        val meanArray = FloatArray(dim) { 0f }
        val stdArray  = FloatArray(dim) { 0f }

        // --- Moyenne par dimension ---
        for (x in X) {
            for (i in 0 until dim) {
                meanArray[i] = meanArray[i] + x[i]   // explicite
            }
        }
        for (i in 0 until dim) {
            meanArray[i] = meanArray[i] / X.size    // explicite
        }

        // --- Variance (somme des carrés des écarts) ---
        for (x in X) {
            for (i in 0 until dim) {
                val d = x[i] - meanArray[i]
                stdArray[i] = stdArray[i] + (d * d)  // explicite
            }
        }

        // --- Écart-type ---
        for (i in 0 until dim) {
            val variance = stdArray[i] / X.size.toFloat()
            val stdValue = kotlin.math.sqrt(variance.toDouble()).toFloat()
            stdArray[i] = if (stdValue == 0f) 1f else stdValue
        }

        // --- Normalisation des vecteurs ---
        val normalized: List<FloatArray> = X.map { x ->
            val out = FloatArray(dim)
            for (i in 0 until dim) {
                out[i] = (x[i] - meanArray[i]) / stdArray[i]
            }
            out
        }

        return Triple(normalized, meanArray, stdArray)
    }
    private fun normalizeInstance(x: FloatArray, mean: FloatArray, std: FloatArray): FloatArray {
        return FloatArray(x.size) { i -> (x[i] - mean[i]) / std[i] }
    }
}

// ---------------- Thresholding & Metrics ----------------

private fun findBestThreshold(scores: List<Double>, labels: List<Int>): Double {
    var bestT = 0.5
    var bestF1 = -1.0
    for (t in 1..99) {
        val thr = t / 100.0
        val f1 = f1At(scores, labels, thr)
        if (f1 > bestF1) { bestF1 = f1; bestT = thr }
    }
    return bestT
}

private data class Metrics(val precision: Double, val recall: Double, val f1: Double)

private fun metricsAt(scores: List<Double>, labels: List<Int>, thr: Double): Metrics {
    var tp = 0; var fp = 0; var fn = 0
    for (i in scores.indices) {
        val pred = if (scores[i] >= thr) 1 else 0
        val y = labels[i]
        if (pred == 1 && y == 1) tp++
        if (pred == 1 && y == 0) fp++
        if (pred == 0 && y == 1) fn++
    }
    val precision = if (tp + fp == 0) 0.0 else tp.toDouble() / (tp + fp)
    val recall = if (tp + fn == 0) 0.0 else tp.toDouble() / (tp + fn)
    val f1 = if (precision + recall == 0.0) 0.0 else 2 * precision * recall / (precision + recall)
    return Metrics(precision, recall, f1)
}

private fun f1At(scores: List<Double>, labels: List<Int>, thr: Double): Double =
    metricsAt(scores, labels, thr).f1



@Component
class SyntheticDataGenerator(
    private val model: String = "Qwen/Qwen3-32B-FP8",
    private val temperature: Double = 0.4,
    private val maxTokens: Int = 2048,
    private val requestTimeoutSec: Long = 180,
    private val parallelism: Int = 2,     // NEW: parallélisme
    private val batchSize: Int = 50,     // NEW: nb phrases par requête
    private val maxRetries: Int = 10       // NEW: retry max par requête
) {
    private val client = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(10))
        .build()
    private val mapper = jacksonObjectMapper()
    private val apiKey: String = System.getenv("LLM_API_KEY") ?: ""

    init {
        require(apiKey.isNotBlank()) {
            "La variable d'environnement LLM_API_KEY est manquante."
        }
    }

    fun loadExamples(path: String): List<TrainingExample> {
        val file = java.io.File(path)
        return mapper.readValue<List<TrainingExample>>(file)
    }
   // Rate limiter global pour toutes les requêtes
   private val requestSemaphore = kotlinx.coroutines.sync.Semaphore(1)

   @OptIn(ExperimentalCoroutinesApi::class)
   fun generateAndSave(
       outputPath: String,
       categories: List<String>,
       countPerCategory: Int = 1000
   ) = runBlocking {
       println("🚀 Génération de ${categories.size} catégories en parallèle...")

       val allExamples = categories
           .asFlow()
           .flatMapMerge(concurrency = parallelism) { category ->
               flow {
                   val examples = generateExamplesForCategory(category, countPerCategory)
                   examples.forEach { text ->
                       emit(TrainingExample(text = text, label = category))
                   }
               }
           }
           .toList()
           .toMutableList()

       // Shuffle pour mélanger les catégories
       allExamples.shuffle()

       // Sauvegarde en JSON
       val file = java.io.File(outputPath)
       file.writeText(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(allExamples))

       println("✅ Sauvegardé ${allExamples.size} exemples dans: $outputPath")
   }

    // Rate limiter : émet un ticket toutes les 1000ms
    @OptIn(DelicateCoroutinesApi::class)
    private val rateLimiter = Channel<Unit>(Channel.RENDEZVOUS).apply {
        kotlinx.coroutines.GlobalScope.launch {
            while (true) {
                send(Unit)
//                delay(1000) // ✅ 1 seconde entre chaque émission
            }
        }
    }

    fun generateExamplesForCategory(category: String, count: Int = 1000): List<String> = runBlocking {
        println("🤖 Génération de $count exemples pour: $category")
        val examplesSeed = getExamplesForCategory(category)
        val totalBatches = ((count + batchSize - 1) / batchSize)

        val all = (0 until totalBatches)
            .asFlow()
            .map { batchIndex ->
                val prompt = buildPrompt(category, batchSize, examplesSeed, styleVariant = randomStyleVariant())

                // ✅ Attend son ticket (1/s max globalement)
                rateLimiter.receive()

                // ✅ Lance la requête SANS bloquer les autres
                withContext(Dispatchers.IO) {
                    retryWithBackoff(
                        maxRetries = maxRetries,
                        initialDelayMs = 500,
                        factor = 2.0
                    ) {
                        performRequest(prompt)
                    }.also {
                        if ((batchIndex + 1) % 5 == 0) {
                            println("   → $category: batch ${batchIndex + 1}/$totalBatches")
                        }
                    }
                }
            }
            .toList()
            .flatten()

        val unique = dedup(all).take(count)
        println("✅ $category: ${unique.size} uniques (avant: ${all.size})")
        unique
    }

   // Version avec rate limiting global

// Version suspend compatible avec Flow
private suspend fun <T> retryWithBackoff(
    maxRetries: Int,
    initialDelayMs: Long = 500,
    factor: Double = 2.0,
    block: suspend () -> T
): T {
    var currentDelay = initialDelayMs
    var lastException: Exception? = null

    repeat(maxRetries) { attempt ->
        try {
            return block()
        } catch (e: Exception) {
            lastException = e
            if (attempt < maxRetries - 1) {
                System.err.println("⚠️  Tentative ${attempt + 1}/$maxRetries échouée: ${e.message}. Retry dans ${currentDelay}ms...")
                delay(currentDelay)
                currentDelay = (currentDelay * factor).toLong().coerceAtMost(10_000)
            }
        }
    }

    throw lastException ?: RuntimeException("Retry épuisé sans exception")
}
    // ---------- Requête asynchrone avec retry/backoff ----------
    private fun submitRequest(prompt: String, pool: java.util.concurrent.ExecutorService): CompletableFuture<List<String>> {
        return CompletableFuture.supplyAsync({
            retry(maxRetries, initialDelayMs = 500) {
                performRequest(prompt)
            }
        }, pool)
    }

    private fun performRequest(prompt: String): List<String> {
        val body = """
            {
              "model": ${mapper.writeValueAsString(model)},
              "temperature": $temperature,
              "max_tokens": $maxTokens,
              "messages": [
                {"role": "system", "content": "Tu réponds UNIQUEMENT en JSON pur (tableau de chaînes), jamais avec ```markdown."},
                {"role": "user", "content": ${mapper.writeValueAsString(prompt)}}
              ]
            }
        """.trimIndent()

        val before = System.currentTimeMillis()
        val request = HttpRequest.newBuilder()
            .uri(URI.create("https://6iifxy15b2tta2-8000.proxy.runpod.net/v1/chat/completions"))
            .header("Authorization", "Bearer $apiKey")
            .header("Content-Type", "application/json")
            .timeout(Duration.ofSeconds(requestTimeoutSec))
            .POST(HttpRequest.BodyPublishers.ofString(body))
            .build()

        val response = client.send(request, HttpResponse.BodyHandlers.ofString())
        println("reponse received after ${System.currentTimeMillis() - before}ms : ${response.body().take(200)}")

        if (response.statusCode() !in 200..299) {
            throw RuntimeException("HTTP ${response.statusCode()} - ${response.body().take(200)}")
        }

        val json = mapper.readTree(response.body())
        var content = json["choices"][0]["message"]["content"].asText()

        // Nettoyage
        content = content
            .replace("```json", "")
            .replace("```", "")
            .trim()

        // Parse robuste
        return parseJsonArrayOfStrings(content)
    }

    private fun <T> retry(times: Int, initialDelayMs: Long = 1000, factor: Double = 2.0, block: () -> T): T {
        var delay = initialDelayMs
        var last: Exception? = null
        for (attempt in 0 until times) {
            try {
                return block()
            } catch (e: Exception) {
                last = e
                if (attempt < times - 1) {
                    Thread.sleep(delay)
                    delay = (delay * factor).toLong().coerceAtMost(10_000)
                }
            }
        }

        throw last ?: RuntimeException("Erreur inconnue dans retry()")
    }


    // ---------- Parsing & nettoyage ----------
    private fun parseJsonArrayOfStrings(content: String): List<String> {
        // Essai direct
        try {
            return mapper.readValue<List<String>>(content).map { it.clean() }.filter { it.isNotBlank() }
        } catch (_: Exception) {
            // Fallback heuristique : extraire le plus grand tableau [...]
            val start = content.indexOf('[')
            val end = content.lastIndexOf(']')
            if (start in 0..<end) {
                val sub = content.substring(start, end + 1)
                return try {
                    mapper.readValue<List<String>>(sub).map { it.clean() }.filter { it.isNotBlank() }
                } catch (e: Exception) {
                    throw RuntimeException("Parsing JSON échoué: ${e.message}\nContenu: ${content.take(200)}")
                }
            } else {
                throw RuntimeException("Réponse sans JSON array: ${content.take(200)}")
            }
        }
    }

    private fun String.clean(): String {
        return this.replace("\n", " ")
            .replace(Regex("\\s+"), " ")
            .trim()
            .removeSurrounding("\"")
    }

    private fun dedup(list: List<String>): List<String> {
        val seen = HashSet<String>()
        val out = ArrayList<String>(list.size)
        for (s in list) {
            val norm = s.lowercase()
            if (seen.add(norm)) out.add(s)
        }
        return out
    }


    private fun randomStyleVariant(): String {
        val styles = listOf(
            "18e-19e siècle français, passé simple",
            "chronique officielle, solennel",
            "registre épistolaire, concis",
            "gazette provinciale, neutre",
            "registre administratif, factuel"
        )
        return styles[Random.nextInt(styles.size)]
    }

    private fun buildPrompt(category: String, count: Int, examples: List<String>, styleVariant: String): String {
        val categoryInstructions = when {
            category.startsWith("temporal_") -> """
        Génère des phrases avec marqueurs temporels ${if (category == "temporal_explicit") "EXPLICITES (dates, années)" else "IMPLICITES (peu après, à l'époque)"}.
                """.trim()
            category.startsWith("coref_") -> """
        Génère des phrases nécessitant résolution de coréférence (pronoms, références anaphoriques).
                """.trim()
            category == "location_explicit" -> """
        Génère des phrases avec mentions de LIEUX PRÉCIS (villes, régions, bâtiments).
                """.trim()
            category in listOf("factual", "opinion", "formal", "subjective", "abstract", "concrete") -> """
        Style: $category. ${when(category) {
                "factual" -> "Énoncés neutres, objectifs, vérifiables."
                "opinion" -> "Jugements, évaluations, appréciations."
                "formal" -> "Registre administratif, solennel."
                "subjective" -> "Émotions, impressions personnelles."
                "abstract" -> "Concepts généraux, notions philosophiques."
                "concrete" -> "Descriptions matérielles, actions physiques."
                else -> ""
            }}
                """.trim()
            category.endsWith("_sentiment") -> """
        Sentiment: ${category.replace("_sentiment", "")}. Lexique connoté.
                """.trim()
            category in listOf("procedural", "technical", "instructional", "definition", "causal") -> """
        Type de contenu: ${when(category) {
                "procedural" -> "Étapes, procédures, déroulement."
                "technical" -> "Termes spécialisés, détails techniques."
                "instructional" -> "Directives, consignes, recommandations."
                "definition" -> "Explications de termes, définitions."
                "causal" -> "Relations cause-effet explicites."
                else -> ""
            }}
                """.trim()
            else -> ""
        }

        return """
        Génère $count phrases COURTES (8-15 mots) en français historique (18e-19e siècle).
        $categoryInstructions
        Style: $styleVariant. Évite répétitions. Variété lexicale maximale.
        
        EXEMPLES:
        ${examples.joinToString("\n") { "- $it" }}
        
        Réponds STRICTEMENT avec un tableau JSON de chaînes, SANS markdown:
        ["phrase1", "phrase2", ...]
            """.trimIndent()
    }

    // ---------- Exemples seed ----------
    private fun getExamplesForCategory(category: String): List<String> = when (category) {
        // === TEMPORAL ===
        "temporal_explicit" -> listOf(
            "Le 14 juillet 1789, la foule s'empara de la Bastille.",
            "En l'an III de la République, on proclama une nouvelle constitution.",
            "Durant l'hiver 1709, le royaume souffrit d'un froid terrible."
        )
        "temporal_implicit" -> listOf(
            "Quelques jours après l'exécution du roi, la guerre éclata.",
            "À l'époque des récoltes, les paysans se soulevèrent.",
            "Peu avant la Révolution, les tensions montaient."
        )
        "planning" -> listOf(
            "On prévoit d'ouvrir les États généraux au printemps prochain.",
            "Le roi envisage de réformer les finances.",
            "L'Assemblée projette d'abolir les privilèges."
        )

        // === SPATIAL ===
        "location_explicit" -> listOf(
            "À Versailles, le roi reçut les députés.",
            "Dans les provinces du Sud, la révolte grondait.",
            "Sur la place de Grève, on dressa l'échafaud."
        )

        // === COREFERENCE ===
        "coref_required" -> listOf(
            "Le ministre démissionna. Il quitta Paris le lendemain.",
            "La reine fut arrêtée. On la conduisit à la Conciergerie.",
            "L'Assemblée vota la loi. Elle entra en vigueur immédiatement."
        )
        "needs_coref_person" -> listOf(
            "Le général ordonna la retraite. Ses troupes obéirent.",
            "Robespierre prit la parole. Son discours dura trois heures.",
            "Le roi refusa de signer. Cette décision provoqua l'indignation."
        )
        "needs_coref_location" -> listOf(
            "La ville fut assiégée. Ses défenseurs résistèrent vaillamment.",
            "Paris se souleva. Ses faubourgs prirent les armes.",
            "Le château fut envahi. Ses occupants prirent la fuite."
        )
        "needs_coref_time" -> listOf(
            "En 1789, le peuple se révolta. Cette année marqua un tournant.",
            "Le mois suivant, on proclama la République.",
            "À cette époque, la famine sévissait partout."
        )

        // === STYLE ===
        "factual" -> listOf(
            "La Convention vota l'abolition de la monarchie le 21 septembre 1792.",
            "Le traité fut signé à Paris en présence des ambassadeurs.",
            "La population de la ville atteignait 600 000 habitants."
        )
        "opinion" -> listOf(
            "Le roi se montra indigne de la confiance du peuple.",
            "Cette mesure parut injuste à la majorité des députés.",
            "On jugea ce décret tyrannique et arbitraire."
        )
        "formal" -> listOf(
            "L'Assemblée nationale décrète ce qui suit.",
            "Sa Majesté a daigné accorder son consentement.",
            "Les soussignés ont l'honneur de porter à la connaissance."
        )
        "subjective" -> listOf(
            "La terreur régnait dans tous les cœurs.",
            "L'espoir d'un changement animait les esprits.",
            "L'angoisse saisit la population."
        )
        "abstract" -> listOf(
            "La liberté triompha sur la tyrannie.",
            "L'égalité devint le fondement de la loi.",
            "La justice exigea réparation."
        )
        "concrete" -> listOf(
            "On dressa la guillotine sur la place.",
            "Les canons tonnèrent toute la nuit.",
            "Le pain manquait dans les boulangeries."
        )

        // === SENTIMENT ===
        "positive_sentiment" -> listOf(
            "La victoire fut accueillie avec enthousiasme.",
            "Le peuple célébra joyeusement sa libération.",
            "Cette nouvelle remplit les cœurs d'espérance."
        )
        "negative_sentiment" -> listOf(
            "Le massacre provoqua l'horreur générale.",
            "La famine causa des souffrances terribles.",
            "Cette défaite plongea le royaume dans le désespoir."
        )
        "uncertainty" -> listOf(
            "On ignore encore l'issue des négociations.",
            "L'avenir du royaume demeure incertain.",
            "Nul ne sait si la paix sera durable."
        )

        // === CONTENT TYPE ===
        "procedural" -> listOf(
            "Pour voter, les députés devaient lever la main.",
            "On commença par lire l'ordre du jour, puis on passa au vote.",
            "Le procès se déroula selon les formes habituelles."
        )
        "technical" -> listOf(
            "Le système électoral reposait sur le suffrage censitaire.",
            "La machine hydraulique utilisait un mécanisme à piston.",
            "Le procédé de fonte nécessitait une température de 1200 degrés."
        )
        "instructional" -> listOf(
            "Les citoyens doivent se présenter au bureau de vote.",
            "Il convient de respecter les formes légales.",
            "On recommande de suivre la procédure établie."
        )
        "definition" -> listOf(
            "On entend par citoyen tout homme jouissant de ses droits.",
            "Le tiers état désigne l'ensemble du peuple.",
            "La Convention est l'assemblée souveraine de la nation."
        )
        "causal" -> listOf(
            "La disette provoqua l'émeute.",
            "C'est parce que le roi refusa que la guerre éclata.",
            "La famine résulta de plusieurs mauvaises récoltes."
        )

        // === DOMAIN ===
        "release_update" -> listOf(
            "La nouvelle version de l'édit fut publiée en mars.",
            "On diffusa une édition corrigée du décret.",
            "Le texte révisé parut dans la gazette officielle."
        )
        "financial" -> listOf(
            "Les finances du royaume étaient exsangues.",
            "On établit un nouvel impôt pour renflouer le trésor.",
            "La dette publique atteignait des sommes considérables."
        )
        "biography_event" -> listOf(
            "Danton naquit à Arcis-sur-Aube en 1759.",
            "À l'âge de vingt ans, il entreprit des études de droit.",
            "Il épousa Antoinette Charpentier en 1787."
        )

        "birth" -> listOf(
            "L'enfant naquit à Versailles le 3 mai 1785.",
            "Marie vint au monde dans la nuit du 12 juillet.",
            "Un nouveau-né poussa son premier cri à l'aube."
        )
        "death" -> listOf(
            "Le roi mourut en son château de Versailles.",
            "L'évêque rendit l'âme le 14 janvier 1793.",
            "Le général périt sur le champ de bataille."
        )
        "marriage" -> listOf(
            "Le prince épousa la duchesse en grande pompe.",
            "Les noces furent célébrées à la cathédrale.",
            "L'union fut scellée devant témoins."
        )
        "baptism" -> listOf(
            "L'enfant fut baptisé en l'église Saint-Pierre.",
            "Le baptême eut lieu le dimanche suivant.",
            "On versa l'eau bénite sur le front du nouveau-né."
        )
        "burial" -> listOf(
            "On enterra le défunt au cimetière paroissial.",
            "La dépouille fut mise en terre le 8 mars.",
            "L'inhumation se déroula en présence de la famille."
        )
        "divorce" -> listOf(
            "Le mariage fut dissous par décision royale.",
            "L'union fut annulée par l'évêque.",
            "Les époux se séparèrent officiellement."
        )
        "dissolution" -> listOf(
            "L'Assemblée fut dissoute par décret royal.",
            "Le parlement cessa ses activités.",
            "L'institution fut abolie en 1792."
        )
        "coup" -> listOf(
            "Le général prit le pouvoir par la force.",
            "Un coup d'État renversa le gouvernement.",
            "Les militaires s'emparèrent du palais."
        )
        "proclamation" -> listOf(
            "Le roi proclama l'état d'urgence.",
            "La République fut proclamée place publique.",
            "On annonça solennellement la nouvelle constitution."
        )
        "abdication" -> listOf(
            "Le roi abdiqua en faveur de son fils.",
            "L'empereur renonça au trône le 6 avril 1814.",
            "Le souverain abandonna le pouvoir."
        )

        "coronation" -> listOf(
            "Le roi fut couronné à Reims.",
            "Le sacre eut lieu en grande pompe à Notre-Dame.",
            "L'empereur reçut la couronne impériale."
        )

        "treaty" -> listOf(
            "Les puissances signèrent un traité de paix.",
            "L'accord fut ratifié par les deux royaumes.",
            "On conclut une alliance entre nations."
        )

        "rebellion" -> listOf(
            "Une révolte éclata dans les provinces du sud.",
            "Les paysans se soulevèrent contre les taxes.",
            "L'insurrection fut matée par l'armée."
        )

        "riot" -> listOf(
            "Une émeute éclata place de Grève.",
            "La foule se souleva contre les autorités.",
            "Des troubles agitèrent le quartier."
        )

        "censure" -> listOf(
            "Le gouvernement interdit la publication.",
            "La censure frappa les gazettes d'opposition.",
            "L'ouvrage fut saisi par la police."
        )

        "exile" -> listOf(
            "Le ministre fut exilé en province.",
            "On bannit le noble du royaume.",
            "Le proscrit quitta le territoire."
        )

        "amnesty" -> listOf(
            "Le roi proclama une amnistie générale.",
            "Les prisonniers politiques furent graciés.",
            "On accorda le pardon aux rebelles."
        )

        "appointment" -> listOf(
            "Le roi nomma un nouveau ministre.",
            "L'évêque fut désigné par Rome.",
            "Le gouverneur prit ses fonctions."
        )
        "famine" -> listOf(
            "Une famine décima la population.",
            "La disette sévit dans toute la province.",
            "Le manque de blé causa des milliers de morts."
        )

        "plague" -> listOf(
            "La peste ravagea la ville durant l'été.",
            "L'épidémie fit des milliers de victimes.",
            "La contagion se répandit rapidement."
        )
        "strike" -> listOf(
            "Les ouvriers cessèrent le travail.",
            "Une grève paralysa les manufactures.",
            "Les artisans refusèrent de reprendre l'ouvrage."
        )

        "tax" -> listOf(
            "Le roi instaura un nouvel impôt.",
            "La taxe sur le sel fut augmentée.",
            "On leva un tribut extraordinaire."
        )

        "trade_agreement" -> listOf(
            "Un accord commercial fut conclu avec l'Espagne.",
            "Les deux puissances signèrent un traité de commerce.",
            "On établit des relations marchandes."
        )

        "bankruptcy" -> listOf(
            "La banque fit faillite en mars.",
            "Le négociant se déclara en banqueroute.",
            "L'entreprise cessa ses paiements."
        )

        "monopoly" -> listOf(
            "Le roi octroya un monopole à la compagnie.",
            "On accorda le privilège exclusif du commerce.",
            "La manufacture reçut le droit unique de production."
        )

        "price_regulation" -> listOf(
            "Les autorités fixèrent le prix du pain.",
            "Un édit réglementa le coût des denrées.",
            "On établit un maximum pour les céréales."
        )
        "excommunication" -> listOf(
            "L'évêque excommunia le noble rebelle.",
            "Le pape lança l'anathème contre le roi.",
            "On retrancha le prêtre de l'Église."
        )

        "canonization" -> listOf(
            "Rome canonisa le bienheureux.",
            "Le pape proclama la sainteté du martyr.",
            "On éleva le défunt aux autels."
        )

        "pilgrimage" -> listOf(
            "Des milliers de fidèles se rendirent en pèlerinage.",
            "La procession atteignit le sanctuaire.",
            "On marcha jusqu'au lieu saint."
        )

        "ordination" -> listOf(
            "L'évêque ordonna trois nouveaux prêtres.",
            "Le diacre reçut les ordres sacrés.",
            "On consacra le clerc."
        )

        "heresy_trial" -> listOf(
            "L'Inquisition jugea le suspect d'hérésie.",
            "Le tribunal ecclésiastique condamna le blasphémateur.",
            "On poursuivit les partisans de doctrines hérétiques."
        )

        "conversion" -> listOf(
            "Le prince abjura la foi protestante.",
            "On se convertit au catholicisme.",
            "Le roi changea de religion."
        )
        "publication" -> listOf(
            "L'auteur publia son ouvrage à Paris.",
            "Le livre parut chez l'éditeur royal.",
            "On imprima le traité en trois volumes."
        )

        "discovery" -> listOf(
            "Le savant découvrit une nouvelle planète.",
            "On mit au jour des ruines antiques.",
            "L'explorateur révéla l'existence d'un continent."
        )

        "invention" -> listOf(
            "L'ingénieur inventa une machine révolutionnaire.",
            "On créa un nouveau procédé technique.",
            "Le mécanicien conçut un dispositif novateur."
        )

        "exhibition" -> listOf(
            "Le Salon exposa les œuvres des artistes.",
            "On présenta les tableaux au public.",
            "L'exposition ouvrit ses portes en mai."
        )

        "performance" -> listOf(
            "La troupe joua la pièce à la Comédie-Française.",
            "On représenta l'opéra devant la cour.",
            "Le spectacle fut donné trois fois."
        )

        "award" -> listOf(
            "L'Académie décerna le prix au savant.",
            "On remit une médaille au lauréat.",
            "Le roi honora l'artiste d'une récompense."
        )

        "censorship_lifting" -> listOf(
            "La censure fut levée sur l'ouvrage.",
            "On autorisa la publication du livre.",
            "L'interdiction fut rapportée."
        )
        "embassy" -> listOf(
            "Une ambassade fut dépêchée à Vienne.",
            "Le roi envoya un émissaire auprès du sultan.",
            "On reçut les représentants étrangers."
        )

        "negotiation" -> listOf(
            "Les puissances négocièrent un armistice.",
            "On entama des pourparlers de paix.",
            "Les diplomates discutèrent des conditions."
        )

        "ultimatum" -> listOf(
            "Le gouvernement adressa un ultimatum à l'ennemi.",
            "On exigea une réponse sous huit jours.",
            "La puissance menaça de représailles."
        )

        "alliance" -> listOf(
            "Les royaumes conclurent une alliance défensive.",
            "On scella un pacte entre nations.",
            "Les souverains s'engagèrent mutuellement."
        )

        "declaration_of_war" -> listOf(
            "Le roi déclara la guerre à l'Autriche.",
            "Les hostilités furent officiellement ouvertes.",
            "On rompit les relations diplomatiques."
        )

        "oath" -> listOf(
            "Le roi prêta serment devant l'Assemblée.",
            "Les députés jurèrent fidélité à la nation.",
            "Le général fit serment d'allégeance."
        )
        "decree" -> listOf(
            "Un décret royal abolit les privilèges.",
            "L'édit fut promulgué le 4 août 1789.",
            "La loi entra en vigueur immédiatement."
        )
        "election" -> listOf(
            "Les députés furent élus au suffrage indirect.",
            "Le scrutin eut lieu dans toutes les provinces.",
            "L'élection se déroula sans incident."
        )
        "battle" -> listOf(
            "L'armée livra bataille près de Valmy.",
            "Le combat fit rage toute la journée.",
            "Les troupes s'affrontèrent avec violence."
        )
        "arrest" -> listOf(
            "La police arrêta le suspect à son domicile.",
            "On appréhenda le prévenu dans la nuit.",
            "Le fuyard fut capturé par la maréchaussée."
        )

        "pardon" -> listOf(
            "Le roi gracia le condamné.",
            "On accorda la clémence au prisonnier.",
            "La peine fut commuée en exil."
        )

        "execution" -> listOf(
            "Le condamné fut exécuté place publique.",
            "On guillotina le traître à l'aube.",
            "La sentence capitale fut appliquée."
        )

        "testimony" -> listOf(
            "Le témoin déposa devant le tribunal.",
            "On recueillit la déposition sous serment.",
            "L'accusé témoigna en sa faveur."
        )
        "surrender" -> listOf(
            "La garnison capitula après trois mois de siège.",
            "L'ennemi se rendit sans conditions.",
            "Les assiégés déposèrent les armes."
        )

        "retreat" -> listOf(
            "L'armée battit en retraite vers le nord.",
            "Les troupes se replièrent en bon ordre.",
            "On abandonna les positions."
        )

        "mutiny" -> listOf(
            "Les soldats se mutinèrent contre leurs officiers.",
            "Une rébellion éclata dans le régiment.",
            "Les marins refusèrent d'obéir."
        )

        "desertion" -> listOf(
            "De nombreux soldats désertèrent les rangs.",
            "On signala la fuite de plusieurs régiments.",
            "Les troupes abandonnèrent le camp."
        )

        "armistice" -> listOf(
            "Les belligérants signèrent un armistice.",
            "Les hostilités furent suspendues.",
            "On conclut une trêve provisoire."
        )


        "attack" -> listOf(
            "L'ennemi attaqua au lever du jour.",
            "Les rebelles assaillirent la garnison.",
            "L'offensive débuta à l'aube."
        )
        "siege" -> listOf(
            "Les troupes assiégèrent la forteresse.",
            "Le siège dura plusieurs semaines.",
            "La ville fut encerclée par l'armée."
        )
        "bombardment" -> listOf(
            "L'artillerie bombarda la citadelle.",
            "Les canons tonnèrent durant trois jours.",
            "La ville subit un pilonnage intensif."
        )
        "skirmish" -> listOf(
            "Une escarmouche éclata à la frontière.",
            "Les avant-postes s'accrochèrent brièvement.",
            "Un léger engagement opposa les patrouilles."
        )
        "conflict" -> listOf(
            "Un conflit éclata entre provinces.",
            "Les tensions dégénérèrent en affrontements.",
            "Une querelle opposa les factions rivales."
        )
        "trial_opening" -> listOf(
            "Le procès du roi s'ouvrit le 11 décembre.",
            "L'audience débuta en présence des juges.",
            "Le tribunal entama les débats."
        )
        "conviction" -> listOf(
            "L'accusé fut reconnu coupable de trahison.",
            "Le verdict tomba après délibération.",
            "On condamna le prévenu aux galères."
        )
        "acquittal" -> listOf(
            "Le tribunal acquitta l'accusé.",
            "Le prévenu fut déclaré innocent.",
            "La cour prononça la relaxe."
        )
        "appeal" -> listOf(
            "L'avocat interjeta appel du jugement.",
            "La défense demanda révision du procès.",
            "On porta l'affaire devant la cour supérieure."
        )
        "dismissal" -> listOf(
            "L'affaire fut classée sans suite.",
            "Le juge rejeta la plainte.",
            "Les charges furent abandonnées."
        )
        "speech" -> listOf(
            "Le député prononça un discours enflammé.",
            "L'orateur harangua la foule.",
            "Le ministre prit la parole devant l'Assemblée."
        )
        "announcement" -> listOf(
            "On annonça la victoire place publique.",
            "Le crieur proclama la nouvelle.",
            "L'annonce fut faite au son du tambour."
        )
        "press_conference" -> listOf(
            "Le ministre reçut les gazettes.",
            "Une déclaration fut faite aux journalistes.",
            "Le porte-parole s'adressa à la presse."
        )
        "statement" -> listOf(
            "Le gouvernement publia un communiqué.",
            "Une déclaration officielle fut diffusée.",
            "Le ministre fit une mise au point."
        )
        "broadcast" -> listOf(
            "La proclamation fut lue dans tout le royaume.",
            "On diffusa l'édit dans les provinces.",
            "La nouvelle se répandit par voie d'affiche."
        )
        "foundation" -> listOf(
            "On fonda une académie des sciences.",
            "L'institution fut créée par décret royal.",
            "La société savante vit le jour en 1775."
        )
        "inauguration" -> listOf(
            "Le roi inaugura le nouveau palais.",
            "L'édifice fut ouvert en grande pompe.",
            "La cérémonie d'ouverture eut lieu le 14 juillet."
        )
        "fire" -> listOf(
            "Un incendie ravagea le quartier.",
            "Le feu détruisit plusieurs immeubles.",
            "Les flammes consumèrent l'entrepôt."
        )
        "earthquake" -> listOf(
            "Un tremblement de terre secoua la région.",
            "Le séisme fit des centaines de victimes.",
            "La terre trembla durant plusieurs minutes."
        )
        "demolition" -> listOf(
            "On démolit la Bastille pierre par pierre.",
            "L'ancienne forteresse fut rasée.",
            "La destruction de l'édifice prit des mois."
        )
        else -> listOf("Exemple générique pour $category")
    }
}

data class NerEntitySpan(
    val start: Int,
    val end: Int,   // end exclusif
    val label: String
)

data class NerSentenceExample(
    val id: String? = null,
    val text: String,
    val entities: List<NerEntitySpan> = emptyList()
)

data class TokenCandidateExample(
    val sentence: String,
    val tokenIndex: Int,
    val tokenStart: Int,
    val tokenEnd: Int,
    val label: String
)

data class TokenWindowConfig(
    val leftTokens: Int = 6,
    val rightTokens: Int = 6,
    val shortSentenceMaxTokens: Int = 20,
    val shortSentenceMaxChars: Int = 220,
    val maxContextChars: Int = 384,
    val addMarkers: Boolean = true,
    val markerLeft: String = "[ENT]",
    val markerRight: String = "[/ENT]"
)

private data class TokOffset(val start: Int, val end: Int)

private fun tokenizeWithOffsets(text: String): List<TokOffset> {
    val out = ArrayList<TokOffset>()
    var i = 0
    while (i < text.length) {
        while (i < text.length && text[i].isWhitespace()) i++
        if (i >= text.length) break
        val start = i
        while (i < text.length && !text[i].isWhitespace()) i++
        val end = i
        out.add(TokOffset(start, end))
    }
    return out
}

private fun normalizeSpaces(s: String): String = s.replace(Regex("\\s+"), " ")

private fun injectMarkers(text: String, start: Int, end: Int, cfg: TokenWindowConfig): String {
    if (!cfg.addMarkers) return text
    require(start in 0..text.length && end in 0..text.length && start < end) {
        "Invalid span $start..$end for text length ${text.length}"
    }
    val left = text.substring(0, start)
    val span = text.substring(start, end)
    val right = text.substring(end)
    return "$left${cfg.markerLeft} $span ${cfg.markerRight}$right"
}

private fun buildTokenContext(sentence: String, tokenIndex: Int, cfg: TokenWindowConfig): String {
    val toks = tokenizeWithOffsets(sentence)
    require(tokenIndex in toks.indices) { "tokenIndex=$tokenIndex hors bornes (0..${toks.lastIndex})" }

    val isShort = toks.size <= cfg.shortSentenceMaxTokens && sentence.length <= cfg.shortSentenceMaxChars
    val (spanStart, spanEnd) = toks[tokenIndex]

    val base = if (isShort) {
        injectMarkers(sentence, spanStart, spanEnd, cfg)
    } else {
        val leftI = (tokenIndex - cfg.leftTokens).coerceAtLeast(0)
        val rightJ = (tokenIndex + cfg.rightTokens).coerceAtMost(toks.lastIndex)
        val ctxStart = toks[leftI].start
        val ctxEnd = toks[rightJ].end

        val window = sentence.substring(ctxStart, ctxEnd)
        val localStart = spanStart - ctxStart
        val localEnd = spanEnd - ctxStart
        injectMarkers(window, localStart, localEnd, cfg)
    }

    val cleaned = normalizeSpaces(base).trim()
    return if (cleaned.length <= cfg.maxContextChars) cleaned else cleaned.take(cfg.maxContextChars)
}

private fun overlaps(aStart: Int, aEnd: Int, bStart: Int, bEnd: Int): Boolean =
    max(aStart, bStart) < minOf(aEnd, bEnd)

private fun labelForToken(
    tokenStart: Int,
    tokenEnd: Int,
    entities: List<NerEntitySpan>,
    outsideLabel: String
): String {
    // Stratégie simple: si le token overlap une entité, on prend le label de l'entité la plus longue.
    var best: NerEntitySpan? = null
    for (e in entities) {
        if (overlaps(tokenStart, tokenEnd, e.start, e.end)) {
            if (best == null || (e.end - e.start) > (best.end - best.start)) best = e
        }
    }
    return best?.label ?: outsideLabel
}

private fun nerToTokenCandidates(
    examples: List<NerSentenceExample>,
    allowedLabels: Set<String>,
    outsideLabel: String = "O",
    maxTokensPerSentence: Int = 256
): List<TokenCandidateExample> {
    val out = ArrayList<TokenCandidateExample>()

    for (ex in examples) {
        val sent = ex.text
        val toks = tokenizeWithOffsets(sent)
        if (toks.isEmpty()) continue

        val limited = if (toks.size > maxTokensPerSentence) toks.take(maxTokensPerSentence) else toks

        for ((i, tok) in limited.withIndex()) {
            val label = labelForToken(tok.start, tok.end, ex.entities, outsideLabel)
            if (label != outsideLabel && label !in allowedLabels) continue
            out.add(
                TokenCandidateExample(
                    sentence = sent,
                    tokenIndex = i,
                    tokenStart = tok.start,
                    tokenEnd = tok.end,
                    label = label
                )
            )
        }
    }
    return out
}

private fun downsampleOutside(
    examples: List<TokenCandidateExample>,
    outsideLabel: String,
    maxOutsideToPosRatio: Int = 3,
    seed: Int = 42
): List<TokenCandidateExample> {
    val pos = examples.filter { it.label != outsideLabel }
    val neg = examples.filter { it.label == outsideLabel }

    if (pos.isEmpty()) return examples // rien à faire

    val maxNeg = (pos.size * maxOutsideToPosRatio).coerceAtLeast(1)
    if (neg.size <= maxNeg) return examples

    val rnd = java.util.Random(seed.toLong())
    val negShuffled = neg.shuffled(rnd)
    return (pos + negShuffled.take(maxNeg)).shuffled(rnd)
}

class TokenCandidateEntityClassifier(
    private val embedder: Embedder,
    private val entityLabels: List<String>,
    private val outsideLabel: String = "O",
    private val tokenWindow: TokenWindowConfig = TokenWindowConfig(),
    private val validationSplit: Double = 0.2,
    private val optimizeThresholds: Boolean = true,
    private val l2NormalizeEmbeddings: Boolean = true
) {
    private val classifiers = ConcurrentHashMap<String, LogisticRegressionClassifier>()

    fun save(path: String) {
        val file = java.io.File(path)
        file.parentFile?.mkdirs()

        val modelData = mapOf(
            "labels" to entityLabels,
            "outsideLabel" to outsideLabel,
            "tokenWindow" to mapOf(
                "leftTokens" to tokenWindow.leftTokens,
                "rightTokens" to tokenWindow.rightTokens,
                "shortSentenceMaxTokens" to tokenWindow.shortSentenceMaxTokens,
                "shortSentenceMaxChars" to tokenWindow.shortSentenceMaxChars,
                "maxContextChars" to tokenWindow.maxContextChars,
                "addMarkers" to tokenWindow.addMarkers,
                "markerLeft" to tokenWindow.markerLeft,
                "markerRight" to tokenWindow.markerRight
            ),
            "classifiers" to classifiers.mapValues { (_, clf) ->
                mapOf(
                    "weights" to clf.weights.toList(),
                    "bias" to clf.bias,
                    "mean" to clf.mean.toList(),
                    "std" to clf.std.toList(),
                    "threshold" to clf.threshold
                )
            }
        )

        val mapper = jacksonObjectMapper()
        file.writeText(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(modelData))
        println("✅ Modèle token-candidat sauvegardé dans: $path")
    }

    companion object {
        fun load(path: String, embedder: Embedder): TokenCandidateEntityClassifier {
            val file = java.io.File(path)
            val mapper = jacksonObjectMapper()
            val modelData = mapper.readTree(file)

            val labels = modelData["labels"].map { it.asText() }
            val outsideLabel = modelData["outsideLabel"].asText("O")

            val w = modelData["tokenWindow"]
            val tokenWindow = TokenWindowConfig(
                leftTokens = w["leftTokens"].asInt(6),
                rightTokens = w["rightTokens"].asInt(6),
                shortSentenceMaxTokens = w["shortSentenceMaxTokens"].asInt(20),
                shortSentenceMaxChars = w["shortSentenceMaxChars"].asInt(220),
                maxContextChars = w["maxContextChars"].asInt(384),
                addMarkers = w["addMarkers"].asBoolean(true),
                markerLeft = w["markerLeft"].asText("[ENT]"),
                markerRight = w["markerRight"].asText("[/ENT]")
            )

            val classifier = TokenCandidateEntityClassifier(
                embedder = embedder,
                entityLabels = labels,
                outsideLabel = outsideLabel,
                tokenWindow = tokenWindow
            )

            modelData["classifiers"].fields().forEach { (lab, data) ->
                val weights = data["weights"].map { it.asDouble().toFloat() }.toFloatArray()
                val bias = data["bias"].asDouble().toFloat()
                val mean = data["mean"].map { it.asDouble().toFloat() }.toFloatArray()
                val std = data["std"].map { it.asDouble().toFloat() }.toFloatArray()
                val threshold = data["threshold"].asDouble()

                val clf = LogisticRegressionClassifier(inputDim = weights.size)
                clf.weights = weights
                clf.bias = bias
                clf.mean = mean
                clf.std = std
                clf.threshold = threshold

                classifier.classifiers[lab] = clf
            }

            println("✅ Modèle token-candidat chargé depuis: $path")
            return classifier
        }
    }

    /**
     * Entraîne un OVR sur des candidats tokens.
     * labels entraînés = entityLabels (sans outsideLabel). Les exemples avec label=outsideLabel servent de négatifs.
     */
    fun trainTokenCandidates(examples: List<TokenCandidateExample>) {
        require(examples.isNotEmpty()) { "Dataset vide" }

        val startTime = System.currentTimeMillis()
        println("🚀 Entraînement token-candidat sur ${examples.size} exemples, labels=${entityLabels.size}, outside='$outsideLabel'")

        // 1) Construire textes contextualisés autour du token
        val ctxTexts = examples.map { ex -> buildTokenContext(ex.sentence, ex.tokenIndex, tokenWindow) }

        // 2) Embeddings
        println("\n📊 Génération embeddings (token-context)...")
        val embeddingStart = System.currentTimeMillis()
        val textChunks = ctxTexts.chunked(128)
        val embeddings = textChunks.flatMapIndexed { index, chunk ->
            println("Iteration embedding ${index + 1}/${textChunks.size} (chunk size=${chunk.size})...")
            embedder.embed(chunk.toRagDocuments()).map { emb ->
                if (l2NormalizeEmbeddings) l2Normalize(emb) else emb
            }
        }
        val yLabels = examples.map { it.label }
        val dim = embeddings.first().size

        val embeddingTime = (System.currentTimeMillis() - embeddingStart) / 1000.0
        println("✅ Embeddings générés en ${String.format("%.1f", embeddingTime)}s (dim=$dim, chunks=${textChunks.size})")

        // 3) Split train/val
        val indices = (examples.indices).toMutableList()
        Collections.shuffle(indices, java.util.Random(42))
        val valSize = (examples.size * validationSplit).toInt().coerceAtLeast(1)
        val valIdx = indices.take(valSize)
        val trainIdx = indices.drop(valSize)

        fun <T> subset(list: List<T>, idxs: List<Int>) = idxs.map { list[it] }

        val Xtrain = subset(embeddings, trainIdx)
        val Xval = subset(embeddings, valIdx)
        val yTrainLabels = subset(yLabels, trainIdx)
        val yValLabels = subset(yLabels, valIdx)

        println("\n📊 Entraînement classifieurs (one-vs-rest) sur tokens...")

        entityLabels.parallelStream().forEach { label ->
            val labelStart = System.currentTimeMillis()

            val yTrain = yTrainLabels.map { if (it == label) 1 else 0 }
            val yVal = yValLabels.map { if (it == label) 1 else 0 }

            val classifier = LogisticRegressionClassifier(
                inputDim = dim,
                learningRate = 0.02,
                iterations = 2000,
                regularization = 0.01,
                verbose = false,
                patience = 100,
                lrDecay = 0.5,
                lrDecayPatience = 60,
                seed = 1234
            )

            classifier.train(Xtrain, yTrain)

            var threshold = 0.5
            if (optimizeThresholds) {
                val scoresVal = Xval.map { classifier.predict(it) }
                threshold = findBestThreshold(scoresVal, yVal)
            }
            classifier.threshold = threshold
            classifiers[label] = classifier

            val labelTime = (System.currentTimeMillis() - labelStart) / 1000.0

            synchronized(this) {
                val scoresVal = Xval.map { classifier.predict(it) }
                val metrics = metricsAt(scoresVal, yVal, threshold)
                println("   ✅ ${label.padEnd(16)} | P=${fmt(metrics.precision)} R=${fmt(metrics.recall)} F1=${fmt(metrics.f1)} thr=${fmt(threshold)} (${String.format("%.1f", labelTime)}s)")
            }
        }

        val totalTime = (System.currentTimeMillis() - startTime) / 1000.0
        println("\n✅ Entraînement token-candidat terminé (total ${String.format("%.1f", totalTime)}s)")
    }

    /**
     * Charge un dataset NER (offsets) et entraîne directement.
     */
    fun trainFromNerDataset(
        nerPath: String,
        maxTokensPerSentence: Int = 256,
        maxOutsideToPosRatio: Int = 3
    ) {
        val ner = NerTokenDatasetIO.loadNerExamples(nerPath)
        val tokenExamples0 = NerTokenDatasetIO.toTokenCandidates(
            nerExamples = ner,
            entityLabels = entityLabels,
            outsideLabel = outsideLabel,
            maxTokensPerSentence = maxTokensPerSentence
        )
        val tokenExamples = downsampleOutside(tokenExamples0, outsideLabel, maxOutsideToPosRatio)
        println("📉 Token candidates: total=${tokenExamples0.size}, après downsample=${tokenExamples.size} (ratio O<=${maxOutsideToPosRatio}x)")
        trainTokenCandidates(tokenExamples)
    }

    /**
     * Score un token candidat (renvoie score par label). Utile pour décodage NER.
     */
    fun scoreToken(sentence: String, tokenIndex: Int): Map<String, Double> {
        val ctx = buildTokenContext(sentence, tokenIndex, tokenWindow)
        var emb = embedder.embed(listOf(ctx).toRagDocuments()).first()
        if (l2NormalizeEmbeddings) emb = l2Normalize(emb)
        return classifiers.mapValues { (_, clf) -> clf.predict(emb) }
    }

    /**
     * Prédit le meilleur label (ou outsideLabel) pour un token.
     */
    fun predictTokenLabel(sentence: String, tokenIndex: Int): Pair<String, Double> {
        val scores = scoreToken(sentence, tokenIndex)
        var bestLab = outsideLabel
        var bestScore = 0.0
        for ((lab, s) in scores) {
            val thr = classifiers[lab]?.threshold ?: 0.5
            if (s >= thr && s > bestScore) {
                bestLab = lab
                bestScore = s
            }
        }
        return bestLab to bestScore
    }
}

/**
 * Helpers de chargement dataset NER (JSON ou JSONL) -> token candidates.
 */
object NerTokenDatasetIO {
    private val mapper = jacksonObjectMapper()

    fun loadNerExamples(path: String): List<NerSentenceExample> {
        val file = java.io.File(path)
        require(file.exists()) { "Fichier introuvable: $path" }

        val content = file.readText()
        // JSONL : une ligne = un exemple
        val isJsonl = content.lineSequence().any { it.trimStart().startsWith("{") } && !content.trimStart().startsWith("[")
        return if (isJsonl) {
            content.lineSequence()
                .filter { it.isNotBlank() }
                .map { line -> mapper.readValue<NerSentenceExample>(line) }
                .toList()
        } else {
            mapper.readValue<List<NerSentenceExample>>(file)
        }
    }

    fun toTokenCandidates(
        nerExamples: List<NerSentenceExample>,
        entityLabels: List<String>,
        outsideLabel: String = "O",
        maxTokensPerSentence: Int = 256
    ): List<TokenCandidateExample> {
        return nerToTokenCandidates(
            examples = nerExamples,
            allowedLabels = entityLabels.toSet(),
            outsideLabel = outsideLabel,
            maxTokensPerSentence = maxTokensPerSentence
        )
    }
}

/**
 * Catalogue ACE 2005 (Event types / subtypes) sous forme de labels plats.
 * Convention: "ACE:<TYPE>/<SUBTYPE>".
 *
 * NB: ACE distingue aussi Entity/Relation/Value/Time; ici on ne gère que les événements.
 */
object Ace2005EventLabels {
    const val PREFIX = "ACE:"

    // Source: ACE 2005 event ontology (8 types, 33 subtypes)
    private val typeToSubtypes: Map<String, List<String>> = linkedMapOf(
        "Life" to listOf(
            "Be-Born",
            "Marry",
            "Divorce",
            "Injure",
            "Die"
        ),
        "Movement" to listOf(
            "Transport"
        ),
        "Transaction" to listOf(
            "Transfer-Ownership",
            "Transfer-Money"
        ),
        "Business" to listOf(
            "Start-Org",
            "Merge-Org",
            "Declare-Bankruptcy",
            "End-Org"
        ),
        "Conflict" to listOf(
            "Attack",
            "Demonstrate"
        ),
        "Contact" to listOf(
            "Meet",
            "Phone-Write"
        ),
        "Personnel" to listOf(
            "Start-Position",
            "End-Position",
            "Nominate",
            "Elect"
        ),
        "Justice" to listOf(
            "Arrest-Jail",
            "Release-Parole",
            "Trial-Hearing",
            "Charge-Indict",
            "Sue",
            "Convict",
            "Sentence",
            "Fine",
            "Execute",
            "Extradite",
            "Acquit",
            "Appeal",
            "Pardon"
        )
    )

    fun allLabels(): List<String> {
        return typeToSubtypes.flatMap { (type, subs) ->
            subs.map { sub -> "$PREFIX$type/$sub" }
        }
    }

    fun isAceLabel(label: String): Boolean = label.startsWith(PREFIX)

    fun isSupported(label: String): Boolean {
        if (!isAceLabel(label)) return false
        val raw = label.removePrefix(PREFIX)
        val parts = raw.split('/')
        if (parts.size != 2) return false
        val type = parts[0]
        val sub = parts[1]
        return typeToSubtypes[type]?.contains(sub) == true
    }

    fun normalize(type: String, subtype: String): String = "$PREFIX$type/$subtype"
}

/**
 * Catalogue ACE 2005 (Entity types / subtypes) sous forme de labels plats.
 * Convention: "ACE-ENT:<TYPE>/<SUBTYPE>".
 *
 * Types ACE (entités): PER, ORG, GPE, LOC, FAC, VEH, WEA.
 * Subtypes: Individual, Group, Organization, Government, Commercial, Educational, Media,
 *           Medical-Science, Non-Governmental, Religious; Nation, State-or-Province, County-or-District,
 *           Population-Center, GPE-Cluster, Continent; Region, Water-Body, Land-Region, Celestial;
 *           Building-Grounds, Airport, Plant, Path, Subarea-Facility; etc.
 *
 * NB: Les sous-types varient selon les ressources; ici on couvre la taxonomie ACE 2005 la plus courante.
 */
object Ace2005EntityLabels {
    const val PREFIX = "ACE-ENT:"

    private val typeToSubtypes: Map<String, List<String>> = linkedMapOf(
        "PER" to listOf(
            "Individual",
            "Group",
            "Indeterminate"
        ),
        "ORG" to listOf(
            "Government",
            "Commercial",
            "Educational",
            "Media",
            "Medical-Science",
            "Non-Governmental",
            "Religious",
            "Sports",
            "Entertainment",
            "Indeterminate"
        ),
        "GPE" to listOf(
            "Nation",
            "State-or-Province",
            "County-or-District",
            "Population-Center",
            "GPE-Cluster",
            "Continent",
            "Indeterminate"
        ),
        "LOC" to listOf(
            "Region",
            "Water-Body",
            "Land-Region",
            "Celestial",
            "Indeterminate"
        ),
        "FAC" to listOf(
            "Building-Grounds",
            "Airport",
            "Plant",
            "Path",
            "Subarea-Facility",
            "Indeterminate"
        ),
        "VEH" to listOf(
            "Air",
            "Land",
            "Water",
            "Subarea-Vehicle",
            "Indeterminate"
        ),
        "WEA" to listOf(
            "Biological",
            "Chemical",
            "Exploding",
            "Nuclear",
            "Projectile",
            "Sharp",
            "Shooting",
            "Indeterminate"
        )
    )

    fun allLabels(): List<String> {
        return typeToSubtypes.flatMap { (type, subs) ->
            subs.map { sub -> "$PREFIX$type/$sub" }
        }
    }

    fun isAceEntityLabel(label: String): Boolean = label.startsWith(PREFIX)

    fun isSupported(label: String): Boolean {
        if (!isAceEntityLabel(label)) return false
        val raw = label.removePrefix(PREFIX)
        val parts = raw.split('/')
        if (parts.size != 2) return false
        val type = parts[0]
        val sub = parts[1]
        return typeToSubtypes[type]?.contains(sub) == true
    }

    fun normalize(type: String, subtype: String): String = "$PREFIX$type/$subtype"
}

object AceEntityDatasetValidator {
    fun validateLabels(examples: List<NerSentenceExample>, outsideLabel: String = "O"): List<String> {
        val errors = ArrayList<String>()
        for ((idx, ex) in examples.withIndex()) {
            for (ent in ex.entities) {
                if (ent.label == outsideLabel) continue
                if (!Ace2005EntityLabels.isSupported(ent.label)) {
                    errors.add("Exemple#${idx + 1}: label ACE-ENT invalide '${ent.label}'")
                }
            }
        }
        return errors
    }
}

