package rag.connectors.ner.onnx

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.slf4j.LoggerFactory
import rag.engine.NerExtractor
import rag.model.Entity
import rag.model.RagDocument
import rag.model.Span
import java.nio.LongBuffer
import java.nio.file.Paths
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.min

// ─────────────────────────────────────────────────────────────────────────────
// Labels — ordre exact identique à labels.py
// ─────────────────────────────────────────────────────────────────────────────

private val COARSE_LABELS = listOf(
    "PER", "LOC", "ORG", "TIME", "EVENT", "OBJECT", "VALUE", "ABSTRACT", "NONE"
)
private val FINE_LABELS = listOf(
    "hint_person_name", "hint_person_role", "hint_norp", "hint_group_role",
    "hint_org_name", "hint_gpe", "hint_fac_name", "hint_loc_generic",
    "hint_weapon", "hint_vehicle", "hint_substance", "hint_food",
    "hint_infra", "hint_tool", "hint_object_generic", "hint_object_name",
    "hint_event_nominal", "hint_event_named",
    "hint_time_date", "hint_time_clock", "hint_time_duration",
    "hint_quantity", "hint_measure", "hint_percentage", "hint_count",
    "hint_money", "hint_rate",
    "hint_law", "hint_work_of_art", "hint_concept", "hint_disease", "hint_language"
)

private val COARSE_NONE_IDX = COARSE_LABELS.indexOf("NONE")

/** mask[coarseIdx][fineIdx] = true si ce label fine est autorisé pour ce coarse */
private val COARSE_FINE_MASK: Array<BooleanArray> = run {
    val mapping: Map<Int, List<Int>> = mapOf(
        0 to listOf(0, 1, 2, 3),           // PER    → person_name, person_role, norp, group_role
        1 to listOf(5, 6, 7, 12),          // LOC    → gpe, fac_name, loc_generic, infra
        2 to listOf(4),                    // ORG    → org_name
        3 to listOf(18, 19, 20),           // TIME   → time_date, time_clock, time_duration
        4 to listOf(16, 17),               // EVENT  → event_nominal, event_named
        5 to listOf(8, 9, 10, 11, 13, 14, 15), // OBJECT → weapon…object_name
        6 to listOf(21, 22, 23, 24, 25, 26),   // VALUE  → quantity…rate
        7 to listOf(27, 28, 29, 30, 31),   // ABSTRACT → law…language
    )
    Array(COARSE_LABELS.size) { c -> BooleanArray(FINE_LABELS.size) { f -> mapping[c]?.contains(f) == true } }
}

/** Seuil boundary minimum pour retenir un span comme entité candidate. */
private const val DEFAULT_TAU_BOUNDARY = 0.70f
/** Max prob NONE au-dessus duquel on abandonne ce span. */
private const val DEFAULT_TAU_NONE = 0.50f
/** Prob coarse minimum. */
private const val DEFAULT_TAU_COARSE = 0.45f
/** Longueur max en tokens par label fine. */
private val MAX_TOK_LEN: Map<String, Int> = mapOf(
    "hint_person_name"    to 6,
    "hint_person_role"    to 4,
    "hint_group_role"     to 4,
    "hint_gpe"            to 5,
    "hint_org_name"       to 8,
    "hint_fac_name"       to 7,
    "hint_time_date"      to 6,
    "hint_time_clock"     to 5,
    "hint_event_nominal"  to 6,
    "hint_object_generic" to 5,
    "hint_percentage"     to 4,
    "hint_money"          to 6,
    "hint_measure"        to 6,
    "hint_count"          to 5,
    "hint_quantity"       to 5,
    "hint_rate"           to 7,
)
/** Seuils fine par label. */
private val FINE_THRESHOLDS: Map<String, Float> = mapOf(
    "hint_person_name"    to 0.90f,
    "hint_org_name"       to 0.90f,
    "hint_gpe"            to 0.70f,
    "hint_fac_name"       to 0.70f,
    "hint_time_date"      to 0.70f,
    "hint_time_clock"     to 0.70f,
    "hint_person_role"    to 0.95f,
    "hint_group_role"     to 0.95f,
    "hint_event_nominal"  to 0.97f,
    "hint_object_generic" to 0.97f,
)
private const val DEFAULT_FINE_THRESHOLD = 0.80f

// ─────────────────────────────────────────────────────────────────────────────
// Data classes internes
// ─────────────────────────────────────────────────────────────────────────────

/** Un span candidat extrait depuis la tokenisation. */
private data class SpanCandidate(
    val exampleIdx: Int,
    val tokStart: Int,
    val tokEnd: Int,
    val charStart: Int,
    val charEnd: Int,
    val spanText: String,
)

/** Résultat brut d'un span après scoring par le modèle. */
private data class SpanResult(
    val candidate: SpanCandidate,
    val pBoundary: Float,
    val coarse: String,
    val pCoarse: Float,
    val fine: String,
    val pFine: Float,
    val score: Float,
)

// ─────────────────────────────────────────────────────────────────────────────
// Extracteur principal
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Extracteur NER basé sur le modèle multi-tête span-based (SpanMultiTaskModel).
 *
 * Interface ONNX attendue :
 *   Inputs  : input_ids [B,L], attention_mask [B,L],
 *             span_starts [N], span_ends [N], span_batch_ids [N]
 *   Outputs : boundary_logits [N,2], coarse_logits [N,9], fine_logits [N,32]
 *
 * Cf. export_onnx_multitask.py pour l'export du modèle.
 */
class OnnxMultiHeadEntityExtractor(
    modelPath: String,
    tokenizerDir: String,
    private val maxSeqLen: Int = 128,
    private val maxSpanLen: Int = 8,
    private val tauBoundary: Float = DEFAULT_TAU_BOUNDARY,
    private val tauNone: Float = DEFAULT_TAU_NONE,
    private val tauCoarse: Float = DEFAULT_TAU_COARSE,
    private val useCoreMl: Boolean = false,
    private val intraOpThreads: Int = Runtime.getRuntime().availableProcessors(),
) : AutoCloseable, NerExtractor {

    private val log = LoggerFactory.getLogger(OnnxMultiHeadEntityExtractor::class.java)

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(modelPath, OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(intraOpThreads)
        if (useCoreMl) tryAddCoreML()
    })
    private val tokenizer: HuggingFaceTokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerDir))

    override fun extractNer(documents: List<RagDocument>): List<List<Entity>> =
        extractFromTexts(documents.map { it.text })

    fun extractFromText(text: String): List<Entity> = extractFromTexts(listOf(text)).first()

    fun extractFromTexts(texts: List<String>): List<List<Entity>> {
        if (texts.isEmpty()) return emptyList()
        val t0 = System.nanoTime()

        // ── 1. Tokenisation + spans candidats ──────────────────────────────
        val tTok = System.nanoTime()
        data class EncodedText(
            val text: String,
            val ids: LongArray,
            val seqLen: Int,
            val offsets: List<Pair<Int, Int>?>,  // (charStart, charEnd) par token, null si spécial
        )

        val encodings = texts.mapIndexed { exIdx, text ->
            val enc = tokenizer.encode(text)
            val seqLen = minOf(enc.ids.size, maxSeqLen)
            // DJL retourne les offsets via enc.characterSpans si disponible,
            // sinon on reconstruit depuis wordIds
            val offsets = buildTokenOffsets(text, enc, seqLen)
            EncodedText(text, enc.ids, seqLen, offsets)
        }
        log.debug("[MH] tokenisation batchSize={} ms={}", texts.size, ms(tTok))

        // ── 2. Candidats spans (plats, tous exemples) ──────────────────────
        val tSpan = System.nanoTime()
        val candidates: List<SpanCandidate> = buildCandidates(encodings.mapIndexed { i, enc ->
            Triple(i, enc.text, enc.offsets)
        })
        if (candidates.isEmpty()) return texts.map { emptyList() }
        log.debug("[MH] candidats N={} ms={}", candidates.size, ms(tSpan))

        // ── 3. Construction des tenseurs ────────────────────────────────────
        val tTensor = System.nanoTime()
        val batchSize = texts.size
        val maxLen    = encodings.maxOf { it.seqLen }
        val N         = candidates.size

        val inputIds   = LongArray(batchSize * maxLen)
        val attMask    = LongArray(batchSize * maxLen)
        val spanStarts = LongArray(N)
        val spanEnds   = LongArray(N)
        val spanBatch  = LongArray(N)

        encodings.forEachIndexed { i, enc ->
            for (j in 0 until enc.seqLen) {
                inputIds[i * maxLen + j] = enc.ids[j]
                attMask [i * maxLen + j] = 1L
            }
        }
        candidates.forEachIndexed { k, c ->
            spanStarts[k] = c.tokStart.toLong()
            spanEnds  [k] = c.tokEnd.toLong()
            spanBatch [k] = c.exampleIdx.toLong()
        }
        log.debug("[MH] tenseurs maxLen={} N={} ms={}", maxLen, N, ms(tTensor))

        // ── 4. Inférence ONNX ───────────────────────────────────────────────
        val tInfer = System.nanoTime()
        val shape2D   = longArrayOf(batchSize.toLong(), maxLen.toLong())
        val shape1D   = longArrayOf(N.toLong())

        val tInputIds  = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), shape2D)
        val tAttMask   = OnnxTensor.createTensor(env, LongBuffer.wrap(attMask),  shape2D)
        val tStarts    = OnnxTensor.createTensor(env, LongBuffer.wrap(spanStarts), shape1D)
        val tEnds      = OnnxTensor.createTensor(env, LongBuffer.wrap(spanEnds),   shape1D)
        val tBatchIds  = OnnxTensor.createTensor(env, LongBuffer.wrap(spanBatch),  shape1D)

        val (boundaryLogits, coarseLogits, fineLogits) = session.run(
            mapOf(
                "input_ids"      to tInputIds,
                "attention_mask" to tAttMask,
                "span_starts"    to tStarts,
                "span_ends"      to tEnds,
                "span_batch_ids" to tBatchIds,
            )
        ).use { result ->
            val b = result["boundary_logits"].get().value as Array<FloatArray>  // [N, 2]
            val c = result["coarse_logits"].get().value   as Array<FloatArray>  // [N, 9]
            val f = result["fine_logits"].get().value     as Array<FloatArray>  // [N, 32]
            Triple(b, c, f)
        }

        listOf(tInputIds, tAttMask, tStarts, tEnds, tBatchIds).forEach { it.close() }
        log.debug("[MH] inférence ONNX ms={}", ms(tInfer))

        // ── 5. Décodage par span ────────────────────────────────────────────
        val tDec = System.nanoTime()
        val rawByExample: Array<MutableList<SpanResult>> = Array(texts.size) { mutableListOf() }

        candidates.forEachIndexed { k, cand ->
            val bLogits = boundaryLogits[k]  // FloatArray(2)
            val cLogits = coarseLogits[k]    // FloatArray(9)
            val fLogits = fineLogits[k]      // FloatArray(32)

            val pBoundary = softmaxProb(bLogits, 1)   // prob classe 1 = "entité"
            if (pBoundary < tauBoundary) return@forEachIndexed

            val cProbs = softmax(cLogits)
            val pNone  = cProbs[COARSE_NONE_IDX]
            if (pNone >= tauNone) return@forEachIndexed

            // Chercher le meilleur coarse non-NONE
            val result = bestCoarseFine(cProbs, fLogits, pBoundary) ?: return@forEachIndexed

            val tokLen = cand.tokEnd - cand.tokStart + 1
            val maxTok = MAX_TOK_LEN[result.fine]
            if (maxTok != null && tokLen > maxTok) return@forEachIndexed

            val fineThresh = FINE_THRESHOLDS.getOrDefault(result.fine, DEFAULT_FINE_THRESHOLD)
            if (result.pFine < fineThresh) return@forEachIndexed

            rawByExample[cand.exampleIdx] += SpanResult(
                candidate  = cand,
                pBoundary  = pBoundary,
                coarse     = result.coarse,
                pCoarse    = result.pCoarse,
                fine       = result.fine,
                pFine      = result.pFine,
                score      = pBoundary * result.pCoarse * result.pFine,
            )
        }

        val results = rawByExample.mapIndexed { exIdx, spans ->
            val sorted = spans.sortedByDescending { it.score }
            val filtered = nmsSpans(sorted)
            filtered.map { toEntity(it) }
        }
        log.debug("[MH] décodage ms={}", ms(tDec))
        log.debug("[MH] total batchSize={} maxLen={} N={} ms={}", batchSize, maxLen, N, ms(t0))
        return results
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers : tokenisation & spans
    // ─────────────────────────────────────────────────────────────────────────

    private fun buildTokenOffsets(
        text: String,
        enc: ai.djl.huggingface.tokenizers.Encoding,
        seqLen: Int,
    ): List<Pair<Int, Int>?> {
        // DJL expose wordIds ; on recalcule les charSpans depuis les wordIds + positions des mots
        val words     = Regex("\\S+").findAll(text).toList()
        val wordRanges = words.map { it.range.first to (it.range.last + 1) }
        val wordIds    = enc.wordIds

        val offsets = mutableListOf<Pair<Int, Int>?>()
        var prevWid = -1L
        for (i in 0 until seqLen) {
            val wid = wordIds[i]
            if (wid < 0 || wid >= wordRanges.size) {
                offsets += null
            } else if (wid != prevWid) {
                offsets += wordRanges[wid.toInt()]
            } else {
                offsets += null   // sous-token : pas le premier → on ignore
            }
            prevWid = wid
        }
        return offsets
    }

    private fun buildCandidates(
        examples: List<Triple<Int, String, List<Pair<Int, Int>?>>>,
    ): List<SpanCandidate> {
        val result = mutableListOf<SpanCandidate>()
        for ((exIdx, text, offsets) in examples) {
            val tokenPositions = offsets.indices.filter { offsets[it] != null }
            for (si in tokenPositions.indices) {
                val tokStart = tokenPositions[si]
                for (ei in si until min(si + maxSpanLen, tokenPositions.size)) {
                    val tokEnd  = tokenPositions[ei]
                    val cStart  = offsets[tokStart]!!.first
                    val cEnd    = offsets[tokEnd]!!.second
                    val spanTxt = text.substring(cStart, cEnd).trim()
                    if (spanTxt.length < 2) continue
                    if (spanTxt.all { !it.isLetterOrDigit() }) continue
                    // frontières de mots
                    if (cStart > 0 && text[cStart - 1].isLetterOrDigit()) continue
                    if (cEnd < text.length && text[cEnd].isLetterOrDigit()) continue

                    result += SpanCandidate(exIdx, tokStart, tokEnd, cStart, cEnd, spanTxt)
                }
            }
        }
        return result
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers : scoring
    // ─────────────────────────────────────────────────────────────────────────

    private data class CoarseFineScore(
        val coarse: String, val pCoarse: Float,
        val fine: String,   val pFine: Float,
    )

    private fun bestCoarseFine(
        cProbs: FloatArray,
        fLogits: FloatArray,
        pBoundary: Float,
    ): CoarseFineScore? {
        var best: CoarseFineScore? = null
        var bestScore = -1f

        for (c in COARSE_LABELS.indices) {
            if (c == COARSE_NONE_IDX) continue
            val pCoarse = cProbs[c]
            if (pCoarse < tauCoarse) continue
            val mask = COARSE_FINE_MASK[c]
            if (mask.none { it }) continue

            val maskedLogits = FloatArray(fLogits.size) { i -> if (mask[i]) fLogits[i] else -1e9f }
            val fProbs = softmax(maskedLogits)
            val fIdx   = fProbs.indices.maxByOrNull { fProbs[it] } ?: continue
            val pFine  = fProbs[fIdx]

            val score = pBoundary * pCoarse * pFine
            if (score > bestScore) {
                bestScore = score
                best = CoarseFineScore(COARSE_LABELS[c], pCoarse, FINE_LABELS[fIdx], pFine)
            }
        }
        return best
    }

    // ─────────────────────────────────────────────────────────────────────────
    // NMS : suppression des overlaps (par IoU)
    // ─────────────────────────────────────────────────────────────────────────

    private fun nmsSpans(spans: List<SpanResult>, iouThreshold: Float = 0.6f): List<SpanResult> {
        val kept = mutableListOf<SpanResult>()
        for (s in spans) {
            var discard = false
            for (k in kept) {
                if (iou(s, k) < iouThreshold) continue
                // Même fine → garder le meilleur score
                discard = (s.score <= k.score)
                if (discard) break
            }
            if (!discard) kept += s
        }
        return kept
    }

    private fun iou(a: SpanResult, b: SpanResult): Float {
        val inter = maxOf(0, minOf(a.candidate.charEnd, b.candidate.charEnd) -
                maxOf(a.candidate.charStart, b.candidate.charStart))
        if (inter == 0) return 0f
        val lenA  = a.candidate.charEnd - a.candidate.charStart
        val lenB  = b.candidate.charEnd - b.candidate.charStart
        return inter.toFloat() / (lenA + lenB - inter)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Conversion vers Entity
    // ─────────────────────────────────────────────────────────────────────────

    private fun toEntity(r: SpanResult): Entity = Entity(
        text  = r.candidate.spanText,
        type  = r.fine,
        span  = Span(r.candidate.charStart, r.candidate.charEnd, emptyList()),
        metadata = mapOf(
            "coarse"       to r.coarse,
            "pBoundary"    to r.pBoundary,
            "pCoarse"      to r.pCoarse,
            "pFine"        to r.pFine,
            "score"        to r.score,
        )
    )

    // ─────────────────────────────────────────────────────────────────────────
    // Math utils
    // ─────────────────────────────────────────────────────────────────────────

    private fun softmax(logits: FloatArray): FloatArray {
        val max  = logits.max()
        val exps = FloatArray(logits.size) { exp((logits[it] - max).toDouble()).toFloat() }
        val sum  = exps.sum()
        return FloatArray(exps.size) { exps[it] / sum }
    }

    private fun softmaxProb(logits: FloatArray, classIdx: Int): Float =
        softmax(logits)[classIdx]

    private fun ms(nanoStart: Long) = (System.nanoTime() - nanoStart) / 1_000_000L

    // ─────────────────────────────────────────────────────────────────────────
    // CoreML & lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    private fun OrtSession.SessionOptions.tryAddCoreML() {
        try {
            addCoreML()
            log.info("[MH] CoreML EP activé")
        } catch (e: Exception) {
            log.warn("[MH] CoreML non disponible : {} → fallback CPU", e.message)
        }
    }

    override fun close() {
        tokenizer.close()
        session.close()
    }
}

