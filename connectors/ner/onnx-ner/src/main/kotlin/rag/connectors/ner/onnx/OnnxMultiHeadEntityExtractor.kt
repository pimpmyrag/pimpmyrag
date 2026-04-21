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

// ─────────────────────────────────────────────────────────────────────────────
// Taxonomie sémantique des labels fine
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Rôle sémantique d'un span détecté dans la taxonomie de l'application :
 *
 * - ENTITY           : entité nommée classique (personne, org, lieu nommé, loi…)
 * - MENTION_ROLE     : mention de rôle ou de groupe (pas une entité nommée, mais un participant typé)
 * - TRIGGER_INFO     : indice sur la présence d'un événement (nominal ou adjectival) —
 *                      sert à typer/déclencher l'extraction d'événement, pas une entité en soi
 * - TRIGGER_ARG      : candidat argument d'un événement (temps, lieu générique, valeur, objet…)
 */
enum class LabelKind { ENTITY, MENTION_ROLE, TRIGGER_INFO, TRIGGER_ARG }

/** Rôle sémantique de chaque label fine dans la taxonomie. */
val LABEL_KIND: Map<String, LabelKind> = mapOf(
    // ── Entités nommées ───────────────────────────────────────────────────────
    "hint_person_name"    to LabelKind.ENTITY,
    "hint_gpe"            to LabelKind.ENTITY,
    "hint_org_name"       to LabelKind.ENTITY,
    "hint_event_named"    to LabelKind.ENTITY,
    "hint_law"            to LabelKind.ENTITY,
    "hint_fac_name"       to LabelKind.ENTITY,       // toponyme de facility (palais, aéroport…)

    // ── Mentions de rôle / groupe ─────────────────────────────────────────────
    "hint_person_role"    to LabelKind.MENTION_ROLE,
    "hint_group_role"     to LabelKind.MENTION_ROLE,
    "hint_norp"           to LabelKind.MENTION_ROLE, // nationalité/appartenance (peut être adj.)

    // ── Indices de trigger événementiel ──────────────────────────────────────
    "hint_event_nominal"  to LabelKind.TRIGGER_INFO,

    // ── Candidats arguments de trigger ───────────────────────────────────────
    "hint_time_date"      to LabelKind.TRIGGER_ARG,
    "hint_time_clock"     to LabelKind.TRIGGER_ARG,
    "hint_time_duration"  to LabelKind.TRIGGER_ARG,
    "hint_loc_generic"    to LabelKind.TRIGGER_ARG,
    "hint_infra"          to LabelKind.TRIGGER_ARG,
    "hint_object_generic" to LabelKind.TRIGGER_ARG,
    "hint_object_name"    to LabelKind.TRIGGER_ARG,
    "hint_vehicle"        to LabelKind.TRIGGER_ARG,
    "hint_substance"      to LabelKind.TRIGGER_ARG,
    "hint_food"           to LabelKind.TRIGGER_ARG,
    "hint_weapon"         to LabelKind.TRIGGER_ARG,
    "hint_tool"           to LabelKind.TRIGGER_ARG,
    "hint_disease"        to LabelKind.TRIGGER_ARG,
    "hint_concept"        to LabelKind.TRIGGER_ARG,
    "hint_work_of_art"    to LabelKind.TRIGGER_ARG,
    "hint_language"       to LabelKind.TRIGGER_ARG,
    "hint_quantity"       to LabelKind.TRIGGER_ARG,
    "hint_measure"        to LabelKind.TRIGGER_ARG,
    "hint_percentage"     to LabelKind.TRIGGER_ARG,
    "hint_count"          to LabelKind.TRIGGER_ARG,
    "hint_money"          to LabelKind.TRIGGER_ARG,
    "hint_rate"           to LabelKind.TRIGGER_ARG,
)

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
private const val DEFAULT_TAU_NONE = 0.99f
/** Prob coarse minimum. */
private const val DEFAULT_TAU_COARSE = 0.45f
/** Score minimum global (pBnd × pCoarse × pFine) pour retenir un span. */
private const val DEFAULT_MIN_SCORE = 0.10f
/** Longueur max en SOUS-TOKENS par label fine (≈ mots × 2 pour DeBERTa fr). */
private val MAX_TOK_LEN: Map<String, Int> = mapOf(
    "hint_person_name"    to 12,
    "hint_person_role"    to 8,
    "hint_group_role"     to 8,
    "hint_gpe"            to 10,
    "hint_org_name"       to 16,
    "hint_fac_name"       to 14,
    "hint_time_date"      to 12,
    "hint_time_clock"     to 10,
    "hint_event_nominal"  to 12,
    "hint_event_named"    to 16,
    "hint_object_generic" to 10,
    "hint_object_name"    to 10,
    "hint_percentage"     to 8,
    "hint_money"          to 12,
    "hint_measure"        to 12,
    "hint_count"          to 10,
    "hint_quantity"       to 10,
    "hint_rate"           to 14,
    "hint_vehicle"        to 12,
    "hint_substance"      to 10,
    "hint_disease"        to 10,
    "hint_loc_generic"    to 12,
    "hint_infra"          to 14,
    "hint_concept"        to 16,
    "hint_law"            to 16,
)
/** Seuils fine par label. */
private val FINE_THRESHOLDS: Map<String, Float> = mapOf(
    "hint_person_name"    to 0.90f,
    "hint_org_name"       to 0.90f,
    "hint_gpe"            to 0.70f,
    "hint_fac_name"       to 0.70f,
    "hint_time_date"      to 0.70f,
    "hint_time_clock"     to 0.70f,
    "hint_person_role"    to 0.90f,
    "hint_group_role"     to 0.90f,
    "hint_event_nominal"  to 0.90f,
    "hint_object_generic" to 0.90f,
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
    private val minScore: Float = DEFAULT_MIN_SCORE,
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
            val wordRanges: List<Pair<Int, Int>>,    // (charStart, charEnd) par MOT via Regex
            val charOffsets: List<Pair<Int, Int>?>,  // charTokenSpans par token (peut être null)
            val wordIds: LongArray,
            val tokens: Array<String>,               // strings de tokens (pour détection ponctuation)
        )

        val encodings = texts.mapIndexed { _, text ->
            val enc    = tokenizer.encode(text)
            val seqLen = minOf(enc.ids.size, maxSeqLen)
            val wordList   = Regex("\\S+").findAll(text).toList()
            val wordRanges = wordList.map { it.range.first to (it.range.last + 1) }
            val charOffsets: List<Pair<Int, Int>?> = enc.charTokenSpans
                .take(seqLen)
                .map { span -> span?.let { it.start to it.end } }
            EncodedText(text, enc.ids, seqLen, wordRanges, charOffsets, enc.wordIds, enc.tokens)
        }
        log.debug("[MH] tokenisation batchSize={} ms={}", texts.size, ms(tTok))

        // ── 2. Candidats spans (plats, tous exemples) ──────────────────────
        val tSpan = System.nanoTime()
        val candidates: List<SpanCandidate> = buildCandidates(encodings.mapIndexed { i, enc ->
            EncodedExample(i, enc.text, enc.wordRanges, enc.charOffsets, enc.wordIds, enc.tokens, enc.seqLen)
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

//        // DEBUG
//        System.err.println("[DBG] N candidates = ${candidates.size}")
//        candidates.take(5).forEachIndexed { k, c ->
//            System.err.println("[DBG]   cand[$k] ex=${c.exampleIdx} tok(${c.tokStart},${c.tokEnd}) char(${c.charStart},${c.charEnd}) '${c.spanText}'")
//        }

        candidates.forEachIndexed { k, cand ->
            val bLogits = boundaryLogits[k]  // FloatArray(2)
            val cLogits = coarseLogits[k]    // FloatArray(9)
            val fLogits = fineLogits[k]      // FloatArray(32)

            val pBoundary = softmaxProb(bLogits, 1)   // prob classe 1 = "entité"
            // DEBUG: log top boundary scores
//            if (k < 20 || pBoundary > 0.1f) {
//                System.err.println("[DBG] span(${cand.tokStart},${cand.tokEnd}) '${cand.spanText}' pBnd=${"%.4f".format(pBoundary)}")
//            }
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

            val score = pBoundary * result.pCoarse * result.pFine
            if (score < minScore) return@forEachIndexed

            rawByExample[cand.exampleIdx] += SpanResult(
                candidate  = cand,
                pBoundary  = pBoundary,
                coarse     = result.coarse,
                pCoarse    = result.pCoarse,
                fine       = result.fine,
                pFine      = result.pFine,
                score      = score,
            )
        }

        val results = rawByExample.map { spans ->
            // Tri par score × √longueur pour favoriser les spans plus complets à score proche.
            // Ex : "tremblement de terre" (0.911 × √20 = 4.07) > "tremblement" (0.915 × √11 = 3.03)
            val sorted = spans.sortedByDescending { sr ->
                sr.score * Math.sqrt((sr.candidate.charEnd - sr.candidate.charStart).toDouble())
            }
            val filtered = nmsSpans(sorted)
            // Ré-trier par score pour l'affichage
            filtered.sortedByDescending { it.score }.map { toEntity(it) }
        }
        log.debug("[MH] décodage ms={}", ms(tDec))
        log.debug("[MH] total batchSize={} maxLen={} N={} ms={}", batchSize, maxLen, N, ms(t0))
        return results
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers : tokenisation & spans
    // ─────────────────────────────────────────────────────────────────────────

    private data class EncodedExample(
        val exIdx: Int,
        val text: String,
        val wordRanges: List<Pair<Int, Int>>,    // (charStart, charEnd) par mot via Regex
        val charOffsets: List<Pair<Int, Int>?>,  // charTokenSpans par token (peut être null)
        val wordIds: LongArray,
        val tokens: Array<String>,               // strings de tokens pour détection ponctuation
        val seqLen: Int,
    )

    /**
     * Construit les spans candidats en énumérant toutes les fenêtres de mots [1..maxSpanLen].
     *
     * - wordRanges (Regex "\\S+") → bornes char fiables pour TOUS les mots
     * - wordIds → groupement tokens→mots (inclus, dernier sous-token correct)
     * - charOffsets (charTokenSpans) → trim fin de ponctuation sur tokEnd (best-effort)
     * - tokens → fallback pour détection ponctuation si charOffsets est null
     *
     * tokStart/tokEnd INCLUSIFS, alignés sur build_multitask_dataset.py Python.
     */
    private fun buildCandidates(examples: List<EncodedExample>): List<SpanCandidate> {
        val result = mutableListOf<SpanCandidate>()

        for ((exIdx, text, wordRanges, charOffsets, wordIds, tokens, seqLen) in examples) {

            // ── Regrouper les tokens en mots via wordIds ──────────────────────
            data class Word(
                val firstTok: Int, val lastTok: Int,
                val charStart: Int, val charEnd: Int,
            )

            val words = mutableListOf<Word>()
            var prevWid  = Long.MIN_VALUE
            var wFirstTok = -1
            var wLastTok  = -1

            fun flushWord() {
                if (wFirstTok < 0) return
                val wid = wordIds[wFirstTok]
                if (wid < 0 || wid >= wordRanges.size) { wFirstTok = -1; return }
                val (cs, ce) = wordRanges[wid.toInt()]
                // Détecter les contractions françaises (l', d', j', etc.) :
                // si le mot commence par une courte particule suivie d'une apostrophe,
                // on ajoute aussi une vue "sans la particule" pour que "Assemblée"
                // soit un point de départ valide.
                val wordText = text.substring(cs, ce)
                val apoIdx   = wordText.indexOfFirst { it == '\'' || it == '\u2019' }
                if (apoIdx in 1..3 && apoIdx < wordText.length - 1) {
                    // Vue tronquée : commence après l'apostrophe
                    val truncStart = cs + apoIdx + 1
                    // Trouver le premier sous-token dont l'offset char commence après l'apostrophe
                    val truncFirstTok = (wFirstTok..wLastTok).firstOrNull { i ->
                        val off = charOffsets.getOrNull(i)
                        off != null && off.first >= truncStart
                    } ?: (wFirstTok + 1).coerceAtMost(wLastTok)
                    words += Word(truncFirstTok, wLastTok, truncStart, ce)
                }
                words += Word(wFirstTok, wLastTok, cs, ce)
                wFirstTok = -1
            }

            for (i in 0 until seqLen) {
                val wid = wordIds[i]
                if (wid < 0) { flushWord(); prevWid = wid; continue }  // token spécial
                if (wid != prevWid) { flushWord(); wFirstTok = i }
                wLastTok = i
                prevWid  = wid
            }
            flushWord()

            // ── Énumérer les fenêtres de mots ────────────────────────────────
            for (si in words.indices) {
                val tokStart  = words[si].firstTok
                val charStart = words[si].charStart

                for (ei in si until min(si + maxSpanLen, words.size)) {
                    var tokEnd  = words[ei].lastTok
                    var charEnd = words[ei].charEnd
                    var spanTxt = text.substring(charStart, charEnd).trim()

                    // Trimmer la ponctuation finale ET reculer tokEnd en conséquence
                    while (spanTxt.isNotEmpty() && !spanTxt.last().isLetterOrDigit()) {
                        spanTxt = spanTxt.dropLast(1).trimEnd()
                        val newCharEnd = charStart + spanTxt.length
                        // Reculer tokEnd si le token courant est ponctuation-only
                        while (tokEnd > tokStart) {
                            val tOff = charOffsets.getOrNull(tokEnd)
                            if (tOff != null) {
                                if (tOff.first >= newCharEnd) tokEnd-- else break
                            } else {
                                // fallback : inspecter la string du token
                                val tok = tokens.getOrNull(tokEnd)?.trimStart('▁')?.trim() ?: break
                                if (tok.isNotEmpty() && tok.all { !it.isLetterOrDigit() }) tokEnd--
                                else break
                            }
                        }
                        charEnd = newCharEnd
                    }

                    if (spanTxt.length < 2) continue
                    if (spanTxt.all { !it.isLetterOrDigit() }) continue
                    if (charStart > 0 && text[charStart - 1].isLetterOrDigit()) continue
                    if (charEnd < text.length && text[charEnd].isLetterOrDigit()) continue

                    result += SpanCandidate(exIdx, tokStart, tokEnd, charStart, charEnd, spanTxt)
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

    private fun nmsSpans(spans: List<SpanResult>, iouThreshold: Float = 0.5f): List<SpanResult> {
        val kept = mutableListOf<SpanResult>()
        for (s in spans) {
            if (kept.none { k -> iouOrContainment(s, k) >= iouThreshold }) kept += s
        }
        return kept
    }

    /** IoU standard OU ratio de containment (le plus grand des deux). */
    private fun iouOrContainment(a: SpanResult, b: SpanResult): Float {
        val inter = maxOf(0, minOf(a.candidate.charEnd, b.candidate.charEnd) -
                maxOf(a.candidate.charStart, b.candidate.charStart))
        if (inter == 0) return 0f
        val lenA  = a.candidate.charEnd - a.candidate.charStart
        val lenB  = b.candidate.charEnd - b.candidate.charStart
        val iou   = inter.toFloat() / (lenA + lenB - inter)
        // containment : quel pourcentage du plus petit span est couvert ?
        val containment = inter.toFloat() / minOf(lenA, lenB)
        return maxOf(iou, containment)
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
            "kind"         to (LABEL_KIND[r.fine] ?: LabelKind.TRIGGER_ARG),
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

