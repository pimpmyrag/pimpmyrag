package rag.connectors.ud.stanza

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.*
import rag.engine.NerExtractorFromUD
import rag.model.*
import java.nio.LongBuffer
import java.nio.file.Paths
import kotlin.math.min

// ------------------------------------------------------------
// ENUM
// ------------------------------------------------------------

enum class EntityType {
    HINT_PERSON_NAME,
    HINT_PERSON_ROLE,
    HINT_NORP,
    HINT_GROUP_ROLE,
    HINT_ORG_NAME,
    HINT_GPE,
    HINT_FAC_NAME,
    HINT_LOC_GENERIC,
    HINT_WEAPON,
    HINT_VEHICLE,
    HINT_SUBSTANCE,
    HINT_FOOD,
    HINT_INFRA,
    HINT_TOOL,
    HINT_OBJECT_GENERIC,
    HINT_OBJECT_NAME,
    HINT_EVENT_NOMINAL,
    HINT_EVENT_NAMED,
    HINT_TIME_DATE,
    HINT_TIME_CLOCK,
    HINT_TIME_DURATION,
    HINT_UNKNOWN,
    HINT_QUANTITY;

    fun isName() = name.endsWith("_NAME")
    fun isHint() = !isName()
}

// ------------------------------------------------------------
// SPAN FILTER
// ------------------------------------------------------------

// PRON exclu : les pronoms (Il, le, sa…) ne doivent pas être des entités NER standalone.

data class SimpleEntityModel(
    val text: String,
    val label: EntityType,
    val start: Int,
    val end: Int,
    val tokens: List<UDToken>,
    val isHint: Boolean
)

private const val COARSE_UNKNOWN: Long = -1L
private val ALL_COARSE_IDS = longArrayOf(0L, 1L, 2L, 3L, 4L, 5L)

// ------------------------------------------------------------
// EXTRACTEUR
// ------------------------------------------------------------

class OnnxSpanNerExtractor(
    modelPath: String,
    tokenizerDir: String,
    private val labelNames: List<String> =
    // 22 labels EXACTS du SpanClassifier (dataset.py::LABELS), dans l'ordre d'index.
    // Tout changement ici doit être synchronisé avec le dict LABELS du training.
        listOf(
            "hint_person_name",    //  0
            "hint_person_role",    //  1
            "hint_norp",           //  2
            "hint_group_role",     //  3
            "hint_org_name",       //  4
            "hint_gpe",            //  5
            "hint_fac_name",       //  6
            "hint_loc_generic",    //  7
            "hint_weapon",         //  8
            "hint_vehicle",        //  9
            "hint_substance",      // 10
            "hint_food",           // 11
            "hint_infra",          // 12
            "hint_tool",           // 13
            "hint_object_generic", // 14
            "hint_object_name",    // 15
            "hint_event_nominal",  // 16
            "hint_event_named",    // 17
            "hint_time_date",      // 18
            "hint_time_clock",     // 19
            "hint_time_duration",  // 20
            "hint_quantity"        // 21
        ),
    /** CoreML EP — Apple Neural Engine + GPU (Mac uniquement).
     *  Beaucoup plus rapide que CPU INT8 sur Apple Silicon.
     *  Ignoré silencieusement si non disponible (Linux/Windows). */
    useCoreMl: Boolean = false,
    intraOpThreads: Int = Runtime.getRuntime().availableProcessors(),
    /** Nombre de phrases traitées en un seul appel ONNX (batching UD sentences). */
    private val sentBatchSize: Int = 8,
    /**
     * Longueur maximale de séquence **par phrase** (en tokens).
     * Comme on décode phrase par phrase, 128 suffit pour la quasi-totalité des phrases
     * et réduit significativement le padding ainsi que le coût d'attention O(n²).
     * Les candidats dont le span dépasse cette limite sont silencieusement écartés.
     */
    private val maxSeqLen: Int = 128,
) : AutoCloseable, NerExtractorFromUD {

    private val log = org.slf4j.LoggerFactory.getLogger(OnnxSpanNerExtractor::class.java)

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = run {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(intraOpThreads)
            if (useCoreMl) {
                try {
                    addCoreML()
                    println("[OnnxSpanNerExtractor] CoreML EP activé (Apple Neural Engine / GPU)")
                } catch (e: Exception) {
                    println("[OnnxSpanNerExtractor] CoreML non disponible : ${e.message} → fallback CPU")
                }
            }
        }
        env.createSession(modelPath, opts)
    }
    private val tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerDir))
    private val hasTokenType = session.inputNames.contains("token_type_ids")
    private val hasCoarseInput = session.inputNames.contains("coarse_ids")



    // ------------------------------------------------------------
    // PUBLIC — API NER
    // ------------------------------------------------------------

    override fun extractNerFromUD(udDocuments: List<UDDocument>): List<List<Entity>> =
        extractSimple(udDocuments).map { list ->
            list.map {
                Entity(
                    text = it.text,
                    type = it.label.name.lowercase(),
                    span = Span(it.start, it.end, it.tokens),
                    metadata = mapOf("isHint" to it.isHint)
                )
            }
        }

    // ------------------------------------------------------------
    // CORE LOGIC
    // ------------------------------------------------------------

    fun extractSimple(docs: List<UDDocument>): List<List<SimpleEntityModel>> {
        val t0 = System.nanoTime()
        fun ms(t: Long) = (System.nanoTime() - t) / 1_000_000L

        data class Cand(val docIdx: Int, val sent: UDSentence, val span: Span)

        val collected = mutableListOf<Cand>()

        // 1) Construire les spans UD NOMINAUX
        // Les tokens avec deprel flat/flat:name/name/compound sont déjà captés par le span
        // de leur tête via reconstructSpan → les ignorer ici évite de générer des sous-spans
        // redondants (ex : "Martin" en flat:name de "Jacques" → sinon "Martin" duplique "Jacques Martin").
        val flatDeprels = setOf("flat", "name", "compound")
        docs.forEachIndexed { di, doc ->
            doc.sentences.forEach { sent ->
                sent.tokens.forEach { tok ->
                    if (!shouldClassifyHead(tok, sent.tokens)) return@forEach

                    val sp = reconstructSpan(sent.tokens, tok.id)
                    if (sp.end > sp.start) {
                        collected += Cand(di, sent, sp)
                    }
                }
            }
        }


        if (collected.isEmpty()) {
            return docs.indices.map { emptyList() }
        }

        val byDoc = collected.groupBy { it.docIdx }
        val result = mutableMapOf<Int, MutableList<SimpleEntityModel>>()

        data class SentInfo(
            val docIdx: Int,
            val sent: UDSentence,
            val sentenceText: String,
            val enc: ai.djl.huggingface.tokenizers.Encoding,
            val starts: LongArray,
            val ends: LongArray,
            val validCands: List<Cand>,
            val coarseArr: LongArray,
            val seqLen: Int
        )

        // ── Phase 1 : tokeniser toutes les phrases de tous les docs ───────────
        val allSentInfos = mutableListOf<SentInfo>()
        for ((docIdx, candsOfDoc) in byDoc) {
            val doc = docs[docIdx]
            result[docIdx] = mutableListOf()

            for ((sent, candsInSent) in candsOfDoc.groupBy { it.sent }) {
                val sentenceText = doc.text.substring(sent.start, sent.end)
                val enc = tokenizer.encode(sentenceText, true, false)
                val seqLen = min(enc.ids.size, maxSeqLen)
                val offsets: List<Pair<Int, Int>?> = enc.charTokenSpans
                    .take(seqLen)
                    .map { span -> span?.let { it.start to it.end } }

                val starts = mutableListOf<Long>()
                val ends = mutableListOf<Long>()
                val validCands = mutableListOf<Cand>()

                for (c in candsInSent) {
                    val relStart = c.span.start - sent.start
                    val relEnd = c.span.end - sent.start
                    val mapped = mapCharToTokenSpan(offsets, relStart, relEnd) ?: continue
                    if (mapped.second > seqLen) continue
                    starts += mapped.first.toLong()
                    ends += mapped.second.toLong()
                    validCands += c
                }
                if (starts.isEmpty()) continue


                val coarseArr = LongArray(validCands.size) { i ->
                    COARSE_UNKNOWN // car ici, c’est UD-only => coarse non fiable
                }


                allSentInfos += SentInfo(
                    docIdx = docIdx,
                    sent = sent,
                    sentenceText = sentenceText,
                    enc = enc,
                    starts = starts.toLongArray(),
                    ends = ends.toLongArray(),
                    validCands = validCands,
                    coarseArr = coarseArr,
                    seqLen = seqLen
                )
            }
        }

        // ── Phase 2 : inférence par chunks globaux (sentBatchSize phrases) ────
        for (chunk in allSentInfos.chunked(sentBatchSize)) {
            val n = chunk.size
            val maxLen = chunk.maxOf { it.seqLen }

            val flatIds = LongArray(n * maxLen) { 0L }
            val flatAtt = LongArray(n * maxLen) { 0L }
            val flatTti = LongArray(n * maxLen) { 0L }

            chunk.forEachIndexed { b, si ->
                for (i in 0 until si.seqLen) {
                    flatIds[b * maxLen + i] = si.enc.ids[i]
                    flatAtt[b * maxLen + i] = si.enc.attentionMask[i].toLong()
                }
                si.enc.typeIds?.let { tti ->
                    for (i in 0 until si.seqLen) {
                        flatTti[b * maxLen + i] = tti[i].toLong()
                    }
                }
            }

            val allSpanStarts = mutableListOf<Long>()
            val allSpanEnds = mutableListOf<Long>()
            val allBatchIdx = mutableListOf<Long>()
            val allCoarseIds = mutableListOf<Long>()
            val allCandRefs = mutableListOf<Pair<Int, Int>>() // (chunkIdx, candIdx)

            data class Ref(val chunkIdx: Int, val candIdx: Int)

            data class RowRef(val ref: Ref, val coarseId: Long)

            val allRowRefs = mutableListOf<RowRef>()


            chunk.forEachIndexed { b, si ->
                for (i in si.starts.indices) {
                    val ref = Ref(b, i)
                    val coarse = si.coarseArr[i]

                    if (hasCoarseInput) {
                        if (coarse == COARSE_UNKNOWN) {
                            // marginalisation: 6 passes
                            for (c in ALL_COARSE_IDS) {
                                allSpanStarts += si.starts[i]
                                allSpanEnds += si.ends[i]
                                allBatchIdx += b.toLong()
                                allCoarseIds += c
                                allRowRefs += RowRef(ref, c)
                            }
                        } else {
                            allSpanStarts += si.starts[i]
                            allSpanEnds += si.ends[i]
                            allBatchIdx += b.toLong()
                            allCoarseIds += coarse
                            allRowRefs += RowRef(ref, coarse)
                        }
                    } else {
                        // si le modèle ne prend pas coarse_ids, comportement inchangé (une seule passe)
                        allSpanStarts += si.starts[i]
                        allSpanEnds += si.ends[i]
                        allBatchIdx += b.toLong()
                        allRowRefs += RowRef(ref, COARSE_UNKNOWN)
                    }
                }
            }


            val inputIdsT = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(flatIds),
                longArrayOf(n.toLong(), maxLen.toLong())
            )
            val attT = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(flatAtt),
                longArrayOf(n.toLong(), maxLen.toLong())
            )
            val ttiT = if (hasTokenType) {
                OnnxTensor.createTensor(
                    env,
                    LongBuffer.wrap(flatTti),
                    longArrayOf(n.toLong(), maxLen.toLong())
                )
            } else null

            val startT = OnnxTensor.createTensor(env, allSpanStarts.toLongArray())
            val endT = OnnxTensor.createTensor(env, allSpanEnds.toLongArray())
            val batchIdxT = OnnxTensor.createTensor(env, allBatchIdx.toLongArray())

            val map = mutableMapOf<String, OnnxTensor>(
                "input_ids" to inputIdsT,
                "attention_mask" to attT,
                "span_starts" to startT,
                "span_ends" to endT,
                "span_batch_idx" to batchIdxT
            )
            if (hasTokenType) map["token_type_ids"] = ttiT!!

            val coarseIdsT = if (hasCoarseInput) {
                OnnxTensor.createTensor(env, allCoarseIds.toLongArray()).also { map["coarse_ids"] = it }
            } else null

            val tInfer = System.nanoTime()
            val rows = to2D(session.run(map).use { it[0].value })
            log.debug(
                "[SpanNER-A] inférence  N={}  maxLen={}  spans={}  ms={}",
                n, maxLen, allCandRefs.size, (System.nanoTime() - tInfer) / 1_000_000L
            )

            // 1) Accumule les logits par candidat original (Ref)
            val sumByRef = mutableMapOf<Ref, FloatArray>()
            val countByRef = mutableMapOf<Ref, Int>()

            for (r in rows.indices) {
                val logits = rows[r]
                if (logits.isEmpty()) continue

                val ref = allRowRefs[r].ref
                val acc = sumByRef.getOrPut(ref) { FloatArray(logits.size) }
                for (j in logits.indices) acc[j] += logits[j]
                countByRef[ref] = (countByRef[ref] ?: 0) + 1
            }

            // 2) Décodage final par Ref, sur logits moyennés
            for ((ref, sum) in sumByRef) {
                val cnt = countByRef[ref] ?: 1
                for (j in sum.indices) sum[j] /= cnt.toFloat()

                val idx = sum.indices.maxBy { sum[it] }
                val raw = if (sum.size == labelNames.size + 1) {
                    if (idx == 0) "O" else labelNames[idx - 1]
                } else {
                    labelNames[idx]
                }

                if (raw == "O") continue

                val si = chunk[ref.chunkIdx]
                val cand = si.validCands[ref.candIdx]

                val txt = si.sentenceText.substring(
                    cand.span.start - si.sent.start,
                    cand.span.end - si.sent.start
                )

                val decoded = decodeLabel(raw)

                result.getOrPut(si.docIdx) { mutableListOf() } += SimpleEntityModel(
                    text = txt,
                    label = decoded,
                    start = cand.span.start,
                    end = cand.span.end,
                    tokens = cand.span.tokens,
                    isHint = decoded.isHint()
                )
            }

            inputIdsT.close()
            attT.close()
            ttiT?.close()
            coarseIdsT?.close()
            startT.close()
            endT.close()
            batchIdxT.close()
        }

        log.debug("[SpanNER-A] total  docs={}  ms={}", docs.size, ms(t0))
        return docs.indices.map { result[it]?.toList() ?: emptyList() }
    }

    // ------------------------------------------------------------
    // PIPELINE NER-LABEL + UD → SPAN CLASSIFIER
    // ------------------------------------------------------------

    /**
     * Classifie les candidats pré-calculés par le pipeline NER-label + UD,
     * plutôt que de générer les spans depuis tous les NOUN/PROPN.
     *
     * Pipeline attendu (par document) :
     *   OnnxBilouEntityExtractor.extractFromText(text)          -> List<Entity>  (PER/LOC/ORG…)
     *     -> mergeNerLabelWithUD(nerEntities, udDoc)            -> List<Entity>  (spans UD raffinés)
     *     -> extractFromCandidates(listOf(udDoc), listOf(above))-> List<SimpleEntityModel>
     *
     * Fallback : si le SpanClassifier retourne "O" pour un candidat,
     * on conserve l'entité en mappant le type NER coarse (entity.type)
     * pour ne pas perdre totalement le candidat.
     *
     * @param docs       documents UD (un par élément).
     * @param candidates candidats pré-calculés, un List<Entity> par document.
     */
    fun extractFromCandidates(
        docs: List<UDDocument>,
        candidates: List<List<Entity>>
    ): List<List<SimpleEntityModel>> {
        val t0 = System.nanoTime()
        fun ms(t: Long) = (System.nanoTime() - t) / 1_000_000L

        data class Cand(val docIdx: Int, val sent: UDSentence, val entity: Entity)

        val collected = mutableListOf<Cand>()

        docs.forEachIndexed { di, doc ->
            candidates.getOrElse(di) { emptyList() }.forEach { entity ->
                val eStart = entity.span?.start ?: return@forEach
                val eEnd = entity.span?.end ?: return@forEach
                if (eStart >= eEnd) return@forEach

                val sent = doc.sentences.firstOrNull { s ->
                    s.start <= eStart && s.end >= eEnd
                } ?: return@forEach

                collected += Cand(di, sent, entity)
            }
        }

        if (collected.isEmpty()) return docs.indices.map { emptyList() }

        val byDoc = collected.groupBy { it.docIdx }
        val result = mutableMapOf<Int, MutableList<SimpleEntityModel>>()

        data class SentInfo(
            val docIdx: Int,
            val doc: UDDocument,
            val sent: UDSentence,
            val enc: ai.djl.huggingface.tokenizers.Encoding,
            val starts: LongArray,
            val ends: LongArray,
            val validCands: List<Cand>,
            val coarseArr: LongArray,
            val seqLen: Int
        )

        // ── Phase 1 : tokeniser toutes les phrases de tous les docs ───────────
        val allSentInfos = mutableListOf<SentInfo>()
        for ((docIdx, candsOfDoc) in byDoc) {
            val doc = docs[docIdx]
            result[docIdx] = mutableListOf()

            for ((sent, candsInSent) in candsOfDoc.groupBy { it.sent }) {
                val enc = tokenizer.encode(doc.text.substring(sent.start, sent.end), true, false)
                val seqLen = min(enc.ids.size, maxSeqLen)
                val offsets: List<Pair<Int, Int>?> = enc.charTokenSpans
                    .take(seqLen)
                    .map { span -> span?.let { it.start to it.end } }

                val starts = mutableListOf<Long>()
                val ends = mutableListOf<Long>()
                val validCands = mutableListOf<Cand>()

                for (c in candsInSent) {
                    val relStart = (c.entity.span?.start ?: continue) - sent.start
                    val relEnd = (c.entity.span?.end ?: continue) - sent.start
                    val mapped = mapCharToTokenSpan(offsets, relStart, relEnd) ?: continue
                    if (mapped.second > seqLen) continue
                    starts += mapped.first.toLong()
                    ends += mapped.second.toLong()
                    validCands += c
                }
                if (starts.isEmpty()) continue

                val coarseArr = LongArray(validCands.size) { i ->
                    nerTypeToCoarseId(validCands[i].entity.type)
                }

                allSentInfos += SentInfo(
                    docIdx = docIdx,
                    doc = doc,
                    sent = sent,
                    enc = enc,
                    starts = starts.toLongArray(),
                    ends = ends.toLongArray(),
                    validCands = validCands,
                    coarseArr = coarseArr,
                    seqLen = seqLen
                )
            }
        }

        // ── Phase 2 : inférence par chunks globaux (sentBatchSize phrases) ────
        for (chunk in allSentInfos.chunked(sentBatchSize)) {
            val n = chunk.size
            val maxLen = chunk.maxOf { it.seqLen }

            val flatIds = LongArray(n * maxLen) { 0L }
            val flatAtt = LongArray(n * maxLen) { 0L }
            val flatTti = LongArray(n * maxLen) { 0L }

            chunk.forEachIndexed { b, si ->
                for (i in 0 until si.seqLen) {
                    flatIds[b * maxLen + i] = si.enc.ids[i]
                    flatAtt[b * maxLen + i] = si.enc.attentionMask[i].toLong()
                }
                si.enc.typeIds?.let { tti ->
                    for (i in 0 until si.seqLen) {
                        flatTti[b * maxLen + i] = tti[i].toLong()
                    }
                }
            }

            val allSpanStarts = mutableListOf<Long>()
            val allSpanEnds = mutableListOf<Long>()
            val allBatchIdx = mutableListOf<Long>()
            val allCoarseIds = mutableListOf<Long>()
            val allCandRefs = mutableListOf<Pair<Int, Int>>()

            chunk.forEachIndexed { b, si ->
                for (i in si.starts.indices) {
                    allSpanStarts += si.starts[i]
                    allSpanEnds += si.ends[i]
                    allBatchIdx += b.toLong()
                    allCoarseIds += si.coarseArr[i]
                    allCandRefs += b to i
                }
            }

            val inputIdsT = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(flatIds),
                longArrayOf(n.toLong(), maxLen.toLong())
            )
            val attT = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(flatAtt),
                longArrayOf(n.toLong(), maxLen.toLong())
            )
            val ttiT = if (hasTokenType) {
                OnnxTensor.createTensor(
                    env,
                    LongBuffer.wrap(flatTti),
                    longArrayOf(n.toLong(), maxLen.toLong())
                )
            } else null
            val startT = OnnxTensor.createTensor(env, allSpanStarts.toLongArray())
            val endT = OnnxTensor.createTensor(env, allSpanEnds.toLongArray())
            val batchIdxT = OnnxTensor.createTensor(env, allBatchIdx.toLongArray())

            val map = mutableMapOf<String, OnnxTensor>(
                "input_ids" to inputIdsT,
                "attention_mask" to attT,
                "span_starts" to startT,
                "span_ends" to endT,
                "span_batch_idx" to batchIdxT
            )
            if (hasTokenType) map["token_type_ids"] = ttiT!!

            val coarseIdsT = if (hasCoarseInput) {
                OnnxTensor.createTensor(env, allCoarseIds.toLongArray()).also { map["coarse_ids"] = it }
            } else null

            val tInfer = System.nanoTime()
            val rows = to2D(session.run(map).use { it[0].value })
            log.debug(
                "[SpanNER-B] inférence  N={}  maxLen={}  spans={}  ms={}",
                n, maxLen, allCandRefs.size, (System.nanoTime() - tInfer) / 1_000_000L
            )

            for (i in rows.indices) {
                val l = rows[i]
                if (l.isEmpty()) continue

                val idx = l.indices.maxBy { l[it] }
                val raw = if (l.size == labelNames.size + 1) {
                    if (idx == 0) "O" else labelNames[idx - 1]
                } else {
                    labelNames[idx]
                }

                val (chunkIdx, ciIdx) = allCandRefs[i]
                val si = chunk[chunkIdx]
                val cand = si.validCands[ciIdx]
                val eSpan = cand.entity.span ?: continue
                val txt = si.doc.text.substring(eSpan.start, eSpan.end)

                val decoded = if (raw == "O") {
                    EntityType.HINT_UNKNOWN
                } else {
                    decodeLabel(raw)
                }


                result.getOrPut(si.docIdx) { mutableListOf() } += SimpleEntityModel(
                    text = txt,
                    label = decoded,
                    start = eSpan.start,
                    end = eSpan.end,
                    tokens = eSpan.tokens,
                    isHint = decoded.isHint()
                )
            }

            inputIdsT.close()
            attT.close()
            ttiT?.close()
            coarseIdsT?.close()
            startT.close()
            endT.close()
            batchIdxT.close()
        }

        log.debug("[SpanNER-B] total  docs={}  ms={}", docs.size, ms(t0))
        return docs.indices.map { result[it]?.toList() ?: emptyList() }
    }


    /**
     * Coarse id depuis le type NER coarse (chaîne lowercase produite par le BIO extractor
     * ou le merge NER+UD).
     *
     * Aligné avec le training Python :
     * - person_name / person_role / norp / group_role -> PER (0)
     * - gpe / fac / loc / infra -> LOC (1)
     * - org -> ORG (2)
     * - time_* -> TIME (3)
     * - event_* -> EVENT (4)
     * - object_* / quantity -> OBJECT (5)
     */
    private fun nerTypeToCoarseId(nerType: String): Long = when (nerType.lowercase()) {
        "per", "person", "person_name", "person_role", "group_role", "norp" -> 0L
        "loc", "gpe", "fac", "loc_generic", "infra" -> 1L
        "org", "org_name" -> 2L
        "time", "date", "time_date", "time_clock", "time_duration" -> 3L
        "event", "event_nominal", "event_named" -> 4L
        "object", "object_generic", "object_name", "quantity", "weapon", "vehicle", "substance", "food", "tool" -> 5L
        else -> 5L
    }

    /**
     * Coarse id estimé depuis l’UD quand aucun NER coarse fiable n’est disponible.
     *
     * Version robuste :
     * - PROPN n'est PAS automatiquement PER
     * - certains collectifs humains fréquents passent en PER
     * - sinon fallback neutre OBJECT
     */
    private fun uposToCoarseId(tok: UDToken): Long {

        return when (tok.upos) {
            UPOS.PRON -> 0L
            UPOS.PROPN -> 5L
            else -> 5L
        }
    }

    // ------------------------------------------------------------
    // Mapping EXACT Python
    // ------------------------------------------------------------

    private fun mapCharToTokenSpan(
        offsets: List<Pair<Int, Int>?>,
        charStart: Int,
        charEnd: Int
    ): Pair<Int, Int>? {
        var ts: Int? = null
        var te: Int? = null

        for (i in offsets.indices) {
            val off = offsets[i] ?: continue
            val (s, e) = off
            if (e > charStart && s <= charStart) {
                ts = i
                break
            }
        }

        for (i in offsets.indices) {
            val off = offsets[i] ?: continue
            val (s, e) = off
            if (e >= charEnd && s < charEnd) {
                te = i + 1
                break
            }
        }

        if (ts == null || te == null) return null
        if (ts >= te) return null
        return ts to te
    }

    private fun to2D(v: Any?): Array<FloatArray> =
        when (v) {
            is Array<*> -> v.map {
                when (it) {
                    is FloatArray -> it
                    is DoubleArray -> it.map(Double::toFloat).toFloatArray()
                    else -> FloatArray(0)
                }
            }.toTypedArray()

            is FloatArray -> arrayOf(v)
            is DoubleArray -> arrayOf(v.map(Double::toFloat).toFloatArray())
            else -> emptyArray()
        }

    // ------------------------------------------------------------
    // LABEL DECODE
    // ------------------------------------------------------------

    /**
     * Décodage robuste :
     * - cas normal : labels fin-grained hint_*
     * - ancien modèle BILOU coarse : fallback conservateur
     */
    private fun decodeLabel(raw: String): EntityType {
        val lower = raw.lowercase()

        return when {
            lower.startsWith("hint_") -> {
                try {
                    EntityType.valueOf(raw.uppercase())
                } catch (_: Exception) {
                    log.warn("Unknown fine-grained label '{}', fallback OBJECT_GENERIC", raw)
                    EntityType.HINT_OBJECT_GENERIC
                }
            }

            raw.startsWith("B-") || raw.startsWith("I-") || raw.startsWith("L-") || raw.startsWith("U-") -> {
                val base = raw.substringAfter('-').uppercase()
                log.warn("Model returned legacy BILOU label '{}', fallback to generic family '{}'", raw, base)
                when (base) {
                    "PER" -> EntityType.HINT_PERSON_NAME
                    "LOC" -> EntityType.HINT_LOC_GENERIC
                    "ORG" -> EntityType.HINT_ORG_NAME
                    "TIME" -> EntityType.HINT_TIME_DATE
                    "EVENT" -> EntityType.HINT_EVENT_NOMINAL
                    "OBJECT" -> EntityType.HINT_OBJECT_GENERIC
                    else -> EntityType.HINT_OBJECT_GENERIC
                }
            }

            else -> {
                try {
                    EntityType.valueOf(raw.uppercase())
                } catch (_: Exception) {
                    log.warn("Unknown label '{}', fallback OBJECT_GENERIC", raw)
                    EntityType.HINT_OBJECT_GENERIC
                }
            }
        }
    }




    override fun close() {
        tokenizer.close()
        session.close()
    }

    private fun baseDeprel(tok: UDToken): String =
        tok.deprel.lowercase().substringBefore(":")

    private fun findHead(tokens: List<UDToken>): UDToken? {
        val ids = tokens.map { it.id }.toSet()
        return tokens.firstOrNull { t -> t.head == 0 || t.head !in ids }
            ?: tokens.firstOrNull()
    }

    private fun shouldClassifyHead(tok: UDToken, sentTokens: List<UDToken>): Boolean {
        if (tok.upos !in setOf(UPOS.NOUN, UPOS.PROPN)) return false

        val rel = baseDeprel(tok)
        val headTok = sentTokens.firstOrNull { it.id == tok.head }

        // Relations internes évidentes
        if (rel in setOf("flat", "name", "compound", "amod")) return false

        // nmod clairement interne à un autre syntagme nominal
        if (rel == "nmod" && headTok != null && headTok.upos in setOf(UPOS.NOUN, UPOS.PROPN, UPOS.NUM)) {
            return false
        }

        // apposition interne à un autre nominal
        if (rel == "appos" && headTok != null && headTok.upos in setOf(UPOS.NOUN, UPOS.PROPN)) {
            return false
        }

        return true
    }

    // ------------------------------------------------------------
    // DEBUG / PROBE DIRECT ONNX (sans UD, sans pipeline)
    // ------------------------------------------------------------

    fun debugPrintSessionInfo() {
        println("=".repeat(100))
        println("ONNX INPUTS")
        session.inputInfo.forEach { (name, info) ->
            println("  - name='${name}' type=${info.info.javaClass.simpleName} info=$info")
        }
        println("ONNX OUTPUTS")
        session.outputInfo.forEach { (name, info) ->
            println("  - name='${name}' type=${info.info.javaClass.simpleName} info=$info")
        }
        println("hasTokenType=$hasTokenType  hasCoarseInput=$hasCoarseInput")
        println("=".repeat(100))
    }

    fun debugProbeSingleSpan(
        text: String,
        charStart: Int,
        charEnd: Int,
        coarseId: Long,
        printTopK: Int = 5
    ) {
        require(charStart >= 0) { "charStart < 0" }
        require(charEnd > charStart) { "charEnd must be > charStart" }
        require(charEnd <= text.length) { "charEnd > text.length" }

        val enc = tokenizer.encode(text, true, false)
        val seqLen = min(enc.ids.size, maxSeqLen)
        val offsets: List<Pair<Int, Int>?> = enc.charTokenSpans
            .take(seqLen)
            .map { span -> span?.let { it.start to it.end } }

        val mapped = mapCharToTokenSpan(offsets, charStart, charEnd)
            ?: error("Impossible de mapper le span char [$charStart,$charEnd) sur les tokens")

        val tokenStart = mapped.first
        val tokenEnd = mapped.second

        val inputIds = LongArray(seqLen) { i -> enc.ids[i] }
        val attMask = LongArray(seqLen) { i -> enc.attentionMask[i].toLong() }
        val typeIds = LongArray(seqLen) { i -> enc.typeIds?.getOrNull(i)?.toLong() ?: 0L }

        val inputIdsT = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(inputIds),
            longArrayOf(1L, seqLen.toLong())
        )
        val attT = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(attMask),
            longArrayOf(1L, seqLen.toLong())
        )
        val ttiT = if (hasTokenType) {
            OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(typeIds),
                longArrayOf(1L, seqLen.toLong())
            )
        } else null

        val startT = OnnxTensor.createTensor(env, longArrayOf(tokenStart.toLong()))
        val endT = OnnxTensor.createTensor(env, longArrayOf(tokenEnd.toLong()))
        val batchIdxT = OnnxTensor.createTensor(env, longArrayOf(0L))
        val coarseIdsT = if (hasCoarseInput) {
            OnnxTensor.createTensor(env, longArrayOf(coarseId))
        } else null

        val inputMap = mutableMapOf<String, OnnxTensor>(
            "input_ids" to inputIdsT,
            "attention_mask" to attT,
            "span_starts" to startT,
            "span_ends" to endT,
            "span_batch_idx" to batchIdxT
        )
        if (hasTokenType) inputMap["token_type_ids"] = ttiT!!
        if (hasCoarseInput) inputMap["coarse_ids"] = coarseIdsT!!

        val rows = session.run(inputMap).use { result ->
            to2D(result[0].value)
        }

        inputIdsT.close()
        attT.close()
        ttiT?.close()
        startT.close()
        endT.close()
        batchIdxT.close()
        coarseIdsT?.close()

        require(rows.isNotEmpty()) { "Aucune ligne de logits retournée" }
        val logits = rows[0]
        require(logits.isNotEmpty()) { "Logits vides" }

        val idx = logits.indices.maxBy { logits[it] }
        val raw = if (logits.size == labelNames.size + 1) {
            if (idx == 0) "O" else labelNames[idx - 1]
        } else {
            labelNames[idx]
        }

        val decoded = if (raw == "O") null else decodeLabel(raw)

        println("=".repeat(100))
        println("TEXT        : '$text'")
        println("SPAN CHARS  : [$charStart, $charEnd) -> '${text.substring(charStart, charEnd)}'")
        println("COARSE ID   : $coarseId")
        println("TOK SPAN    : [$tokenStart, $tokenEnd)")
        println("PRED IDX    : $idx")
        println("RAW LABEL   : $raw")
        println("DECODED     : $decoded")
        println("TOP $printTopK       :")
        topKLabels(logits, printTopK).forEach { (i, label, score) ->
            println("  - idx=${"%2d".format(i)}  ${label.padEnd(22)}  ${"%+.6f".format(score)}")
        }

        if (coarseId == 0L && logits.size >= 4) {
            println(
                "PER LOGITS  : {\n" +
                        "  \"person_name\": ${logits[0]},\n" +
                        "  \"person_role\": ${logits[1]},\n" +
                        "  \"norp\": ${logits[2]},\n" +
                        "  \"group_role\": ${logits[3]}\n" +
                        "}"
            )
        }

        println("TOKENS/OFFSETS:")
        for (i in 0 until seqLen) {
            val tok = enc.tokens[i]
            val off = offsets[i]
            val offStr = if (off != null) "(${off.first}, ${off.second})" else "null"
            val marker = if (i in tokenStart until tokenEnd) " <SPAN>" else ""
            println("  [${"%02d".format(i)}] ${tok.padEnd(20)} offset=$offStr$marker")
        }
    }

    private fun topKLabels(logits: FloatArray, k: Int = 5): List<Triple<Int, String, Float>> {
        return logits.indices
            .map { i -> Triple(i, "", logits[i]) }
            .sortedByDescending { it.third }
            .take(k)
            .map { (i, _, v) ->
                val label = if (logits.size == labelNames.size + 1) {
                    if (i == 0) "O" else labelNames[i - 1]
                } else {
                    labelNames[i]
                }
                Triple(i, label, v)
            }
    }

}

fun main(args: Array<String>) {
    if (false) {
        println(
            """
            Usage:
              kotlin ...OnnxSpanNerExtractorKt <onnxPath> <tokenizerDir> [text] [charStart] [charEnd] [coarse]
            
            Examples:
              kotlin ...OnnxSpanNerExtractorKt best_model.onnx tokenizer_dir
              kotlin ...OnnxSpanNerExtractorKt best_model.onnx tokenizer_dir "Ahmed Benali" 0 12 PER
            
            If only onnxPath + tokenizerDir are provided, a default probe set is used:
              - Ahmed Benali  -> PER
              - juge          -> PER
              - policiers     -> PER
              - Kurdes        -> PER
            """.trimIndent()
        )
        return
    }
    //modelPath: "/Users/simon_longuet/IdeaProjects/pimpmyrag/training/training_package/training_output_deberta/best_model_v6.onnx"
    //tokenizerDir: "/Users/simon_longuet/IdeaProjects/pimpmyrag/debertav3-ner/tokenizer_from_hf"

    val onnxPath = "/Users/simon_longuet/IdeaProjects/pimpmyrag/training/training_package/training_output_deberta/best_model_v6.onnx"
    val tokenizerDir = "/Users/simon_longuet/IdeaProjects/pimpmyrag/deberta/tokenizer_export"

    OnnxSpanNerExtractor(
        modelPath = onnxPath,
        tokenizerDir = tokenizerDir,
        useCoreMl = false
    ).use { extractor ->

        extractor.debugPrintSessionInfo()

        if (args.size >= 6) {
            val text = args[2]
            val charStart = args[3].toInt()
            val charEnd = args[4].toInt()
            val coarse = args[5].uppercase()

            val coarseId = when (coarse) {
                "PER" -> 0L
                "LOC" -> 1L
                "ORG" -> 2L
                "TIME" -> 3L
                "EVENT" -> 4L
                "OBJECT" -> 5L
                else -> error("Unknown coarse '$coarse' (expected PER/LOC/ORG/TIME/EVENT/OBJECT)")
            }

            extractor.debugProbeSingleSpan(
                text = text,
                charStart = charStart,
                charEnd = charEnd,
                coarseId = coarseId
            )
        } else {
            val probes = listOf(
                Triple("Ahmed Benali", 0 to "Ahmed Benali".length, 0L),
                Triple("juge", 0 to "juge".length, 0L),
                Triple("policiers", 0 to "policiers".length, 0L),
                Triple("Kurdes", 0 to "Kurdes".length, 0L),
            )

            probes.forEach { (text, span, coarseId) ->
                extractor.debugProbeSingleSpan(
                    text = text,
                    charStart = span.first,
                    charEnd = span.second,
                    coarseId = coarseId
                )
            }
        }
    }
}
