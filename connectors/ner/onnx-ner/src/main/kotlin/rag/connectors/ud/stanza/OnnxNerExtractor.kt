package rag.connectors.ud.stanza

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.*
import rag.engine.NerExtractorFromUD
import rag.model.*
import java.nio.LongBuffer
import java.nio.file.Paths

// ------------------------------------------------------------
// ENUM (inchangé)
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
    HINT_QUANTITY;

    fun isName() = name.endsWith("_NAME")
    fun isHint() = !isName()
}

// ------------------------------------------------------------
// SPAN FILTER
// ------------------------------------------------------------

fun shouldClassify(tok: UDToken) =
    tok.upos == UPOS.NOUN || tok.upos == UPOS.PROPN
    // PRON exclu : les pronoms (Il, le, sa…) ne doivent pas être des entités NER standalone.

data class SimpleEntityModel(
    val text: String,
    val label: EntityType,
    val start: Int,
    val end: Int,
    val tokens: List<UDToken>,
    val isHint: Boolean
)

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
            "hint_person_name",   //  0
            "hint_person_role",   //  1
            "hint_norp",          //  2
            "hint_group_role",    //  3
            "hint_org_name",      //  4
            "hint_gpe",           //  5
            "hint_fac_name",      //  6
            "hint_loc_generic",   //  7
            "hint_weapon",        //  8
            "hint_vehicle",       //  9
            "hint_substance",     // 10
            "hint_food",          // 11
            "hint_infra",         // 12
            "hint_tool",          // 13
            "hint_object_generic",// 14
            "hint_object_name",   // 15
            "hint_event_nominal", // 16
            "hint_event_named",   // 17
            "hint_time_date",     // 18
            "hint_time_clock",    // 19
            "hint_time_duration", // 20
            "hint_quantity"       // 21
        )
) : AutoCloseable, NerExtractorFromUD {

    private val env = OrtEnvironment.getEnvironment()
    private val session = env.createSession(modelPath, OrtSession.SessionOptions())
    private val tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerDir))
    private val hasTokenType   = session.inputNames.contains("token_type_ids")
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
                    if (!shouldClassify(tok)) return@forEach
                    val baseDeprel = tok.deprel.lowercase().substringBefore(":")
                    if (baseDeprel in flatDeprels) return@forEach
                    val sp = reconstructSpan(sent.tokens, tok.id)
                    if (sp.end > sp.start)
                        collected += Cand(di, sent, sp)
                }
            }
        }

        if (collected.isEmpty())
            return docs.indices.map { emptyList<SimpleEntityModel>() }

        // Groupement par document
        val byDoc = collected.groupBy { it.docIdx }
        val result = mutableMapOf<Int, MutableList<SimpleEntityModel>>()

        for ((docIdx, candsOfDoc) in byDoc) {
            val doc = docs[docIdx]
            val outList = mutableListOf<SimpleEntityModel>()
            result[docIdx] = outList

            val bySent = candsOfDoc.groupBy { it.sent }

            // 2) Pour chaque phrase : tokenisation + batch de spans
            for ((sent, candsInSent) in bySent) {

                // ✅ texte de la phrase SEULEMENT
                val sentenceText = doc.text.substring(sent.start, sent.end)
                val enc = tokenizer.encode(sentenceText)
                val offsets = enc.charTokenSpans.map { it.start to it.end }

                // Spans ONNX
                val starts = mutableListOf<Long>()
                val ends = mutableListOf<Long>()
                val validCands = mutableListOf<Cand>()

                for (c in candsInSent) {
                    val relStart = c.span.start - sent.start
                    val relEnd   = c.span.end   - sent.start

                    val mapped = mapCharToTokenSpan(offsets, relStart, relEnd)
                    if (mapped != null) {
                        val (ts, te) = mapped
                        starts += ts.toLong()
                        ends += te.toLong()
                        validCands += c
                    }
                }

                if (starts.isEmpty()) continue

                // ✅ Tensors ONNX
                val ids = LongArray(enc.ids.size) { i -> enc.ids[i] }
                val att = LongArray(enc.attentionMask.size) { i -> enc.attentionMask[i].toLong() }
                val tti = if (enc.typeIds != null) LongArray(enc.typeIds!!.size) { i -> enc.typeIds!![i].toLong() } else LongArray(ids.size)

                val inputIdsT = tensor2d(ids)
                val attT = tensor2d(att)
                val ttiT = if (hasTokenType) tensor2d(tti) else null

                val startT = OnnxTensor.createTensor(env, starts.toLongArray())
                val endT   = OnnxTensor.createTensor(env, ends.toLongArray())
                val batchIdxT = OnnxTensor.createTensor(env, LongArray(starts.size) { 0L })

                val map = mutableMapOf<String, OnnxTensor>(
                    "input_ids" to inputIdsT,
                    "attention_mask" to attT,
                    "span_starts" to startT,
                    "span_ends" to endT,
                    "span_batch_idx" to batchIdxT
                )
                if (hasTokenType) map["token_type_ids"] = ttiT!!

                // --- COARSE IDS : 1D tensor (S,), un id par span ---
                val coarseIdsT: OnnxTensor? = if (hasCoarseInput) {
                    // Pour extractSimple on n'a pas de contexte NER coarse →
                    // on estime depuis le UPOS du token tête de chaque span.
                    val coarseArr = LongArray(validCands.size) { i ->
                        val headTok = validCands[i].span.tokens.firstOrNull()
                        if (headTok != null) uposToCoarseId(headTok) else 5L
                    }
                    OnnxTensor.createTensor(env, coarseArr)
                        .also { map["coarse_ids"] = it }
                } else null

                val logits = session.run(map).use { it[0].value }
                val rows = to2D(logits)

                // 3) Convert logits -> labels
                for (i in rows.indices) {
                    val l = rows[i]
                    if (l.isEmpty()) continue

                    val maxEntry = l.withIndex().maxBy { it.value }
                    val idx = maxEntry.index

                    val raw =
                        if (l.size == labelNames.size + 1) {
                            if (idx == 0) "O" else labelNames[idx - 1]
                        } else {
                            labelNames[idx]
                        }

                    if (raw == "O") continue

                    val cand = validCands[i]

                    // Map raw label → EntityType
                    val enum = decodeLabel(raw)

                    // ✅ Texte local à la phrase, pas au document
                    val rs = cand.span.start - sent.start
                    val re = cand.span.end   - sent.start
                    val txt = sentenceText.substring(rs, re)

                    outList += SimpleEntityModel(
                        text = txt,
                        label = enum,
                        start = cand.span.start,
                        end = cand.span.end,
                        tokens = cand.span.tokens,
                        isHint = enum.isHint()
                    )
                }

                inputIdsT.close(); attT.close()
                ttiT?.close(); coarseIdsT?.close()
                startT.close(); endT.close(); batchIdxT.close()
            }
        }

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
     *   OnnxBilouEntityExtractor.extractFromText(text)          → List<Entity>  (PER/LOC/ORG…)
     *     → mergeNerLabelWithUD(nerEntities, udDoc)             → List<Entity>  (spans UD raffinés)
     *     → extractFromCandidates(listOf(udDoc), listOf(above)) → List<SimpleEntityModel>
     *
     * Fallback : si le SpanClassifier retourne "O" pour un candidat,
     * on conserve l'entité en mappant le type NER coarse (entity.type)
     * pour ne pas perdre de candidat.
     *
     * @param docs       documents UD (un par élément).
     * @param candidates candidats pré-calculés, un List<Entity> par document.
     */
    fun extractFromCandidates(
        docs: List<UDDocument>,
        candidates: List<List<Entity>>
    ): List<List<SimpleEntityModel>> {

        data class Cand(val docIdx: Int, val sent: UDSentence, val entity: Entity)

        val collected = mutableListOf<Cand>()

        docs.forEachIndexed { di, doc ->
            candidates.getOrElse(di) { emptyList() }.forEach { entity ->
                val eStart = entity.span?.start ?: return@forEach
                val eEnd   = entity.span?.end   ?: return@forEach
                if (eStart >= eEnd) return@forEach
                val sent = doc.sentences.firstOrNull { s -> s.start <= eStart && s.end >= eEnd }
                    ?: return@forEach
                collected += Cand(di, sent, entity)
            }
        }

        if (collected.isEmpty()) return docs.indices.map { emptyList() }

        val byDoc  = collected.groupBy { it.docIdx }
        val result = mutableMapOf<Int, MutableList<SimpleEntityModel>>()

        for ((docIdx, candsOfDoc) in byDoc) {
            val doc     = docs[docIdx]
            val outList = mutableListOf<SimpleEntityModel>()
            result[docIdx] = outList

            for ((sent, candsInSent) in candsOfDoc.groupBy { it.sent }) {
                val sentenceText = doc.text.substring(sent.start, sent.end)
                val enc          = tokenizer.encode(sentenceText)
                val offsets      = enc.charTokenSpans.map { it.start to it.end }

                val starts     = mutableListOf<Long>()
                val ends       = mutableListOf<Long>()
                val validCands = mutableListOf<Cand>()

                for (c in candsInSent) {
                    val relStart = (c.entity.span?.start ?: continue) - sent.start
                    val relEnd   = (c.entity.span?.end   ?: continue) - sent.start
                    val mapped   = mapCharToTokenSpan(offsets, relStart, relEnd) ?: continue
                    starts     += mapped.first.toLong()
                    ends       += mapped.second.toLong()
                    validCands += c
                }

                if (starts.isEmpty()) continue

                val ids = LongArray(enc.ids.size) { enc.ids[it] }
                val att = LongArray(enc.attentionMask.size) { enc.attentionMask[it].toLong() }
                val tti = enc.typeIds?.let { t -> LongArray(t.size) { t[it].toLong() } }
                    ?: LongArray(ids.size)

                val inputIdsT = tensor2d(ids)
                val attT      = tensor2d(att)
                val ttiT      = if (hasTokenType) tensor2d(tti) else null
                val startT    = OnnxTensor.createTensor(env, starts.toLongArray())
                val endT      = OnnxTensor.createTensor(env, ends.toLongArray())
                val batchIdxT = OnnxTensor.createTensor(env, LongArray(starts.size) { 0L })

                val map = mutableMapOf<String, OnnxTensor>(
                    "input_ids"      to inputIdsT,
                    "attention_mask" to attT,
                    "span_starts"    to startT,
                    "span_ends"      to endT,
                    "span_batch_idx" to batchIdxT
                )
                if (hasTokenType) map["token_type_ids"] = ttiT!!

                // --- COARSE IDS : 1D tensor (S,), dérivé du type NER coarse ---
                val coarseIdsT: OnnxTensor? = if (hasCoarseInput) {
                    val coarseArr = LongArray(validCands.size) { i ->
                        nerTypeToCoarseId(validCands[i].entity.type)
                    }
                    OnnxTensor.createTensor(env, coarseArr)
                        .also { map["coarse_ids"] = it }
                } else null

                val rows = to2D(session.run(map).use { it[0].value })

                for (i in rows.indices) {
                    val l = rows[i]
                    if (l.isEmpty()) continue

                    val idx = l.indices.maxBy { l[it] }
                    val raw = if (l.size == labelNames.size + 1) {
                        if (idx == 0) "O" else labelNames[idx - 1]
                    } else labelNames[idx]

                    val cand  = validCands[i]
                    val eSpan = cand.entity.span ?: continue

                    // Fallback sur le type NER coarse si le SpanClassifier dit "O"
                    // → on ne perd aucun candidat déjà filtré par le NER label
                    val enum = if (raw == "O")
                        coarseNerTypeToEntityType(cand.entity.type)
                    else
                        decodeLabel(raw)

                    outList += SimpleEntityModel(
                        text   = doc.text.substring(eSpan.start, eSpan.end),
                        label  = enum,
                        start  = eSpan.start,
                        end    = eSpan.end,
                        tokens = eSpan.tokens,
                        isHint = enum.isHint()
                    )
                }

                inputIdsT.close(); attT.close(); ttiT?.close()
                coarseIdsT?.close()
                startT.close(); endT.close(); batchIdxT.close()
            }
        }

        return docs.indices.map { result[it]?.toList() ?: emptyList() }
    }

    // ------------------------------------------------------------
    // HELPERS — coarse_ids
    // Indices : 0=PER  1=LOC  2=ORG  3=TIME  4=EVENT  5=OBJECT
    // ------------------------------------------------------------

    /** Fallback : mappe le type NER coarse vers un EntityType générique quand le SpanClassifier dit "O". */
    private fun coarseNerTypeToEntityType(nerType: String): EntityType = when (nerType.lowercase()) {
        "per"    -> EntityType.HINT_PERSON_NAME
        "loc"    -> EntityType.HINT_LOC_GENERIC
        "org"    -> EntityType.HINT_ORG_NAME
        "time"   -> EntityType.HINT_TIME_DATE
        "event"  -> EntityType.HINT_EVENT_NOMINAL
        "object" -> EntityType.HINT_OBJECT_GENERIC
        else     -> EntityType.HINT_OBJECT_GENERIC
    }

    /** Coarse id depuis le type NER coarse (chaîne lowercase produite par le BIO extractor). */
    private fun nerTypeToCoarseId(nerType: String): Long = when (nerType.lowercase()) {
        "per"    -> 0L
        "loc"    -> 1L
        "org"    -> 2L
        "time"   -> 3L
        "event"  -> 4L
        "object" -> 5L
        else     -> 5L   // OBJECT par défaut
    }

    /** Coarse id estimé depuis le UPOS quand aucun contexte NER n'est disponible. */
    private fun uposToCoarseId(tok: UDToken): Long = when (tok.upos) {
        UPOS.PROPN -> 0L   // nom propre → PER (meilleure hypothèse)
        UPOS.PRON  -> 0L   // pronom     → PER
        else       -> 5L   // NOUN etc.  → OBJECT
    }

    // ------------------------------------------------------------
    // Mapping EXACT Python
    // ------------------------------------------------------------

    private fun mapCharToTokenSpan(
        offsets: List<Pair<Int,Int>>,
        charStart: Int,
        charEnd: Int
    ): Pair<Int,Int>? {

        var ts: Int? = null
        var te: Int? = null

        for (i in offsets.indices) {
            val (s, e) = offsets[i]
            if (e > charStart && s <= charStart) { ts = i; break }
        }

        for (i in offsets.indices) {
            val (s, e) = offsets[i]
            if (e >= charEnd && s < charEnd) { te = i + 1; break }
        }

        if (ts == null || te == null) return null
        if (ts >= te) return null
        return ts to te
    }

    private fun tensor2d(arr: LongArray): OnnxTensor {
        val buf = LongBuffer.allocate(arr.size)
        buf.put(arr)
        buf.flip()
        return OnnxTensor.createTensor(env, buf, longArrayOf(1, arr.size.toLong()))
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

    private fun decodeLabel(raw: String): EntityType {

        // Older models used hint_* names directly. Newer ones use BILOU.
        // We map both to our EntityType.

        return when {
            raw.startsWith("B-") || raw.startsWith("I-") || raw.startsWith("L-") || raw.startsWith("U-") -> {
                val base = raw.substringAfter('-').uppercase()
                when (base) {
                    "PER" -> EntityType.HINT_PERSON_NAME
                    "LOC" -> EntityType.HINT_LOC_GENERIC
                    "OBJECT" -> EntityType.HINT_OBJECT_GENERIC
                    "ORG" -> EntityType.HINT_ORG_NAME
                    "TIME" -> EntityType.HINT_TIME_DATE
                    "EVENT" -> EntityType.HINT_EVENT_NOMINAL
                    else -> EntityType.HINT_OBJECT_GENERIC
                }
            }
            raw.lowercase().startsWith("hint_") -> {
                try {
                    EntityType.valueOf(raw.uppercase())
                } catch (_: Exception) {
                    EntityType.HINT_OBJECT_GENERIC
                }
            }
            else -> {
                try {
                    EntityType.valueOf(raw.uppercase())
                } catch (_: Exception) {
                    EntityType.HINT_OBJECT_GENERIC
                }
            }
        }
    }

    override fun close() {
        tokenizer.close()
        session.close()
    }
}