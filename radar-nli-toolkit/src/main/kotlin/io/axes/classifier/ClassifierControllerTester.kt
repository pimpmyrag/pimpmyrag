package io.axes.classifier

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import org.slf4j.LoggerFactory
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import rag.connectors.ud.stanza.EntityCandidate
import rag.connectors.ud.stanza.NerCoarseType
import rag.connectors.ud.stanza.OnnxSpanNerExtractor
import rag.connectors.ud.stanza.SimpleEntityModel
import rag.connectors.ud.stanza.buildEntityCandidates
import rag.connectors.ud.stanza.mergeNerLabelWithUD
import rag.connectors.ud.stanza.mergeNerLabelWithUDV2
import rag.engine.Embedder
import rag.engine.NerExtractor
import rag.engine.NerExtractorFromUD
import rag.engine.UDParser
import rag.model.Entity
import rag.model.Span
import rag.model.UDDocument
import rag.model.UPOS

@RestController
@RequestMapping("/api/classify")
class ClassifierController(
    private val embedder: Embedder,
    private val nerExtractor: NerExtractorFromUD,
    private val nerLabelExtractor: NerExtractor,
    private val udParser: UDParser,
    private val multiAxisTrainingService: MultiAxisTrainingService
) {
    private val log = LoggerFactory.getLogger(ClassifierController::class.java)
    private var classifier: MultiClassEventClassifier? = null
    private var multiClassClassifier: MultiClassEventClassifier? = null
    private var multiAxisClassifier: MultiAxisTextClassifier? = null
    private var langAxisClassifier: MultiClassEventClassifier? = null

    init {
        // Tentative de chargement du modèle existant
        try {
            classifier = MultiClassEventClassifier.load("data/model-results_content-type.json", embedder)
            multiClassClassifier = MultiClassEventClassifier.load("data/model-results_label_type.json", embedder)
            langAxisClassifier = MultiClassEventClassifier.load("data/model-results_langage-type.json", embedder)
            println("✅ Modèle chargé depuis data/model-results.json")
        } catch (e: Exception) {
            println("⚠️  Aucun modèle trouvé. Utilisez /api/classify/train pour entraîner.")
        }
    }

    // ========== Training Endpoints ==========


    @PostMapping("/train/axis")
    fun trainMultiAxis(@RequestParam("basePath", defaultValue = "data") basePath: String, @RequestParam("axis", defaultValue = "") axis: String): MultiAxisTrainResponse {
        val classifiers = multiAxisTrainingService.trainAllAxes(
            basePath =basePath,
            axisName = axis
        )

        // Construction du MultiAxisTextClassifier
         MultiAxisTextClassifier(
            embedder = embedder,
            classifiers = classifiers,
            l2NormalizeEmbeddings = true
        )

        return MultiAxisTrainResponse(
            message = "Entraînement multi-axes terminé",
            axesTrained = classifiers.keys.toList()
        )
    }

    // ========== Classification Endpoints ==========

    @PostMapping("/single")
    fun classifySingle(@RequestBody request: ClassifyRequest): ClassifyResponse {
        requireNotNull(classifier) { "Aucun modèle chargé. Entraînez d'abord avec /train/event-detection" }

        val scores = classifier!!.classify(request.text)
        val topCategories = scores.toList()
            .sortedByDescending { it.second }
            .take(5)
            .map { CategoryScore(it.first, it.second) }

        return ClassifyResponse(topCategories)
    }

    @PostMapping("/extract")
    fun extractSingle(@RequestBody request: ExtractRequest): ExtractResponse {
        return ExtractResponse(
            udParser.parse(listOf(request.text.toRagDocuments())).let { nerExtractor.extractNerFromUD(it) }.first()
        ).also {
            val toto = nerLabelExtractor.extractNer(listOf(request.text.toRagDocuments())).first()
            println(toto.joinToString())
        }
    }

    /**
     * Endpoint complet : classification content type + multi-class event + extraction NER
     * - Content type : top 2 résultats avec scores
     * - Multi-class event : top 3 résultats avec scores
     * - NER extraction : toutes les entités trouvées
     */
    @PostMapping("/full-analysis")
    fun fullAnalysis(@RequestBody request: ExtractRequest): FullAnalysisResponse {


        // 1. Classification Content Type (top 2)
        val scores = classifier!!.classify(request.text)
        val topCategories = scores.toList()
            .sortedByDescending { it.second }
            .take(5)
            .map { CategoryScore(it.first, it.second) }

        // 2. Multi-class Event Classification (top 3)
        val scoresEvents = multiClassClassifier!!.classify(request.text)
        val topEventCategories = scoresEvents.toList()
            .sortedByDescending { it.second }
            .filter { it.second > 0.2 }
            .take(5)
            .map { CategoryScore(it.first, it.second) }

        val scoresLang = langAxisClassifier!!.classify(request.text)
        val topLangtCategories = scoresLang.toList()
            .sortedByDescending { it.second }
            .filter { it.second > 0.2 }
            .take(2)
            .map { CategoryScore(it.first, it.second) }

        // 3. NER Extraction
        val entities = try {
            udParser.parse(listOf(request.text.toRagDocuments())).let { nerExtractor.extractNerFromUD(it) }.first()
        } catch (e: Exception) {
            println("⚠️  NER extraction failed: ${e.message}")
            emptyList()
        }

        return FullAnalysisResponse(
            text = request.text,
            contentType = topCategories,
            events = topEventCategories,
            lang = topLangtCategories,
            entities = entities
        )
    }

    // ========== NER candidates pipeline ==========

    /**
     * Pipeline hybride NER label + UD → EntityCandidate.
     *
     *  A. NER label (XLM-RoBERTa) → mergeNerLabelWithUD → SpanClassifier
     *  B. UD NOUN/PROPN (extractSimple) → SpanClassifier          ← fallback
     *
     * Les candidats B comblent ce que le modèle BILOU rate
     * (ex : "la police", "Jacques Martin" en sujet passif).
     * La déduplication se fait par chevauchement de span.
     */
    @PostMapping("/extract/candidates")
    fun extractCandidates(@RequestBody request: ExtractRequest): EntityCandidatesResponse {
        return try {
            extractCandidatesInternal(request)
        } catch (e: Exception) {
            println("❌ /extract/candidates ERROR [${e.javaClass.simpleName}]: ${e.message}")
            e.printStackTrace()
            EntityCandidatesResponse(text = request.text, count = 0, candidates = emptyList())
        }
    }

    /**
     * Version batch du pipeline hybride NER label + UD → EntityCandidate.
     *
     * Chaque niveau (UD, NER-label ONNX, SpanNER ONNX) reçoit **l'ensemble du batch**
     * en un seul appel, ce qui maximise l'utilisation du modèle et réduit les allers-retours :
     *
     *  1. udParser.parse(allRagDocs)                              → 1 appel HTTP UD
     *  2. nerLabelExtractor.extractNer(allRagDocs)               → 1 inférence ONNX BILOU
     *  3. mergeNerLabelWithUD (par doc, O(1))
     *  4. spanNer.extractFromCandidates(allUdDocs, allEnriched)  → 1 inférence ONNX span (Pipeline A)
     *  5. spanNer.extractSimple(allUdDocs)                       → 1 inférence ONNX span (Pipeline B)
     *  6. Zip par index + déduplication + buildCandidateFromSimple
     */
    @PostMapping("/extract/candidates/batch")
    fun extractCandidatesBatch(@RequestBody request: BatchExtractRequest): BatchEntityCandidatesResponse {
        if (request.texts.isEmpty()) return BatchEntityCandidatesResponse(emptyList())
        return try {
            BatchEntityCandidatesResponse(extractCandidatesBatchInternal(request.texts))
        } catch (e: Exception) {
            println("❌ /extract/candidates/batch ERROR [${e.javaClass.simpleName}]: ${e.message}")
            e.printStackTrace()
            BatchEntityCandidatesResponse(
                request.texts.map { EntityCandidatesResponse(text = it, count = 0, candidates = emptyList()) }
            )
        }
    }

    private fun extractCandidatesBatchInternal(texts: List<String>): List<EntityCandidatesResponse> {
        val spanNer = nerExtractor as? OnnxSpanNerExtractor
        val t0 = System.nanoTime()
        fun ms(t: Long) = (System.nanoTime() - t) / 1_000_000L

        val ragDocs = texts.map { it.toRagDocuments() }

        // ── Étapes 1+2 en parallèle : UD et BILOU sont indépendants ──────────
        val tPar1 = System.nanoTime()
        val (udDocs, nerByDoc) = runBlocking(Dispatchers.IO) {
            val deferredUd    = async { udParser.parse(ragDocs) }
            val deferredBilou = async { nerLabelExtractor.extractNer(ragDocs) }
            deferredUd.await() to deferredBilou.await()
        }
        log.debug("[pipeline] UD+BILOU //      ms={}", ms(tPar1))

        // ── Étape 3 : merge (pur calcul) ─────────────────────────────────────
        val tMerge = System.nanoTime()
        val enrichedByDoc = nerByDoc.zip(udDocs).map { (entities, udDoc) ->
            mergeNerLabelWithUDV2(entities, udDoc)
        }
        log.debug("[pipeline] merge NER×UD     ms={}", ms(tMerge))

        // ── Étapes 4+5 en parallèle : SpanNER A et B sont indépendants ───────
        val tPar2 = System.nanoTime()
        val (fromNerByDoc, fromUdByDoc) = runBlocking(Dispatchers.IO) {
            val fromNerByDoc = spanNer?.extractFromCandidates(udDocs, enrichedByDoc)
                ?: udDocs.map { emptyList() }

            val fromUdByDoc = spanNer?.extractSimple(udDocs)
                ?: udDocs.map { emptyList() }
            fromNerByDoc to fromUdByDoc
        }
        log.debug("[pipeline] SpanNER A+B //   ms={}", ms(tPar2))

        // ── Étape 6 : assemblage ──────────────────────────────────────────────
        val tAssemble = System.nanoTime()
        val result = texts.indices.map { i ->
            val text     = texts[i]
            val udDoc    = udDocs[i]
            val enriched = enrichedByDoc[i]
            val fromNer  = fromNerByDoc[i]
            val fromUd   = fromUdByDoc[i]
            try {
                val candidatesA   = buildEntityCandidates(enriched, fromNer, udDoc)
                val coveredRanges = candidatesA.map { it.span.start until it.span.end }
                val udOnly        = fromUd.filter { simple ->
                    coveredRanges.none { r -> simple.start < r.last + 1 && simple.end > r.first }
                }
                val candidatesB = udOnly.mapNotNull { buildCandidateFromSimple(it, udDoc) }
                // Inclure les candidats fallback issus du pipeline UD (candidatesB)
                val candidates  = (candidatesA + candidatesB).sortedBy { it.span.start }
                EntityCandidatesResponse(text = text, count = candidates.size,
                    candidates = candidates.map { it.toDto() })
            } catch (e: Exception) {
                println("❌ batch[doc=$i] ERROR [${e.javaClass.simpleName}]: ${e.message}")
                EntityCandidatesResponse(text = text, count = 0, candidates = emptyList())
            }
        }
        log.debug("[pipeline] assemblage       ms={}", ms(tAssemble))
        log.debug("[pipeline] TOTAL            batchSize={}  ms={}", texts.size, ms(t0))
        return result
    }


    private fun extractCandidatesInternal(request: ExtractRequest): EntityCandidatesResponse {

        val ragDoc  = request.text.toRagDocuments()
        val udDocs  = udParser.parse(listOf(ragDoc))
        val udDoc   = udDocs.first()
        val spanNer = nerExtractor as? OnnxSpanNerExtractor

        // ── Pipeline A : NER label ────────────────────────────────────────────
        val nerEntities = nerLabelExtractor.extractNer(listOf(ragDoc)).first()
        val enriched    = mergeNerLabelWithUD(nerEntities, udDoc)
        val fromNer     = spanNer
            ?.extractFromCandidates(udDocs, listOf(enriched))
            ?.first() ?: emptyList()
        val candidatesA = buildEntityCandidates(enriched, fromNer, udDoc)

        // ── Pipeline B : UD NOUN/PROPN (fallback) ────────────────────────────
        val fromUd  = spanNer?.extractSimple(udDocs)?.first() ?: emptyList()

        // Garder uniquement les spans UD qui ne chevauchent aucun candidat A
        val coveredRanges = candidatesA.map { it.span.start until it.span.end }
        val udOnly = fromUd.filter { simple ->
            coveredRanges.none { r -> simple.start < r.last + 1 && simple.end > r.first }
        }
        val candidatesB = udOnly.mapNotNull { buildCandidateFromSimple(it, udDoc) }

        val candidates = (candidatesA + candidatesB).sortedBy { it.span.start }

        return EntityCandidatesResponse(
            text       = request.text,
            count      = candidates.size,
            candidates = candidates.map { it.toDto() }
        )
    }

    /** Construit un [EntityCandidate] depuis un [SimpleEntityModel] issu du pipeline UD pur. */
    private fun buildCandidateFromSimple(simple: SimpleEntityModel, udDoc: UDDocument): EntityCandidate? {
        val head = simple.tokens.firstOrNull { it.upos == UPOS.PROPN }
            ?: simple.tokens.firstOrNull { it.upos == UPOS.NOUN }
            ?: simple.tokens.firstOrNull()
            ?: return null

        val sentence = udDoc.sentences.firstOrNull { s -> s.start <= simple.start && s.end >= simple.end }

        val nerCoarse = when {
            simple.label.name.contains("PERSON") -> NerCoarseType.PER
            simple.label.name.contains("GROUP")  -> NerCoarseType.PER
            simple.label.name.contains("NORP")   -> NerCoarseType.PER

            simple.label.name.contains("GPE")    -> NerCoarseType.LOC
            simple.label.name.contains("LOC")    -> NerCoarseType.LOC
            simple.label.name.contains("FAC")    -> NerCoarseType.LOC
            simple.label.name.contains("INFRA")  -> NerCoarseType.LOC

            simple.label.name.contains("ORG")    -> NerCoarseType.ORG

            simple.label.name.contains("TIME")   -> NerCoarseType.TIME
            simple.label.name.contains("EVENT")  -> NerCoarseType.EVENT

            else -> NerCoarseType.OBJECT
        }


        return EntityCandidate(
            text         = simple.text,
            lemma        = head.lemma ?: simple.text.lowercase(),
            span         = Span(simple.start, simple.end, simple.tokens),
            nerType      = nerCoarse,
            nerHint      = simple.label,
            isName       = simple.label.isName(),
            head         = head,
            headUpos     = head.upos,
            headDeprel   = head.deprel,
            isPropn      = head.upos == UPOS.PROPN,
            isPron       = head.upos == UPOS.PRON,
            gender       = head.feats?.gender,
            number       = head.feats?.number,
            feats        = head.feats,
            sentenceSpan = sentence?.let { it.start until it.end } ?: (simple.start until simple.end),
            confidence   =  0.5f
        )
    }

//    @PostMapping("/multi-axis")
//    fun classifyMultiAxis(@RequestBody request: ClassifyRequest): MultiAxisPrediction {
//        requireNotNull(multiAxisClassifier) { "Aucun classifieur multi-axes. Entraînez avec /train/multi-axis" }
//
//        return multiAxisClassifier!!.classify(request.text)
//    }

//    @PostMapping("/batch")
//    fun classifyBatch(@RequestBody request: BatchClassifyRequest): BatchClassifyResponse {
//        requireNotNull(classifier) { "Aucun modèle chargé" }
//
//        val results = classifier!!.classifyBatch(request.texts).map { scores ->
//            scores.toList()
//                .sortedByDescending { it.second }
//                .take(5)
//                .map { CategoryScore(it.first, it.second) }
//        }
//
//        return BatchClassifyResponse(results)
//    }
}

// ========== DTOs ==========

data class TrainRequest(
    val categories: List<String>? = null,
    val countPerCategory: Int? = null
)

data class TrainResponse(
    val message: String,
    val totalExamples: Int
)

data class MultiAxisTrainResponse(
    val message: String,
    val axesTrained: List<String>
)

data class ExtractRequest(val text: String)
data class BatchExtractRequest(val texts: List<String>)
data class ExtractResponse(val categories: List<Entity>)

data class FullAnalysisResponse(
    val text: String,
    val contentType: List<CategoryScore>,
    val events: List<CategoryScore>,
    val entities: List<Entity>,
    val lang: List<CategoryScore>
)

data class ClassifyRequest(val text: String)
data class ClassifyResponse(val categories: List<CategoryScore>)
data class CategoryScore(val category: String, val score: Double)
data class BatchClassifyRequest(val texts: List<String>)
data class BatchClassifyResponse(val results: List<List<CategoryScore>>)
data class EntityCandidatesResponse(
    val text:       String,
    val count:      Int,
    val candidates: List<EntityCandidateDto>
)

data class BatchEntityCandidatesResponse(
    val results: List<EntityCandidatesResponse>
)

/** DTO sérialisable (pas de UDToken / Enum non-standard en racine) pour l'endpoint /extract/candidates. */
data class EntityCandidateDto(
    // Surface
    val text:          String,
    val lemma:         String,
    val start:         Int,
    val end:           Int,
    // NER
    val nerType:       String,   // PER / LOC / ORG / TIME / EVENT / OBJECT
    val nerHint:       String,   // HINT_PERSON_NAME, HINT_LOC_GENERIC, …
    val isName:        Boolean,  // nom propre direct vs rôle/générique
    // UD syntaxique
    val headText:      String?,
    val headLemma:     String?,
    val headUpos:      String?,
    val headDeprel:    String?,
    val isPropn:       Boolean,
    val isPron:        Boolean,
    // Morphologie
    val gender:        String?,  // MASC / FEM / NEUT
    val number:        String?,  // SG / PL
    // Contexte
    val sentenceStart: Int,
    val sentenceEnd:   Int,
    // Tokens UD du span (forme + POS pour debug/affichage)
    val spanTokens:    List<SpanTokenDto>
)

data class SpanTokenDto(val text: String, val lemma: String?, val upos: String?, val deprel: String)

// ── Mapper EntityCandidate → DTO sérialisable ─────────────────────────────────

fun EntityCandidate.toDto() = EntityCandidateDto(
    text          = text,
    lemma         = lemma,
    start         = span.start,
    end           = span.end,
    nerType       = nerType.name,
    nerHint       = nerHint.name,
    isName        = isName,
    headText      = head?.text,
    headLemma     = head?.lemma,
    headUpos      = headUpos?.name,
    headDeprel    = headDeprel,
    isPropn       = isPropn,
    isPron        = isPron,
    gender        = gender?.name,
    number        = number?.name,
    sentenceStart = sentenceSpan.first,
    sentenceEnd   = sentenceSpan.last,
    spanTokens    = span.tokens.map { t ->
        SpanTokenDto(
            text   = t.text,
            lemma  = t.lemma,
            upos   = t.upos?.name,
            deprel = t.deprel
        )
    }
)


