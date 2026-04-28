package rag.demo

import org.slf4j.LoggerFactory
import org.springframework.beans.factory.DisposableBean
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Service
import rag.connectors.ner.onnx.ExtractionResult
import rag.connectors.ner.onnx.ExtractionThresholds
import rag.connectors.ner.onnx.OnnxMultiHeadEntityExtractor
import rag.connectors.ner.onnx.SvoSpan
import ai.onnxruntime.OrtSession
import rag.model.Entity
import com.ibm.icu.text.BreakIterator
import java.nio.file.Files
import java.nio.file.Paths
import java.util.Locale
import java.util.concurrent.atomic.AtomicReference

// ── Spans SVO enrichis (réconciliation NER↔SVO) ───────────────────────────────
/**
 * Span SVO enrichi portant optionnellement l'entité NER qui lui est associée.
 *
 * [entity] est renseigné par `reconcile()` dans deux cas :
 *   - Phase 1 (inline) : l'entité a été scorée directement sur la tête SVO pendant
 *     le forward pass NER → c'est la source la plus fiable.
 *   - Phase 2 (snap positionnel) : un span SVO brut a été "snappé" sur la meilleure
 *     entité NER voisine par recouvrement de 60%.
 * Lorsque [entity] est null, le span SVO n'a aucune entité NER associée (verbe,
 * pronom, ou argument sans entité détectée).
 */
data class EnrichedSvoSpan(
    val base: SvoSpan,
    val nerOverride: String? = null,
    val nerOverrideScore: Float? = null,
    val fromNer: Boolean = false,
    /** Entité NER fusionnée dans ce span SVO (null = aucune entité associée). */
    val entity: Entity? = null,
) {
    val text: String        get() = base.text
    val charStart: Int      get() = base.charStart
    val charEnd: Int        get() = base.charEnd
    val role: String        get() = base.role
    val voice: String       get() = base.voice
    val gender: String?     get() = base.gender
    val number: String?     get() = base.number
    val svoBoundaryProb: Float get() = base.svoBoundaryProb
    val roleProb: Float     get() = base.roleProb
    val voiceProb: Float    get() = base.voiceProb
}

data class AnnotatedSentence(
    val text: String,
    val entities: List<Entity>,
    val svoSpans: List<EnrichedSvoSpan>,
)

@Service
class NerService(
    @Value("\${ner.model-path}")              modelPath: String,
    @Value("\${ner.tokenizer-path}")          tokenizerPath: String,
    @Value("\${ner.max-seq-len:128}")         maxSeqLen: Int,
    @Value("\${ner.max-span-len:12}")         maxSpanLen: Int,
    @Value("\${ner.tau-boundary:0.70}")       tauBoundary: Float,
    @Value("\${ner.tau-none:0.99}")           tauNone: Float,
    @Value("\${ner.tau-coarse:0.45}")         tauCoarse: Float,
    @Value("\${ner.tau-svo-boundary:0.50}")   tauSvo: Float,
    @Value("\${ner.tau-svo-role-forced:0.0}") tauSvoRoleForced: Float,
    @Value("\${ner.batch-size:8}")            batchSize: Int,
    @Value("\${ner.intra-op-threads:-1}")     intraOpThreads: Int,
    @Value("\${ner.cpu-arena:true}")          cpuArena: Boolean,
    @Value("\${ner.opt-level:ALL_OPT}")       optLevelName: String,
) : DisposableBean {

    private val log = LoggerFactory.getLogger(NerService::class.java)

    init {
        val model = Paths.get(modelPath)
        val tokenizer = Paths.get(tokenizerPath)
        require(Files.isRegularFile(model)) {
            "ner.model-path invalide: '$modelPath'. Le fichier ONNX est introuvable (non versionne dans Git)."
        }
        require(Files.isDirectory(tokenizer)) {
            "ner.tokenizer-path invalide: '$tokenizerPath'. Le dossier tokenizer est introuvable (non versionne dans Git)."
        }
    }

    private val extractor = OnnxMultiHeadEntityExtractor(
        modelPath      = modelPath,
        tokenizerDir   = tokenizerPath,
        maxSeqLen      = maxSeqLen,
        maxSpanLen     = maxSpanLen,
        tauBoundary    = tauBoundary,
        tauNone        = tauNone,
        tauCoarse      = tauCoarse,
        tauSvoBoundary    = tauSvo,
        tauSvoRoleForced  = tauSvoRoleForced,
        intraOpThreads    = if (intraOpThreads < 1) Runtime.getRuntime().availableProcessors() else intraOpThreads,
        cpuArena       = cpuArena,
        optLevel       = runCatching { OrtSession.SessionOptions.OptLevel.valueOf(optLevelName) }
                             .getOrDefault(OrtSession.SessionOptions.OptLevel.ALL_OPT),
    ).also { log.info("✅ Modèle NER chargé depuis {} [opt={} arena={} intraThreads={}]",
        modelPath, optLevelName, cpuArena, intraOpThreads) }

    private val _config = AtomicReference(
        DemoConfig(
            tauBoundary    = tauBoundary,
            tauNone        = tauNone,
            tauCoarse      = tauCoarse,
            tauSvoBoundary = tauSvo,
            batchSize      = batchSize,
        )
    )

    val config: DemoConfig get() = _config.get()

    fun updateConfig(cfg: DemoConfig) {
        _config.set(cfg)
        log.info("Config mise à jour : {}", cfg)
    }

    fun runtimeInfo(): Map<String, Any> = extractor.runtimeInfo()

    // ── Sentence splitting (ICU4J) ────────────────────────────────────────────

    fun splitSentences(text: String): List<String> {
        val bi = BreakIterator.getSentenceInstance(Locale.FRENCH)
        bi.setText(text)
        val result = mutableListOf<String>()
        var start = bi.first(); var end = bi.next()
        while (end != BreakIterator.DONE) {
            val s = text.substring(start, end).trim()
            if (s.isNotBlank()) result += s
            start = end; end = bi.next()
        }
        return result.ifEmpty { listOf(text.trim()) }
    }

    // ── Analyse unique ────────────────────────────────────────────────────

    fun analyse(text: String): ExtractionResult {
        val cfg = _config.get()
        val t0  = System.currentTimeMillis()
        return extractor.extractWithSvoFromTexts(listOf(text), cfg.toThresholds()).first().also {
            log.info("Inférence : {} entités, {} SVO — {}ms",
                it.entities.size, it.svoSpans.size, System.currentTimeMillis() - t0)
        }
    }

    /**
     * Analyse un texte unique avec réconciliation NER↔SVO complète.
     * Contrairement à [analyse] qui renvoie un [ExtractionResult] brut, cette méthode
     * applique [reconcile] et [fillNullSubjects], produisant des [EnrichedSvoSpan] avec
     * les associations entité↔rôle syntaxique — utile pour les outils MCP et les previews SVO.
     */
    fun analyseSingle(text: String): AnnotatedSentence {
        var result: AnnotatedSentence? = null
        analyseStream(listOf(text)) { _, results -> result = results.firstOrNull() }
        return result ?: AnnotatedSentence(text, emptyList(), emptyList())
    }

    // ── Streaming batch avec post-traitements ─────────────────────────────────

    fun analyseStream(
        sentences: List<String>,
        onBatchReady: (startIdx: Int, results: List<AnnotatedSentence>) -> Unit,
    ) {
        val cfg = _config.get()
        var i = 0
        while (i < sentences.size) {
            val end   = minOf(i + cfg.batchSize, sentences.size)
            val batch = sentences.subList(i, end)
            val t0    = System.currentTimeMillis()
            val raw   = extractor.extractWithSvoFromTexts(batch, cfg.toThresholds())
            log.info("Batch [{}-{}] {}ms", i, end - 1, System.currentTimeMillis() - t0)

            val annotated = batch.zip(raw).map { (sent, res) ->
                val enrichedSvo = if (cfg.showSvo && cfg.doReconcile)
                    reconcile(res.svoSpans, res.entities, cfg) +
                    fillNullSubjects(res.svoSpans, res.entities, cfg)
                else res.svoSpans.map { EnrichedSvoSpan(it) }

                AnnotatedSentence(sent, res.entities, enrichedSvo)
            }
            onBatchReady(i, annotated)
            i += cfg.batchSize
        }
    }

    // ── Réconciliation NER ↔ SVO ──────────────────────────────────────────────

    private val subjectRoles = setOf("svo_subject", "pron_subj")
    private val objectRoles  = setOf("svo_object", "svo_iobj", "pron_obj")
    private val subjCoarse   = setOf("PER", "ORG", "EVENT", "ABSTRACT")
    private val objCoarse    = setOf("PER", "ORG", "LOC", "EVENT", "OBJECT", "ABSTRACT", "VALUE", "TIME")

    private fun reconcile(
        svoSpans: List<SvoSpan>,
        nerEntities: List<Entity>,
        cfg: DemoConfig,
    ): List<EnrichedSvoSpan> {

        // ── Phase 1 : entités scorées directement sur la tête SVO ─────────────────
        // Pendant le forward pass NER, chaque candidat span est aussi passé par la tête
        // SVO. Le résultat est stocké dans entity.metadata["svoRole"] / "syntacticRole".
        // C'est la vraie réponse à "la tête SVO pense quoi de cette entité ?"
        //
        // On filtre par pBoundary (lu dans les métadonnées de l'entité) en utilisant
        // le seuil COURANT du config (cfg.tauBoundary) plutôt qu'un seuil composeite
        // (minNerScoreReconcile) qui agissait comme un seuil boundary implicite ~0.84.
        // Pour les entités svoAnchored (pBoundary < tauBoundary), on tolère jusqu'à
        // tauSvoAnchoredBoundary — leur légitimité vient de la tête SVO, pas NER.
        val inlineFromEntities: List<EnrichedSvoSpan> = nerEntities.mapNotNull { e ->
            val svoRole = e.metadata["svoRole"] as? String ?: return@mapNotNull null
            if (svoRole !in subjectRoles && svoRole !in objectRoles) return@mapNotNull null
            val coarse  = e.metadata["coarse"] as? String ?: return@mapNotNull null
            val allowed = if (svoRole in subjectRoles) subjCoarse else objCoarse
            if (coarse !in allowed) return@mapNotNull null
            val pBoundary  = e.metadata["pBoundary"] as? Float ?: 0f
            val isAnchored = e.metadata["svoAnchored"] as? Boolean ?: false
            // Seuil dynamique : tauBoundary courant pour les entités normales,
            // tauSvoAnchoredBoundary pour les entités promues par SVO.
            val minBoundary = if (isAnchored) cfg.tauSvoAnchoredBoundary else cfg.tauBoundary
            if (pBoundary < minBoundary) return@mapNotNull null
            EnrichedSvoSpan(
                base = SvoSpan(
                    text            = e.text,
                    charStart       = e.span.start,
                    charEnd         = e.span.end,
                    role            = svoRole,
                    roleProb        = e.metadata["svoRoleProb"]     as? Float ?: 0f,
                    svoBoundaryProb = e.metadata["svoBoundaryScore"] as? Float ?: 0f,
                    voice = "ACTIVE", voiceProb = 0f,
                    gender = e.metadata["gender"] as? String,
                    number = e.metadata["number"] as? String,
                ),
                nerOverride      = e.type,
                nerOverrideScore = e.metadata["score"] as? Float,
                entity           = e,   // ← entité NER fusionnée (scoring inline, même forward pass)
            )
        }

        // ── Phase 2 : spans SVO bruts du modèle non couverts par le scoring inline ─
        // Cas où la tête SVO a détecté un argument mais pas la tête NER (boundary NER
        // sous le seuil) → on tente l'alignement positionnel sur les entités NER.
        val rawFallback: List<EnrichedSvoSpan> = svoSpans.mapNotNull { svo ->
            // Verbes et spans sans rôle argumental : wrappés tels quels
            if (svo.role !in subjectRoles && svo.role !in objectRoles)
                return@mapNotNull EnrichedSvoSpan(svo)

            // Déjà couvert → on évite le doublon.
            // Deux cas :
            //  a) Une entité a été incluse dans inlineFromEntities (Phase 1) et chevauche ce span SVO.
            //  b) Une entité a un svoRole inline mais n'a pas passé le filtre Phase 1
            //     (ex. svoAnchored avec pBoundary juste sous le seuil SVO) → on la vérifie aussi,
            //     car son svoRole badge NER couvre déjà le même span (nsubj, obj, iobj).
            val covered = inlineFromEntities.any { inline ->
                minOf(svo.charEnd, inline.charEnd) - maxOf(svo.charStart, inline.charStart) > 0
            } || nerEntities.any { e ->
                e.metadata["svoRole"] != null &&
                minOf(svo.charEnd, e.span.end) - maxOf(svo.charStart, e.span.start) > 0
            }
            if (covered) return@mapNotNull null

            // Réconciliation positionnelle (snap aux bornes de l'entité NER la plus proche)
            val allowed = if (svo.role in subjectRoles) subjCoarse else objCoarse
            val svoLen  = (svo.charEnd - svo.charStart).coerceAtLeast(1)
            val best    = nerEntities.filter { e ->
                val coarse  = e.metadata["coarse"] as? String ?: return@filter false
                val overlap = minOf(svo.charEnd, e.span.end) - maxOf(svo.charStart, e.span.start)
                val minLen  = minOf(svoLen, (e.span.end - e.span.start).coerceAtLeast(1))
                (e.metadata["score"] as? Float ?: 0f) >= cfg.minNerScoreReconcile &&
                coarse in allowed &&
                overlap.toFloat() / minLen >= 0.60f
            }.maxByOrNull { e -> e.metadata["score"] as? Float ?: 0f }

            if (best != null) {
                val coarse = best.metadata["coarse"] as? String ?: ""
                EnrichedSvoSpan(
                    base = SvoSpan(
                        text = best.text, charStart = best.span.start, charEnd = best.span.end,
                        role = svo.role, roleProb = svo.roleProb, svoBoundaryProb = svo.svoBoundaryProb,
                        voice = svo.voice, voiceProb = svo.voiceProb, gender = svo.gender, number = svo.number,
                    ),
                    nerOverride      = best.type.ifBlank { coarse },
                    nerOverrideScore = best.metadata["score"] as? Float,
                    entity           = best,  // ← entité NER fusionnée (snap positionnel)
                )
            } else EnrichedSvoSpan(svo)
        }

        return inlineFromEntities + rawFallback
    }

    private fun fillNullSubjects(
        svoSpans: List<SvoSpan>,
        nerEntities: List<Entity>,
        cfg: DemoConfig,
    ): List<EnrichedSvoSpan> {
        val verbs    = svoSpans.filter { it.role == "svo_verb" }
        val subjects = svoSpans.filter { it.role in subjectRoles }
        return verbs.mapNotNull { v ->
            if (subjects.any { it.charEnd <= v.charStart }) return@mapNotNull null
            val best = nerEntities
                .filter { e ->
                    val coarse = e.metadata["coarse"] as? String ?: return@filter false
                    coarse in subjCoarse &&
                    e.span.end <= v.charStart &&
                    v.charStart - e.span.end <= cfg.maxGapChars &&
                    (e.metadata["score"] as? Float ?: 0f) >= cfg.minNerScoreFill
                }
                .maxByOrNull { it.span.end }
            best?.let { e ->
                EnrichedSvoSpan(
                    base = SvoSpan(
                        text = e.text, charStart = e.span.start, charEnd = e.span.end,
                        role = "svo_subject", roleProb = 0f, svoBoundaryProb = 0f,
                        voice = v.voice, voiceProb = 0f, gender = null, number = null,
                    ),
                    nerOverride = e.type.ifBlank { e.metadata["coarse"] as? String },
                    nerOverrideScore = e.metadata["score"] as? Float,
                    fromNer = true,
                )
            }
        }
    }

    // ── Utils ─────────────────────────────────────────────────────────────────

    private fun DemoConfig.toThresholds() = ExtractionThresholds(
        tauBoundary             = tauBoundary,
        tauNone                 = tauNone,
        tauCoarse               = tauCoarse,
        tauSvoBoundary          = tauSvoBoundary,
        tauSvoAnchoredBoundary  = tauSvoAnchoredBoundary,
        scoreByCoarse           = scoreByCoarse,
    )

    override fun destroy() = extractor.close()
}
