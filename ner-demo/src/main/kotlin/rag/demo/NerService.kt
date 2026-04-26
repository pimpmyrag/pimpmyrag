package rag.demo

import org.slf4j.LoggerFactory
import org.springframework.beans.factory.DisposableBean
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Service
import rag.connectors.ner.onnx.ExtractionResult
import rag.connectors.ner.onnx.ExtractionThresholds
import rag.connectors.ner.onnx.OnnxMultiHeadEntityExtractor
import rag.connectors.ner.onnx.SvoSpan
import rag.model.Entity
import com.ibm.icu.text.BreakIterator
import java.nio.file.Files
import java.nio.file.Paths
import java.util.Locale
import java.util.concurrent.atomic.AtomicReference

// ── Spans SVO enrichis (réconciliation NER↔SVO) ───────────────────────────────
data class EnrichedSvoSpan(
    val base: SvoSpan,
    val nerOverride: String? = null,
    val nerOverrideScore: Float? = null,
    val fromNer: Boolean = false,
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
    @Value("\${ner.batch-size:8}")            batchSize: Int,
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
        tauSvoBoundary = tauSvo,
    ).also { log.info("✅ Modèle NER chargé depuis {}", modelPath) }

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

    // ── Analyse unique ────────────────────────────────────────────────────────

    fun analyse(text: String): ExtractionResult {
        val cfg = _config.get()
        val t0  = System.currentTimeMillis()
        return extractor.extractWithSvoFromTexts(listOf(text), cfg.toThresholds()).first().also {
            log.info("Inférence : {} entités, {} SVO — {}ms",
                it.entities.size, it.svoSpans.size, System.currentTimeMillis() - t0)
        }
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
    ): List<EnrichedSvoSpan> = svoSpans.map { svo ->
        if (svo.role !in subjectRoles && svo.role !in objectRoles) return@map EnrichedSvoSpan(svo)

        val allowed = if (svo.role in subjectRoles) subjCoarse else objCoarse
        val best = nerEntities
            .filter { e ->
                val coarse = e.metadata["coarse"] as? String ?: return@filter false
                (e.metadata["score"] as? Float ?: 0f) >= cfg.minNerScoreReconcile &&
                coarse in allowed &&
                minOf(svo.charEnd, e.span!!.end) - maxOf(svo.charStart, e.span.start) > 0 &&
                (e.span.end - e.span.start) >= (svo.charEnd - svo.charStart)
            }
            .maxByOrNull { e -> e.metadata["score"] as? Float ?: 0f }

        if (best != null && best.span!!.start <= svo.charStart && best.span.end >= svo.charEnd) {
            val coarse = best.metadata["coarse"] as? String ?: ""
            EnrichedSvoSpan(
                base = SvoSpan(
                    text = best.text, charStart = best.span.start, charEnd = best.span.end,
                    role = svo.role, roleProb = svo.roleProb, svoBoundaryProb = svo.svoBoundaryProb,
                    voice = svo.voice, voiceProb = svo.voiceProb, gender = svo.gender, number = svo.number,
                ),
                nerOverride = best.type.ifBlank { coarse },
                nerOverrideScore = best.metadata["score"] as? Float,
            )
        } else EnrichedSvoSpan(svo)
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
                    e.span!!.end <= v.charStart &&
                    v.charStart - e.span.end <= cfg.maxGapChars &&
                    (e.metadata["score"] as? Float ?: 0f) >= cfg.minNerScoreFill
                }
                .maxByOrNull { it.span!!.end }
            best?.let { e ->
                EnrichedSvoSpan(
                    base = SvoSpan(
                        text = e.text, charStart = e.span!!.start, charEnd = e.span.end,
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
        tauBoundary    = tauBoundary,
        tauNone        = tauNone,
        tauCoarse      = tauCoarse,
        tauSvoBoundary = tauSvoBoundary,
        scoreByCoarse  = scoreByCoarse,
    )

    override fun destroy() = extractor.close()
}
