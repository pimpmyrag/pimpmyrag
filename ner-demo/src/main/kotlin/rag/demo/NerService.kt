package rag.demo

import org.slf4j.LoggerFactory
import org.springframework.beans.factory.DisposableBean
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Service
import rag.connectors.ner.onnx.ExtractionResult
import rag.connectors.ner.onnx.ExtractionThresholds
import rag.connectors.ner.onnx.OnnxMultiHeadEntityExtractor
import rag.connectors.ner.onnx.SvoSpan
import rag.connectors.ner.onnx.Eventlet
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
    /** Label syntaxique v4 : verb_trigger | pron_subj | pron_obj */
    val synLabel: String    get() = base.synLabel
    /** Rôle argumental v4 : SUBJECT | OBJECT | OBLIQUE | OBLIQUE_AGENT | OBLIQUE_CAUSE | APPOS | NONE */
    val role: String        get() = base.role
    val voice: String       get() = base.voice
    /** Modalité v4 : certain | modal | denied */
    val certainty: String   get() = base.certainty
    val gender: String?     get() = base.gender
    val number: String?     get() = base.number
    val person: String?     get() = base.person
    val govVerbCharStart: Int?  get() = base.govVerbCharStart
    val govVerbText: String?    get() = base.govVerbText
    val svoBoundaryProb: Float get() = base.svoBoundaryProb
    val roleProb: Float     get() = base.roleProb
    val voiceProb: Float    get() = base.voiceProb

    /**
     * Score de confiance SVO "unifié" :
     * • Pour les verbes (role == NONE)  → svoBoundaryProb  (verb-detector head)
     * • Pour les arguments (role != NONE) → roleProb        (role-head, le seul signal fiable en v4)
     * Ce score est utilisé pour le NMS, la déduplication et l'affichage.
     */
    val svoConfidence: Float get() = if (role == "NONE") svoBoundaryProb else roleProb
}

data class AnnotatedSentence(
    val text: String,
    val entities: List<Entity>,
    val svoSpans: List<EnrichedSvoSpan>,
    val eventlets: List<Eventlet> = emptyList(),
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
                // Filtrer les spans SVO — pipeline v4 deux signaux indépendants :
                // • Verbes  : svoBoundaryProb >= tau, role=NONE (imposé par l'extracteur pour isActualVerb)
                // • Args NP : role != NONE (chemin forcé classique ou pron_subj/obj avec rôle)
                // pron_subj/obj avec role=NONE sont dropés par l'extracteur → plus de bruit pronominal.
                val svoFiltered = res.svoSpans.filter { svo ->
                    svo.svoBoundaryProb >= cfg.tauSvoBoundary  // verbe confirmé
                    || svo.role != "NONE"                       // argument avec rôle utile
                }
                // L'extracteur force déjà role=NONE sur les vrais verbes → pas de normalisation ici.

                val enrichedSvo = if (cfg.showSvo && cfg.doReconcile)
                    reconcile(svoFiltered, res.entities, cfg) +
                    fillNullSubjects(svoFiltered, res.entities, cfg)
                else svoFiltered.map { EnrichedSvoSpan(it) }

                // Eventlets calculés sur les spans ENRICHIS (post-reconcile) et non sur les spans
                // bruts du modèle. Objectif : utiliser les rôles corrigés par reconcile()
                // (ex : Bruxelles LOC promu de SUBJECT→OBLIQUE) et les SUBJECT inférés par
                // fillNullSubjects() (ex : "ministre" sans govVerbCharStart récupéré depuis NER).
                val eventlets = if (cfg.showSvo) {
                    res.copy(svoSpans = enrichedSvo.map { it.base }).eventlets()
                } else emptyList()

                // Patcher syntacticRole="appos" sur les entités NER dont le span SVO APPOS
                // a été résolu par reconcile() Phase 2 (chemin forcé, pSvo trop bas pour la
                // passe inline → syntacticRole non renseigné par l'extracteur).
                val apposEntityIds: Set<Pair<Int,Int>> = enrichedSvo
                    .filter { it.role == "APPOS" && it.entity != null && it.entity.metadata["syntacticRole"] == null }
                    .mapNotNull { it.entity?.let { e -> e.span.start to e.span.end } }
                    .toSet()
                val finalEntities = if (apposEntityIds.isEmpty()) res.entities else
                    res.entities.map { e ->
                        if ((e.span.start to e.span.end) in apposEntityIds)
                            e.copy(metadata = e.metadata + ("syntacticRole" to "appos"))
                        else e
                    }

                AnnotatedSentence(sent, finalEntities, enrichedSvo, eventlets)
            }
            onBatchReady(i, annotated)
            i += cfg.batchSize
        }
    }

    // ── Réconciliation NER ↔ SVO ──────────────────────────────────────────────

    // Helper v4 : un SvoSpan est sujet si role=SUBJECT ou synLabel=pron_subj
    private fun SvoSpan.isSubject() = role == "SUBJECT" || synLabel == "pron_subj"
    private fun SvoSpan.isObject()  = role in setOf(
        "OBJECT",
        "OBLIQUE", "OBLIQUE_AGENT", "OBLIQUE_CAUSE",
        "OBLIQUE_ADVERSARY", "OBLIQUE_BENEFICIARY", "OBLIQUE_COMITATIVE",
        "OBLIQUE_DOMAIN", "OBLIQUE_SOURCE", "OBLIQUE_TIME", "OBLIQUE_LOC",
        "APPOS",
    ) || synLabel == "pron_obj"

    private val subjCoarse   = setOf("PER", "ORG", "EVENT", "ABSTRACT")
    private val objCoarse    = setOf("PER", "ORG", "LOC", "EVENT", "OBJECT", "ABSTRACT", "VALUE", "TIME", "WORK")

    private fun reconcile(
        svoSpans: List<SvoSpan>,
        nerEntities: List<Entity>,
        cfg: DemoConfig,
    ): List<EnrichedSvoSpan> {

        // ── Phase 1 : entités scorées directement sur la tête SVO ─────────────────
        // Pendant le forward pass NER, chaque candidat span est aussi passé par la tête
        // SVO. Le résultat est stocké dans entity.metadata["svoRole"] (labels v4 :
        // SUBJECT, OBJECT, OBLIQUE, OBLIQUE_AGENT, OBLIQUE_CAUSE, APPOS).
        // C'est la vraie réponse à "la tête SVO pense quoi de cette entité ?"
        //
        // On filtre par pBoundary (lu dans les métadonnées de l'entité) en utilisant
        // le seuil COURANT du config (cfg.tauBoundary) plutôt qu'un seuil composite
        // (minNerScoreReconcile) qui agissait comme un seuil boundary implicite ~0.84.
        // Pour les entités svoAnchored (pBoundary < tauBoundary), on tolère jusqu'à
        // tauSvoAnchoredBoundary — leur légitimité vient de la tête SVO, pas NER.
        val inlineFromEntities: List<EnrichedSvoSpan> = nerEntities.mapNotNull { e ->
            val svoRole = e.metadata["svoRole"] as? String ?: return@mapNotNull null
            // svoRole contient les nouveaux labels v4 : SUBJECT, OBJECT, OBLIQUE, etc.
            val isSubj = svoRole == "SUBJECT"
            val isObj  = svoRole in setOf(
                "OBJECT",
                "OBLIQUE", "OBLIQUE_AGENT", "OBLIQUE_CAUSE",
                "OBLIQUE_ADVERSARY", "OBLIQUE_BENEFICIARY", "OBLIQUE_COMITATIVE",
                "OBLIQUE_DOMAIN", "OBLIQUE_SOURCE", "OBLIQUE_TIME", "OBLIQUE_LOC",
                "APPOS",
            )
            if (!isSubj && !isObj) return@mapNotNull null
            val coarse  = e.metadata["coarse"] as? String ?: return@mapNotNull null
            val allowed = if (isSubj) subjCoarse else objCoarse
            if (coarse !in allowed) return@mapNotNull null
            val pBoundary  = e.metadata["pBoundary"] as? Float ?: 0f
            val isAnchored = e.metadata["svoAnchored"] as? Boolean ?: false
            // Seuil dynamique : tauBoundary courant pour les entités normales,
            // tauSvoAnchoredBoundary pour les entités promues par SVO.
            val minBoundary = if (isAnchored) cfg.tauSvoAnchoredBoundary else cfg.tauBoundary
            if (pBoundary < minBoundary) return@mapNotNull null

            // Note : Les entités NER n'ont pas de synLabel (verb_trigger/pron_subj/pron_obj)
            // car ce sont des types syntaxiques distincts détectés sur d'autres spans.
            // On utilise un synLabel factice pour construire le SvoSpan.
            EnrichedSvoSpan(
                base = SvoSpan(
                    text            = e.text,
                    charStart       = e.span.start,
                    charEnd         = e.span.end,
                    synLabel        = "NER",  // Factice : pas un vrai synLabel
                    synProb         = 0f,
                    role            = svoRole,  // SUBJECT, OBJECT, OBLIQUE, etc.
                    roleProb        = e.metadata["svoRoleProb"]     as? Float ?: 0f,
                    svoBoundaryProb = e.metadata["svoBoundaryScore"] as? Float ?: 0f,
                    voice           = "active",
                    voiceProb       = 0f,
                    certainty       = "certain",
                    certaintyProb   = 0f,
                    gender          = e.metadata["gender"] as? String,
                    number          = e.metadata["number"] as? String,
                    person          = e.metadata["person"] as? String,
                    govVerbCharStart = null,
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
            if (!svo.isSubject() && !svo.isObject())
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
            val allowed = if (svo.isSubject()) subjCoarse else objCoarse
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
                // Le texte retenu est celui du span SVO brut si ce span est PLUS LARGE que
                // l'entité NER snappée (ex. "La police" [0:9] > "police" [3:9]).
                // Cela préserve les déterminants/articles que NER n'inclut pas dans son span.
                val svoIsWider = svo.charStart < best.span.start || svo.charEnd > best.span.end
                val mergedText      = if (svoIsWider) svo.text            else best.text
                val mergedCharStart = if (svoIsWider) svo.charStart       else best.span.start
                val mergedCharEnd   = if (svoIsWider) svo.charEnd         else best.span.end
                EnrichedSvoSpan(
                    base = SvoSpan(
                        text = mergedText, charStart = mergedCharStart, charEnd = mergedCharEnd,
                        synLabel = svo.synLabel, synProb = svo.synProb,
                        role = svo.role, roleProb = svo.roleProb, svoBoundaryProb = svo.svoBoundaryProb,
                        voice = svo.voice, voiceProb = svo.voiceProb,
                        certainty = svo.certainty, certaintyProb = svo.certaintyProb,
                        gender = svo.gender, number = svo.number, person = svo.person,
                        govVerbCharStart = svo.govVerbCharStart,
                        govVerbText      = svo.govVerbText,
                    ),
                    nerOverride      = best.type.ifBlank { coarse },
                    nerOverrideScore = best.metadata["score"] as? Float,
                    entity           = best,  // ← entité NER fusionnée (snap positionnel)
                )
            } else EnrichedSvoSpan(svo)
        }

        // ── Déduplication finale ──────────────────────────────────────────────
        // Deux rôles sont "en conflit" si identiques OU si deux sous-types OBLIQUE_*
        // (même span ne peut pas avoir deux rôles obliques différents).
        val OBLIQUE_ROLES = setOf(
            "OBLIQUE", "OBLIQUE_AGENT", "OBLIQUE_CAUSE",
            "OBLIQUE_ADVERSARY", "OBLIQUE_BENEFICIARY", "OBLIQUE_COMITATIVE",
            "OBLIQUE_DOMAIN", "OBLIQUE_SOURCE", "OBLIQUE_TIME", "OBLIQUE_LOC",
        )
        fun rolesConflict(a: String, b: String) =
            a == b || (a in OBLIQUE_ROLES && b in OBLIQUE_ROLES)

        // Phase 1 vs Phase 2 : si un span Phase 2 chevauche un span Phase 1 avec rôle
        // en conflit, on garde le plus confiant (svoConfidence).
        val rawDeduped = rawFallback.filter { raw ->
            val overlapWithInline = inlineFromEntities.any { inline ->
                rolesConflict(raw.role, inline.role) &&
                minOf(raw.charEnd, inline.charEnd) - maxOf(raw.charStart, inline.charStart) > 0
            }
            if (!overlapWithInline) return@filter true
            val matchingInline = inlineFromEntities.firstOrNull { inline ->
                rolesConflict(raw.role, inline.role) &&
                minOf(raw.charEnd, inline.charEnd) - maxOf(raw.charStart, inline.charStart) > 0
            }
            raw.svoConfidence > (matchingInline?.svoConfidence ?: 0f)
        }
        val inlineFinal = inlineFromEntities.filter { inline ->
            rawDeduped.none { raw ->
                rolesConflict(raw.role, inline.role) &&
                minOf(raw.charEnd, inline.charEnd) - maxOf(raw.charStart, inline.charStart) > 0 &&
                raw.svoConfidence > inline.svoConfidence
            }
        }

        // Phase 2 intra-NMS : parmi les spans rawDeduped qui se chevauchent avec rôle en conflit,
        // ne garder que le plus confiant (tri décroissant → le plus confiant passe en premier).
        val rawNmsed = mutableListOf<EnrichedSvoSpan>()
        for (candidate in rawDeduped.sortedByDescending { it.svoConfidence }) {
            val dominated = rawNmsed.any { kept ->
                rolesConflict(kept.role, candidate.role) &&
                minOf(candidate.charEnd, kept.charEnd) - maxOf(candidate.charStart, kept.charStart) > 0
            }
            if (!dominated) rawNmsed += candidate
        }

        return inlineFinal + rawNmsed
    }

    private fun fillNullSubjects(
        svoSpans: List<SvoSpan>,
        nerEntities: List<Entity>,
        cfg: DemoConfig,
    ): List<EnrichedSvoSpan> {
        // Après normalisation, les vrais verbes ont synLabel=verb_trigger ET role=NONE.
        // Les spans synLabel=verb_trigger avec role!=NONE sont des args NP (artefact v4).
        val verbs    = svoSpans.filter { it.synLabel == "verb_trigger" && it.role == "NONE" }
        val subjects = svoSpans.filter { it.isSubject() }
        // Entités déjà présentes comme args SVO (OBLIQUE, OBJECT, etc.) : ne pas les recycler en SUBJECT.
        // Ex : "épidémiologie" déjà OBLIQUE → fillNullSubjects ne doit pas le remettre en SUBJECT.
        val alreadyUsedAsArg = svoSpans
            .filter { it.role !in setOf("NONE", "SUBJECT") }
            .map { it.charStart to it.charEnd }
            .toSet()

        return verbs.mapNotNull { v ->
            if (subjects.any { it.charEnd <= v.charStart }) return@mapNotNull null
            val best = nerEntities
                .filter { e ->
                    val coarse = e.metadata["coarse"] as? String ?: return@filter false
                    coarse in subjCoarse &&
                    e.span.end <= v.charStart &&
                    v.charStart - e.span.end <= cfg.maxGapChars &&
                    (e.metadata["score"] as? Float ?: 0f) >= cfg.minNerScoreFill &&
                    // Exclure les entités déjà assignées à un autre rôle SVO
                    (e.span.start to e.span.end) !in alreadyUsedAsArg
                }
                .maxByOrNull { it.span.end }
            best?.let { e ->
                EnrichedSvoSpan(
                    base = SvoSpan(
                        text = e.text, charStart = e.span.start, charEnd = e.span.end,
                        synLabel = "NER", synProb = 0f,   // sujet inféré depuis NER, pas un verb_trigger
                        role = "SUBJECT", roleProb = 0f, svoBoundaryProb = 0f,
                        voice = v.voice, voiceProb = 0f,
                        certainty = v.certainty, certaintyProb = 0f,
                        gender = null, number = null, person = null,
                        govVerbCharStart = v.charStart,
                        govVerbText     = v.text,
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
