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
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.file.Paths
import kotlin.math.exp
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
// Labels SVO / Voice / Morpho — ordre identique à labels.py
// ─────────────────────────────────────────────────────────────────────────────

private val SVO_LABELS = listOf(
    "svo_verb", "svo_subject", "svo_object", "svo_iobj", "pron_subj", "pron_obj"
)
private val VOICE_LABELS = listOf("ACTIVE", "PASSIVE")
private val GENDER_LABELS = listOf("Masc", "Fem", "NONE")
private val NUMBER_LABELS = listOf("Sing", "Plur", "NONE")

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

/** Indices coarse valides (pas NONE, et avec au moins un label fine autorisé) — pré-calculé. */
private val VALID_COARSE_INDICES: IntArray = COARSE_LABELS.indices
    .filter { c -> c != COARSE_NONE_IDX && COARSE_FINE_MASK[c].any { it } }
    .toIntArray()

/**
 * Buckets de longueur de séquence utilisés pour le dynamic padding.
 * Chaque groupe de textes dont seqLen ≤ bucket est traité avec maxLen = bucket,
 * évitant de pader 100 phrases courtes à 128 tokens à cause d'un seul outlier.
 */
private val LENGTH_BUCKETS = intArrayOf(24, 32, 48, 64, 80, 96, 112, 128, 192, 256, 384, 512)

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

/** Un span syntaxique SVO brut après scoring. */
private data class RawSvoResult(
    val candidate: SpanCandidate,
    val svoBoundaryProb: Float,
    val role: String,
    val roleProb: Float,
    val voice: String,
    val voiceProb: Float,
    val gender: String?,
    val number: String?,
)

/**
 * Span syntaxique SVO (verbe, sujet, objet, oblique ou pronom) détecté par le modèle.
 *
 * [role] : svo_verb | svo_subject | svo_object | svo_iobj | pron_subj | pron_obj
 * [voice] : ACTIVE | PASSIVE (surtout pertinent pour svo_verb)
 * [gender] / [number] : Masc/Fem/NONE et Sing/Plur/NONE (morphologie coréf)
 */
data class SvoSpan(
    val text: String,
    val charStart: Int,
    val charEnd: Int,
    val role: String,
    val roleProb: Float,
    val svoBoundaryProb: Float,
    val voice: String,
    val voiceProb: Float,
    val gender: String?,
    val number: String?,
)

/**
 * Entité NER enrichie de son rôle syntaxique reconcilié depuis la tête SVO.
 *
 * [syntacticRole] : "nsubj" (sujet), "obj" (objet direct), "iobj" (objet indirect) ou null.
 * [svoSpan]       : le SvoSpan source du role, pour accéder aux features morpho (gender, number, voice…).
 * [overlapRatio]  : fraction de l'entité couverte par le SvoSpan (0..1).
 */
data class EntityWithRole(
    val entity: Entity,
    val syntacticRole: String?,
    val svoSpan: SvoSpan?,
    val overlapRatio: Float,
)

/**
 * Résultat complet pour un texte : entités NER + spans SVO.
 */
data class ExtractionResult(
    val entities: List<Entity>,
    val svoSpans: List<SvoSpan>,
) {
    /** Reconstruit les triplets (sujet, verbe, objet) de façon greedy. */
    fun svoTriplets(): List<SvoTriplet> {
        val verbs    = svoSpans.filter { it.role == "svo_verb" }
        val subjects = svoSpans.filter { it.role in listOf("svo_subject", "pron_subj") }
        val objects  = svoSpans.filter { it.role in listOf("svo_object", "svo_iobj", "pron_obj") }
        return verbs.map { v ->
            val subj = subjects.filter { it.charEnd  <= v.charStart }.maxByOrNull { it.charStart }
            val obj  = objects .filter { it.charStart >= v.charEnd   }.minByOrNull { it.charStart }
            SvoTriplet(subject = subj, verb = v, obj = obj)
        }
    }

    /**
     * Réconciliation nsubj / obj / iobj pour chaque entité NER.
     *
     * Pour chaque entité, cherche le SvoSpan argumental (svo_subject, svo_object,
     * svo_iobj, pron_subj, pron_obj) qui maximise le taux de recouvrement avec l'entité.
     * Le rôle SVO est normalisé en rôle syntaxique universel :
     *   svo_subject / pron_subj → "nsubj"
     *   svo_object  / pron_obj  → "obj"
     *   svo_iobj                → "iobj"
     *
     * Ce croisement est complémentaire de l'enrichissement inline (métadonnée "svoRole"
     * stockée dans l'entité quand la tête SVO a tiré sur le même candidat span) :
     * il peut réconcilier des entités dont les bornes diffèrent légèrement du SvoSpan.
     */
    fun reconcileSvoRoles(): List<EntityWithRole> {
        val argumentRoles = setOf("svo_subject", "svo_object", "svo_iobj", "pron_subj", "pron_obj")

        fun toSyntactic(role: String): String? = when (role) {
            "svo_subject", "pron_subj" -> "nsubj"
            "svo_object",  "pron_obj"  -> "obj"
            "svo_iobj"                 -> "iobj"
            else                       -> null
        }

        return entities.map { entity ->
            val eStart    = entity.span.start
            val eEnd      = entity.span.end
            if (eStart < 0 || eEnd <= eStart)
                return@map EntityWithRole(entity, null, null, 0f)

            val entityLen = (eEnd - eStart).coerceAtLeast(1)

            // Score = roleProb × overlapRatio → favorise les spans précis ET confiants
            val best = svoSpans
                .filter { it.role in argumentRoles }
                .mapNotNull { svo ->
                    val overlap = minOf(svo.charEnd, eEnd) - maxOf(svo.charStart, eStart)
                    if (overlap <= 0) null
                    else {
                        val ratio = overlap.toFloat() / entityLen
                        Triple(svo, ratio, svo.roleProb * ratio)
                    }
                }
                .maxByOrNull { it.third }

            EntityWithRole(
                entity        = entity,
                syntacticRole = best?.let { toSyntactic(it.first.role) },
                svoSpan       = best?.first,
                overlapRatio  = best?.second ?: 0f,
            )
        }
    }
}

data class SvoTriplet(
    val subject: SvoSpan?,
    val verb: SvoSpan,
    val obj: SvoSpan?,
)

/**
 * Seuils d'inférence passables en override pour un appel spécifique.
 * Toute valeur null → repli sur la valeur de construction de l'extracteur.
 */
data class ExtractionThresholds(
    val tauBoundary: Float? = null,
    val tauNone: Float? = null,
    val tauCoarse: Float? = null,
    val tauSvoBoundary: Float? = null,
    /**
     * Seuil NER boundary abaissé appliqué uniquement aux spans que la tête SVO a identifiés
     * comme arguments non-pronominaux (svo_subject, svo_object, svo_iobj).
     * Permet de typer des entités borderline que NER n'aurait pas retenues seul.
     * null → repli sur la valeur de construction de l'extracteur (défaut 0.40).
     */
    val tauSvoAnchoredBoundary: Float? = null,
    /** Score minimum (pBnd × pCoarse × pFine) par label coarse. Si absent → minScore global. */
    val scoreByCoarse: Map<String, Float> = emptyMap(),
)

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
    /**
     * Rôle SVO détecté sur ce même span via la tête SVO (mêmes logits, même forward pass).
     * svo_subject | svo_object | svo_iobj | pron_subj | pron_obj — jamais svo_verb.
     * Null si la tête SVO ne détecte pas de boundary sur ce span.
     */
    val svoRole: String? = null,
    val svoRoleProb: Float? = null,
    val svoBoundaryScore: Float? = null,
    /**
     * Genre et nombre morphologiques lus depuis les têtes gender/number (mêmes logits).
     * Disponibles indépendamment du SVO boundary — features morphologiques de l'entité.
     */
    val svoGender: String? = null,
    val svoNumber: String? = null,
    /**
     * true si cette entité a été promue par la tête SVO (boundary NER sous le seuil normal
     * mais au-dessus du seuil abaissé tauSvoAnchoredBoundary) sur un span argumental
     * non-pronominal. Les entités svoAnchored sont moins certaines côté NER.
     */
    val svoAnchored: Boolean = false,
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
    private val tauSvoBoundary: Float = 0.50f,
    /**
     * Seuil NER boundary abaissé réservé aux spans que la tête SVO a classés comme
     * arguments non-pronominaux (svo_subject, svo_object, svo_iobj).
     * Défaut 0.40 : en-dessous du seuil normal (0.70) mais au-dessus du bruit.
     */
    private val tauSvoAnchoredBoundary: Float = 0.40f,
    private val useCoreMl: Boolean = false,
    /**
     * Nombre de threads intra-op (parallélisme au sein d'un seul opérateur).
     * Sur Apple Silicon il n'y a pas d'hyperthreading : availableProcessors == cœurs physiques.
     * Sur x86/HT, availableProcessors == cœurs logiques ; utiliser /2 si nécessaire.
     * Défaut : tous les cœurs disponibles (identique au comportement Python ORT).
     */
    private val intraOpThreads: Int = Runtime.getRuntime().availableProcessors(),
    /**
     * Nombre de threads inter-op (parallélisme entre opérateurs indépendants du graph).
     * 1 est optimal pour des inférences séquentielles ; augmenter seulement si très gros modèle
     * avec branches parallèles et machine multi-socket.
     */
    private val interOpThreads: Int = 1,
    /**
     * Niveau d'optimisation du graph ORT.
     * - ALL_OPT   : fusions maximales (LayerNorm, GELU, Attention…) → pic mémoire au chargement ~2× modèle
     * - EXTENDED_OPT : fusions node-level sans layout → bon compromis latence/mémoire
     * - BASIC_OPT : constant folding + shape inference → empreinte mémoire minimale
     * Défaut : ALL_OPT (max performance, local/ML machines).
     * En démo contrainte mémoire, utiliser BASIC_OPT.
     */
    private val optLevel: OrtSession.SessionOptions.OptLevel = OrtSession.SessionOptions.OptLevel.ALL_OPT,
    /**
     * Active l'arène mémoire CPU ORT (pool de blocs réutilisés entre inférences).
     * true  → performance maximale, mais la RAM de pointe reste allouée entre les requêtes.
     * false → la RAM est restituée à l'OS après chaque inférence (légèrement plus lent, bien
     *          adapté aux démos à faible concurrence sur des machines à mémoire limitée).
     */
    private val cpuArena: Boolean = true,
) : AutoCloseable, NerExtractor {

    private val log = LoggerFactory.getLogger(OnnxMultiHeadEntityExtractor::class.java)

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(modelPath, OrtSession.SessionOptions().apply {
        // ── Graph-level optimizations ──────────────────────────────────────
        // ALL_OPT = Basic + Extended + Layout optimisations + fused kernels
        //  → fusionne LayerNorm, GELU, Attention, MatMul+Add, etc.
        //  → peut donner 30-60% de gain sur DeBERTa par rapport au défaut (BASIC_OPT)
        //  → pic mémoire au chargement ~2× taille modèle ; utiliser BASIC_OPT sur machines contraintes
        setOptimizationLevel(optLevel)

        // ── Thread configuration ───────────────────────────────────────────
        setIntraOpNumThreads(intraOpThreads)
        setInterOpNumThreads(interOpThreads)

        // SEQUENTIAL : optimal pour batch séquentiels + intra-op threading
        setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)

        // ── Memory optimizations ───────────────────────────────────────────
        // CPU Arena : pool d'allocations réutilisées entre inférences.
        //   true  → perf maximale, mais la RAM reste allouée entre les calls.
        //   false → RAM restituée à l'OS à chaque fin d'inférence (démos mémoire-contrainte).
        setCPUArenaAllocator(cpuArena)
        // Memory pattern : mémorise le plan d'allocation du graph (un seul malloc groupé)
        setMemoryPatternOptimization(true)

        if (useCoreMl) tryAddCoreML()
    })
    private val tokenizer: HuggingFaceTokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerDir))

    /** Informations sur le runtime ONNX et la configuration matérielle. */
    fun runtimeInfo(): Map<String, Any> = mapOf(
        "provider"       to if (useCoreMl) "CoreML (Apple Neural Engine + GPU)" else "CPU (ONNX Runtime)",
        "intraOpThreads" to intraOpThreads,
        "interOpThreads" to interOpThreads,
        "optimization"   to optLevel.name,
        "cpuArena"       to cpuArena,
        "maxSeqLen"      to maxSeqLen,
        "maxSpanLen"     to maxSpanLen,
        "availableProcessors" to Runtime.getRuntime().availableProcessors(),
    )

    // ── Buffers thread-local : zéro allocation dans les hot paths ────────────
    private val tlBufCoarse   = ThreadLocal.withInitial { FloatArray(COARSE_LABELS.size) }
    private val tlBufFine     = ThreadLocal.withInitial { FloatArray(FINE_LABELS.size)  }
    private val tlBufFineMask = ThreadLocal.withInitial { FloatArray(FINE_LABELS.size)  }
    /** Buffer dédié au chargement de la ligne fine depuis le FloatBuffer flat.
     *  Distinct de tlBufFine/tlBufFineMask pour éviter l'aliasing dans bestCoarseFine. */
    private val tlBufFineRow  = ThreadLocal.withInitial { FloatArray(FINE_LABELS.size)  }
    private val tlBufSvo      = ThreadLocal.withInitial { FloatArray(SVO_LABELS.size)   }
    private val tlBufVoice    = ThreadLocal.withInitial { FloatArray(VOICE_LABELS.size) }
    private val tlBufGender   = ThreadLocal.withInitial { FloatArray(GENDER_LABELS.size)}
    private val tlBufNumber   = ThreadLocal.withInitial { FloatArray(NUMBER_LABELS.size)}

    init {
        // ── Warmup JIT + compilation CoreML ─────────────────────────────────
        // La première passe ONNX + JVM JIT est 3-5× plus lente.
        // Sur macOS avec CoreML, la première passe compile le modèle Core ML
        // (peut prendre 10-60s selon la taille du modèle) — mis en cache ensuite.
        try {
            log.info("[MH] warmup (coreML={}, threads={} intra / {} inter)…",
                useCoreMl, intraOpThreads, interOpThreads)
            val t0 = System.nanoTime()
            extractAllFromTexts(listOf("Le président de la République est à Paris ."))
            log.info("[MH] warmup done in {}ms", (System.nanoTime() - t0) / 1_000_000L)
        } catch (e: Exception) {
            log.warn("[MH] warmup failed (non-bloquant) : {}", e.message)
        }
    }


    override fun extractNer(documents: List<RagDocument>): List<List<Entity>> =
        extractFromTexts(documents.map { it.text })

    fun extractFromText(text: String): List<Entity> = extractFromTexts(listOf(text)).first()

    fun extractFromTexts(texts: List<String>): List<List<Entity>> =
        extractAllFromTexts(texts).map { it.entities }

    /** Extraction NER + SVO pour un seul texte. */
    fun extractWithSvo(text: String): ExtractionResult =
        extractWithSvoFromTexts(listOf(text)).first()

    /** Extraction NER + SVO pour un batch de textes. */
    fun extractWithSvoFromTexts(texts: List<String>): List<ExtractionResult> =
        extractAllFromTexts(texts)

    /** Extraction NER + SVO pour un batch avec seuils overridés à la volée. */
    fun extractWithSvoFromTexts(texts: List<String>, overrides: ExtractionThresholds): List<ExtractionResult> =
        extractAllFromTexts(texts, overrides)

    // ─── implémentation commune ───────────────────────────────────────────────

    private fun extractAllFromTexts(
        texts: List<String>,
        overrides: ExtractionThresholds? = null,
    ): List<ExtractionResult> {
        val effTauBoundary         = overrides?.tauBoundary            ?: tauBoundary
        val effTauNone             = overrides?.tauNone               ?: tauNone
        val effTauCoarse           = overrides?.tauCoarse             ?: tauCoarse
        val effTauSvoBoundary      = overrides?.tauSvoBoundary        ?: tauSvoBoundary
        val effTauSvoAnchored      = overrides?.tauSvoAnchoredBoundary ?: tauSvoAnchoredBoundary
        val effScoreByCoarse       = overrides?.scoreByCoarse         ?: emptyMap()
        if (texts.isEmpty()) return emptyList()
        val t0 = System.nanoTime()

        // ── 1. Tokenisation unique pour tous les textes ─────────────────────
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

        // ── 2. Grouper par bucket de longueur pour minimiser le padding ─────
        // Les textes courts ne sont plus padés à 128 à cause d'un seul outlier long :
        //   ex. 80 phrases de 40 tokens + 20 de 128 → 2 appels ONNX [80,48] + [20,128]
        //   au lieu de 1 appel [100,128] → ~3-5× moins de calcul d'attention.
        val bucketGroups: Map<Int, List<Int>> = texts.indices
            .groupBy { i -> LENGTH_BUCKETS.firstOrNull { it >= encodings[i].seqLen } ?: encodings[i].seqLen }
            .toSortedMap()

        if (log.isDebugEnabled) {
            val summary = bucketGroups.entries.joinToString(", ") { (k, v) -> "$k→${v.size}" }
            log.debug("[MH] buckets: {}", summary)
        }

        // Structure de sortie : FloatBuffer flat au lieu de Array<FloatArray>
        // → 1 objet Java par sortie (vs N floatArrays), moins de JNI overhead
        data class OnnxOutputsFlat(
            val boundaryFlat: FloatBuffer,  // [N * 2]
            val coarseFlat:   FloatBuffer,  // [N * nCoarse]
            val fineFlat:     FloatBuffer,  // [N * nFine]
            val svoBndFlat:   FloatBuffer?, // [N * 2]
            val svoFlat:      FloatBuffer?, // [N * nSvo]
            val voiceFlat:    FloatBuffer?, // [N * nVoice]
            val genderFlat:   FloatBuffer?, // [N * nGender]
            val numberFlat:   FloatBuffer?, // [N * nNumber]
        )

        val resultsByIdx = arrayOfNulls<ExtractionResult>(texts.size)

        // ── 3. Traiter chaque bucket indépendamment ─────────────────────────
        for ((bucketMaxLen, origIndices) in bucketGroups) {
            val tBucket = System.nanoTime()
            val subEncodings = origIndices.map { encodings[it] }
            val batchSize    = origIndices.size

            // ── 3a. Candidats spans (indexés localement 0..batchSize-1) ─────
            val tCand = System.nanoTime()
            val candidates: List<SpanCandidate> = buildCandidates(
                subEncodings.mapIndexed { localIdx, enc ->
                    EncodedExample(localIdx, enc.text, enc.wordRanges, enc.charOffsets, enc.wordIds, enc.tokens, enc.seqLen)
                }
            )
            val msCand = ms(tCand)
            if (candidates.isEmpty()) {
                origIndices.forEach { resultsByIdx[it] = ExtractionResult(emptyList(), emptyList()) }
                continue
            }

            // ── 3b. Construction des tenseurs ────────────────────────────────
            val tTensors = System.nanoTime()
            val N          = candidates.size
            val inputIds   = LongArray(batchSize * bucketMaxLen)
            val attMask    = LongArray(batchSize * bucketMaxLen)
            val spanStarts = LongArray(N)
            val spanEnds   = LongArray(N)
            val spanBatch  = LongArray(N)

            subEncodings.forEachIndexed { li, enc ->
                for (j in 0 until enc.seqLen) {
                    inputIds[li * bucketMaxLen + j] = enc.ids[j]
                    attMask [li * bucketMaxLen + j] = 1L
                }
            }
            candidates.forEachIndexed { k, c ->
                spanStarts[k] = c.tokStart.toLong()
                spanEnds  [k] = c.tokEnd.toLong()
                spanBatch [k] = c.exampleIdx.toLong()
            }
            val msTensors = ms(tTensors)

            // ── 3c. Inférence ONNX ───────────────────────────────────────────
            val shape2D  = longArrayOf(batchSize.toLong(), bucketMaxLen.toLong())
            val shape1D  = longArrayOf(N.toLong())

            val tInputIds = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), shape2D)
            val tAttMask  = OnnxTensor.createTensor(env, LongBuffer.wrap(attMask),  shape2D)
            val tStarts   = OnnxTensor.createTensor(env, LongBuffer.wrap(spanStarts), shape1D)
            val tEnds     = OnnxTensor.createTensor(env, LongBuffer.wrap(spanEnds),   shape1D)
            val tBatchIds = OnnxTensor.createTensor(env, LongBuffer.wrap(spanBatch),  shape1D)

            val tOnnxRun = System.nanoTime()
            val onnxOut = session.run(
                mapOf(
                    "input_ids"      to tInputIds,
                    "attention_mask" to tAttMask,
                    "span_starts"    to tStarts,
                    "span_ends"      to tEnds,
                    "span_batch_ids" to tBatchIds,
                )
            ).use { result ->
                // Utiliser FloatBuffer (flat, potentiellement natif) au lieu de Array<FloatArray>
                // → 1 objet Java au lieu de N, moins de JNI overhead
                @Suppress("UNCHECKED_CAST")
                fun flatBuf(key: String): FloatBuffer? =
                    (result[key].orElse(null) as? OnnxTensor)?.floatBuffer

                OnnxOutputsFlat(
                    boundaryFlat = (result["boundary_logits"].get() as OnnxTensor).floatBuffer,
                    coarseFlat   = (result["coarse_logits"]  .get() as OnnxTensor).floatBuffer,
                    fineFlat     = (result["fine_logits"]    .get() as OnnxTensor).floatBuffer,
                    svoBndFlat   = flatBuf("svo_boundary_logits"),
                    svoFlat      = flatBuf("svo_logits"),
                    voiceFlat    = flatBuf("voice_logits"),
                    genderFlat   = flatBuf("gender_logits"),
                    numberFlat   = flatBuf("number_logits"),
                )
            }
            val msOnnx = ms(tOnnxRun)
            listOf(tInputIds, tAttMask, tStarts, tEnds, tBatchIds).forEach { it.close() }

            // ── 3d. Décodage par span ───────────────────────────────────────
            val tDecode = System.nanoTime()
            val rawByLocal: Array<MutableList<SpanResult>>    = Array(batchSize) { mutableListOf() }
            val svoByLocal: Array<MutableList<RawSvoResult>>  = Array(batchSize) { mutableListOf() }

            val nCoarse = COARSE_LABELS.size
            val nFine   = FINE_LABELS.size
            val nSvo    = SVO_LABELS.size
            val nVoice  = VOICE_LABELS.size
            val nGender = GENDER_LABELS.size
            val nNumber = NUMBER_LABELS.size

            candidates.forEachIndexed { k, cand ->
                // ── NER ──────────────────────────────────────────────────────
                // softmaxProbFlat sur les 2 boundary logits — zéro allocation
                val pBoundary = softmaxProbFlat(onnxOut.boundaryFlat, k * 2, 2, 1)
                if (pBoundary >= effTauBoundary) {
                    // Charger la ligne coarse dans le buffer TL puis softmax in-place
                    val cProbs = tlBufCoarse.get()
                    loadRow(onnxOut.coarseFlat, k * nCoarse, cProbs)
                    softmaxInto(cProbs, cProbs)
                    val pNone = cProbs[COARSE_NONE_IDX]
                    if (pNone < effTauNone) {
                        // Charger la ligne fine dans un buffer dédié (≠ tlBufFine/tlBufFineMask)
                        val fLogits = tlBufFineRow.get()
                        loadRow(onnxOut.fineFlat, k * nFine, fLogits)
                        val topResults = bestCoarseFine(cProbs, fLogits, effTauCoarse)

                        // ── SVO enrichment inline ──────────────────────────────────────────
                        // Pour ce même candidat NER (même k, même forward pass), on lit les
                        // logits SVO et on récupère le rôle argumental si la tête SVO tire.
                        // On exclut svo_verb : une entité NER verb serait sémantiquement incohérent.
                        // Genre et nombre sont lus indépendamment du boundary SVO :
                        // ce sont des features morphologiques valables pour toute entité.
                        var nerSvoRole:    String? = null
                        var nerSvoRoleProb: Float? = null
                        var nerSvoBndScore: Float? = null
                        var nerGender:     String? = null
                        var nerNumber:     String? = null
                        if (topResults.isNotEmpty()) {
                            // ── Genre (toujours lu si la tête est disponible) ──────────────
                            nerGender = onnxOut.genderFlat?.let {
                                val p = tlBufGender.get()
                                loadRow(it, k * nGender, p); softmaxInto(p, p)
                                var gi = 0; for (j in 1 until nGender) if (p[j] > p[gi]) gi = j
                                GENDER_LABELS.getOrElse(gi) { "NONE" }.takeUnless { s -> s == "NONE" }
                            }
                            // ── Nombre (toujours lu si la tête est disponible) ─────────────
                            nerNumber = onnxOut.numberFlat?.let {
                                val p = tlBufNumber.get()
                                loadRow(it, k * nNumber, p); softmaxInto(p, p)
                                var ni = 0; for (j in 1 until nNumber) if (p[j] > p[ni]) ni = j
                                NUMBER_LABELS.getOrElse(ni) { "NONE" }.takeUnless { s -> s == "NONE" }
                            }
                            // ── Rôle SVO (uniquement si boundary SVO suffisant) ─────────────
                            val svoBndLocal = onnxOut.svoBndFlat
                            if (svoBndLocal != null && onnxOut.svoFlat != null) {
                                val pSvo = softmaxProbFlat(svoBndLocal, k * 2, 2, 1)
                                if (pSvo >= effTauSvoBoundary) {
                                    val p = tlBufSvo.get()
                                    loadRow(onnxOut.svoFlat, k * nSvo, p); softmaxInto(p, p)
                                    var ri = 0
                                    for (j in 1 until nSvo) if (p[j] > p[ri]) ri = j
                                    val name = SVO_LABELS.getOrElse(ri) { "svo_verb" }
                                    if (name != "svo_verb") {
                                        nerSvoRole     = name
                                        nerSvoRoleProb = p[ri]
                                        nerSvoBndScore = pSvo
                                    }
                                }
                            }
                        }

                        for (result in topResults) {
                            val tokLen     = cand.tokEnd - cand.tokStart + 1
                            val maxTok     = MAX_TOK_LEN[result.fine]
                            val fineThresh = FINE_THRESHOLDS.getOrDefault(result.fine, DEFAULT_FINE_THRESHOLD)
                            val score      = pBoundary * result.pCoarse * result.pFine
                            val minScoreEff = effScoreByCoarse[result.coarse] ?: minScore
                            if ((maxTok == null || tokLen <= maxTok) && result.pFine >= fineThresh && score >= minScoreEff) {
                                rawByLocal[cand.exampleIdx] += SpanResult(
                                    candidate = cand,
                                    pBoundary = pBoundary,
                                    coarse    = result.coarse,
                                    pCoarse   = result.pCoarse,
                                    fine      = result.fine,
                                    pFine     = result.pFine,
                                    score     = score,
                                    svoRole          = nerSvoRole,
                                    svoRoleProb      = nerSvoRoleProb,
                                    svoBoundaryScore = nerSvoBndScore,
                                    svoGender        = nerGender,
                                    svoNumber        = nerNumber,
                                )
                            }
                        }
                    }
                }

                // ── SVO (si têtes disponibles dans ce modèle ONNX) ───────────
                val svoBndFlat = onnxOut.svoBndFlat
                if (svoBndFlat != null) {
                    val pSvoB = softmaxProbFlat(svoBndFlat, k * 2, 2, 1)
                    if (pSvoB >= effTauSvoBoundary) {
                        val roleName: String
                        val roleProb: Float
                        if (onnxOut.svoFlat != null) {
                            val p = tlBufSvo.get()
                            loadRow(onnxOut.svoFlat, k * nSvo, p); softmaxInto(p, p)
                            var ri = 0; for (j in 1 until nSvo) if (p[j] > p[ri]) ri = j
                            roleName = SVO_LABELS.getOrElse(ri) { "svo_verb" }; roleProb = p[ri]
                        } else { roleName = "svo_verb"; roleProb = 0f }

                        val voiceName: String
                        val voiceProb: Float
                        if (onnxOut.voiceFlat != null) {
                            val p = tlBufVoice.get()
                            loadRow(onnxOut.voiceFlat, k * nVoice, p); softmaxInto(p, p)
                            var vi = 0; for (j in 1 until nVoice) if (p[j] > p[vi]) vi = j
                            voiceName = VOICE_LABELS.getOrElse(vi) { "ACTIVE" }; voiceProb = p[vi]
                        } else { voiceName = "ACTIVE"; voiceProb = 0f }

                        val gender: String? = onnxOut.genderFlat?.let {
                            val p = tlBufGender.get()
                            loadRow(it, k * nGender, p); softmaxInto(p, p)
                            var gi = 0; for (j in 1 until nGender) if (p[j] > p[gi]) gi = j
                            GENDER_LABELS.getOrElse(gi) { "NONE" }.takeUnless { s -> s == "NONE" }
                        }
                        val number: String? = onnxOut.numberFlat?.let {
                            val p = tlBufNumber.get()
                            loadRow(it, k * nNumber, p); softmaxInto(p, p)
                            var ni = 0; for (j in 1 until nNumber) if (p[j] > p[ni]) ni = j
                            NUMBER_LABELS.getOrElse(ni) { "NONE" }.takeUnless { s -> s == "NONE" }
                        }

                        svoByLocal[cand.exampleIdx] += RawSvoResult(
                            candidate       = cand,
                            svoBoundaryProb = pSvoB,
                            role            = roleName,
                            roleProb        = roleProb,
                            voice           = voiceName,
                            voiceProb       = voiceProb,
                            gender          = gender,
                            number          = number,
                        )

                        // ── SVO-anchored NER ──────────────────────────────────────────
                        // Si le span est un argument non-pronominal (nsubj/obj/iobj) ET que
                        // NER boundary n'a pas tiré au seuil normal mais dépasse le seuil
                        // abaissé → on score quand même la tête NER pour obtenir un type.
                        // Ces entités sont taguées svoAnchored=true (confiance NER réduite).
                        val isNonPronounArg = roleName in setOf("svo_subject", "svo_object", "svo_iobj")
                        if (isNonPronounArg
                            && pBoundary >= effTauSvoAnchored
                            && pBoundary < effTauBoundary
                        ) {
                            val cProbs = tlBufCoarse.get()
                            loadRow(onnxOut.coarseFlat, k * nCoarse, cProbs)
                            softmaxInto(cProbs, cProbs)
                            if (cProbs[COARSE_NONE_IDX] < effTauNone) {
                                val fLogits = tlBufFineRow.get()
                                loadRow(onnxOut.fineFlat, k * nFine, fLogits)
                                val topResults = bestCoarseFine(cProbs, fLogits, effTauCoarse)
                                for (result in topResults) {
                                    val tokLen      = cand.tokEnd - cand.tokStart + 1
                                    val maxTok      = MAX_TOK_LEN[result.fine]
                                    // Seuil fine légèrement assoupli (×0.85) en mode anchored
                                    val fineThresh  = FINE_THRESHOLDS.getOrDefault(result.fine, DEFAULT_FINE_THRESHOLD) * 0.85f
                                    val score       = pBoundary * result.pCoarse * result.pFine
                                    val minScoreEff = (effScoreByCoarse[result.coarse] ?: minScore) * 0.60f
                                    if ((maxTok == null || tokLen <= maxTok) && result.pFine >= fineThresh && score >= minScoreEff) {
                                        rawByLocal[cand.exampleIdx] += SpanResult(
                                            candidate        = cand,
                                            pBoundary        = pBoundary,
                                            coarse           = result.coarse,
                                            pCoarse          = result.pCoarse,
                                            fine             = result.fine,
                                            pFine            = result.pFine,
                                            score            = score,
                                            svoRole          = roleName,
                                            svoRoleProb      = roleProb,
                                            svoBoundaryScore = pSvoB,
                                            svoGender        = gender,
                                            svoNumber        = number,
                                            svoAnchored      = true,
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
            val msDecode = ms(tDecode)

            // ── 3e. NMS compound + conversion + stockage ────────────────────
            val bucketResults = rawByLocal.zip(svoByLocal).map { (nerSpans, svoSpans) ->
                // nmsSpansCompound trie par longueur en interne → pas besoin de pré-trier
                val nerEntities = nmsSpansCompound(nerSpans)
                    // Ordre de sortie : position puis parent avant enfant
                    .sortedWith(compareBy<NmsResult> { it.span.candidate.charStart }
                        .thenByDescending { it.span.candidate.charEnd - it.span.candidate.charStart })
                    .map { toEntity(it) }
                val svoFiltered = nmsRawSvo(svoSpans.sortedByDescending { it.svoBoundaryProb })
                    .sortedBy { it.candidate.charStart }
                    .map { toSvoSpan(it) }
                ExtractionResult(entities = nerEntities, svoSpans = svoFiltered)
            }

            origIndices.forEachIndexed { bi, origIdx -> resultsByIdx[origIdx] = bucketResults[bi] }
            // ⚠️ Timing INFO pour diagnostiquer les goulots d'étranglement
            log.info("[MH-PERF] bucket maxLen={} n={} N={}  cand={}ms  tensors={}ms  onnx={}ms  decode={}ms  total={}ms",
                bucketMaxLen, batchSize, N, msCand, msTensors, msOnnx, msDecode, ms(tBucket))
        }

        log.info("[MH-PERF] TOTAL batchSize={} ms={}", texts.size, ms(t0))
        return resultsByIdx.map { it!! }
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
        tauCoarseEff: Float = tauCoarse,
        topK: Int = 2,
    ): List<CoarseFineScore> {
        // ── Étape 1 : sélection des top-K coarse parmi ceux qui passent tauCoarseEff ──
        //
        // On collecte tous les labels coarse dont la probabilité dépasse tauCoarseEff,
        // triés par probabilité décroissante, et on en prend au plus topK.
        // On ne fait PAS concourir plusieurs familles via le score composite,
        // car les pFine re-softmaxés ne sont pas comparables entre familles ayant
        // un nombre différent de labels autorisés :
        //   - ORG (1 label autorisé)  → pFine ≈ 1.0 par construction
        //   - PER (4 labels autorisés) → pFine ≤ 0.90 (masse distribuée)
        // Pour chaque candidat coarse retenu, on calcule le meilleur fine dans sa famille.
        val coarseCandidates = mutableListOf<Pair<Int, Float>>() // (coarseIdx, pCoarse)
        for (c in VALID_COARSE_INDICES) {
            val pC = cProbs[c]
            if (pC < tauCoarseEff) continue
            coarseCandidates += c to pC
        }
        if (coarseCandidates.isEmpty()) return emptyList()
        coarseCandidates.sortByDescending { it.second }

        // ── Étape 2 : pour chaque coarse retenu, meilleur fine dans sa famille ─
        // Les buffers thread-local sont réutilisés séquentiellement (valeurs scalaires extraites
        // dans CoarseFineScore avant la prochaine itération → pas d'aliasing).
        val masked = tlBufFineMask.get()
        val fProbs = tlBufFine.get()
        return coarseCandidates.take(topK).map { (bestCoarseIdx, bestPCoarse) ->
            val mask = COARSE_FINE_MASK[bestCoarseIdx]
            for (i in fLogits.indices) masked[i] = if (mask[i]) fLogits[i] else -1e9f
            softmaxInto(masked, fProbs)
            var fIdx = 0
            var fMax = fProbs[0]
            for (i in 1 until fProbs.size) if (fProbs[i] > fMax) { fMax = fProbs[i]; fIdx = i }
            CoarseFineScore(COARSE_LABELS[bestCoarseIdx], bestPCoarse, FINE_LABELS[fIdx], fMax)
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // NMS : suppression des overlaps + d��tection des spans imbriqués (compound)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Résultat NMS enrichi : chaque span est soit un parent top-level, soit un span
     * imbriqué (fully contained) dans un parent — conservé comme compound sub-span.
     * Les overlaps partiels sont toujours supprimés.
     */
    private data class NmsResult(
        val span: SpanResult,
        /** Parent si ce span est entièrement inclus dans un autre span conservé. */
        val parent: SpanResult?,
    ) {
        val isNested: Boolean get() = parent != null
    }

    /**
     * NMS avec remontée des spans imbriqués.
     *
     * Algorithme :
     *  1. Trier par longueur décroissante → les spans les plus longs deviennent parents
     *  2. Pour chaque span s :
     *     - S'il est entièrement contenu dans un span déjà gardé → compound (nested), conservé
     *     - Sinon, s'il n'a pas de chevauchement partiel significatif → nouveau parent
     *     - Sinon → supprimé (overlap partiel, même comportement qu'avant)
     *
     *  Le tri par longueur garantit que "Nations Unies" (court, ORG) devient un enfant de
     *  "secrétaire général des Nations Unies" (long, PER_ROLE) → les deux sont renvoyés.
     */
    private fun nmsSpansCompound(
        spans: List<SpanResult>,
        iouThreshold: Float = 0.5f,
    ): List<NmsResult> {
        // Priorité aux spans les plus longs pour être parents ; à score égal, confiance d'abord
        val byLength = spans.sortedWith(
            compareByDescending<SpanResult> { it.candidate.charEnd - it.candidate.charStart }
                .thenByDescending { it.score }
        )
        val kept    = mutableListOf<SpanResult>()
        val results = mutableListOf<NmsResult>()

        for (s in byLength) {
            val sStart = s.candidate.charStart
            val sEnd   = s.candidate.charEnd

            // Fully contained dans un span parent déjà conservé ?
            val parent = kept.firstOrNull { k ->
                k.candidate.charStart <= sStart && k.candidate.charEnd >= sEnd
            }
            if (parent != null) {
                // Ne garder comme compound que si le type fin est DIFFÉRENT du parent.
                // Même fine = doublon (ex. deux EVENT_NAMED imbriqués) → on discarde simplement.
                if (s.fine != parent.fine) {
                    results += NmsResult(s, parent)
                }
                continue
            }

            // Overlap partiel avec un span déjà conservé → supprimé (NMS classique)
            if (kept.any { k -> iouPartialOnly(s, k) >= iouThreshold }) continue

            // Nouveau span top-level
            kept    += s
            results += NmsResult(s, null)
        }
        return results
    }

    /**
     * IoU "overlap partiel" uniquement — NE tient PAS compte du containment complet.
     * Un span A entièrement inclus dans B donne ici IoU < threshold afin d'être traité
     * comme compound plutôt que supprimé.
     */
    private fun iouPartialOnly(a: SpanResult, b: SpanResult): Float {
        val aStart = a.candidate.charStart; val aEnd = a.candidate.charEnd
        val bStart = b.candidate.charStart; val bEnd = b.candidate.charEnd
        val inter  = maxOf(0, minOf(aEnd, bEnd) - maxOf(aStart, bStart))
        if (inter == 0) return 0f
        val lenA = aEnd - aStart
        val lenB = bEnd - bStart
        // Containment total → retourner 0 pour NE PAS déclencher la suppression NMS
        if (aStart >= bStart && aEnd <= bEnd) return 0f  // a dans b
        if (bStart >= aStart && bEnd <= aEnd) return 0f  // b dans a
        // Overlap partiel standard
        return inter.toFloat() / (lenA + lenB - inter)
    }

    /** IoU standard OU ratio de containment — conservé pour la compatibilité SVO. */
    private fun iouOrContainment(a: SpanResult, b: SpanResult): Float {
        val inter = maxOf(0, minOf(a.candidate.charEnd, b.candidate.charEnd) -
                maxOf(a.candidate.charStart, b.candidate.charStart))
        if (inter == 0) return 0f
        val lenA  = a.candidate.charEnd - a.candidate.charStart
        val lenB  = b.candidate.charEnd - b.candidate.charStart
        val iou   = inter.toFloat() / (lenA + lenB - inter)
        val containment = inter.toFloat() / minOf(lenA, lenB)
        return maxOf(iou, containment)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Conversion vers Entity
    // ─────────────────────────────────────────────────────────────��───────────

    private fun toEntity(r: NmsResult): Entity = Entity(
        text  = r.span.candidate.spanText,
        type  = r.span.fine,
        span  = Span(r.span.candidate.charStart, r.span.candidate.charEnd, emptyList()),
        metadata = buildMap {
            put("coarse",    r.span.coarse)
            put("kind",      LABEL_KIND[r.span.fine] ?: LabelKind.TRIGGER_ARG)
            put("pBoundary", r.span.pBoundary)
            put("pCoarse",   r.span.pCoarse)
            put("pFine",     r.span.pFine)
            put("score",     r.span.score)
            // ── Rôle SVO inline (même candidat span, même forward pass) ──────────
            // svoRole        : svo_subject | svo_object | svo_iobj | pron_subj | pron_obj
            // syntacticRole  : "nsubj" | "obj" | "iobj" (normalisation UD)
            if (r.span.svoRole != null) {
                val syntactic = when (r.span.svoRole) {
                    "svo_subject", "pron_subj" -> "nsubj"
                    "svo_object",  "pron_obj"  -> "obj"
                    "svo_iobj"                 -> "iobj"
                    else                       -> null
                }
                put("svoRole",         r.span.svoRole)
                put("svoRoleProb",     r.span.svoRoleProb)
                put("svoBoundaryScore",r.span.svoBoundaryScore)
                if (syntactic != null) put("syntacticRole", syntactic)
            }
            // ── Morphologie (genre / nombre) — lus indépendamment du rôle SVO ───
            r.span.svoGender?.let { put("gender", it) }
            r.span.svoNumber?.let { put("number", it) }
            // ── Entité promue par SVO (boundary NER sous seuil normal) ────────────
            if (r.span.svoAnchored) put("svoAnchored", true)
            if (r.isNested) {
                put("nested",       true)
                put("parentText",   r.parent!!.candidate.spanText)
                put("parentStart",  r.parent.candidate.charStart)
                put("parentEnd",    r.parent.candidate.charEnd)
                put("parentFine",   r.parent.fine)
                put("parentCoarse", r.parent.coarse)
            }
        }
    )

    // ─────────────────────────────────────────────────────────────────────────
    // Math utils
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Softmax in-place dans [dst] (même taille que [src]).
     * Zéro allocation — utiliser avec les buffers thread-local.
     */
    private fun softmaxInto(src: FloatArray, dst: FloatArray): FloatArray {
        var maxV = src[0]
        for (v in src) if (v > maxV) maxV = v
        var sum = 0.0
        for (i in src.indices) {
            val e = exp((src[i] - maxV).toDouble()).toFloat()
            dst[i] = e
            sum   += e
        }
        val invSum = (1.0 / sum).toFloat()
        for (i in dst.indices) dst[i] *= invSum
        return dst
    }

    /**
     * Prob softmax pour un seul classIdx, sans aucune allocation.
     * Conservée pour usage éventuel hors hot-path.
     */
    @Suppress("unused")
    private fun softmaxProb(logits: FloatArray, classIdx: Int): Float {
        var maxV = logits[0]
        for (v in logits) if (v > maxV) maxV = v
        val target = exp((logits[classIdx] - maxV).toDouble()).toFloat()
        var sum = 0f
        for (v in logits) sum += exp((v - maxV).toDouble()).toFloat()
        return target / sum
    }

    /**
     * Prob softmax sur une ligne d'un FloatBuffer flat (offset, size), zéro allocation.
     * Utilisé pour boundary (taille 2) et svo_boundary directement sur le buffer ONNX.
     */
    private fun softmaxProbFlat(buf: FloatBuffer, offset: Int, size: Int, classIdx: Int): Float {
        var maxV = buf.get(offset)
        for (i in 1 until size) { val v = buf.get(offset + i); if (v > maxV) maxV = v }
        val target = exp((buf.get(offset + classIdx) - maxV).toDouble()).toFloat()
        var sum = 0f
        for (i in 0 until size) sum += exp((buf.get(offset + i) - maxV).toDouble()).toFloat()
        return target / sum
    }

    /** Copie une ligne [srcOffset .. srcOffset+dst.size) du FloatBuffer flat dans dst. */
    private fun loadRow(src: FloatBuffer, srcOffset: Int, dst: FloatArray) {
        for (i in dst.indices) dst[i] = src.get(srcOffset + i)
    }

    /** softmax allouant — conservé uniquement pour usages non-critiques éventuels en dehors du hot path. */
    @Suppress("unused")
    private fun softmax(logits: FloatArray): FloatArray =
        softmaxInto(logits, FloatArray(logits.size))

    private fun ms(nanoStart: Long) = (System.nanoTime() - nanoStart) / 1_000_000L

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers : SVO NMS + conversion
    // ─────────────────────────────────────────────────────────────────────────

    /** Suppression naïve des spans SVO qui se chevauchent strictement : garde le plus probable. */
    private fun nmsRawSvo(sorted: List<RawSvoResult>): List<RawSvoResult> {
        val kept = mutableListOf<RawSvoResult>()
        for (s in sorted) {
            val overlap = kept.any { k ->
                val inter = maxOf(0, minOf(s.candidate.charEnd, k.candidate.charEnd) -
                        maxOf(s.candidate.charStart, k.candidate.charStart))
                // Ne supprimer que si même rôle ET overlap > 50% du plus court span
                if (s.role != k.role) false
                else {
                    val minLen = minOf(
                        s.candidate.charEnd - s.candidate.charStart,
                        k.candidate.charEnd - k.candidate.charStart,
                    ).coerceAtLeast(1)
                    inter.toFloat() / minLen > 0.5f
                }
            }
            if (!overlap) kept += s
        }
        return kept
    }

    private fun toSvoSpan(r: RawSvoResult) = SvoSpan(
        text            = r.candidate.spanText,
        charStart       = r.candidate.charStart,
        charEnd         = r.candidate.charEnd,
        role            = r.role,
        roleProb        = r.roleProb,
        svoBoundaryProb = r.svoBoundaryProb,
        voice           = r.voice,
        voiceProb       = r.voiceProb,
        gender          = r.gender,
        number          = r.number,
    )

    // ─────────────────────────────────────────────────────────────────────────
    // CoreML & lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    private fun OrtSession.SessionOptions.tryAddCoreML() {
        // Sur Apple Silicon, CoreML route le modèle sur le Neural Engine / GPU Metal
        // → typiquement 5-15× plus rapide que CPU pur pour les transformers.
        // La PREMIÈRE inférence compile le modèle Core ML → peut prendre 10-60 secondes,
        // mais le résultat est mis en cache pour les démarrages suivants.
        try {
            addCoreML()
            log.info("[MH] CoreML EP activé — première inférence peut être lente (compilation Core ML → cache)")
        } catch (e: Exception) {
            log.warn("[MH] CoreML non disponible : {} → fallback CPU", e.message)
        }
    }

    override fun close() {
        tokenizer.close()
        session.close()
    }
}

