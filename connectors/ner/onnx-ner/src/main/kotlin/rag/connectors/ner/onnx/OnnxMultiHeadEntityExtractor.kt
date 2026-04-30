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
import kotlin.math.abs
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
// Labels SVO / Voice / Morpho v4 — ordre identique à labels.py
// ─────────────────────────────────────────────────────────────────────────────

private val SYN_LABELS = listOf(
    "verb_trigger", // 0  verbe (gold Claude v4)
    "pron_subj",    // 1  pronom sujet
    "pron_obj",     // 2  pronom objet
)
private val ROLE_LABELS = listOf(
    "SUBJECT",       // 0  sujet
    "OBJECT",        // 1  objet direct
    "OBLIQUE",       // 2  oblique/complément prépositionnel
    "OBLIQUE_AGENT", // 3  agent passif introduit par "par"
    "OBLIQUE_CAUSE", // 4  cause introduite par "à cause de" / "en raison de"
    "APPOS",         // 5  apposition
    "NONE",          // 6
)
private val VOICE_LABELS     = listOf("active", "passive")
private val CERTAINTY_LABELS = listOf("certain", "modal", "denied")
private val GENDER_LABELS    = listOf("M", "F", "N")
private val NUMBER_LABELS    = listOf("SG", "PL")
private val PERSON_LABELS    = listOf("1", "2", "3")

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

/** Un span syntaxique SVO brut après scoring (v4 gold). */
private data class RawSvoResult(
    val candidate: SpanCandidate,
    val svoBoundaryProb: Float,
    val synLabel: String,           // verb_trigger | pron_subj | pron_obj
    val synProb: Float,
    val role: String,                // SUBJECT | OBJECT | OBLIQUE | OBLIQUE_AGENT | OBLIQUE_CAUSE | APPOS | NONE
    val roleProb: Float,
    val voice: String,               // active | passive
    val voiceProb: Float,
    val certainty: String,           // certain | modal | denied
    val certaintyProb: Float,
    val gender: String?,
    val number: String?,
    val person: String?,             // personne grammaticale (pron_subj / pron_obj)
    val govVerbCharStart: Int?,      // charStart du verbe gouverneur (verb pointer, null si non prédit)
)

/**
 * Span syntaxique SVO (verbe, sujet, objet, oblique ou pronom) détecté par le modèle v4.
 *
 * [synLabel]  : verb_trigger | pron_subj | pron_obj
 * [role]      : SUBJECT | OBJECT | OBLIQUE | OBLIQUE_AGENT | OBLIQUE_CAUSE | APPOS | NONE
 * [voice]     : active | passive (surtout pertinent pour verb_trigger)
 * [certainty] : certain | modal | denied (verb_trigger uniquement)
 * [gender] / [number] : M/F/N et SG/PL (morphologie coréf)
 * [person]    : 1/2/3 — personne grammaticale (pronoms uniquement)
 * [govVerbCharStart] : charStart du verbe gouverneur prédit par le verb pointer (null pour verb_trigger)
 * [govVerbText]      : texte du verbe gouverneur résolu après NMS (null si non résolu)
 */
data class SvoSpan(
    val text: String,
    val charStart: Int,
    val charEnd: Int,
    val synLabel: String,
    val synProb: Float,
    val role: String,
    val roleProb: Float,
    val svoBoundaryProb: Float,
    val voice: String,
    val voiceProb: Float,
    val certainty: String,
    val certaintyProb: Float,
    val gender: String?,
    val number: String?,
    val person: String?,           // personne grammaticale (pron_subj / pron_obj / pron_dem)
    val govVerbCharStart: Int?,    // charStart du verbe gouverneur (verb pointer)
    val govVerbText: String? = null, // texte du verbe gouverneur (résolu après NMS)
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
    /** Reconstruit les triplets (sujet, verbe, objet) de façon greedy.
     *  Priorité au verb pointer (govVerbCharStart) quand disponible, sinon heuristique positionnelle. */
    fun svoTriplets(): List<SvoTriplet> {
        val verbs    = svoSpans.filter { it.synLabel == "verb_trigger" }
        val subjects = svoSpans.filter { it.role in listOf("SUBJECT") || it.synLabel == "pron_subj" }
        val objects  = svoSpans.filter { it.role in listOf("OBJECT") || it.synLabel == "pron_obj" }
        return verbs.map { v ->
            // Si l'argument a un govVerbCharStart qui pointe sur ce verbe → liaison directe
            val subj = subjects
                .filter { it.govVerbCharStart == v.charStart || (it.govVerbCharStart == null && it.charEnd <= v.charStart) }
                .maxByOrNull { if (it.govVerbCharStart != null) 1f else 0f + it.charStart }
            val obj = objects
                .filter { it.govVerbCharStart == v.charStart || (it.govVerbCharStart == null && it.charStart >= v.charEnd) }
                .minByOrNull { if (it.govVerbCharStart != null) -1f else 0f + it.charStart }
            SvoTriplet(subject = subj, verb = v, obj = obj)
        }
    }

    /**
     * Réconciliation nsubj / obj / iobj pour chaque entité NER.
     *
     * Pour chaque entité, cherche le SvoSpan argumental (role SUBJECT/OBJECT/OBLIQUE
     * ou synLabel pron_subj/pron_obj) qui maximise le taux de recouvrement avec l'entité.
     * Le rôle SVO est normalisé en rôle syntaxique universel :
     *   role SUBJECT / synLabel pron_subj → "nsubj"
     *   role OBJECT  / synLabel pron_obj  → "obj"
     *   role OBLIQUE*                     → "iobj" / "obl"
     *
     * Ce croisement est complémentaire de l'enrichissement inline (métadonnée "svoRole"
     * stockée dans l'entité quand la tête SVO a tiré sur le même candidat span) :
     * il peut réconcilier des entités dont les bornes diffèrent légèrement du SvoSpan.
     */
    fun reconcileSvoRoles(): List<EntityWithRole> {
        val argumentRoles = setOf("SUBJECT", "OBJECT", "OBLIQUE", "OBLIQUE_AGENT", "OBLIQUE_CAUSE", "APPOS")
        val argumentSynLabels = setOf("pron_subj", "pron_obj")

        fun toSyntactic(svo: SvoSpan): String? = when {
            svo.role == "SUBJECT" || svo.synLabel == "pron_subj" -> "nsubj"
            svo.role == "OBJECT"  || svo.synLabel == "pron_obj"  -> "obj"
            svo.role == "OBLIQUE" -> "obl"
            svo.role == "OBLIQUE_AGENT" -> "obl:agent"
            svo.role == "OBLIQUE_CAUSE" -> "obl:cause"
            svo.role == "APPOS" -> "appos"
            else -> null
        }

        return entities.map { entity ->
            val eStart    = entity.span.start
            val eEnd      = entity.span.end
            if (eStart < 0 || eEnd <= eStart)
                return@map EntityWithRole(entity, null, null, 0f)

            val entityLen = (eEnd - eStart).coerceAtLeast(1)

            // Score = roleProb × overlapRatio → favorise les spans précis ET confiants
            val best = svoSpans
                .filter { it.role in argumentRoles || it.synLabel in argumentSynLabels }
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
                syntacticRole = best?.let { toSyntactic(it.first) },
                svoSpan       = best?.first,
                overlapRatio  = best?.second ?: 0f,
            )
        }
    }

    /**
     * Construit les eventlets en groupant les SvoSpans par verbe gouverneur.
     *
     * Algorithme :
     *  1. Grouper tous les spans argumentaux par [govVerbCharStart] (verb pointer).
     *  2. Pour chaque svo_verb, collecter ses arguments pointés + orphans positionnels.
     *  3. Pour chaque slot, matcher la meilleure entité NER par taux de recouvrement.
     *  4. Pronoms sans entité → EventletSlot(resolved=false) pour la coref async.
     *  5. Négation : un span "neg" situé dans ±50 chars du verbe → negated=true.
     *
     * Les eventlets sont indépendants et non-destructifs : un pronom non résolu reste
     * dans le graph avec resolved=false jusqu'à la passe de coref (rejouable).
     */
    fun eventlets(): List<Eventlet> {
        // ── Labels v4 gold ────────────────────────────────────────────────────
        // Verbes : synLabel == "verb_trigger"  (role est NONE sur les verb spans)
        // Args nominaux : role in {SUBJECT, OBJECT, OBLIQUE, OBLIQUE_AGENT, OBLIQUE_CAUSE, APPOS}
        // Args pronominaux : synLabel in {pron_subj, pron_obj}
        val SLOT_ROLES = setOf(
            "SUBJECT", "OBJECT", "OBLIQUE", "OBLIQUE_AGENT", "OBLIQUE_CAUSE", "APPOS",
            "pron_subj", "pron_obj",
        )
        val PRON_ROLES = setOf("pron_subj", "pron_obj")

        // v4 : les verbes sont identifiés par synLabel == "verb_trigger" ET role == "NONE".
        // Après normalisation dans NerService, les vrais verbes ont role=NONE ;
        // les arguments NP ont aussi synLabel=verb_trigger (artefact v4) mais role != NONE.
        // Sans ce filtre sur role, les args NP apparaissent comme des verbes dans les eventlets.
        val verbSpans = svoSpans.filter { it.synLabel == "verb_trigger" && it.role == "NONE" }.sortedBy { it.charStart }
        // v4 : pas de label "neg" → pas de négation détectée pour l'instant
        val negSpans: List<SvoSpan> = emptyList()

        // Construire un EventletSlot en matchant l'entité NER la plus longue parmi celles
        // qui overlappent le span SVO. Quand des entités sont imbriquées (ex : "2026" dans
        // "sommet mondial Earth Summit 2026"), on préfère la plus longue — elle est plus
        // informative et correspond généralement mieux à la tête sémantique du groupe nominal.
        fun mkSlot(svo: SvoSpan): EventletSlot {
            val svoLen = (svo.charEnd - svo.charStart).coerceAtLeast(1)
            val candidates = entities.mapNotNull { e ->
                val overlap = minOf(svo.charEnd, e.span.end) - maxOf(svo.charStart, e.span.start)
                if (overlap > 0) Pair(e, overlap.toFloat() / svoLen) else null
            }
            // Parmi les entités en overlap, prendre la plus longue (span la plus étendue).
            // Si plusieurs ont la même longueur, tiebreak sur le score NER.
            val bestNer = candidates.maxWithOrNull(
                compareBy(
                    { it.first.span.end - it.first.span.start },          // 1. plus longue
                    { (it.first.metadata["score"] as? Float) ?: 0f },     // 2. score NER le plus élevé
                )
            )

            val isPron    = svo.synLabel in PRON_ROLES
            val ratio     = bestNer?.second ?: 0f
            return EventletSlot(
                svoSpan      = svo,
                nerEntity    = bestNer?.first,
                overlapRatio = ratio,
                confidence   = svo.roleProb * if (bestNer != null) ratio else 1f,
                resolved     = bestNer != null || !isPron,
            )
        }

        // Résolution du verbe gouverneur pour un argument SVO.
        fun resolveVerb(svo: SvoSpan): SvoSpan? {
            if (verbSpans.isEmpty()) return null
            val ptr = svo.govVerbCharStart
            if (ptr != null) {
                val containing = verbSpans.firstOrNull { v -> ptr >= v.charStart && ptr < v.charEnd }
                if (containing != null) return containing
                return verbSpans.minByOrNull { abs(ptr - it.charStart) }
            }
            val svoCenter = (svo.charStart + svo.charEnd) / 2
            return verbSpans.minByOrNull { v -> abs(svoCenter - (v.charStart + v.charEnd) / 2) }
        }

        // Grouper les spans argumentaux par verbe résolu
        val argSpans = svoSpans.filter { it.role in SLOT_ROLES || it.synLabel in PRON_ROLES }
        val byVerb: Map<Int, List<SvoSpan>> = argSpans
            .groupBy { resolveVerb(it)?.charStart ?: -1 }
            .filterKeys { it >= 0 }

        // NMS intra-rôle : parmi des spans qui se chevauchent pour le même rôle,
        // garder seulement le plus confiant (roleProb). Évite d'avoir 3× le même arg
        // quand "Fondation Horizon", "Fondation" et "Horizon" sont tous OBLIQUE.
        // NMS sur les SvoSpans : supprime les spans qui se chevauchent (garde le plus confiant).
        fun nmsSpans(spans: List<SvoSpan>): List<SvoSpan> {
            val kept = mutableListOf<SvoSpan>()
            for (cand in spans.sortedByDescending { it.roleProb }) {
                val overlaps = kept.any { k ->
                    minOf(cand.charEnd, k.charEnd) - maxOf(cand.charStart, k.charStart) > 0
                }
                if (!overlaps) kept += cand
            }
            return kept
        }

        // Dédupe par entité NER : si deux slots résolvent la même entité NER (même charStart+charEnd),
        // on garde celui avec la meilleure confidence. Évite les doublons quand des spans adjacents
        // ("trois" + "suspects") se retrouvent tous les deux mappés sur la même entité NER
        // ("trois suspects") après mkSlot — nmsSpans ne les capturait pas car non-chevauchants.
        fun dedupeByNer(slots: List<EventletSlot>): List<EventletSlot> {
            data class NerKey(val start: Int, val end: Int)
            val seen = mutableMapOf<NerKey, EventletSlot>()
            for (slot in slots) {
                val nerEntity = slot.nerEntity
                if (nerEntity != null) {
                    val key = NerKey(nerEntity.span.start, nerEntity.span.end)
                    val prev = seen[key]
                    if (prev == null || slot.confidence > prev.confidence) seen[key] = slot
                } else {
                    // Pas d'entité NER : on garde le slot tel quel (pas de dédupe possible)
                    // On l'insère avec une clé unique basée sur la position SVO
                    val key = NerKey(slot.svoSpan.charStart, slot.svoSpan.charEnd)
                    val prev = seen[key]
                    if (prev == null || slot.confidence > prev.confidence) seen[key] = slot
                }
            }
            // Préserver l'ordre original (tri par charStart)
            return seen.values.sortedBy { it.svoSpan.charStart }
        }

        return verbSpans.map { verb ->
            val all = byVerb[verb.charStart] ?: emptyList()

            val negated = negSpans.any { neg ->
                neg.charStart in (verb.charStart - 50)..(verb.charEnd + 20)
            }

            Eventlet(
                verb        = verb,
                voice       = verb.voice,
                negated     = negated,
                // v4 : utiliser uniquement le champ `role` (role_head) pour discriminer subject/obj.
                // Le synLabel (verb_trigger/pron_subj/pron_obj) est le type syntaxique détecté par la
                // syn_head, qui peut classifier des NP nominaux comme "pron_obj" ou "pron_subj" par
                // confusion — en particulier avec tauSvoRoleForced actif (spans forcés).
                // La role_head est le signal fiable en v4 : SUBJECT/OBJECT/OBLIQUE/… toujours assigné.
                subject     = all.filter { it.role == "SUBJECT" }
                                 .maxByOrNull { it.roleProb }?.let { mkSlot(it) },
                obj         = all.filter { it.role == "OBJECT" }
                                 .maxByOrNull { it.roleProb }?.let { mkSlot(it) },
                iobjs       = dedupeByNer(nmsSpans(all.filter { it.role in setOf("OBLIQUE", "OBLIQUE_AGENT") }).map { mkSlot(it) }),
                tcomps      = emptyList(),
                lcomps      = emptyList(),
                causes      = dedupeByNer(nmsSpans(all.filter { it.role == "OBLIQUE_CAUSE" }).map { mkSlot(it) }),
                appositions = nmsSpans(all.filter { it.role == "APPOS" }).map { mkSlot(it) },
            )
        }
    }
}

data class SvoTriplet(
    val subject: SvoSpan?,
    val verb: SvoSpan,
    val obj: SvoSpan?,
)

// ─────────────────────────────────────────────────────────────────────────────
// Eventlet — structure event-centrique (NER × rôle SVO groupé par verbe)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Slot d'un eventlet : un argument SVO avec l'entité NER qui le remplit (si détectée).
 *
 * [svoSpan]     : le span SVO argumental (svo_subject, svo_object, svo_iobj, etc.)
 * [nerEntity]   : l'entité NER qui overlap le mieux ce span (null si pronom non résolu
 *                 ou si aucune entité NER ne couvre ce span)
 * [overlapRatio]: fraction du span SVO couverte par l'entité NER (0..1)
 * [confidence]  : roleProb × overlap si NER présent ; roleProb seul si pronom attendu
 * [resolved]    : true si l'entité NER est connue OU si ce n'est pas un pronom ;
 *                 false pour un pronom sans antécédent → à résoudre par coref async
 */
data class EventletSlot(
    val svoSpan: SvoSpan,
    val nerEntity: Entity?,
    val overlapRatio: Float,
    val confidence: Float,
    val resolved: Boolean,
)

/**
 * Eventlet : événement élémentaire extrait à la phrase, centré sur un verbe trigger.
 *
 * Tous les slots sont groupés par [govVerbCharStart] (verb pointer du modèle) — si le
 * pointer est absent (silver legacy), une heuristique positionnelle prend le relais.
 *
 * Les slots dont [resolved]=false sont des pronoms non encore liés à une entité nommée ;
 * la coref asynchrone (graph Neo4j) les résoudra de façon non-destructive.
 *
 * [verb]       : span svo_verb (trigger)
 * [voice]      : ACTIVE | PASSIVE
 * [negated]    : true si un span neg porte sur ce verbe
 * [subject]    : slot svo_subject / pron_subj (agent en actif, patient en passif)
 * [obj]        : slot svo_object / pron_obj / attr
 * [iobjs]      : obliques prépositionnels (svo_iobj) — peut être multiple
 * [tcomps]     : compléments de temps (svo_tcomp)
 * [lcomps]     : compléments de lieu (svo_lcomp)
 * [causes]     : propositions / GN causaux (svo_cause)
 * [appositions]: appositions NE→rôle issues d'un participant (ent_appos)
 */
data class Eventlet(
    val verb: SvoSpan,
    val voice: String,
    val negated: Boolean,
    val subject: EventletSlot?,
    val obj: EventletSlot?,
    val iobjs: List<EventletSlot>,
    val tcomps: List<EventletSlot>,
    val lcomps: List<EventletSlot>,
    val causes: List<EventletSlot>,
    val appositions: List<EventletSlot>,
) {
    /** Tous les slots dans l'ordre canonique (sujet, objet, obliques, temps, lieu, causes). */
    val allSlots: List<EventletSlot>
        get() = listOfNotNull(subject, obj) + iobjs + tcomps + lcomps + causes + appositions

    /** true si au moins un slot est un pronom encore non résolu (attend la coref async). */
    val hasUnresolvedMentions: Boolean
        get() = allSlots.any { !it.resolved }
}

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
    /**
     * Seuil de roleProb pour le chemin SVO "forcé" : quand une entité NER a un bon score
     * mais que svoBoundaryProb est en-dessous de tauSvoBoundary, on score quand même la
     * tête SVO et on retient le rôle si roleProb ≥ tauSvoRoleForced.
     * Capture notamment les agents passifs "par le président" (p_bnd ~0.20, roleProb ~0.99).
     * 0f ou null → chemin forcé désactivé (comportement historique).
     */
    val tauSvoRoleForced: Float? = null,
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
     * Genre, nombre et personne grammaticale lus depuis les têtes morpho (mêmes logits).
     * Disponibles indépendamment du SVO boundary — features morphologiques de l'entité.
     */
    val svoGender: String? = null,
    val svoNumber: String? = null,
    val svoPerson: String? = null,
    /**
     * true si cette entité a été promue par la tête SVO (boundary NER sous le seuil normal
     * mais au-dessus du seuil abaissé tauSvoAnchoredBoundary) sur un span argumental.
     * Les entités svoAnchored sont moins certaines côté NER.
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
    /**
     * Chemin SVO "forcé" : pour les entités NER avec un bon score (pBoundary ≥ tauBoundary déjà
     * garanti), si svoBoundaryProb < tauSvoBoundary, on score quand même la tête de rôle SVO
     * et on retient le résultat si roleProb ≥ tauSvoRoleForced.
     * Capture les agents passifs "par le président" (p_bnd ~0.20, roleProb ~0.99).
     * 0f → désactivé (comportement historique).
     */
    private val tauSvoRoleForced: Float = 0f,
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
    private val tlBufSyn      = ThreadLocal.withInitial { FloatArray(SYN_LABELS.size)    }
    private val tlBufVoice    = ThreadLocal.withInitial { FloatArray(VOICE_LABELS.size)  }
    private val tlBufGender   = ThreadLocal.withInitial { FloatArray(GENDER_LABELS.size) }
    private val tlBufNumber   = ThreadLocal.withInitial { FloatArray(NUMBER_LABELS.size) }
    private val tlBufPerson   = ThreadLocal.withInitial { FloatArray(PERSON_LABELS.size) }

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
        val effTauSvoRoleForced    = overrides?.tauSvoRoleForced       ?: tauSvoRoleForced
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
            val boundaryFlat:   FloatBuffer,  // [N * 2]
            val coarseFlat:     FloatBuffer,  // [N * nCoarse]
            val fineFlat:       FloatBuffer,  // [N * nFine]
            val svoBndFlat:     FloatBuffer?, // [N * 2]
            val synFlat:        FloatBuffer?, // [N * 3] verb_trigger/pron_subj/pron_obj  (v4 models)
            val roleFlat:       FloatBuffer?, // [N * 7] SUBJECT/OBJECT/OBLIQUE...       (v4 models)
            val voiceFlat:      FloatBuffer?, // [N * 2]
            val certaintyFlat:  FloatBuffer?, // [N * 3] certain/modal/denied
            val genderFlat:     FloatBuffer?, // [N * 3]
            val numberFlat:     FloatBuffer?, // [N * 2]
            val personFlat:     FloatBuffer?, // [N * 3]
            val verbPtrFlat:    FloatBuffer?, // [N * seqLen] — verb pointer logits
            // ── v3 backward-compat ────────────────────────────────────────────
            // Old models exported a single combined svo_logits [N * nSvoLegacy]
            // instead of the separate syn_logits + role_logits used by v4 models.
            // When present (and synFlat is absent) the v3 labels are mapped to v4
            // synLabel / roleName during decoding.
            val svoLegacyFlat:  FloatBuffer?, // [N * nSvoLegacy] — v3 compat: svo_logits
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
                    svoBndFlat    = flatBuf("svo_boundary_logits"),
                    synFlat       = flatBuf("syn_logits"),
                    roleFlat      = flatBuf("role_logits"),
                    voiceFlat     = flatBuf("voice_logits"),
                    certaintyFlat = flatBuf("certainty_logits"),
                    genderFlat    = flatBuf("gender_logits"),
                    numberFlat    = flatBuf("number_logits"),
                    personFlat    = flatBuf("person_logits"),
                    verbPtrFlat   = flatBuf("verb_ptr_logits"),
                    svoLegacyFlat = flatBuf("svo_logits"),        // v3 compat
                )
            }
            val msOnnx = ms(tOnnxRun)
            listOf(tInputIds, tAttMask, tStarts, tEnds, tBatchIds).forEach { it.close() }

            // ── 3d. Décodage par span ───────────────────────────────────────
            val tDecode = System.nanoTime()
            val rawByLocal: Array<MutableList<SpanResult>>    = Array(batchSize) { mutableListOf() }
            val svoByLocal: Array<MutableList<RawSvoResult>>  = Array(batchSize) { mutableListOf() }

            val nCoarse  = COARSE_LABELS.size
            val nFine    = FINE_LABELS.size
            val nSyn     = SYN_LABELS.size
            val nRole    = ROLE_LABELS.size
            val nVoice   = VOICE_LABELS.size
            val nCertainty = CERTAINTY_LABELS.size
            val nGender  = GENDER_LABELS.size
            val nNumber  = NUMBER_LABELS.size
            val nPerson  = PERSON_LABELS.size
            // seqLen du bucket courant (verb pointer a [N * seqLen] éléments)
            val nSeqLen  = bucketMaxLen
            // v3 compat: svo_logits width (0 when absent, i.e. v4 model)
            val nSvoLegacy = if (onnxOut.svoLegacyFlat != null && N > 0)
                onnxOut.svoLegacyFlat.capacity() / N else 0

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
                        var nerPerson:     String? = null
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
                            // ── Personne grammaticale (toujours lu si la tête est disponible) ─
                            nerPerson = onnxOut.personFlat?.let {
                                val p = tlBufPerson.get()
                                loadRow(it, k * nPerson, p); softmaxInto(p, p)
                                var pi = 0; for (j in 1 until nPerson) if (p[j] > p[pi]) pi = j
                                PERSON_LABELS.getOrElse(pi) { "NONE" }.takeUnless { s -> s == "NONE" }
                            }
                            // ── Rôle SVO (uniquement si boundary SVO suffisant) ─────────────
                            // Note : le chemin "forcé" (faible p_bnd mais roleProb ≥ tauSvoRoleForced)
                            // est géré dans le bloc SVO standalone ci-dessous, qui lit les logits
                            // SVO aux BONNES bornes token (span SVO ≠ span NER pour le même texte).
                            val svoBndLocal = onnxOut.svoBndFlat
                            if (svoBndLocal != null && onnxOut.roleFlat != null) {
                                val pSvo = softmaxProbFlat(svoBndLocal, k * 2, 2, 1)
                                if (pSvo >= effTauSvoBoundary) {
                                    // Lire le rôle directement
                                    val pRole = FloatArray(nRole)
                                    loadRow(onnxOut.roleFlat, k * nRole, pRole); softmaxInto(pRole, pRole)
                                    var ri = 0
                                    for (j in 1 until nRole) if (pRole[j] > pRole[ri]) ri = j
                                    val roleName = ROLE_LABELS.getOrElse(ri) { "NONE" }
                                    // Ignorer NONE et les rôles non argumentaux — on veut les vrais arguments
                                    if (roleName !in setOf("NONE") && pRole[ri] > 0.3f) {
                                        nerSvoRole     = roleName
                                        nerSvoRoleProb = pRole[ri]
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
                                    svoPerson        = nerPerson,
                                )
                            }
                        }
                    }
                }

                // ── SVO (si têtes disponibles dans ce modèle ONNX) ───────────
                val svoBndFlat = onnxOut.svoBndFlat
                if (svoBndFlat != null) {
                    val pSvoB = softmaxProbFlat(svoBndFlat, k * 2, 2, 1)
                    // Chemin standard  : boundary SVO ≥ tauSvoBoundary.
                    // Chemin forcé     : boundary SVO faible MAIS roleProb ≥ tauSvoRoleForced.
                    //   Capture les agents passifs "par X" (p_bnd ~0.20, roleProb ~0.99).
                    //   Ces spans sont émis dans svoByLocal ; reconcile Phase 2 les snappera
                    //   sur l'entité NER voisine — ce qui évite le problème de décalage de bornes
                    //   token entre le span NER et le span SVO pour le même mot.
                    val aboveSvoBnd = pSvoB >= effTauSvoBoundary
                    // Chemin forcé : SVO boundary faible MAIS roleProb ≥ tauSvoRoleForced.
                    // Pré-condition : le span doit aussi passer le NER boundary (pBoundary ≥ tauBoundary).
                    // Sans cette garde, des spans verbe avec pBoundary ~0 mais roleProb élevé
                    // par confusion du modèle fuitent comme arguments → bruit pur.
                    val trySvoForced = !aboveSvoBnd && effTauSvoRoleForced > 0f && pBoundary >= effTauBoundary
                    if (aboveSvoBnd || trySvoForced) {
                        val synLabel: String
                        val synProb: Float
                        if (onnxOut.synFlat != null) {
                            // v4 model path: dedicated syn_logits head
                            val p = FloatArray(nSyn)
                            loadRow(onnxOut.synFlat, k * nSyn, p); softmaxInto(p, p)
                            var si = 0; for (j in 1 until nSyn) if (p[j] > p[si]) si = j
                            synLabel = SYN_LABELS.getOrElse(si) { "verb_trigger" }; synProb = p[si]
                        } else if (onnxOut.svoLegacyFlat != null && nSvoLegacy > 0) {
                            // v3 compat path: combined svo_logits — map label index to v4 synLabel
                            // v3 label 0 (svo_verb) → verb_trigger; 10/12 (pron_subj/pron_dem) → pron_subj;
                            // 11 (pron_obj) → pron_obj; all NP args (1-9) → verb_trigger (artefact, like v4).
                            val p = FloatArray(nSvoLegacy)
                            loadRow(onnxOut.svoLegacyFlat, k * nSvoLegacy, p); softmaxInto(p, p)
                            var li = 0; for (j in 1 until nSvoLegacy) if (p[j] > p[li]) li = j
                            synLabel = when (li) {
                                10, 12 -> "pron_subj"
                                11     -> "pron_obj"
                                else   -> "verb_trigger"
                            }; synProb = p[li]
                        } else { synLabel = "verb_trigger"; synProb = 0f }

                        // Architecture v4 — arbitrage synLabel vs svo_boundary :
                        // • synLabel == "verb_trigger" AND aboveSvoBnd → vrai verbe, role=NONE forcé
                        //   (le role_head est confus sur les verbes, prédit OBJECT/SUBJECT par erreur)
                        // • synLabel == "pron_subj" | "pron_obj" → pronom, même si p_svob est élevé
                        //   (svo_boundary peut faussement tirer sur des relatifs/"qui" etc.)
                        //   → On lit role depuis role_head et on applique le seuil forcé.
                        // • Chemin forcé classique (aboveSvoBnd=false) → role depuis role_head.
                        val isActualVerb = aboveSvoBnd && synLabel == "verb_trigger"
                        val roleName: String
                        val roleProb: Float
                        if (isActualVerb) {
                            // Vrai verbe → rôle vaut NONE par définition
                            roleName = "NONE"; roleProb = 0f
                        } else if (onnxOut.roleFlat != null) {
                            // v4 model path: dedicated role_logits head
                            val p = FloatArray(nRole)
                            loadRow(onnxOut.roleFlat, k * nRole, p); softmaxInto(p, p)
                            var ri = 0; for (j in 1 until nRole) if (p[j] > p[ri]) ri = j
                            roleName = ROLE_LABELS.getOrElse(ri) { "NONE" }; roleProb = p[ri]
                        } else if (onnxOut.svoLegacyFlat != null && nSvoLegacy > 0) {
                            // v3 compat path: derive role from combined svo_logits argmax
                            // v3 label → v4 ROLE_LABELS mapping:
                            //   0 svo_verb         → NONE (handled by isActualVerb above)
                            //   1 svo_subject       → SUBJECT
                            //   2 svo_object        → OBJECT
                            //   3 svo_iobj          → OBLIQUE
                            //   4 svo_tcomp         → OBLIQUE
                            //   5 svo_lcomp         → OBLIQUE
                            //   6 svo_cause         → OBLIQUE_CAUSE
                            //   7 attr / 8 nom_event→ SUBJECT
                            //   9 ent_appos         → APPOS
                            //  10 pron_subj/12 dem  → SUBJECT
                            //  11 pron_obj          → OBJECT
                            //  13 neg               → NONE
                            val p = FloatArray(nSvoLegacy)
                            loadRow(onnxOut.svoLegacyFlat, k * nSvoLegacy, p); softmaxInto(p, p)
                            var li = 0; for (j in 1 until nSvoLegacy) if (p[j] > p[li]) li = j
                            roleName = when (li) {
                                1             -> "SUBJECT"       // svo_subject
                                2             -> "OBJECT"        // svo_object
                                3, 4, 5       -> "OBLIQUE"       // svo_iobj, svo_tcomp, svo_lcomp
                                6             -> "OBLIQUE_CAUSE" // svo_cause
                                7, 8, 10, 12  -> "SUBJECT"       // attr, nom_event, pron_subj, pron_dem
                                9             -> "APPOS"         // ent_appos
                                11            -> "OBJECT"        // pron_obj
                                else          -> "NONE"          // svo_verb (0) or unknown
                            }; roleProb = p[li]
                        } else { roleName = "NONE"; roleProb = 0f }

                        // Guard : pronoms et args forcés doivent passer le seuil roleProb.
                        // isActualVerb est exempté (il émet toujours avec role=NONE).
                        if (!isActualVerb && (roleProb < effTauSvoRoleForced || roleName == "NONE")) {
                            // roleProb insuffisant ou role=NONE → on n'émet pas ce span
                        } else {

                        val voiceName: String
                        val voiceProb: Float
                        if (onnxOut.voiceFlat != null) {
                            val p = FloatArray(nVoice)
                            loadRow(onnxOut.voiceFlat, k * nVoice, p); softmaxInto(p, p)
                            var vi = 0; for (j in 1 until nVoice) if (p[j] > p[vi]) vi = j
                            voiceName = VOICE_LABELS.getOrElse(vi) { "active" }; voiceProb = p[vi]
                        } else { voiceName = "active"; voiceProb = 0f }

                        val certaintyName: String
                        val certaintyProb: Float
                        if (onnxOut.certaintyFlat != null) {
                            val p = FloatArray(nCertainty)
                            loadRow(onnxOut.certaintyFlat, k * nCertainty, p); softmaxInto(p, p)
                            var ci = 0; for (j in 1 until nCertainty) if (p[j] > p[ci]) ci = j
                            certaintyName = CERTAINTY_LABELS.getOrElse(ci) { "certain" }; certaintyProb = p[ci]
                        } else { certaintyName = "certain"; certaintyProb = 0f }

                        val gender: String? = onnxOut.genderFlat?.let {
                            val p = FloatArray(nGender)
                            loadRow(it, k * nGender, p); softmaxInto(p, p)
                            var gi = 0; for (j in 1 until nGender) if (p[j] > p[gi]) gi = j
                            GENDER_LABELS.getOrElse(gi) { "N" }.takeUnless { s -> s == "N" }
                        }
                        val number: String? = onnxOut.numberFlat?.let {
                            val p = FloatArray(nNumber)
                            loadRow(it, k * nNumber, p); softmaxInto(p, p)
                            var ni = 0; for (j in 1 until nNumber) if (p[j] > p[ni]) ni = j
                            NUMBER_LABELS.getOrElse(ni) { "SG" }
                        }
                        val person: String? = onnxOut.personFlat?.let {
                            val p = FloatArray(nPerson)
                            loadRow(it, k * nPerson, p); softmaxInto(p, p)
                            var pi = 0; for (j in 1 until nPerson) if (p[j] > p[pi]) pi = j
                            PERSON_LABELS.getOrElse(pi) { "3" }
                        }
                        // ── Verb pointer : tok argmax → charStart du verbe gouverneur ──
                        // Supervisé sur tous les arguments, qu'ils soient pron_subj/pron_obj OU NP args.
                        // En v4, les args NP ont synLabel=verb_trigger (artefact) mais ne sont PAS des
                        // vrais verbes (isActualVerb=false) → on lit quand même le verb pointer.
                        // Seuls les vrais verbes (isActualVerb=true) n'ont pas de verbe gouverneur.
                        val govVerbCharStart: Int? = if (!isActualVerb && onnxOut.verbPtrFlat != null) {
                            val buf    = onnxOut.verbPtrFlat
                            val offset = k * nSeqLen
                            var bestTok = 0; var bestVal = buf.get(offset)
                            for (j in 1 until nSeqLen) {
                                val v = buf.get(offset + j)
                                if (v > bestVal) { bestVal = v; bestTok = j }
                            }
                            // Convertir tok → charStart via charOffsets de l'exemple local
                            subEncodings[cand.exampleIdx].charOffsets.getOrNull(bestTok)?.first
                        } else null

                        svoByLocal[cand.exampleIdx] += RawSvoResult(
                            candidate        = cand,
                            svoBoundaryProb  = pSvoB,
                            synLabel         = synLabel,
                            synProb          = synProb,
                            role             = roleName,
                            roleProb         = roleProb,
                            voice            = voiceName,
                            voiceProb        = voiceProb,
                            certainty        = certaintyName,
                            certaintyProb    = certaintyProb,
                            gender           = gender,
                            number           = number,
                            person           = person,
                            govVerbCharStart  = govVerbCharStart,
                        )

                        // ── SVO-anchored NER ──────────────────────────────────────────
                        // Si le span porte un rôle argumental (v4 labels) ET que NER boundary
                        // n'a pas tiré au seuil normal mais dépasse le seuil abaissé → on score
                        // quand même la tête NER. Ces entités sont taguées svoAnchored=true.
                        // Note v4 : les roles pronominaux (pron_subj/pron_obj) sont portés par
                        // synLabel, pas roleName → on exclut NONE et verbes (role forcé NONE ci-dessus).
                        val isNonPronounArg = roleName in setOf(
                            "SUBJECT", "OBJECT", "OBLIQUE", "OBLIQUE_AGENT", "OBLIQUE_CAUSE", "APPOS",
                        )
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
                                            svoPerson        = person,
                                            svoAnchored      = true,
                                        )
                                    }
                                }
                            }
                        }
                        } // end else (roleProb sufficient)
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
                    .let { spans ->
                        // Résolution verb pointer : charStart → texte du verbe gouverneur
                        val verbByCharStart = spans
                            .filter { it.synLabel == "verb_trigger" }
                            .associateBy { it.charStart }
                        spans.map { s ->
                            val govText = s.govVerbCharStart?.let { verbByCharStart[it]?.text }
                            if (govText != null) s.copy(govVerbText = govText) else s
                        }
                    }
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

    /**
     * Équivalent de Python str.isalnum() pour un caractère unique :
     * inclut les lettres, les chiffres décimaux ET les autres nombres Unicode
     * (catégorie No — exposants ², ³, ¹, ⁰, ⁴…⁹, fractions ½, ¼…).
     * En Java/Kotlin, Char.isLetterOrDigit() omet la catégorie No, ce qui
     * entraîne un mismatch avec Python pour les bornes de span.
     */
    private fun Char.isWordEndChar(): Boolean =
        isLetterOrDigit() ||
        category == CharCategory.OTHER_NUMBER ||   // ², ³, ¹, ½, ¼… (No)
        category == CharCategory.LETTER_NUMBER     // Ⅰ, Ⅱ… numéraux romains (Nl)

    /**
     * Ponctuations de fin de phrase/syntagme qu'il est sûr de stripper en queue de span.
     * Exclut délibérément les symboles d'unités %, °, €, $, £, ‰ etc.
     * Python (code d'entraînement) ne trimme jamais — il génère des candidats à la
     * granularité du token. Ce whitelist mimique ce comportement au niveau mot :
     * seule la ponctuation grammaticale terminale est retirée.
     */
    private val PHRASE_FINAL_PUNCT = setOf(
        '.', ',', ';', ':', '!', '?',
        ')', ']', '}',
        '"', '\u201C', '\u201D',   // guillemets anglais
        '\u2019',                  // apostrophe droite / guillemet fermant
        '\u00BB', '\u00AB',        // guillemets français » «
        '\u2014', '\u2013',        // tirets em/en
    )

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

                    // Trimmer la ponctuation grammaticale finale (. , ; ! ? etc.) ET reculer
                    // tokEnd en conséquence. On utilise PHRASE_FINAL_PUNCT (whitelist) et
                    // NON pas !isLetterOrDigit() ni !isWordEndChar() pour ne PAS stripper
                    // les symboles d'unités : %, °, €, $ restent dans le span.
                    // Python (entraînement) n'a pas de trim — il génère des candidats par token.
                    while (spanTxt.isNotEmpty() && spanTxt.last() in PHRASE_FINAL_PUNCT) {
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
                                if (tok.isNotEmpty() && tok.all { it in PHRASE_FINAL_PUNCT }) tokEnd--
                                else break
                            }
                        }
                        charEnd = newCharEnd
                    }

                    if (spanTxt.length < 2) continue
                    if (spanTxt.all { !it.isWordEndChar() }) continue
                    if (charStart > 0 && text[charStart - 1].isWordEndChar()) continue
                    if (charEnd < text.length && text[charEnd].isWordEndChar()) continue

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
            // svoRole        : rôle SVO v4 (SUBJECT, OBJECT, OBLIQUE, OBLIQUE_AGENT, OBLIQUE_CAUSE, APPOS)
            // syntacticRole  : normalisation UD ("nsubj", "obj", "obl", "obl:agent", "obl:cause", "appos")
            if (r.span.svoRole != null) {
                val syntactic = when (r.span.svoRole) {
                    // Nouveaux labels v4
                    "SUBJECT"        -> "nsubj"
                    "OBJECT"         -> "obj"
                    "OBLIQUE"        -> "obl"
                    "OBLIQUE_AGENT"  -> "obl:agent"
                    "OBLIQUE_CAUSE"  -> "obl:cause"
                    "APPOS"          -> "appos"
                    // Compat anciens labels (si modèle ancien)
                    "svo_subject", "pron_subj" -> "nsubj"
                    "svo_object",  "pron_obj"  -> "obj"
                    "svo_iobj"                 -> "iobj"
                    "attr"                     -> "attr"
                    "ent_appos"                -> "appos"
                    "nom_event"                -> "nmod"
                    "svo_tcomp"                -> "obl:tmod"
                    "svo_lcomp"                -> "obl:lmod"
                    "svo_cause"                -> "obl:caus"
                    else                       -> null
                }
                put("svoRole",         r.span.svoRole)
                put("svoRoleProb",     r.span.svoRoleProb)
                put("svoBoundaryScore",r.span.svoBoundaryScore)
                if (syntactic != null) put("syntacticRole", syntactic)
            }
            // ── Morphologie (genre / nombre / personne) — lus indépendamment du rôle SVO ───
            r.span.svoGender?.let { put("gender", it) }
            r.span.svoNumber?.let { put("number", it) }
            r.span.svoPerson?.let { put("person", it) }
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
        text             = r.candidate.spanText,
        charStart        = r.candidate.charStart,
        charEnd          = r.candidate.charEnd,
        synLabel         = r.synLabel,
        synProb          = r.synProb,
        role             = r.role,
        roleProb         = r.roleProb,
        svoBoundaryProb  = r.svoBoundaryProb,
        voice            = r.voice,
        voiceProb        = r.voiceProb,
        certainty        = r.certainty,
        certaintyProb    = r.certaintyProb,
        gender           = r.gender,
        number           = r.number,
        person           = r.person,
        govVerbCharStart = r.govVerbCharStart,
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

