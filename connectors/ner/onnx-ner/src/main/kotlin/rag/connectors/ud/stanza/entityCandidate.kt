package rag.connectors.ud.stanza

import rag.model.*

// ─────────────────────────────────────────────────────────────────────────────
// NER coarse — 6 types (issu du modèle XLM-RoBERTa fine-tuné)
// ─────────────────────────────────────────────────────────────────────────────

enum class NerCoarseType {
    PER, LOC, ORG, TIME, EVENT, OBJECT, UNKNOWN;

    companion object {
        fun from(s: String): NerCoarseType = when (s.trim().uppercase()) {
            "PER"    -> PER
            "LOC"    -> LOC
            "ORG"    -> ORG
            "TIME"   -> TIME
            "EVENT"  -> EVENT
            "OBJECT" -> OBJECT
            else     -> UNKNOWN
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constantes hop
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Valeur sentinelle : la tête du candidat n'est pas reliée au trigger
 * dans l'arbre UD en moins de MAX_HOP arcs, ou aucun trigger n'a encore
 * été associé à ce candidat.
 */
const val HOP_UNREACHABLE = 99

// ─────────────────────────────────────────────────────────────────────────────
// EntityCandidate — type unifié pour le pipeline PK LR
// (Parsing · Knowledge-linking · co-Reference resolution)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Candidat entité agrégé depuis les trois étapes du pipeline :
 *
 *  1. OnnxBilouEntityExtractor  → texte + span + type coarse (6)
 *  2. mergeNerLabelWithUD       → span UD raffiné, tête syntaxique, morphologie
 *  3. OnnxSpanNerExtractor      → hint fin-grained (× 22 EntityType)
 *
 * Contient tout ce qu'il faut pour :
 *   • identifier / normaliser l'entité    (text, lemma, span)
 *   • la catégoriser à deux granularités  (nerType × 6, nerHint × 22)
 *   • résoudre la co-référence            (gender, number, isPron, sentenceSpan)
 *   • la lier à une base de connaissances (isPropn, isName, lemma, head)
 */
data class EntityCandidate(

    // ── Surface ──────────────────────────────────────────────────────────────
    /** Texte exact tel qu'il apparaît dans le document (span raffiné UD). */
    val text: String,

    /**
     * Forme canonique : lemme de la tête syntaxique si disponible,
     * sinon texte de surface en minuscules.
     * Utilisée pour la déduplication et le linking.
     */
    val lemma: String,

    /**
     * Span dans le document (start/end char + tokens UD peuplés).
     * Les tokens permettent le surlignage exact et l'accès aux features morpho.
     */
    val span: Span,

    // ── NER coarse — 6 types ─────────────────────────────────────────────────
    /** Catégorie NER coarse issue d'XLM-RoBERTa : PER / LOC / ORG / TIME / EVENT / OBJECT. */
    val nerType: NerCoarseType,

    // ── NER hint — 22 types fin-grained ──────────────────────────────────────
    /**
     * Type fin-grained issu du SpanClassifier DeBERTa (22 EntityType).
     * Ex : HINT_PERSON_NAME, HINT_LOC_GENERIC, HINT_ORG_NAME,
     *      HINT_TIME_DATE, HINT_EVENT_NOMINAL, HINT_GPE, …
     */
    val nerHint: EntityType,

    /**
     * true  → [nerHint] est un nom propre (HINT_*_NAME) : entité nommée directe.
     * false → rôle, générique ou mention non-propre (HINT_PERSON_ROLE, HINT_LOC_GENERIC…).
     * Utile pour décider si l'entité peut être résolue sans contexte.
     */
    val isName: Boolean,

    // ── Tête syntaxique UD ───────────────────────────────────────────────────
    /** Token tête du syntagme nominal (ancre pour la coreférence et le linking). */
    val head: UDToken?,

    /** POS universel de la tête : NOUN, PROPN, PRON, NUM… */
    val headUpos: UPOS?,

    /** Relation de dépendance de la tête dans la phrase (nsubj, obj, obl:agent…). */
    val headDeprel: String?,

    /** true si la tête est un PROPN (nom propre attesté par le parseur UD). */
    val isPropn: Boolean,

    /** true si la tête est un PRON (mention pronominale → nécessite résolution). */
    val isPron: Boolean,

    // ── Morphologie UD ───────────────────────────────────────────────────────
    /** Genre grammatical de la tête (MASC / FEM / NEUT). Clé pour la coreférence. */
    val gender: GenderValue?,

    /** Nombre grammatical de la tête (SG / PL). Clé pour la coreférence. */
    val number: NumberValue?,

    /** Features UD complètes de la tête (personne, temps, mode, etc.). */
    val feats: UDFeats?,

    // ── Contexte ─────────────────────────────────────────────────────────────
    /**
     * Span de la phrase UD contenant l'entité (start until end dans le document).
     * Utile pour extraire le contexte d'une fenêtre glissante ou pour le scoring LR.
     */
    val sentenceSpan: IntRange,

    // ── Distance au trigger (eventlet) ───────────────────────────────────────
    /**
     * Distance en arcs dans l'arbre UD entre la tête de ce candidat et le token
     * trigger de l'eventlet courant.
     *
     * - 0  : le candidat EST le trigger (rare, auto-référence)
     * - 1  : argument direct du trigger  (nsubj, obj, obl…) — argument noyau
     * - 2  : argument indirect (ex. nmod d'un nsubj, ou arg d'un trigger nominal)
     * - 3+ : périphérique, peu probable comme arg core
     * - [HOP_UNREACHABLE] (99) : non relié ou pas encore calculé (défaut)
     *
     * Calculé à la demande via [withHopFrom] — ne dépend pas de l'extraction NER.
     * Clé pour le vecteur de features du LR argument.
     */
    val hopFromTrigger: Int = HOP_UNREACHABLE,
) {
    /**
     * true si ce candidat est un enfant direct du trigger dans l'arbre UD.
     * Raccourci LR pour la feature booléenne `is_direct_child_of_trigger`.
     */
    val isDirectChildOfTrigger: Boolean get() = hopFromTrigger == 1
}

// ─────────────────────────────────────────────────────────────────────────────
// Hop distance — BFS bidirectionnel sur l'arbre UD
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Distance en arcs entre deux tokens dans l'arbre UD de la phrase.
 *
 * Traverse l'arbre dans **les deux sens** (enfant→parent via `head` ET
 * parent→enfants via scan de la liste) pour trouver le chemin le plus court
 * passant par l'ancêtre commun le plus bas (LCA).
 *
 * Cas typiques pour le LR argument :
 * ```
 * "La police arrêta Martin"
 *   arrêta (trigger, id=3)
 *     ├─ police (nsubj, id=2, head=3) → hop=1  Agent
 *     └─ Martin (obj,   id=4, head=3) → hop=1  Patient
 *
 * "L'arrestation de Martin par la police"
 *   arrestation (trigger nominal, id=2)
 *     ├─ Martin (nmod, id=4, head=2) → hop=1  Patient
 *     └─ police (nmod, id=7, head=2) → hop=1  Agent
 * ```
 *
 * @param fromId         Id UD du token de départ (tête du candidat).
 * @param toId           Id UD du token cible (trigger).
 * @param sentenceTokens Tous les tokens de la phrase UD contenant les deux tokens.
 * @param maxHops        Profondeur maximale de recherche (défaut 5).
 *                       Au-delà → [HOP_UNREACHABLE].
 * @return Distance en arcs, ou [HOP_UNREACHABLE] si non connecté dans maxHops.
 */
fun hopDistance(
    fromId:         Int,
    toId:           Int,
    sentenceTokens: List<UDToken>,
    maxHops:        Int = 5,
): Int {
    if (fromId == toId) return 0

    // Index id → token pour les lookups O(1)
    val byId = sentenceTokens.associateBy { it.id }

    // BFS bidirectionnel : on parcourt parent + enfants directs à chaque nœud
    val visited = mutableMapOf(fromId to 0)
    val queue   = ArrayDeque<Int>().also { it.add(fromId) }

    while (queue.isNotEmpty()) {
        val cur  = queue.removeFirst()
        val dist = visited[cur]!!
        if (dist >= maxHops) continue

        // Voisins = parent (arc montant) + enfants directs (arcs descendants)
        val neighbors = mutableListOf<Int>()
        byId[cur]?.head?.takeIf { it > 0 }?.let { neighbors += it }          // ← parent
        sentenceTokens.filter { it.head == cur }.mapTo(neighbors) { it.id }  // ← enfants

        for (n in neighbors) {
            if (n !in visited) {
                val d = dist + 1
                if (n == toId) return d
                visited[n] = d
                queue.add(n)
            }
        }
    }

    return HOP_UNREACHABLE
}

// ─────────────────────────────────────────────────────────────────────────────
// Extension : associer un candidat à un trigger
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Retourne une copie de ce [EntityCandidate] enrichie avec [hopFromTrigger]
 * calculé depuis le token trigger donné.
 *
 * Usage dans la couche eventlet :
 * ```kotlin
 * val enriched = candidates.map { it.withHopFrom(triggerToken.id, sentence.tokens) }
 * val args     = enriched.filter { it.hopFromTrigger <= 2 }
 * ```
 *
 * @param triggerTokenId Id UD du token trigger de l'eventlet (VERB root ou NOUN nominal).
 * @param sentenceTokens Tokens de la phrase UD contenant trigger et candidat.
 */
fun EntityCandidate.withHopFrom(
    triggerTokenId: Int,
    sentenceTokens: List<UDToken>,
): EntityCandidate {
    val headId = head?.id ?: return copy(hopFromTrigger = HOP_UNREACHABLE)
    return copy(hopFromTrigger = hopDistance(headId, triggerTokenId, sentenceTokens))
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructeur agrégateur
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Construit la liste finale d'[EntityCandidate] depuis les trois sorties du pipeline.
 *
 * La jointure entre [enrichedEntities] et [classifiedSpans] se fait par position
 * (span.start, span.end) → insensible à l'ordre de traitement par phrase.
 *
 * @param enrichedEntities  Sortie de [mergeNerLabelWithUD] :
 *                          Entity avec span UD raffiné, tokens peuplés,
 *                          et metadata (head, headLemma, headUpos, gender, number…).
 * @param classifiedSpans   Sortie de [OnnxSpanNerExtractor.extractFromCandidates] :
 *                          SimpleEntityModel avec nerHint (22 types) pour chaque candidat.
 * @param udDoc             Document UD source (pour retrouver la phrase de chaque entité).
 */
fun buildEntityCandidates(
    enrichedEntities: List<Entity>,
    classifiedSpans:  List<SimpleEntityModel>,
    udDoc:            UDDocument,
): List<EntityCandidate> {

    // Index des spans classifiés par position (start, end) pour la jointure O(1)
    val hintBySpan: Map<Pair<Int, Int>, SimpleEntityModel> =
        classifiedSpans.associateBy { it.start to it.end }

    return enrichedEntities.mapNotNull { entity ->
        val eSpan  = entity.span              ?: return@mapNotNull null
        val eStart = eSpan.start.takeIf { it >= 0 } ?: return@mapNotNull null
        val eEnd   = eSpan.end.takeIf   { it > eStart } ?: return@mapNotNull null

        // ── NER coarse et hint ──────────────────────────────────────────────
        val nerCoarse = NerCoarseType.from(entity.type)
        val classified = hintBySpan[eStart to eEnd] ?: return@mapNotNull null
        val nerHint    = classified?.label ?: coarseToHint(nerCoarse)

        // ── Récupération des infos UD depuis metadata (mergeNerLabelWithUD) ─
        val meta       = entity.metadata
        val headUpos   = (meta["headUpos"]  as? String)?.let { UPOS.from(it) }
        val headDeprel = meta["headDeprel"] as? String
        val rawLemma   = meta["headLemma"]  as? String
        val headId     = meta["headId"]     as? Int

        // Chercher la tête dans les tokens UD peuplés
        val head = when {
            headId != null -> eSpan.tokens.firstOrNull { it.id == headId }
            else           -> eSpan.tokens.firstOrNull { it.text == meta["head"] as? String }
                ?: eSpan.tokens.firstOrNull()
        }

        // Lemme : depuis la tête UD si dispo, sinon metadata, sinon surface
        val lemma = head?.lemma
            ?: rawLemma
            ?: entity.text.lowercase()

        // ── Morphologie ─────────────────────────────────────────────────────
        // Priorité : tête UD → metadata → null
        val gender = head?.feats?.gender
            ?: (meta["gender"] as? String)
                ?.let { g -> GenderValue.entries.firstOrNull { it.name == g } }

        val number = head?.feats?.number
            ?: (meta["number"] as? String)
                ?.let { n -> NumberValue.entries.firstOrNull { it.name == n } }

        // ── Phrase UD contenante ────────────────────────────────────────────
        val sentence = udDoc.sentences.firstOrNull { s ->
            s.start <= eStart && s.end >= eEnd
        }
        val sentenceSpan = if (sentence != null) sentence.start until sentence.end
                           else eStart until eEnd

        EntityCandidate(
            text         = entity.text,
            lemma        = lemma,
            span         = eSpan,
            nerType      = nerCoarse,
            nerHint      = nerHint,
            isName       = nerHint.isName(),
            head         = head,
            headUpos     = headUpos,
            headDeprel   = headDeprel,
            isPropn      = headUpos == UPOS.PROPN,
            isPron       = headUpos == UPOS.PRON,
            gender       = gender,
            number       = number,
            feats        = head?.feats,
            sentenceSpan = sentenceSpan,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers privés
// ─────────────────────────────────────────────────────────────────────────────

/** Dérive un [EntityType] hint depuis le type NER coarse (utilisé comme fallback). */
private fun coarseToHint(t: NerCoarseType): EntityType = when (t) {
    NerCoarseType.PER     -> EntityType.HINT_PERSON_NAME
    NerCoarseType.LOC     -> EntityType.HINT_LOC_GENERIC
    NerCoarseType.ORG     -> EntityType.HINT_ORG_NAME
    NerCoarseType.TIME    -> EntityType.HINT_TIME_DATE
    NerCoarseType.EVENT   -> EntityType.HINT_EVENT_NOMINAL
    NerCoarseType.OBJECT,
    NerCoarseType.UNKNOWN -> EntityType.HINT_OBJECT_GENERIC
}
