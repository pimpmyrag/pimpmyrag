package rag.connectors.ud.stanza

import rag.model.*

/**
 * Fusionne les entités NER label (OnnxBilouEntityExtractor → type + char span)
 * avec l'analyse UD pour produire les candidats les plus précis possible.
 *
 * Pour chaque entité NER :
 *  1. Retrouver la phrase UD qui contient le span.
 *  2. Collecter les tokens UD chevauchant le span NER.
 *  3. Rogner DET/ADP/PUNCT aux extrémités.
 *  4. Identifier la tête syntaxique du groupe.
 *  5. Valider UPOS/type : drop les entités linguistiquement impossibles
 *     (PRON taggé EVENT/LOC/ORG, VERB/ADV/CCONJ… taggés comme entités).
 *  6a. Split PER : si la tête est un NOUN et le span contient des PROPN en
 *      flat:name / flat / name → émettre deux entités séparées :
 *      - le span du rôle (NOUN + ses dépendants hors PROPN flat)
 *      - le span du nom propre (cluster PROPN flat:name)
 *  6b. Span final = span NER rogné (frontières NER conservées).
 *  7. Enrichir Entity.span.tokens + metadata.
 *
 * Si l'entité ne peut pas être alignée sur UD, elle est retournée telle quelle.
 * Retourne une liste vide si l'entité est linguistiquement invalide (artefact BIO).
 *
 * ⚠ La fonction retourne désormais List<Entity> (flatMap) : un candidat NER peut
 * produire 0, 1 ou 2 entités enrichies (cas du split rôle/nom).
 */
fun mergeNerLabelWithUD(
    nerEntities: List<Entity>,
    udDoc: UDDocument
): List<Entity> = nerEntities.flatMap { entity -> enrichOne(entity, udDoc) }

// ─────────────────────────────────────────────────────────────────────────────

/**
 * UPOS qui indiquent un mot grammatical pur — jamais tête d'une entité NER valide.
 * (ADP/DET sont déjà rognés en étape 3, mais un filtre défensif ici ne coûte rien.)
 */
private val FUNCTION_UPOS = setOf(
    UPOS.ADP, UPOS.DET, UPOS.CCONJ, UPOS.SCONJ,
    UPOS.PART, UPOS.INTJ, UPOS.PUNCT, UPOS.AUX
)

/**
 * Deprels UD qui signalent qu'un PROPN est la partie "nom propre"
 * d'un syntagme [TITRE NOUN] + [NOM PROPN].
 * On exclut volontairement "appos" (trop ambigu : "le président, chef de…").
 */
private val FLAT_DEPRELS = setOf("flat", "flat:name", "name")

// ─────────────────────────────────────────────────────────────────────────────

private fun enrichOne(entity: Entity, udDoc: UDDocument): List<Entity> {

    val eStart = entity.span?.start ?: return listOf(entity)
    val eEnd   = entity.span?.end   ?: return listOf(entity)
    if (eStart >= eEnd) return listOf(entity)

    // 1. Phrase UD qui contient entièrement le span NER
    val sentence = udDoc.sentences.firstOrNull { s ->
        s.start <= eStart && s.end >= eEnd
    } ?: return listOf(entity)   // cross-phrase ou hors document → pas de raffinement

    // 2. Tokens UD chevauchant le span NER (chevauchement strict)
    val overlapping = sentence.tokens.filter { t ->
        t.start < eEnd && t.end > eStart
    }
    if (overlapping.isEmpty()) return listOf(entity)

    // 3. Rogner aux extrémités :
    //    - début : PUNCT, DET (articles contractés), ADP (prépositions)
    //    - fin   : PUNCT uniquement
    val skipAtStart = setOf(UPOS.PUNCT, UPOS.DET, UPOS.ADP)
    val trimmed = overlapping
        .dropWhile    { it.upos in skipAtStart }
        .dropLastWhile { it.upos == UPOS.PUNCT }
    if (trimmed.isEmpty()) return listOf(entity)

    val trimStart = trimmed.first().start
    val trimEnd   = trimmed.last().end

    // 4. Tête syntaxique = token dont le parent est hors du groupe (ou root)
    val overlappingIds = overlapping.map { it.id }.toSet()
    val head = trimmed.firstOrNull { t ->
        t.head == 0 || t.head !in overlappingIds
    } ?: trimmed.first()

    // 5. Validation UPOS/type — filtre les artefacts du modèle BIO sans liste hardcodée.
    //
    //    Règle A : les mots purement grammaticaux (conjonctions, particules…)
    //              ne peuvent jamais être la tête d'une entité NER.
    //    Règle B : un PRON ne peut être entité que si le type coarse est PER
    //              (ex. "Il" → PER/HINT_PERSON_ROLE pour coref est valide ;
    //               "se" → EVENT ou "y" → LOC sont des artefacts BIO).
    //    Règle C : un VERB ou ADV en tête d'entité est un artefact
    //              (verbe conjugué ou adverbe taggé à tort).
    if (head.upos in FUNCTION_UPOS) return emptyList()
    if (head.upos == UPOS.PRON  && entity.type.lowercase() != "per") return emptyList()
    if (head.upos == UPOS.VERB  || head.upos == UPOS.ADV) return emptyList()

    // 6a. Règle de split PER : NOUN-head + cluster PROPN flat:name
    //     "général De Gaulle"    → ["général"]      + ["De Gaulle"]
    //     "président Macron"     → ["président"]    + ["Macron"]
    //     "Premier ministre…"    → ["Premier ministre"] + ["Jean Castex"]
    //     "Assemblée nationale"  → NOUN + ADJ → pas de PROPN flat:name → pas de split ✓
    //     "Cour de cassation"    → NOUN + nmod NOUN → pas de PROPN flat:name → pas de split ✓
    if (entity.type.lowercase() == "per" && head.upos == UPOS.NOUN) {
        val split = trySplitRoleAndName(entity, trimmed, head, udDoc)
        if (split != null) return split
    }

    // 6b. Span UD via reconstructSpan (depuis la tête) — uniquement pour les métadonnées
    val udSpan = reconstructSpan(sentence.tokens, head.id)

    // 7. Span final : on fait confiance aux frontières NER (rognées de ponctuation).
    val finalTokens = sentence.tokens
        .filter { it.start >= trimStart && it.end <= trimEnd }
        .dropWhile    { it.upos == UPOS.PUNCT }
        .dropLastWhile { it.upos == UPOS.PUNCT }

    if (finalTokens.isEmpty()) return listOf(entity)

    val finalStart = finalTokens.first().start
    val finalEnd   = finalTokens.last().end

    if (finalEnd > udDoc.text.length || finalStart < 0) return listOf(entity)

    return listOf(
        entity.copy(
            text = udDoc.text.substring(finalStart, finalEnd),
            span = Span(finalStart, finalEnd, finalTokens),
            metadata = buildEntityMeta(entity, head, udDoc, udSpan)
        )
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Split PER : rôle (NOUN) + nom propre (PROPN flat:name)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Tente de décomposer un span PER de la forme [TITRE NOUN] [NOM PROPN] en deux
 * entités distinctes.
 *
 * Algorithme :
 *  1. Collecter les PROPN dont le deprel est dans [flat, flat:name, name] et dont
 *     la tête pointe vers un token *à l'intérieur* du span (NOUN head ou autre PROPN
 *     du cluster déjà identifié).
 *  2. Étendre le cluster transitivement (PROPN → PROPN flat:name → …).
 *  3. Séparer les tokens du span en roleTokens / nameTokens.
 *  4. Construire deux Entity et retourner la paire.
 *
 * Retourne null si le split n'est pas applicable (cluster vide, span non-valide…)
 * → l'appelant continuera avec le chemin normal mono-entité.
 *
 * Cas traités :
 *   "général De Gaulle"         → général | De Gaulle
 *   "président Macron"          → président | Macron
 *   "ministre Élisabeth Borne"  → ministre | Élisabeth Borne
 *   "Premier ministre Jean Castex" → Premier ministre | Jean Castex
 *
 * Cas NON-splittés (correct) :
 *   "Assemblée nationale"  → ADJ amod, pas de PROPN flat → pas de split
 *   "Cour de cassation"    → nmod NOUN, pas de PROPN flat → pas de split
 *   "Marie de Médicis"     → head = PROPN → condition NOUN non vérifiée
 *   "Duc de Berry"         → "Berry" est nmod pas flat:name → pas de split
 */
private fun trySplitRoleAndName(
    entity:   Entity,
    trimmed:  List<UDToken>,
    nounHead: UDToken,
    udDoc:    UDDocument,
): List<Entity>? {

    val trimmedIds = trimmed.map { it.id }.toSet()

    // ── Étape 1 : PROPN avec flat:name dont la tête est dans le span ────────
    val clusterIds = mutableSetOf<Int>()
    trimmed.forEach { t ->
        if (t.upos == UPOS.PROPN
            && t.deprel.lowercase() in FLAT_DEPRELS
            && t.head in trimmedIds
        ) clusterIds += t.id
    }

    if (clusterIds.isEmpty()) return null

    // ── Étape 2 : extension transitive (PROPN → PROPN flat:name) ─────────────
    var changed = true
    while (changed) {
        changed = false
        trimmed.forEach { t ->
            if (t.id !in clusterIds
                && t.upos == UPOS.PROPN
                && t.deprel.lowercase() in FLAT_DEPRELS
                && t.head in clusterIds
            ) { clusterIds += t.id; changed = true }
        }
    }

    // ── Étape 3 : séparer role / nom ─────────────────────────────────────────
    // roleTokens : tout ce qui n'est pas dans le cluster PROPN
    // On rogne les ADP/PUNCT qui pourraient traîner à la fin du rôle
    val roleTokens = trimmed
        .filter { it.id !in clusterIds }
        .dropLastWhile { it.upos in setOf(UPOS.PUNCT, UPOS.ADP) }

    // nameTokens : le cluster PROPN dans l'ordre de position
    val nameTokens = trimmed.filter { it.id in clusterIds }

    if (roleTokens.isEmpty() || nameTokens.isEmpty()) return null

    // ── Étape 4 : construire les deux Entity ─────────────────────────────────
    val roleStart = roleTokens.first().start
    val roleEnd   = roleTokens.last().end
    if (roleEnd > udDoc.text.length || roleStart < 0) return null

    val nameStart = nameTokens.first().start
    val nameEnd   = nameTokens.last().end
    if (nameEnd > udDoc.text.length || nameStart < 0) return null

    val roleUdSpan = reconstructSpan(trimmed, nounHead.id)
    val nameHead   = nameTokens.firstOrNull { it.upos == UPOS.PROPN } ?: nameTokens.first()
    val nameUdSpan = reconstructSpan(trimmed, nameHead.id)

    val roleEntity = entity.copy(
        text     = udDoc.text.substring(roleStart, roleEnd),
        span     = Span(roleStart, roleEnd, roleTokens),
        metadata = buildEntityMeta(entity, nounHead, udDoc, roleUdSpan)
    )
    val nameEntity = entity.copy(
        text     = udDoc.text.substring(nameStart, nameEnd),
        span     = Span(nameStart, nameEnd, nameTokens),
        metadata = buildEntityMeta(entity, nameHead, udDoc, nameUdSpan)
    )

    return listOf(roleEntity, nameEntity)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Construit la map de metadata enrichie pour une entité et sa tête UD. */
private fun buildEntityMeta(
    entity: Entity,
    head:   UDToken,
    udDoc:  UDDocument,
    udSpan: Span,
): Map<String, Any> {
    val udSpanText = if (udSpan.start >= 0 && udSpan.end > udSpan.start)
        udDoc.text.substring(
            udSpan.start.coerceAtLeast(0),
            udSpan.end.coerceAtMost(udDoc.text.length)
        )
    else entity.text

    return buildMap<String, Any?> {
        putAll(entity.metadata)
        put("nerRawText", entity.text)
        put("head",       head.text)
        put("headLemma",  head.lemma ?: head.text)
        put("headUpos",   head.upos?.name)
        put("headDeprel", head.deprel)
        put("udSpanText", udSpanText)
        put("gender",     head.feats?.gender?.name)
        put("number",     head.feats?.number?.name)
        put("headId",     head.id)
    }.filterValues { it != null }.mapValues { it.value as Any }
}
