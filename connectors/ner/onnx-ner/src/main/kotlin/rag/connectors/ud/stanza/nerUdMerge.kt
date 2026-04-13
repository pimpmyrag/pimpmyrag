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
): List<Entity> = mergeNerLabelWithUDV2(nerEntities = nerEntities, udDoc = udDoc)

/**
 * V2 :
 * - garde le trim / les filtres / le split rôle+nom
 * - émet plusieurs candidats enrichis au lieu d'un seul span figé
 * - ne repose pas sur des heuristiques métier hardcodées
 */
fun mergeNerLabelWithUDV2(
    nerEntities: List<Entity>,
    udDoc: UDDocument
): List<Entity> {
    val raw = nerEntities.flatMap { enrichOneV2(it, udDoc) }.pruneOverlapsByConfidence()
    return dedupeEntities(raw)
}

private fun enrichOneV2(entity: Entity, udDoc: UDDocument): List<Entity> {
    val eStart = entity.span?.start ?: return listOf(entity)
    val eEnd = entity.span?.end ?: return listOf(entity)
    if (eStart >= eEnd) return listOf(entity)

    // 1) Trouver la phrase UD
    val sentence = udDoc.sentences.firstOrNull { s ->
        s.start <= eStart && s.end >= eEnd
    } ?: return listOf(entity)

    // 2) Tokens qui chevauchent le span
    val overlapping = sentence.tokens.filter { t ->
        t.start < eEnd && t.end > eStart
    }
    if (overlapping.isEmpty()) return listOf(entity)

    // 3) Rogner parasites en bord de span
    val trimmed = trimBoundaryTokens(overlapping)
    if (trimmed.isEmpty()) return emptyList()

    // 4) Trouver la tête syntaxique
    val head = findHead(trimmed) ?: trimmed.first()

    // 5) Filtrer les artefacts manifestes
    if (!isValidEntityHead(entity, head)) return emptyList()

    val candidates = mutableListOf<Entity>()

    // ─────────────────────────────────────────────────────────────
    // CANDIDAT 1 : span NER rogné (baseline fiable)
    // ─────────────────────────────────────────────────────────────
    val baseEntity = buildEntityFromTokens(entity, trimmed, head, udDoc)
    candidates += baseEntity

    // ─────────────────────────────────────────────────────────────
    // CANDIDAT 2 : split rôle + nom (PER, tête nominale)
    // ─────────────────────────────────────────────────────────────
    if (entity.type.lowercase() == "per" && head.upos == UPOS.NOUN) {
        trySplitRoleAndNameV2(entity, trimmed, head, udDoc)?.let { split ->
            candidates += split
        }
    }

    // ─────────────────────────────────────────────────────────────
    // CANDIDAT 3+ : expansions syntaxiques nominales
    // ─────────────────────────────────────────────────────────────
    candidates += generateNominalExpansions(entity, sentence, trimmed, head, udDoc)

    // ─────────────────────────────────────────────────────────────
    // CANDIDAT 4+ : expansions temporelles
    // ─────────────────────────────────────────────────────────────
    if (entity.type.lowercase() == "time") {
        candidates += generateTimeExpansions(entity, sentence, trimmed, head, udDoc)
    }

    return dedupeEntities(candidates)
}
private fun trimBoundaryTokens(tokens: List<UDToken>): List<UDToken> {
    val skipAtStart = setOf(UPOS.PUNCT, UPOS.DET, UPOS.ADP)
    return tokens
        .dropWhile { it.upos in skipAtStart }
        .dropLastWhile { it.upos in setOf(UPOS.PUNCT, UPOS.ADP) }
}

private fun findHead(tokens: List<UDToken>): UDToken? {
    val ids = tokens.map { it.id }.toSet()
    return tokens.firstOrNull { t -> t.head == 0 || t.head !in ids }
}


private fun isValidEntityHead(entity: Entity, head: UDToken): Boolean {
    if (head.upos in FUNCTION_UPOS) return false
    if (head.upos == UPOS.PRON && entity.type.lowercase() != "per") return false
    if (head.upos == UPOS.VERB || head.upos == UPOS.ADV) return false
    return true
}


private fun buildEntityFromTokens(
    entity: Entity,
    tokens: List<UDToken>,
    head: UDToken,
    udDoc: UDDocument
): Entity {
    val start = tokens.first().start
    val end = tokens.last().end
    val udSpan = Span(start, end, tokens)

    return entity.copy(
        text = udDoc.text.substring(start, end),
        span = udSpan,
        metadata = buildEntityMeta(entity, head, udDoc, udSpan)
    )
}

private fun generateNominalExpansions(
    entity: Entity,
    sentence: UDSentence,
    trimmed: List<UDToken>,
    head: UDToken,
    udDoc: UDDocument
): List<Entity> {
    val out = mutableListOf<Entity>()
    val allTokens = sentence.tokens
    val tokenById = allTokens.associateBy { it.id }

    // 1) Span reconstruit simple autour de la tête
    val reconstructed = reconstructSpanV2(allTokens, head.id)

    if (entity.type.equals("loc", true)) {
        val coord = splitCoordinatedPropns(entity, sentence, reconstructed, udDoc, includeGroup = true)
        out += coord
    }

    if (reconstructed.tokens.isNotEmpty()) {
        out += entity.copy(
            text = udDoc.text.substring(reconstructed.start, reconstructed.end),
            span = reconstructed,
            metadata = buildEntityMeta(entity, head, udDoc, reconstructed) +
                    mapOf("candidateSource" to "ud_reconstruct")
        )
    }

    // 2) NOUN + PROPN / appos / flat / nmod
    if (head.upos == UPOS.NOUN) {
        val nounPropn = expandNounWithProperName(allTokens, head.id)
        if (nounPropn != null) {
            out += entity.copy(
                text = udDoc.text.substring(nounPropn.start, nounPropn.end),
                span = nounPropn,
                metadata = buildEntityMeta(entity, head, udDoc, nounPropn) +
                        mapOf("candidateSource" to "noun_propn")
            )
        }
    }

    // 3) NOUN + nmod(case=de/du/d') + X
    if (head.upos == UPOS.NOUN) {
        val deChain = expandNounWithDeChain(allTokens, head.id)
        if (deChain != null) {
            out += entity.copy(
                text = udDoc.text.substring(deChain.start, deChain.end),
                span = deChain,
                metadata = buildEntityMeta(entity, head, udDoc, deChain) +
                        mapOf("candidateSource" to "noun_de_chain")
            )
        }
    }

    // 4) NOUN + code / nummod / PROPN alphanumérique
    if (head.upos == UPOS.NOUN) {
        val withCode = expandNounWithCode(allTokens, head.id)
        if (withCode != null) {
            out += entity.copy(
                text = udDoc.text.substring(withCode.start, withCode.end),
                span = withCode,
                metadata = buildEntityMeta(entity, head, udDoc, withCode) +
                        mapOf("candidateSource" to "noun_code")
            )
        }
    }

    return dedupeEntities(out)
}

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

// ─────────────────────────────────────────────────────────────────────────────
// Split PER : rôle (NOUN) + nom propre (PROPN flat:name)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Tente de décomposer un span PER de la forme [TITRE NOUN] [NOM PROPN] en deux
 * entités distinctes.
 *
 * Algorithme :
 *  1. Collecter les PROPN dont le deprel est dans [flat, flat:name, name, nmod, appos] et dont
 *     la tête pointe vers un token *à l'intérieur* du span (NOUN head ou autre PROPN
 *     du cluster déjà identifié).
 *     - "nmod"  : Stanza parse "de Gaulle" (particle minuscule) en ADP(case)→PROPN(nmod).
 *     - "appos" : Stanza parse "président Hollande" en NOUN(président)→PROPN(appos=Hollande).
 *       Le risque d'appos ambigu (ex. "président, chef de…") est écarté par le
 *       garde upos==PROPN : "chef" est NOUN et n'est donc jamais capturé.
 *  2. Étendre le cluster transitivement (PROPN → PROPN flat:name/nmod/appos → …).
 *  3. Collecter les ADP "case" qui s'attachent au cluster (ex. "de" dans "de Gaulle")
 *     et les inclure dans le span du nom propre.
 *  4. Séparer les tokens du span en roleTokens / nameTokens.
 *  5. Construire deux Entity et retourner la paire.
 *
 * Retourne null si le split n'est pas applicable (cluster vide, span non-valide…)
 * → l'appelant continuera avec le chemin normal mono-entité.
 *
 * Cas traités :
 *   "général De Gaulle"            → général | De Gaulle         (PROPN flat:name)
 *   "Général de Gaulle"            → Général | de Gaulle         (PROPN nmod + ADP case)
 *   "président Hollande"           → président | Hollande        (PROPN appos)
 *   "président Macron"             → président | Macron          (PROPN appos ou flat:name)
 *   "ministre Élisabeth Borne"     → ministre | Élisabeth Borne
 *   "Premier ministre Jean Castex" → Premier ministre | Jean Castex
 *
 * Cas NON-splittés (correct) :
 *   "Assemblée nationale"  → ADJ amod, pas de PROPN → pas de split
 *   "Cour de cassation"    → nmod NOUN (pas PROPN) → pas de split
 *   "Marie de Médicis"     → head = PROPN → condition NOUN non vérifiée
 *   "président, chef de…"  → "chef" est NOUN pas PROPN → pas de split
 */
private fun trySplitRoleAndName(
    entity: Entity,
    trimmed: List<UDToken>,
    nounHead: UDToken,
    udDoc: UDDocument,
): List<Entity>? {

    val trimmedIds = trimmed.map { it.id }.toSet()

    // Deprels reconnus pour identifier un PROPN faisant partie du nom propre.
    // - "nmod"  : "de Gaulle" → de(ADP case) + Gaulle(PROPN nmod)
    // - "appos" : "président Hollande" → Hollande(PROPN appos of président)
    //   Pas de risque de faux positif : le garde upos==PROPN élimine "chef" (NOUN) dans
    //   "le président, chef de l'État".
    val nameDeprels = FLAT_DEPRELS + "nmod" + "appos"

    // ── Étape 1 : PROPN avec deprel nom-propre dont la tête est dans le span ─
    val clusterIds = mutableSetOf<Int>()
    trimmed.forEach { t ->
        if (t.upos == UPOS.PROPN
            && t.deprel.lowercase() in nameDeprels
            && t.head in trimmedIds
        ) clusterIds += t.id
    }

    if (clusterIds.isEmpty()) return null

    // ── Étape 2 : extension transitive (PROPN → PROPN flat:name/nmod → …) ────
    var changed = true
    while (changed) {
        changed = false
        trimmed.forEach { t ->
            if (t.id !in clusterIds
                && t.upos == UPOS.PROPN
                && t.deprel.lowercase() in nameDeprels
                && t.head in clusterIds
            ) {
                clusterIds += t.id; changed = true
            }
        }
    }

    // ── Étape 2b : ADP "case" attachés au cluster ────────────────────────────
    // Ex. "de" dans "de Gaulle" : ADP(case) → Gaulle(PROPN, dans cluster)
    // Ces particules font partie du span du nom propre mais pas du rôle.
    val caseADPIds = trimmed
        .filter { it.upos == UPOS.ADP && it.deprel.lowercase() == "case" && it.head in clusterIds }
        .map { it.id }
        .toSet()

    // ── Étape 3 : séparer role / nom ─────────────────────────────────────────
    // roleTokens : tout ce qui n'est ni dans le cluster PROPN ni une particule "case"
    // On rogne les ADP/PUNCT qui pourraient traîner à la fin du rôle
    val roleTokens = trimmed
        .filter { it.id !in clusterIds && it.id !in caseADPIds }
        .dropLastWhile { it.upos in setOf(UPOS.PUNCT, UPOS.ADP) }

    // nameTokens : cluster PROPN + particules "case" dans l'ordre de position
    val nameTokens = trimmed.filter { it.id in clusterIds || it.id in caseADPIds }

    if (roleTokens.isEmpty() || nameTokens.isEmpty()) return null

    // ── Étape 4 : construire les deux Entity ─────────────────────────────────
    val roleStart = roleTokens.first().start
    val roleEnd = roleTokens.last().end
    if (roleEnd > udDoc.text.length || roleStart < 0) return null

    val nameStart = nameTokens.first().start
    val nameEnd = nameTokens.last().end
    if (nameEnd > udDoc.text.length || nameStart < 0) return null

    val roleUdSpan = reconstructSpan(trimmed, nounHead.id)
    val nameHead = nameTokens.firstOrNull { it.upos == UPOS.PROPN } ?: nameTokens.first()
    val nameUdSpan = reconstructSpan(trimmed, nameHead.id)

    val roleEntity = entity.copy(
        text = udDoc.text.substring(roleStart, roleEnd),
        span = Span(roleStart, roleEnd, roleTokens),
        metadata = buildEntityMeta(entity, nounHead, udDoc, roleUdSpan)
    )
    val nameEntity = entity.copy(
        text = udDoc.text.substring(nameStart, nameEnd),
        span = Span(nameStart, nameEnd, nameTokens),
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
    head: UDToken,
    udDoc: UDDocument,
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
        put("head", head.text)
        put("headLemma", head.lemma ?: head.text)
        put("headUpos", head.upos?.name)
        put("headDeprel", head.deprel)
        put("udSpanText", udSpanText)
        put("gender", head.feats?.gender?.name)
        put("number", head.feats?.number?.name)
        put("headId", head.id)
    }.filterValues { it != null }.mapValues { it.value as Any }
}

private fun reconstructSpanV2(tokens: List<UDToken>, headId: Int): Span {
    val head = tokens.firstOrNull { it.id == headId } ?: return Span(0, 0, emptyList())

    val keep = linkedMapOf<Int, UDToken>()
    keep[head.id] = head

    fun base(d: String): String = d.lowercase().substringBefore(":")

    fun visit(node: UDToken) {
        for (child in tokens.filter { it.head == node.id }) {
            val rel = base(child.deprel)

            val include = when {
                rel in setOf("amod", "compound", "flat", "name", "nummod") -> true
                rel == "appos" && child.upos == UPOS.PROPN -> true
                rel == "nmod" && isNominalChainCandidate(child, tokens) -> true
                else -> false
            }
            if (!include) continue

            if (keep.putIfAbsent(child.id, child) == null) {
                visit(child)

                // Garde case/det pour l'info, mais offsets parfois (0,0)
                tokens.filter { it.head == child.id && base(it.deprel) in setOf("case", "det") }
                    .forEach { keep[it.id] = it }
            }
        }
    }

    visit(head)

    val kept = keep.values.toList()
    if (kept.isEmpty()) return Span(0, 0, emptyList())

    // 👇 Le seul critère fiable : end > start
    val anchored = kept.filter { it.end > it.start }

    val (start, end) = if (anchored.isNotEmpty()) {
        anchored.minOf { it.start } to anchored.maxOf { it.end }
    } else {
        head.start to head.end
    }

    // Tri “propre” (les (0,0) à la fin)
    val sortedForSpan = kept.sortedWith(
        compareBy<UDToken> { if (it.end > it.start) 0 else 1 }
            .thenBy { if (it.end > it.start) it.start else it.id }
    )

    return Span(start, end, sortedForSpan)
}


private fun isNominalChainCandidate(child: UDToken, tokens: List<UDToken>): Boolean {
    val children = tokens.filter { it.head == child.id }

    val hasCase = children.any { it.deprel.lowercase().substringBefore(":") == "case" }
    val nominalEnough = child.upos in setOf(UPOS.NOUN, UPOS.PROPN, UPOS.NUM, UPOS.ADJ)

    return hasCase && nominalEnough
}

private fun trySplitRoleAndNameV2(
    entity: Entity,
    trimmed: List<UDToken>,
    nounHead: UDToken,
    udDoc: UDDocument
): List<Entity>? {
    return trySplitRoleAndName(entity, trimmed, nounHead, udDoc)
}

private fun dedupeEntities(entities: List<Entity>): List<Entity> {
    val seen = linkedSetOf<String>()
    val out = mutableListOf<Entity>()

    for (e in entities) {
        val s = e.span?.start ?: continue
        val ed = e.span?.end ?: continue
        val key = "${e.type}|$s|$ed|${e.text.lowercase()}"
        if (key !in seen) {
            seen += key
            out += e
        }
    }
    return out.sortedWith(compareBy({ it.span?.start ?: Int.MAX_VALUE }, { it.span?.end ?: Int.MAX_VALUE }))
}

fun List<Entity>.pruneOverlapsByConfidence(
    delta: Float = 0.02f
): List<Entity> {

    fun conf(e: Entity) = (e.metadata["confidence"] as? Number)?.toFloat() ?: 0f
    fun len(e: Entity) = (e.span?.end ?: 0) - (e.span?.start ?: 0)

    fun overlap(a: Entity, b: Entity): Boolean {
        val as_ = a.span?.start ?: return false
        val ae = a.span?.end ?: return false
        val bs = b.span?.start ?: return false
        val be = b.span?.end ?: return false
        return as_ < be && ae > bs
    }

    fun better(a: Entity, b: Entity): Entity {
        // retourne le "meilleur" entre a et b (même type et overlap)
        val ca = conf(a)
        val cb = conf(b)

        return when {
            ca > cb + delta -> a
            cb > ca + delta -> b
            else -> {
                // confiance trop proche => tie-breakers
                val la = len(a)
                val lb = len(b)

                when {
                    la != lb -> if (la > lb) a else b   // plus long gagne
                    else -> if (ca >= cb) a else b       // sinon conf
                }
            }
        }
    }

    // Tri utile : start asc, end desc (longs d’abord quand même start)
    val sorted = this.sortedWith(
        compareBy<Entity> { it.span?.start ?: Int.MAX_VALUE }
            .thenByDescending { it.span?.end ?: Int.MIN_VALUE }
    )

    val out = mutableListOf<Entity>()

    for (e in sorted) {
        val s = e.span?.start ?: continue
        val en = e.span?.end ?: continue

        // cherche le dernier candidat dans out qui overlap + même type
        val idx = out.indexOfLast { o ->
            o.type == e.type && overlap(o, e)
        }

        if (idx >= 0) {
            val chosen = better(e, out[idx])
            out[idx] = chosen
        } else {
            out += e
        }
    }
    return out
}


private fun expandTimePhrase(tokens: List<UDToken>, baseTokens: List<UDToken>): Span? {
    if (baseTokens.isEmpty()) return null

    val keep = linkedMapOf<Int, UDToken>()
    baseTokens.forEach { keep[it.id] = it }

    val first = baseTokens.first()
    val last = baseTokens.last()

    // étendre à droite : "du matin", "du soir", etc.
    val rightNeighbors = tokens.filter { it.start >= first.start && it.end <= (last.end + 30) }

    // inclure ADP + NOUN/ADJ proches
    for (i in rightNeighbors.indices) {
        val t = rightNeighbors[i]
        if (t.id in keep) continue

        val rel = t.deprel.lowercase().substringBefore(":")
        if (t.upos == UPOS.ADP || t.upos == UPOS.NOUN || t.upos == UPOS.ADJ || t.upos == UPOS.NUM) {
            // petit garde-fou : proximité stricte
            if (t.start - last.end <= 6 || t.start <= last.end + 6) {
                keep[t.id] = t
            }
        }
    }

    val sorted = keep.values.sortedBy { it.start }
    if (sorted.size <= baseTokens.size) return null
    return Span(sorted.first().start, sorted.last().end, sorted)
}

private fun expandTimePhraseUdAware(
    tokens: List<UDToken>,
    baseTokens: List<UDToken>,
    allowLeftExpansion: Boolean = false, // false = on respecte le start du coarse
): Span? {
    if (baseTokens.isEmpty()) return null

    fun baseRel(dep: String): String = dep.lowercase().substringBefore(":")

    val baseStart = baseTokens.minOf { it.start }
    val baseEnd   = baseTokens.maxOf { it.end }

    val keep = linkedMapOf<Int, UDToken>()
    baseTokens.forEach { keep[it.id] = it }

    // Relations "internes" raisonnables pour une expression de temps
    val allowed = setOf("case", "amod", "compound", "nummod", "flat", "name", "appos", "nmod")

    var changed = true
    while (changed) {
        changed = false
        val keepIds = keep.keys.toSet()

        for (t in tokens) {
            if (t.id in keepIds) continue
            if (t.head !in keepIds) continue
            if (baseRel(t.deprel) !in allowed) continue

            // ✅ Clamp : ne pas étendre à gauche si le coarse a déjà des frontières fiables
            if (!allowLeftExpansion && t.start < baseStart) continue

            keep[t.id] = t
            changed = true
        }
    }

    // ⚠️ Re-filtrer au cas où un token "case/det" se serait glissé avant baseStart
    val filtered = keep.values
        .filter { allowLeftExpansion || it.start >= baseStart }
        .sortedBy { it.start }

    // Si l'expansion n'apporte rien (ou ne fait que tenter d'étendre à gauche), on renvoie null
    if (filtered.isEmpty()) return null

    val newStart = filtered.first().start
    val newEnd   = filtered.last().end

    // Pas d'extension ou extension nulle => null
    if (newStart == baseStart && newEnd == baseEnd) return null

    return Span(newStart, newEnd, filtered)
}


private fun expandNounWithProperName(tokens: List<UDToken>, headId: Int): Span? {
    val head = tokens.firstOrNull { it.id == headId } ?: return null
    val keep = linkedMapOf<Int, UDToken>()
    keep[head.id] = head

    fun base(d: String): String = d.lowercase().substringBefore(":")

    // enfants PROPN directs ou en apposition / flat
    val propnKids = tokens.filter { t ->
        t.head == head.id &&
                t.upos == UPOS.PROPN &&
                base(t.deprel) in setOf("flat", "name", "appos", "nmod")
    }

    if (propnKids.isEmpty()) return null

    propnKids.forEach { keep[it.id] = it }

    // récupérer flat:name transitifs entre PROPN
    var changed = true
    while (changed) {
        changed = false
        val currentIds = keep.keys.toSet()
        tokens.filter { t ->
            t.upos == UPOS.PROPN &&
                    t.head in currentIds &&
                    base(t.deprel) in setOf("flat", "name", "nmod", "appos")
        }.forEach {
            if (it.id !in keep) {
                keep[it.id] = it
                changed = true
            }
        }
    }

    val sorted = keep.values.sortedBy { it.start }
    return Span(sorted.first().start, sorted.last().end, sorted)
}

private fun generateTimeExpansions(
    entity: Entity,
    sentence: UDSentence,
    trimmed: List<UDToken>,
    head: UDToken,
    udDoc: UDDocument
): List<Entity> {
    val out = mutableListOf<Entity>()
    val allTokens = sentence.tokens

    val expanded = expandTimePhraseUdAware(allTokens, trimmed)
    if (expanded != null) {
        out += entity.copy(
            text = udDoc.text.substring(expanded.start, expanded.end),
            span = expanded,
            metadata = buildEntityMeta(entity, head, udDoc, expanded) +
                    mapOf("candidateSource" to "time_expand")
        )
    }

    return out
}

private fun splitCoordinatedPropns(
    entity: Entity,
    sentence: UDSentence,
    span: Span,
    udDoc: UDDocument,
    includeGroup: Boolean = true
): List<Entity> {
    fun baseRel(dep: String): String = dep.lowercase().substringBefore(":")

    val toks = sentence.tokens
    val spanIds = span.tokens.map { it.id }.toSet()

    val head = span.tokens.firstOrNull { it.upos == UPOS.PROPN } ?: return emptyList()

    val conjs = toks.filter {
        it.id in spanIds &&
                it.upos == UPOS.PROPN &&
                baseRel(it.deprel) == "conj" &&
                it.head == head.id
    }
    if (conjs.isEmpty()) return emptyList()

    // atomiques
    val atomicHeads = listOf(head) + conjs
    val atomics = atomicHeads.map { h ->
        val cl = propnCoreCluster(toks, h.id)
        Span(cl.first().start, cl.last().end, cl)
    }

    fun entityFromSpan(s: Span, headTok: UDToken, source: String) =
        entity.copy(
            text = udDoc.text.substring(s.start, s.end),
            span = s,
            metadata = buildEntityMeta(entity, headTok, udDoc, s) + mapOf("candidateSource" to source)
        )

    val atomicEntities = atomicHeads.zip(atomics).map { (h, s) ->
        entityFromSpan(s, h, "coord_atomic")
    }

    if (!includeGroup) return atomicEntities

    // group = union atomiques + cc/punct attachés à head/conj
    val keepIds = linkedSetOf<Int>()
    atomics.flatMap { it.tokens }.forEach { keepIds += it.id }

    toks.filter { it.id in spanIds && baseRel(it.deprel) in setOf("cc", "punct") }
        .filter { t -> t.head == head.id || conjs.any { c -> t.head == c.id } }
        .forEach { keepIds += it.id }

    val groupTokens = toks.filter { it.id in keepIds }.sortedBy { it.start }
    val groupSpan = Span(groupTokens.first().start, groupTokens.last().end, groupTokens)

    val groupEntity = entityFromSpan(groupSpan, head, "coord_group")
    return atomicEntities + groupEntity
}

private fun propnCoreCluster(tokens: List<UDToken>, headId: Int): List<UDToken> {
    fun baseRel(dep: String): String = dep.lowercase().substringBefore(":")

    val keep = linkedSetOf<Int>()
    fun includeChild(ch: UDToken): Boolean {
        val rel = baseRel(ch.deprel)
        return when (rel) {
            "flat", "name", "compound" -> true
            "amod" -> ch.upos == UPOS.ADJ
            "det"  -> ch.upos == UPOS.DET
            else -> false
        }
    }

    fun dfs(id: Int) {
        if (!keep.add(id)) return
        tokens.filter { it.head == id }.forEach { ch ->
            if (includeChild(ch)) dfs(ch.id)
        }
    }

    dfs(headId)
    return tokens.filter { it.id in keep }.sortedBy { it.start }
}

private fun expandNounWithDeChain(tokens: List<UDToken>, headId: Int): Span? {
    val head = tokens.firstOrNull { it.id == headId } ?: return null
    val keep = linkedMapOf<Int, UDToken>()
    keep[head.id] = head

    fun base(d: String): String = d.lowercase().substringBefore(":")
    fun anchored(t: UDToken): Boolean = t.end > t.start  // <-- CRUCIAL (0..2 est valide, 0..0 non)

    val nmods = tokens.filter { t ->
        t.head == head.id && base(t.deprel) == "nmod"
    }
    if (nmods.isEmpty()) return null

    var found = false
    for (nmod in nmods) {
        // case (de/d') + det(le) possible dans ton UD (du => de + le)
        val caseOrDet = tokens.filter {
            it.head == nmod.id && base(it.deprel) in setOf("case", "det")
        }

        if (caseOrDet.isNotEmpty()) {
            found = true
            keep[nmod.id] = nmod
            caseOrDet.forEach { keep[it.id] = it }

            // expansion transitive : "Trente Ans", etc.
            tokens.filter { t ->
                t.head == nmod.id &&
                        base(t.deprel) in setOf("flat", "name", "compound", "amod", "nummod", "appos")
            }.forEach { keep[it.id] = it }
        }
    }

    if (!found) return null

    val kept = keep.values.toList()
    val anchoredTokens = kept.filter(::anchored)

    // Bornes UNIQUEMENT sur tokens ancrés
    val (start, end) = if (anchoredTokens.isNotEmpty()) {
        anchoredTokens.minOf { it.start } to anchoredTokens.maxOf { it.end }
    } else {
        // fallback : au moins le head
        head.start to head.end
    }

    // Tri stable : ancrés d'abord, puis non-ancrés (0,0) à la fin
    val sorted = kept.sortedWith(
        compareBy<UDToken> { if (anchored(it)) 0 else 1 }
            .thenBy { if (anchored(it)) it.start else it.id }
    )

    return Span(start, end, sorted)
}


private fun expandNounWithCode(tokens: List<UDToken>, headId: Int): Span? {
    val head = tokens.firstOrNull { it.id == headId } ?: return null
    val keep = linkedMapOf<Int, UDToken>()
    keep[head.id] = head

    fun base(d: String): String = d.lowercase().substringBefore(":")

    val codeLikeChildren = tokens.filter { t ->
        t.head == head.id &&
                (
                        t.upos == UPOS.PROPN ||
                                t.upos == UPOS.NUM ||
                                Regex("^[A-Z0-9-]+$").matches(t.text)
                        ) &&
                base(t.deprel) in setOf("flat", "name", "nummod", "appos", "compound")
    }

    if (codeLikeChildren.isEmpty()) return null

    codeLikeChildren.forEach { keep[it.id] = it }

    val sorted = keep.values.sortedBy { it.start }
    return Span(sorted.first().start, sorted.last().end, sorted)
}
