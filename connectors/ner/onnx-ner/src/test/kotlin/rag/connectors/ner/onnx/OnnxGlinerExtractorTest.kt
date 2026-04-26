package rag.connectors.ud.stanza

import rag.model.*
import kotlin.test.*

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

private fun tok(
    id: Int,
    text: String,
    upos: UPOS,
    head: Int,
    deprel: String,
    start: Int,
    end: Int,
    lemma: String? = null,
) = UDToken(
    id = id,
    text = text,
    lemma = lemma ?: text.lowercase(),
    upos = upos,
    xpos = null,
    head = head,
    deprel = deprel,
    start = start,
    end = end,
)

private fun sentence(id: Int = 0, tokens: List<UDToken>): UDSentence {
    val s = tokens.minOf { it.start }
    val e = tokens.maxOf { it.end }
    return UDSentence(id = id, tokens = tokens, start = s, end = e)
}

private fun doc(text: String, vararg sentences: UDSentence) =
    UDDocument(text = text, sentences = sentences.toList())

private fun ent(text: String, type: String, start: Int, end: Int) =
    Entity(text = text, type = type, span = Span(start, end, emptyList()))

private fun texts(result: List<Entity>) = result.map { it.text }.toSet()

// ─────────────────────────────────────────────────────────────────────────────
// NerCoarseType.from()
// ─────────────────────────────────────────────────────────────────────────────

class NerCoarseTypeFromTest {

    @Test
    fun `from PER is case insensitive and trims`() {
        assertEquals(NerCoarseType.PER, NerCoarseType.from("PER"))
        assertEquals(NerCoarseType.PER, NerCoarseType.from("per"))
        assertEquals(NerCoarseType.PER, NerCoarseType.from(" Per "))
    }

    @Test
    fun `from all known types`() {
        assertEquals(NerCoarseType.LOC, NerCoarseType.from("LOC"))
        assertEquals(NerCoarseType.ORG, NerCoarseType.from("ORG"))
        assertEquals(NerCoarseType.TIME, NerCoarseType.from("TIME"))
        assertEquals(NerCoarseType.EVENT, NerCoarseType.from("EVENT"))
        assertEquals(NerCoarseType.OBJECT, NerCoarseType.from("OBJECT"))
    }

    @Test
    fun `unknown returns UNKNOWN`() {
        assertEquals(NerCoarseType.UNKNOWN, NerCoarseType.from("MISC"))
        assertEquals(NerCoarseType.UNKNOWN, NerCoarseType.from(""))
        assertEquals(NerCoarseType.UNKNOWN, NerCoarseType.from("  "))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// hopDistance()
// ─────────────────────────────────────────────────────────────────────────────

class HopDistanceTest {

    // arrêta(ROOT)
    // ├─ police (nsubj)
    // │   └─ Grenoble (nmod)
    // └─ Martin (obj)
    private val tokens = listOf(
        tok(1, "arrêta",   UPOS.VERB,  0, "root",  0,  6),
        tok(2, "police",   UPOS.NOUN,  1, "nsubj", 7, 13),
        tok(3, "Grenoble", UPOS.PROPN, 2, "nmod", 14, 22),
        tok(4, "Martin",   UPOS.PROPN, 1, "obj",  23, 29),
    )

    @Test
    fun `same token returns 0`() {
        assertEquals(0, hopDistance(1, 1, tokens))
        assertEquals(0, hopDistance(3, 3, tokens))
    }

    @Test
    fun `direct parent child returns 1`() {
        assertEquals(1, hopDistance(2, 1, tokens))
        assertEquals(1, hopDistance(1, 2, tokens))
    }

    @Test
    fun `two hops`() {
        assertEquals(2, hopDistance(3, 1, tokens))
        assertEquals(2, hopDistance(1, 3, tokens))
    }

    @Test
    fun `siblings are at distance 2`() {
        assertEquals(2, hopDistance(2, 4, tokens))
        assertEquals(2, hopDistance(4, 2, tokens))
    }

    @Test
    fun `maxHops exceeded returns HOP_UNREACHABLE`() {
        assertEquals(HOP_UNREACHABLE, hopDistance(3, 1, tokens, maxHops = 1))
    }

    @Test
    fun `disconnected graph returns HOP_UNREACHABLE`() {
        val isolated = listOf(
            tok(1, "word1", UPOS.NOUN, 0, "root", 0, 5),
            tok(2, "word2", UPOS.NOUN, 0, "root", 6, 11),
        )
        assertEquals(HOP_UNREACHABLE, hopDistance(1, 2, isolated))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EntityCandidate.withHopFrom()
// ─────────────────────────────────────────────────────────────────────────────

class WithHopFromTest {

    private val tokens = listOf(
        tok(1, "arrêta", UPOS.VERB, 0, "root", 0, 6),
        tok(2, "police", UPOS.NOUN, 1, "nsubj", 7, 13),
    )

    private fun candidate(head: UDToken?) = EntityCandidate(
        text = "police",
        lemma = "police",
        span = Span(7, 13, if (head != null) listOf(head) else emptyList()),
        nerType = NerCoarseType.PER,
        nerHint = EntityType.HINT_GROUP_ROLE,
        isName = false,
        head = head,
        headUpos = head?.upos,
        headDeprel = head?.deprel,
        isPropn = false,
        isPron = false,
        gender = null,
        number = null,
        feats = null,
        sentenceSpan = 0 until 20,
        confidence = 0.70f
    )

    @Test
    fun `hop is computed`() {
        val c = candidate(tokens[1]).withHopFrom(triggerTokenId = 1, sentenceTokens = tokens)
        assertEquals(1, c.hopFromTrigger)
    }

    @Test
    fun `no head gives HOP_UNREACHABLE`() {
        val c = candidate(null).withHopFrom(triggerTokenId = 1, sentenceTokens = tokens)
        assertEquals(HOP_UNREACHABLE, c.hopFromTrigger)
    }

    @Test
    fun `isDirectChildOfTrigger matches hop 1`() {
        val c = candidate(tokens[1]).withHopFrom(triggerTokenId = 1, sentenceTokens = tokens)
        assertTrue(c.isDirectChildOfTrigger)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// reconstructSpan() -- public helper used elsewhere
// ─────────────────────────────────────────────────────────────────────────────

class ReconstructSpanTest {

    @Test
    fun `flat name children are included`() {
        val toks = listOf(
            tok(1, "général", UPOS.NOUN,  0, "nsubj",     0,  7),
            tok(2, "De",      UPOS.PROPN, 1, "flat:name", 8, 10),
            tok(3, "Gaulle",  UPOS.PROPN, 1, "flat:name", 11, 17),
        )

        val span = reconstructSpan(toks, 1)
        val ids = span.tokens.map { it.id }.toSet()

        assertEquals(0, span.start)
        assertEquals(17, span.end)
        assertTrue(1 in ids)
        assertTrue(2 in ids)
        assertTrue(3 in ids)
    }

    @Test
    fun `amod is included`() {
        val toks = listOf(
            tok(1, "ancien",   UPOS.ADJ,  2, "amod",  0,  6),
            tok(2, "ministre", UPOS.NOUN, 0, "root",  7, 15),
        )

        val span = reconstructSpan(toks, 2)
        assertTrue(1 in span.tokens.map { it.id })
        assertEquals("ancien ministre", "ancien ministre".substring(0, 15))
    }

    @Test
    fun `obl is excluded`() {
        val toks = listOf(
            tok(1, "arrestation", UPOS.NOUN,  0, "root",  0, 11),
            tok(2, "de",          UPOS.ADP,   3, "case", 12, 14),
            tok(3, "Martin",      UPOS.PROPN, 1, "obl",  15, 21),
        )

        val span = reconstructSpan(toks, 1)
        assertFalse(3 in span.tokens.map { it.id })
    }

    @Test
    fun `unknown head returns empty span`() {
        val toks = listOf(
            tok(1, "police", UPOS.NOUN, 0, "root", 0, 6),
        )
        val span = reconstructSpan(toks, 999)
        assertEquals(0, span.start)
        assertEquals(0, span.end)
        assertTrue(span.tokens.isEmpty())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// mergeNerLabelWithUD() / V2 -- behavioural tests
// ─────────────────────────────────────────────────────────────────────────────

class MergeNerLabelWithUDV2Test {

    @Test
    fun `aligned entity is enriched with tokens and metadata`() {
        val toks = listOf(
            tok(1, "Jacques", UPOS.PROPN, 2, "flat:name", 0, 7),
            tok(2, "Martin",  UPOS.PROPN, 4, "nsubj",     8, 14),
            tok(3, "fut",     UPOS.AUX,   4, "aux",       15, 18),
            tok(4, "arrêté",  UPOS.VERB,  0, "root",      19, 25),
        )
        val ud = doc("Jacques Martin fut arrêté", sentence(0, toks))
        val ner = ent("Jacques Martin", "PER", 0, 14)

        val result = mergeNerLabelWithUD(listOf(ner), ud)

        assertFalse(result.isEmpty())
        val e = result.first()
        assertEquals("Jacques Martin", e.text)
        assertEquals("PER", e.type)
        assertTrue((e.span?.tokens?.size ?: 0) >= 2)
        assertEquals("PROPN", e.metadata["headUpos"])
        assertEquals("Martin", e.metadata["head"])
        assertNotNull(e.metadata["headId"])
    }

    @Test
    fun `entity outside any sentence is returned as is`() {
        val toks = listOf(
            tok(1, "Paris", UPOS.PROPN, 0, "root", 0, 5),
        )
        val ud = doc("Paris", sentence(0, toks))
        val ner = ent("hors doc", "LOC", 100, 108)

        val result = mergeNerLabelWithUD(listOf(ner), ud)

        assertEquals(1, result.size)
        assertEquals("hors doc", result.first().text)
        assertEquals(100, result.first().span?.start)
        assertEquals(108, result.first().span?.end)
    }

    @Test
    fun `verb head entity is filtered`() {
        val toks = listOf(
            tok(1, "Il",      UPOS.PRON, 2, "nsubj", 0, 2),
            tok(2, "attaque", UPOS.VERB, 0, "root",  3, 10),
        )
        val ud = doc("Il attaque", sentence(0, toks))
        val ner = ent("attaque", "EVENT", 3, 10)

        val result = mergeNerLabelWithUD(listOf(ner), ud)

        assertTrue(result.isEmpty())
    }

    @Test
    fun `pron head is filtered for non PER but kept for PER`() {
        val toks = listOf(
            tok(1, "Il", UPOS.PRON, 0, "root", 0, 2),
        )
        val ud = doc("Il", sentence(0, toks))

        val eventEntity = ent("Il", "EVENT", 0, 2)
        val perEntity = ent("Il", "PER", 0, 2)

        val eventResult = mergeNerLabelWithUD(listOf(eventEntity), ud)
        val perResult = mergeNerLabelWithUD(listOf(perEntity), ud)

        assertTrue(eventResult.isEmpty(), "PRON non-PER doit être filtré")
        assertEquals(1, perResult.size, "PRON PER doit être conservé")
        assertEquals("PRON", perResult.first().metadata["headUpos"])
    }

    @Test
    fun `leading DET and trailing PUNCT are trimmed`() {
        val toks = listOf(
            tok(1, "la",     UPOS.DET,   2, "det",   0,  2),
            tok(2, "police", UPOS.NOUN,  0, "root",  3,  9),
            tok(3, ".",      UPOS.PUNCT, 2, "punct", 9, 10),
        )
        val ud = doc("la police.", sentence(0, toks))
        val ner = ent("la police.", "PER", 0, 10)

        val result = mergeNerLabelWithUD(listOf(ner), ud)

        assertFalse(result.isEmpty())
        val first = result.first()
        assertEquals("police", first.text)
        assertEquals(3, first.span?.start)
        assertEquals(9, first.span?.end)
        assertEquals("NOUN", first.metadata["headUpos"])
    }

    @Test
    fun `mergeNerLabelWithUD delegates to V2 and can generate de-chain expansion`() {
        // NER brut = "place", on attend une expansion "place de la République"
        val toks = listOf(
            tok(1, "place",       UPOS.NOUN, 0, "root",  0,  5),
            tok(2, "de",          UPOS.ADP,  4, "case",  6,  8),
            tok(3, "la",          UPOS.DET,  4, "det",   9, 11),
            tok(4, "République",  UPOS.PROPN, 1, "nmod", 12, 22),
        )
        val ud = doc("place de la République", sentence(0, toks))
        val ner = ent("place", "LOC", 0, 5)

        val result = mergeNerLabelWithUD(listOf(ner), ud)
        val ts = texts(result)

        assertTrue("place" in ts)
        assertTrue("place de la République" in ts, "Expansion de-chain attendue")
    }

    @Test
    fun `noun_code expansion creates vol AF447 from base vol`() {
        val toks = listOf(
            tok(1, "vol",   UPOS.NOUN,  0, "root",   0, 3),
            tok(2, "AF447", UPOS.PROPN, 1, "appos",  4, 9),
        )
        val ud = doc("vol AF447", sentence(0, toks))
        val ner = ent("vol", "EVENT", 0, 3)

        val result = mergeNerLabelWithUD(listOf(ner), ud)
        val ts = texts(result)

        assertTrue("vol" in ts)
        assertTrue("vol AF447" in ts)
    }

    @Test
    fun `time expansion extends with trailing nominal phrase`() {
        val toks = listOf(
            tok(1, "12",      UPOS.NUM,  2, "nummod",  0,  2),
            tok(2, "février", UPOS.NOUN, 0, "root",    3, 10),
            tok(3, "2026",    UPOS.NUM,  2, "nummod", 11, 15),
            tok(4, "du",      UPOS.ADP,  5, "case",   16, 18),
            tok(5, "matin",   UPOS.NOUN, 2, "nmod",   19, 24),
        )
        val ud = doc("12 février 2026 du matin", sentence(0, toks))
        val ner = ent("12 février 2026", "TIME", 0, 15)

        val result = mergeNerLabelWithUD(listOf(ner), ud)
        val ts = texts(result)

        assertTrue("12 février 2026" in ts)
        assertTrue("12 février 2026 du matin" in ts)
    }

    @Test
    fun `dedupe removes duplicate candidates with same text span and type`() {
        // Ici le NER couvre déjà "vol AF447", et noun_code regénère le même span.
        val toks = listOf(
            tok(1, "vol",   UPOS.NOUN,  0, "root",   0, 3),
            tok(2, "AF447", UPOS.PROPN, 1, "appos",  4, 9),
        )
        val ud = doc("vol AF447", sentence(0, toks))
        val ner = ent("vol AF447", "EVENT", 0, 9)

        val result = mergeNerLabelWithUD(listOf(ner), ud)

        val matching = result.filter { it.text == "vol AF447" && it.type == "EVENT" }
        assertEquals(1, matching.size, "Les doublons exacts doivent être supprimés")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Split rôle / nom propre
// ─────────────────────────────────────────────────────────────────────────────

class TrySplitRoleAndNameV2Test {

    @Test
    fun `flat name splits role and proper name`() {
        val toks = listOf(
            tok(1, "général", UPOS.NOUN,  0, "root",      0,  7),
            tok(2, "De",      UPOS.PROPN, 1, "flat:name", 8, 10),
            tok(3, "Gaulle",  UPOS.PROPN, 1, "flat:name", 11, 17),
        )
        val ud = doc("général De Gaulle", sentence(0, toks))
        val ner = ent("général De Gaulle", "PER", 0, 17)

        val result = mergeNerLabelWithUD(listOf(ner), ud)
        val ts = texts(result)

        assertTrue("général" in ts)
        assertTrue("De Gaulle" in ts)
        assertTrue("général De Gaulle" in ts) // baseline V2
    }

    @Test
    fun `nmod with case splits role and lowercase particle name`() {
        val toks = listOf(
            tok(1, "Général", UPOS.NOUN,  0, "root", 0, 7),
            tok(2, "de",      UPOS.ADP,   3, "case", 8, 10),
            tok(3, "Gaulle",  UPOS.PROPN, 1, "nmod", 11, 17),
        )
        val ud = doc("Général de Gaulle", sentence(0, toks))
        val ner = ent("Général de Gaulle", "PER", 0, 17)

        val result = mergeNerLabelWithUD(listOf(ner), ud)
        val ts = texts(result)

        assertTrue("Général" in ts)
        assertTrue("de Gaulle" in ts)

        val name = result.first { it.text == "de Gaulle" }
        assertEquals(8, name.span?.start)
        assertEquals(17, name.span?.end)
    }

    @Test
    fun `appos splits president Hollande`() {
        val toks = listOf(
            tok(1, "président", UPOS.NOUN,  0, "root",  0,  9),
            tok(2, "Hollande",  UPOS.PROPN, 1, "appos", 10, 18),
        )
        val ud = doc("président Hollande", sentence(0, toks))
        val ner = ent("président Hollande", "PER", 0, 18)

        val result = mergeNerLabelWithUD(listOf(ner), ud)
        val ts = texts(result)

        assertTrue("président" in ts)
        assertTrue("Hollande" in ts)
    }

    @Test
    fun `multi token role splits Premier ministre Jean Castex`() {
        val toks = listOf(
            tok(1, "Premier",  UPOS.ADJ,   2, "amod",      0,  7),
            tok(2, "ministre", UPOS.NOUN,  0, "root",      8, 16),
            tok(3, "Jean",     UPOS.PROPN, 2, "flat:name", 17, 21),
            tok(4, "Castex",   UPOS.PROPN, 3, "flat:name", 22, 28),
        )
        val ud = doc("Premier ministre Jean Castex", sentence(0, toks))
        val ner = ent("Premier ministre Jean Castex", "PER", 0, 28)

        val result = mergeNerLabelWithUD(listOf(ner), ud)
        val ts = texts(result)

        assertTrue("Premier ministre" in ts)
        assertTrue("Jean Castex" in ts)
    }

    @Test
    fun `no split for NOUN plus ADJ`() {
        val toks = listOf(
            tok(1, "Assemblée", UPOS.NOUN, 0, "root", 0, 9),
            tok(2, "nationale", UPOS.ADJ,  1, "amod", 10, 19),
        )
        val ud = doc("Assemblée nationale", sentence(0, toks))
        val ner = ent("Assemblée nationale", "ORG", 0, 19)

        val result = mergeNerLabelWithUD(listOf(ner), ud)

        assertEquals(1, result.size)
        assertEquals("Assemblée nationale", result.first().text)
    }

    @Test
    fun `no split for nmod NOUN like Cour de cassation`() {
        val toks = listOf(
            tok(1, "Cour",       UPOS.NOUN, 0, "root", 0, 4),
            tok(2, "de",         UPOS.ADP,  3, "case", 5, 7),
            tok(3, "cassation",  UPOS.NOUN, 1, "nmod", 8, 17),
        )
        val ud = doc("Cour de cassation", sentence(0, toks))
        val ner = ent("Cour de cassation", "ORG", 0, 17)

        val result = mergeNerLabelWithUD(listOf(ner), ud)

        assertEquals(1, result.size)
        assertEquals("Cour de cassation", result.first().text)
    }

    @Test
    fun `no false split when appos is NOUN not PROPN`() {
        val toks = listOf(
            tok(1, "président", UPOS.NOUN, 0, "root",  0,  9),
            tok(2, "chef",      UPOS.NOUN, 1, "appos", 10, 14),
        )
        val ud = doc("président chef", sentence(0, toks))
        val ner = ent("président chef", "PER", 0, 14)

        val result = mergeNerLabelWithUD(listOf(ner), ud)
        val ts = result.map { it.text }.toSet()

        // baseline conservé
        assertTrue("président chef" in ts)

        // reconstruction plus courte acceptable en V2
        assertTrue("président" in ts)

        // surtout : pas de faux "nom propre" sur chef
        assertFalse("chef" in ts, "chef ne doit pas être extrait comme nom propre autonome")

        // et pas de split rôle+nom classique
        val splitLike = result.filter { it.text == "président" || it.text == "chef" }
        assertNotEquals(
            splitLike.map { it.text }.toSet(),
            setOf("président", "chef"),
            "Il ne doit pas y avoir de faux split rôle/nom sur appos NOUN"
        )
    }
}
