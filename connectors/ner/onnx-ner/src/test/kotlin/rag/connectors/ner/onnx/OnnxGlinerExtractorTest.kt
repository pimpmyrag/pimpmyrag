package rag.connectors.ner.onnx

import rag.connectors.ud.stanza.*
import rag.model.*
import kotlin.test.*

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

private fun tok(
    id: Int, text: String, upos: UPOS, head: Int, deprel: String,
    start: Int, end: Int, lemma: String? = null
) = UDToken(
    id = id, text = text, lemma = lemma ?: text.lowercase(),
    upos = upos, xpos = null, head = head, deprel = deprel,
    start = start, end = end
)

private fun sentence(id: Int = 0, tokens: List<UDToken>): UDSentence {
    val s = tokens.minOf { it.start }
    val e = tokens.maxOf { it.end }
    return UDSentence(id = id, tokens = tokens, start = s, end = e)
}

private fun doc(text: String, vararg sentences: UDSentence) =
    UDDocument(text = text, sentences = sentences.toList())

// ─────────────────────────────────────────────────────────────────────────────
// NerCoarseType.from()
// ─────────────────────────────────────────────────────────────────────────────

class NerCoarseTypeFromTest {

    @Test fun `from PER - case insensitive`() {
        assertEquals(NerCoarseType.PER, NerCoarseType.from("PER"))
        assertEquals(NerCoarseType.PER, NerCoarseType.from("per"))
        assertEquals(NerCoarseType.PER, NerCoarseType.from("Per"))
    }

    @Test fun `from all known types`() {
        val expected = mapOf(
            "LOC"    to NerCoarseType.LOC,
            "ORG"    to NerCoarseType.ORG,
            "TIME"   to NerCoarseType.TIME,
            "EVENT"  to NerCoarseType.EVENT,
            "OBJECT" to NerCoarseType.OBJECT,
        )
        expected.forEach { (input, want) ->
            assertEquals(want, NerCoarseType.from(input), "from($input)")
        }
    }

    @Test fun `from unknown returns UNKNOWN`() {
        assertEquals(NerCoarseType.UNKNOWN, NerCoarseType.from("MISC"))
        assertEquals(NerCoarseType.UNKNOWN, NerCoarseType.from(""))
        assertEquals(NerCoarseType.UNKNOWN, NerCoarseType.from("  "))
    }

    @Test fun `from trims whitespace`() {
        assertEquals(NerCoarseType.PER, NerCoarseType.from("  PER  "))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// hopDistance()
// ─────────────────────────────────────────────────────────────────────────────

class HopDistanceTest {

    // Tree:  root(1) ─nsubj─> police(2) ─nmod─> Grenoble(3)
    //                └─obj──> Martin(4)
    private val tokens = listOf(
        tok(1, "arrêta",   UPOS.VERB,  0, "root",  0,  6),
        tok(2, "police",   UPOS.NOUN,  1, "nsubj", 7, 13),
        tok(3, "Grenoble", UPOS.PROPN, 2, "nmod", 14, 22),
        tok(4, "Martin",   UPOS.PROPN, 1, "obj",  23, 29),
    )

    @Test fun `same token returns 0`() {
        assertEquals(0, hopDistance(1, 1, tokens))
        assertEquals(0, hopDistance(3, 3, tokens))
    }

    @Test fun `direct parent-child returns 1`() {
        assertEquals(1, hopDistance(2, 1, tokens))   // police → arrêta
        assertEquals(1, hopDistance(1, 2, tokens))   // arrêta → police (reverse)
    }

    @Test fun `two hops`() {
        assertEquals(2, hopDistance(3, 1, tokens))   // Grenoble → police → arrêta
        assertEquals(2, hopDistance(1, 3, tokens))   // reverse
    }

    @Test fun `siblings share parent - two hops`() {
        // Martin and police are both children of arrêta → distance = 2
        assertEquals(2, hopDistance(4, 2, tokens))
    }

    @Test fun `maxHops exceeded returns HOP_UNREACHABLE`() {
        assertEquals(HOP_UNREACHABLE, hopDistance(3, 4, tokens, maxHops = 1))
    }

    @Test fun `disconnected graph returns HOP_UNREACHABLE`() {
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

    private fun candidate(head: UDToken?) = EntityCandidate(
        text = "police", lemma = "police",
        span = Span(7, 13, if (head != null) listOf(head) else emptyList()),
        nerType = NerCoarseType.PER, nerHint = EntityType.HINT_GROUP_ROLE,
        isName = false,
        head = head, headUpos = head?.upos, headDeprel = head?.deprel,
        isPropn = false, isPron = false,
        gender = null, number = null, feats = null,
        sentenceSpan = 0 until 30,
    )

    private val tokens = listOf(
        tok(1, "arrêta", UPOS.VERB,  0, "root",  0,  6),
        tok(2, "police", UPOS.NOUN,  1, "nsubj", 7, 13),
    )

    @Test fun `hop calculated correctly`() {
        val head = tokens[1]  // police id=2, head=1
        val c = candidate(head).withHopFrom(triggerTokenId = 1, sentenceTokens = tokens)
        assertEquals(1, c.hopFromTrigger)
    }

    @Test fun `no head returns HOP_UNREACHABLE`() {
        val c = candidate(null).withHopFrom(triggerTokenId = 1, sentenceTokens = tokens)
        assertEquals(HOP_UNREACHABLE, c.hopFromTrigger)
    }

    @Test fun `isDirectChildOfTrigger true when hop=1`() {
        val c = candidate(tokens[1]).withHopFrom(1, tokens)
        assertTrue(c.isDirectChildOfTrigger)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// reconstructSpan()
// ─────────────────────────────────────────────────────────────────────────────

class ReconstructSpanTest {

    // "général De Gaulle" : général(NOUN) <-flat:name- De(PROPN) <-flat:name- Gaulle(PROPN)
    private val tokens = listOf(
        tok(1, "général", UPOS.NOUN,  0, "nsubj",    0,  7),
        tok(2, "De",      UPOS.PROPN, 1, "flat:name", 8, 10),
        tok(3, "Gaulle",  UPOS.PROPN, 1, "flat:name",11, 17),
    )

    @Test fun `flat-name children are included in span`() {
        val span = reconstructSpan(tokens, headId = 1)
        val ids = span.tokens.map { it.id }
        assertTrue(2 in ids, "De should be in span")
        assertTrue(3 in ids, "Gaulle should be in span")
    }

    @Test fun `span covers full extent`() {
        val span = reconstructSpan(tokens, headId = 1)
        assertEquals(0, span.start)
        assertEquals(17, span.end)
    }

    @Test fun `amod is included`() {
        // "ancien ministre" : ancien(ADJ, amod) → ministre(NOUN, head)
        val toks = listOf(
            tok(1, "ancien",   UPOS.ADJ,  2, "amod",   0,  6),
            tok(2, "ministre", UPOS.NOUN, 0, "nsubj",  7, 15),
        )
        val span = reconstructSpan(toks, headId = 2)
        assertTrue(1 in span.tokens.map { it.id }, "ancien should be in span")
    }

    @Test fun `obl is excluded`() {
        // "arrestation de Martin" : arrestation(NOUN) -nmod-> de(ADP) -case-> Martin
        // obl children should NOT be in the NP span
        val toks = listOf(
            tok(1, "arrestation", UPOS.NOUN,  0, "root",  0, 11),
            tok(2, "de",          UPOS.ADP,   1, "case", 12, 14),
            tok(3, "Martin",      UPOS.PROPN, 1, "obl",  15, 21),
        )
        val span = reconstructSpan(toks, headId = 1)
        assertFalse(3 in span.tokens.map { it.id }, "Martin (obl) should NOT be in span")
    }

    @Test fun `unknown head id returns empty span`() {
        val span = reconstructSpan(tokens, headId = 99)
        assertEquals(0, span.start)
        assertEquals(0, span.end)
        assertTrue(span.tokens.isEmpty())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// mergeNerLabelWithUD() — smoke tests on pure logic
// ─────────────────────────────────────────────────────────────────────────────

class MergeNerLabelWithUDTest {

    // Sentence: "Jacques Martin fut arrêté"  (offsets 0..24)
    //   0: Jacques (PROPN, flat:name of Martin)
    //   1: Martin  (PROPN, nsubj of fut)
    //   2: fut     (AUX,   aux   of arrêté)
    //   3: arrêté  (VERB,  root)
    private val toks = listOf(
        tok(1, "Jacques", UPOS.PROPN, 2, "flat:name", 0,  7),
        tok(2, "Martin",  UPOS.PROPN, 4, "nsubj",     8, 14),
        tok(3, "fut",     UPOS.AUX,   4, "aux",       15, 18),
        tok(4, "arrêté",  UPOS.VERB,  0, "root",      19, 25),
    )
    private val sent = sentence(0, toks)
    private val udDoc = doc("Jacques Martin fut arrêté", sent)

    @Test fun `entity aligned to UD sentence is returned enriched`() {
        val ner = Entity("Jacques Martin", "PER", Span(0, 14, emptyList()))
        val result = mergeNerLabelWithUD(listOf(ner), udDoc)
        assertFalse(result.isEmpty(), "Should return at least one enriched entity")
        val e = result.first()
        assertEquals("Jacques Martin", e.text)
        assertEquals("PER", e.type)
    }

    @Test fun `entity outside any UD sentence is returned as-is`() {
        val ner = Entity("hors doc", "LOC", Span(500, 510, emptyList()))
        val result = mergeNerLabelWithUD(listOf(ner), udDoc)
        assertEquals(1, result.size)
        assertEquals("hors doc", result.first().text)
    }

    @Test fun `VERB head entity is filtered out`() {
        // arrêté (VERB, id=4, offset 19-25) tagged as EVENT → should be dropped
        val ner = Entity("arrêté", "EVENT", Span(19, 25, emptyList()))
        val result = mergeNerLabelWithUD(listOf(ner), udDoc)
        assertTrue(result.isEmpty(), "VERB-head EVENT entity should be filtered")
    }

    @Test fun `leading DET is trimmed`() {
        // Sentence with DET before NOUN: "la police" (DET at 0-2, NOUN at 3-9)
        val tokens2 = listOf(
            tok(1, "la",     UPOS.DET,  2, "det",   0,  2),
            tok(2, "police", UPOS.NOUN, 3, "nsubj", 3,  9),
            tok(3, "arriva", UPOS.VERB, 0, "root", 10, 16),
        )
        val ud = doc("la police arriva", sentence(0, tokens2))
        val ner = Entity("la police", "PER", Span(0, 9, emptyList()))
        val result = mergeNerLabelWithUD(listOf(ner), ud)
        // Entity should be kept (NOUN head), text may be trimmed to "police"
        assertFalse(result.isEmpty())
        // Head token should NOT be the DET
        val head = result.first().metadata["headUpos"] as? String
        assertNotEquals("DET", head)
    }

    @Test fun `empty NER list returns empty`() {
        val result = mergeNerLabelWithUD(emptyList(), udDoc)
        assertTrue(result.isEmpty())
    }

    @Test fun `multiple entities are all processed`() {
        val entities = listOf(
            Entity("Jacques Martin", "PER",  Span(0, 14, emptyList())),
            Entity("arrêté",         "EVENT", Span(19, 25, emptyList())),
        )
        val result = mergeNerLabelWithUD(entities, udDoc)
        // PER entity should survive, EVENT/VERB should be filtered
        assertTrue(result.any { it.type == "PER" })
    }
}

