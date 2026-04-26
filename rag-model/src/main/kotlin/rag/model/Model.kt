package rag.model

enum class RagUnitType { DOCUMENT, PAGE, PARAGRAPH, SENTENCE, TOKEN_GROUP, UNKNOWN }

// --------------------------
// UDDocument = racine du parseur UD
// --------------------------
data class UDDocument(
    val text: String,                           // texte brut complet
    val sentences: List<UDSentence>,            // segmentation UD
    val language: String? = null,               // langue détectée ou fournie
    val metadata: Map<String, String>? = null   // champs libres (source, doc id…)
)


// --------------------------
// UDSentence = phrase UD
// --------------------------
data class UDSentence(
    val id: Int,
    val tokens: List<UDToken>,
    val start: Int,      // offset global dans UDDocument.text
    val end: Int         // offset global fin
)


// --------------------------
// UDToken = token UD complet
// --------------------------
data class UDToken(
    val id: Int,                 // ID dans la phrase (UD = 1..n)
    val text: String,            // forme de surface
    val lemma: String?,          // lemme normalisé
    val upos: UPOS?,             // POS universel
    val xpos: String?,           // POS spécifique au modèle (Stanza/spaCy)
    val head: Int,               // ID du parent (0 = root)
    val deprel: String,          // relation UD brute ("nsubj", "obl:loc", "acl:relcl"...)
    val start: Int,              // offset UTF-16 dans UDDocument.text
    val end: Int,                // offset UTF-16 fin
    val feats: UDFeats? = null,  // morpho UD complète
    val misc: Map<String, String>? = null // infos optionnelles ex: SpaceAfter=No
)


// --------------------------
// Morphological Features UD v2 (Tu avais déjà la bonne base)
// --------------------------
data class UDFeats(
    val number: NumberValue? = null,
    val gender: GenderValue? = null,
    val person: PersonValue? = null,
    val tense: TenseValue? = null,
    val mood: MoodValue? = null,
    val aspect: AspectValue? = null,
    val degree: DegreeValue? = null,
    val case_: String? = null,
    val voice: VoiceValue? = null,
    val polarity: PolarityValue? = null,
    val pronType: PronTypeValue? = null,
    val verbForm: String? = null,       // Inf, Part, Fin, Ger, etc.
    val numType: String? = null,        // Ord, Card, Mult, Frac…
    val animacy: String? = null,        // Animate/Inanimate (slavic languages)
    val reflex: Boolean? = null,        // Reflexif (fr: "se battre")
    val other: Map<String, String>? = null // pour futures extensions UD
)


// --------------------------
// Enums UD v2 - POS universels
// --------------------------
enum class UPOS {
    ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM,
    PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X;

    companion object {
        @JvmStatic
        fun from(value: String?): UPOS? {
            if (value == null) return null
            val v = value.trim().uppercase()
            return entries.firstOrNull { it.name == v }
        }
    }
}


// --------------------------
// Enums pour UDFeats
// --------------------------
enum class NumberValue { SG, PL, DUAL }
enum class GenderValue { MASC, FEM, NEUT }
enum class PersonValue { ONE, TWO, THREE }
enum class TenseValue { PRES, PAST, FUT, IMPF }
enum class MoodValue { IND, SUB, IMP, COND }
enum class DegreeValue { POS, COMP, SUP }
enum class AspectValue { PERF, PROG, IPFV }
enum class VoiceValue { ACT, PASS, MID }
enum class PolarityValue { POS, NEG }
enum class PronTypeValue { PRS, REL, DEM, INT, INDF, TOT, NEG, RECIP }




data class Entity(
    val text: String,
    val type: String,
    val span: Span = Span(),
    val metadata: Map<String, Any?> = emptyMap())

data class Span(
    val start: Int = -1,
    val end: Int = -1,
//    val text: String,
    val tokens: List<UDToken> = emptyList()
)

data class Layout(
    val pageNumber: Int? = null,
    val blockId: String? = null,
    val bbox: List<Float>? = null, // [x0, y0, x1, y1]
    val orientation: Float? = null,
    val readingOrder: Int? = null
)

data class RagElement(
    val type: String,
    val text: String,
    val metadata: Map<String, Any?> = emptyMap(),
    val span: Span? = null,
    val layout: Layout? = null,
    val parentId: String? = null,
    val id: String
)

data class RagDocument(
    val id: String,
    val text: String,
    val type: RagUnitType = RagUnitType.SENTENCE,
    val metadata: Map<String, Any?> = emptyMap(),
    val elements: List<RagElement> = emptyList(),
    val span: Span? = null,
    val layout: Layout? = null,
    val source: String? = null,
    val parentId: String? = null
)
