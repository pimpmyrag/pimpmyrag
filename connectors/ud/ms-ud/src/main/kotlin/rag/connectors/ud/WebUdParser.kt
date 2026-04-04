package rag.connectors.ud

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.Dispatchers
import org.slf4j.LoggerFactory
import rag.engine.UDParser
import rag.model.UDDocument
import rag.model.UDSentence
import rag.model.UDToken    as ModelUDToken
import rag.model.UDFeats    as ModelUDFeats
import rag.model.UPOS       as ModelUPOS
import rag.model.NumberValue  as ModelNumberValue
import rag.model.GenderValue  as ModelGenderValue
import rag.model.PersonValue  as ModelPersonValue
import rag.model.TenseValue   as ModelTenseValue
import rag.model.MoodValue    as ModelMoodValue
import rag.model.AspectValue  as ModelAspectValue
import rag.model.DegreeValue  as ModelDegreeValue
import rag.model.VoiceValue   as ModelVoiceValue
import rag.model.PolarityValue as ModelPolarityValue
import rag.model.PronTypeValue as ModelPronTypeValue
import rag.model.RagDocument

/**
 * Simple UDParser implementation that uses the UdWebClient to call a remote UD service
 * and maps the returned tokens into the internal rag.model.UDDocument structure.
 *
 * Note: the remote response is expected to match UdWebClient. This implementation
 * groups all tokens into a single sentence for now. It is intentionally conservative
 * and sets parsed morphological features into the `misc`/other field if parsing
 * into the strongly-typed ModelUDFeats is non-trivial.
 */
class WebUdParser(
    private val udClient: UdWebClient,
    private val defaultLang: String = "fr"
) : UDParser {

    private val log = LoggerFactory.getLogger(WebUdParser::class.java)

    override fun parse(documents: List<RagDocument>): List<UDDocument> {
        if (documents.isEmpty()) return emptyList()
        val texts = documents.map { it.text }
        log.debug("WebUdParser.parse running on thread={} - offloading UD calls to IO dispatcher", Thread.currentThread().name)
        // Offload the suspend calls to Dispatchers.IO to avoid blocking Netty event-loop threads
        val results = runBlocking(Dispatchers.IO) { udClient.parseBatchConcurrent(texts, lang = defaultLang) }
        log.debug("WebUdParser.parse returned on thread={} resultsSize={}", Thread.currentThread().name, results.size)

        return documents.mapIndexed { idx, doc ->
            val res = results.getOrNull(idx)
            val tokens = when (res) {
                is UdWebClient.UdResult.Success -> res.resp.tokens
                else -> emptyList()
            }
            // Map connector UDToken -> model.UDToken
            val modelTokens = tokens.map { t ->
                ModelUDToken(
                    id = t.id ?: 0,
                    text = t.text,
                    lemma = t.lemma,
                    upos = ModelUPOS.from(t.upos?.name),
                    xpos = t.xpos,
                    head = t.head ?: 0,
                    deprel = t.deprel ?: "",
                    start = t.start,
                    end = t.end,
                    feats = mapToModelFeats(t.feats),
                    misc = null
                )
            }

            val sentence = UDSentence(
                id = 0,
                tokens = modelTokens,
                start = 0,
                end = doc.text.length
            )

            UDDocument(
                text = doc.text,
                sentences = listOf(sentence),
                language = defaultLang,
                metadata = doc.metadata.filterValues { it != null }.mapValues { it.value.toString() }
            )
        }
    }

    private fun mapToModelFeats(feats: Map<String, String>?): ModelUDFeats? {
        if (feats.isNullOrEmpty()) return null

        val known = setOf(
            "Number", "Gender", "Person", "Tense", "Mood",
            "Aspect", "Degree", "Case", "Voice", "Polarity",
            "PronType", "VerbForm", "NumType", "Animacy", "Reflex"
        )

        return ModelUDFeats(
            number   = when (feats["Number"]?.uppercase()) {
                "SG", "SING", "SINGULAR" -> ModelNumberValue.SG
                "PL", "PLUR", "PLURAL"   -> ModelNumberValue.PL
                "DUAL"                   -> ModelNumberValue.DUAL
                else -> null
            },
            gender   = when (feats["Gender"]?.uppercase()) {
                "MASC", "M", "MASCULINE" -> ModelGenderValue.MASC
                "FEM",  "F", "FEMININE"  -> ModelGenderValue.FEM
                "NEUT", "N", "NEUTRAL"   -> ModelGenderValue.NEUT
                else -> null
            },
            person   = when (feats["Person"]) {
                "1", "1st" -> ModelPersonValue.ONE
                "2", "2nd" -> ModelPersonValue.TWO
                "3", "3rd" -> ModelPersonValue.THREE
                else -> null
            },
            tense    = when (feats["Tense"]?.uppercase()) {
                "PRES"         -> ModelTenseValue.PRES
                "PAST", "PST"  -> ModelTenseValue.PAST
                "FUT"          -> ModelTenseValue.FUT
                "IMP", "IMPF"  -> ModelTenseValue.IMPF
                else -> null
            },
            mood     = when (feats["Mood"]?.uppercase()) {
                "IND"          -> ModelMoodValue.IND
                "SUB", "SBJV"  -> ModelMoodValue.SUB
                "IMP"          -> ModelMoodValue.IMP
                "COND"         -> ModelMoodValue.COND
                else -> null
            },
            aspect   = when (feats["Aspect"]?.uppercase()) {
                "PERF"         -> ModelAspectValue.PERF
                "PROG"         -> ModelAspectValue.PROG
                "IPFV", "IMP"  -> ModelAspectValue.IPFV
                else -> null
            },
            degree   = when (feats["Degree"]?.uppercase()) {
                "POS"          -> ModelDegreeValue.POS
                "CMP", "COMP"  -> ModelDegreeValue.COMP
                "SUP"          -> ModelDegreeValue.SUP
                else -> null
            },
            case_    = feats["Case"],
            voice    = when (feats["Voice"]?.uppercase()) {
                "ACT"          -> ModelVoiceValue.ACT
                "PASS"         -> ModelVoiceValue.PASS
                "MID"          -> ModelVoiceValue.MID
                else -> null
            },
            polarity = when (feats["Polarity"]?.uppercase()) {
                "POS"          -> ModelPolarityValue.POS
                "NEG"          -> ModelPolarityValue.NEG
                else -> null
            },
            pronType = when (feats["PronType"]?.uppercase()) {
                "PRS"          -> ModelPronTypeValue.PRS
                "REL"          -> ModelPronTypeValue.REL
                "DEM"          -> ModelPronTypeValue.DEM
                "INT"          -> ModelPronTypeValue.INT
                "IND", "INDF"  -> ModelPronTypeValue.INDF
                "TOT"          -> ModelPronTypeValue.TOT
                "NEG"          -> ModelPronTypeValue.NEG
                "RECIP"        -> ModelPronTypeValue.RECIP
                else -> null
            },
            verbForm = feats["VerbForm"],
            numType  = feats["NumType"],
            animacy  = feats["Animacy"],
            reflex   = feats["Reflex"]?.uppercase() == "YES",
            other    = feats.filterKeys { it !in known }.takeIf { it.isNotEmpty() }
        )
    }
}
