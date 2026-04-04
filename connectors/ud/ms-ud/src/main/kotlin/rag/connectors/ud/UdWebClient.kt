package rag.connectors.ud

import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.core.JsonParser
import com.fasterxml.jackson.databind.DeserializationContext
import com.fasterxml.jackson.databind.JsonDeserializer
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.annotation.JsonDeserialize
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.asFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.coroutines.withTimeout
import org.springframework.web.reactive.function.client.WebClient
import org.springframework.web.reactive.function.client.WebClientResponseException
import org.springframework.web.reactive.function.client.awaitBody
import org.slf4j.LoggerFactory
import java.net.ConnectException
import java.net.SocketTimeoutException
import kotlin.time.Duration.Companion.milliseconds

// -----------------------------
// Top-level enums & data types
// -----------------------------

enum class UPOS {
    ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X;

    companion object {
        @JvmStatic
        @JsonCreator
        fun from(value: String?): UPOS? {
            if (value == null) return null
            val v = value.trim().replace('.', ' ').uppercase()
            return UPOS.entries.firstOrNull { it.name == v }
        }
    }
}

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

data class UDFeats(
    val number: NumberValue? = null,
    val gender: GenderValue? = null,
    val case_: String? = null,
    val person: PersonValue? = null,
    val tense: TenseValue? = null,
    val mood: MoodValue? = null,
    val degree: DegreeValue? = null,
    val verbForm: String? = null,
    val polarity: PolarityValue? = null,
    val pronType: PronTypeValue? = null,
    val reflex: String? = null,
    val animacy: String? = null,
    val numType: String? = null,
    val aspect: AspectValue? = null,
    val voice: VoiceValue? = null,
    val other: Map<String, String>? = null
)

// light parsers for common feature string values
private fun parseNumber(s: String?): NumberValue? {
    if (s == null) return null
    return when (s.trim().uppercase()) {
        "SG", "SING", "SINGULAR" -> NumberValue.SG
        "PL", "PLUR", "PLURAL" -> NumberValue.PL
        "DUAL" -> NumberValue.DUAL
        else -> null
    }
}

private fun parseGender(s: String?): GenderValue? {
    if (s == null) return null
    return when (s.trim().uppercase()) {
        "MASC", "M", "MASCULINE" -> GenderValue.MASC
        "FEM", "F", "FEMININE" -> GenderValue.FEM
        "NEUT", "N", "NEUTRAL" -> GenderValue.NEUT
        else -> null
    }
}

private fun parsePerson(s: String?): PersonValue? {
    if (s == null) return null
    return when (s.trim().uppercase()) {
        "1", "1ST", "ONE" -> PersonValue.ONE
        "2", "2ND", "TWO" -> PersonValue.TWO
        "3", "3RD", "THREE" -> PersonValue.THREE
        else -> null
    }
}

private fun parseTense(s: String?): TenseValue? {
    if (s == null) return null
    return when (s.trim().uppercase()) {
        "PRES", "PRESENT" -> TenseValue.PRES
        "PAST", "PERF" -> TenseValue.PAST
        "FUT", "FUTURE" -> TenseValue.FUT
        "IMPF", "IMPERF" -> TenseValue.IMPF
        else -> null
    }
}

private fun parseMood(s: String?): MoodValue? {
    if (s == null) return null
    return when (s.trim().uppercase()) {
        "IND" -> MoodValue.IND
        "SUB" -> MoodValue.SUB
        "IMP" -> MoodValue.IMP
        "COND" -> MoodValue.COND
        else -> null
    }
}

private fun parseDegree(s: String?): DegreeValue? {
    if (s == null) return null
    return when (s.trim().uppercase()) {
        "POS", "POSITIVE" -> DegreeValue.POS
        "CMP", "COMPARATIVE" -> DegreeValue.COMP
        "SUP", "SUPERLATIVE" -> DegreeValue.SUP
        else -> null
    }
}

private fun parseAspect(s: String?): AspectValue? {
    if (s == null) return null
    return when (s.trim().uppercase()) {
        "PERF", "PERFECT" -> AspectValue.PERF
        "PROG", "PROGRESSIVE" -> AspectValue.PROG
        "IPFV", "IMPERFECTIVE" -> AspectValue.IPFV
        else -> null
    }
}

private fun parseVoice(s: String?): VoiceValue? {
    if (s == null) return null
    return when (s.trim().uppercase()) {
        "ACT", "ACTIVE" -> VoiceValue.ACT
        "PASS", "PASSIVE" -> VoiceValue.PASS
        "MID", "MIDDLE" -> VoiceValue.MID
        else -> null
    }
}

private fun parsePolarity(s: String?): PolarityValue? = when (s?.trim()?.uppercase()) {
    "NEG", "-" -> PolarityValue.NEG
    "+", "POS", "P" -> PolarityValue.POS
    else -> null
}

private fun parsePronType(s: String?): PronTypeValue? = when (s?.trim()?.uppercase()) {
    "PRS" -> PronTypeValue.PRS
    "REL" -> PronTypeValue.REL
    "DEM" -> PronTypeValue.DEM
    "INT" -> PronTypeValue.INT
    "INDF" -> PronTypeValue.INDF
    "TOT" -> PronTypeValue.TOT
    "NEG" -> PronTypeValue.NEG
    "RECIP" -> PronTypeValue.RECIP
    else -> null
}

private fun Map<String, String>?.toUDFeats(): UDFeats {
    if (this == null) return UDFeats()
    fun fetch(k: String) = this.entries.firstOrNull { it.key.equals(k, ignoreCase = true) }?.value
    val number = parseNumber(fetch("Number") ?: fetch("number"))
    val gender = parseGender(fetch("Gender") ?: fetch("gender"))
    val casev = fetch("Case") ?: fetch("case")
    val person = parsePerson(fetch("Person") ?: fetch("person"))
    val tense = parseTense(fetch("Tense") ?: fetch("tense"))
    val mood = parseMood(fetch("Mood") ?: fetch("mood"))
    val degree = parseDegree(fetch("Degree") ?: fetch("degree"))
    val verbForm = fetch("VerbForm") ?: fetch("verbForm")
    val polarity = parsePolarity(fetch("Polarity") ?: fetch("polarity"))
    val pronType = parsePronType(fetch("PronType") ?: fetch("pronType"))
    val reflex = fetch("Reflex") ?: fetch("reflex")
    val animacy = fetch("Animacy") ?: fetch("animacy")
    val numType = fetch("NumType") ?: fetch("numType")
    val aspect = parseAspect(fetch("Aspect") ?: fetch("aspect"))
    val voice = parseVoice(fetch("Voice") ?: fetch("voice"))
    val others = this.filterKeys { k -> listOf("Number","Gender","Case","Person","Tense","Mood","Degree","VerbForm","Polarity","PronType","Reflex","Animacy","NumType","Aspect","Voice").none { it.equals(k, ignoreCase = true) } }
    return UDFeats(
        number = number,
        gender = gender,
        case_ = casev,
        person = person,
        tense = tense,
        mood = mood,
        degree = degree,
        verbForm = verbForm,
        polarity = polarity,
        pronType = pronType,
        reflex = reflex,
        animacy = animacy,
        numType = numType,
        aspect = aspect,
        voice = voice,
        other = if (others.isEmpty()) null else others
    )
}

// -----------------------------
// Request/response DTOs
// -----------------------------

data class UDRequest(val text: String, val lang: String)

data class UDToken(
    val id: Int? = null,
    val text: String,
    val lemma: String? = null,
    val upos: UPOS? = null,
    val xpos: String? = null,
    val start: Int,
    val end: Int,
    val head: Int? = null,
    val deprel: String? = null,
    @field:JsonDeserialize(using = FeatsDeserializer::class)
    val feats: Map<String, String>? = null
) {
    @Suppress("unused")
    fun typedFeats(): UDFeats = feats.toUDFeats()
}

// Custom Jackson deserializer that accepts either a JSON object or a pipe-separated string
class FeatsDeserializer : JsonDeserializer<Map<String, String>?>() {
    override fun deserialize(p: JsonParser, ctxt: DeserializationContext): Map<String, String>? {
        val node: JsonNode = p.codec.readTree(p)
        if (node.isNull) return null
        if (node.isObject) {
            val res = mutableMapOf<String, String>()
            val fields = node.fields()
            while (fields.hasNext()) {
                val e = fields.next()
                val key = e.key
                val value = if (e.value.isNull) "" else e.value.asText()
                res[key] = value
            }
            return res
        }
        if (node.isTextual) {
            val text = node.asText().trim()
            if (text.isEmpty()) return emptyMap()
            val res = mutableMapOf<String, String>()
            text.split('|').forEach { part ->
                val kv = part.split('=', limit = 2)
                if (kv.size == 2) {
                    val k = kv[0].trim()
                    val v = kv[1].trim()
                    if (k.isNotEmpty()) res[k] = v
                }
            }
            return res
        }
        // Unknown format: return null to let caller handle
        return null
    }
}

data class UDResponse(val tokens: List<UDToken>)

data class BatchUDRequest(val texts: List<String>, val lang: String)

// -----------------------------
// Client
// -----------------------------

class UdWebClient(
    private val webClient: WebClient,
    private val basePath: String = "/ud",
    private val batchSize: Int = 16
) {

    private val log = LoggerFactory.getLogger(UdWebClient::class.java)

    sealed class UdError {
        data class Http(val status: Int, val body: String?) : UdError()
        data class Timeout(val cause: Throwable) : UdError()
        data class Network(val cause: Throwable) : UdError()
        data class Decode(val cause: Throwable, val body: String?) : UdError()
        data class Unknown(val cause: Throwable) : UdError()
    }

    sealed class UdResult {
        data class Success(val resp: UDResponse) : UdResult()
        data class Failure(val error: UdError) : UdResult()
    }

    suspend fun parse(text: String, lang: String = "fr", requestTimeoutMs: Long = DEFAULT_REQUEST_TIMEOUT_MS): UdResult {
        val req = UDRequest(text = text, lang = lang)
        return try {
            // diagnostic logging: log the basePath and request payload
            log.debug("UdWebClient.parse POST {} payload={}", basePath, if (text.length > 200) text.substring(0, 200) + "..." else text)
            // withTimeout garantit qu'on ne pend pas indéfiniment si le service ne répond pas
            // (Reactor Netty lève ReadTimeoutException et non SocketTimeoutException)
            val resp = withTimeout(requestTimeoutMs.milliseconds) {
                webClient.post()
                    .uri(basePath)
                    .bodyValue(req)
                    .retrieve()
                    .awaitBody<UDResponse>()
            }
            log.debug("UdWebClient.parse received tokens={} for textLen={}", resp.tokens.size, text.length)
            UdResult.Success(resp)
        } catch (e: TimeoutCancellationException) {
            // timeout coroutine (withTimeout) — doit être attrapé AVANT Throwable
            log.warn("UD request timed out after {}ms for textLen={}", requestTimeoutMs, text.length)
            UdResult.Failure(UdError.Timeout(e))
        } catch (e: WebClientResponseException) {
            val body = try { e.responseBodyAsString } catch (_: Throwable) { null }
            log.warn("UD HTTP error status={} body={}", e.statusCode.value(), body)
            log.debug("Stacktrace:", e)
            UdResult.Failure(UdError.Http(e.statusCode.value(), body))
        } catch (e: SocketTimeoutException) {
            log.warn("UD request timeout (socket): {}", e.message)
            UdResult.Failure(UdError.Timeout(e))
        } catch (e: ConnectException) {
            log.warn("UD network/connection error: {}", e.message)
            UdResult.Failure(UdError.Network(e))
        } catch (e: Throwable) {
            log.error("UD unknown error while calling {}: {}", basePath, e.message, e)
            UdResult.Failure(UdError.Unknown(e))
        }
    }

    suspend fun parseBatchConcurrent(
        texts: List<String>,
        lang: String = "fr",
        parallelism: Int = 4,
        maxRetries: Int = 2,
        baseDelayMs: Long = 200,
        requestTimeoutMs: Long = DEFAULT_REQUEST_TIMEOUT_MS
    ): List<UdResult> = coroutineScope {
        val sem = Semaphore(parallelism)
        val deferred = texts.map { txt ->
            async {
                var attempt = 0
                var last: UdResult? = null
                while (attempt <= maxRetries) {
                    // Le sémaphore n'est tenu QUE pendant l'appel réseau,
                    // PAS pendant le délai de backoff → évite la famine/lock.
                    val r = sem.withPermit { parse(txt, lang, requestTimeoutMs) }
                    if (r is UdResult.Success) return@async r
                    val shouldRetry = when ((r as UdResult.Failure).error) {
                        is UdError.Network -> true
                        is UdError.Timeout -> true
                        is UdError.Http -> (r.error as UdError.Http).status in 500..599
                        is UdError.Decode -> false
                        is UdError.Unknown -> false
                    }
                    last = r
                    if (!shouldRetry) break
                    // Backoff HORS du sémaphore : les autres coroutines peuvent avancer
                    delay((baseDelayMs * (1L shl attempt)).milliseconds)
                    attempt++
                }
                last ?: UdResult.Failure(UdError.Unknown(RuntimeException("Unknown")))
            }
        }
        deferred.awaitAll()
    }

    suspend fun parseBatchFlow(texts: List<String>, lang: String = "fr"): List<UDResponse> {
        if (texts.isEmpty()) return emptyList()
        return texts.chunked(batchSize).asFlow().map { batch ->
            val req = BatchUDRequest(texts = batch, lang = lang)
            try {
                val arr = webClient.post()
                    .uri("$basePath/batch")
                    .bodyValue(req)
                    .retrieve()
                    .awaitBody<Array<UDResponse>>()
                arr.toList()
            } catch (e: WebClientResponseException) {
                val body = try { e.responseBodyAsString } catch (_: Throwable) { null }
                log.warn("UD batch HTTP error: status={} body={}", e.statusCode.value(), body)
                batch.map { UDResponse(tokens = emptyList()) }
            } catch (e: Throwable) {
                log.error("UD batch exception: {}", e.message)
                // On network/timeout/etc, fallback to empty results for this batch (caller can inspect length/order)
                batch.map { UDResponse(tokens = emptyList()) }
            }
        }.toList().flatten()
    }

    // Simple diagnostic helper: send a small ping request to the UD endpoint and return raw response or error
    suspend fun ping(timeoutMs: Long = 2000): Pair<Boolean, String?> {
        return try {
            val req = UDRequest(text = "ping", lang = "fr")
            val resp = webClient.post()
                .uri(basePath)
                .bodyValue(req)
                .retrieve()
                .awaitBody<UDResponse>()
            log.info("UD ping success, tokens=${resp.tokens.size}")
            Pair(true, "ok: tokens=${resp.tokens.size}")
        } catch (e: Throwable) {
            log.warn("UD ping failed: {}", e.message)
            Pair(false, e.message)
        }
    }

    companion object {
        const val DEFAULT_REQUEST_TIMEOUT_MS: Long = 10_000L

        @JvmStatic
        fun main(args: Array<String>) {
            val client = WebClient.builder().baseUrl("http://localhost:8000").build()
            val ud = UdWebClient(client)
            runBlocking {
                val texts = listOf(
                    "Le capitaine des gardes a arrêté deux manifestants hier soir à Paris.",
                    "Les pourparlers avaient démarré en décembre dernier.",
                    "Un incendie a ravagé l'entrepôt près de Marseille.",
                    "La France a remporté le match hier.",
                    "Une manifestation a eu lieu devant l'Assemblée nationale."
                )
//                val r1 = ud.parse(texts)
//                println("BatchFlow size=${r1.size}")
                val r2 = ud.parseBatchConcurrent(texts)
                println("Concurrent size=${r2.size}")
                println(r2.joinToString("\n"))
            }
        }
    }
}
