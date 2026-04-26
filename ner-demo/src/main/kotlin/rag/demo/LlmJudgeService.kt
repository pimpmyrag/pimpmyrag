package rag.demo

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration

/** Configuration LLM saisie par l'utilisateur dans le dialog. */
data class LlmJudgeConfig(
    val baseUrl: String = "https://api.openai.com/v1",
    val apiKey: String  = "",
    val model:  String  = "gpt-4o-mini",
)

@Service
class LlmJudgeService(private val mapper: ObjectMapper) {

    private val log  = LoggerFactory.getLogger(LlmJudgeService::class.java)
    private val http = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(30))
        .build()

    // ── Prompt ────────────────────────────────────────────────────────────────

    fun buildPrompt(results: List<AnnotatedSentence>): String {
        val sb = StringBuilder()
        sb.appendLine("""
            You are an expert NLP evaluator specializing in Named Entity Recognition (NER) and Subject-Verb-Object (SVO) extraction.
            Below are the results produced by a DeBERTa-based model on the user's text.
            
            For each sentence, evaluate:
            1. **NER quality** — Are entities correctly identified? Any obvious false positives (wrong type, spurious span)? Any apparent false negatives (missed entities)?
            2. **SVO quality** — Are subject/verb/object roles coherent with the sentence meaning? Any misassigned roles or missed structures?
            3. **Global assessment** — Overall precision/recall tradeoff, coherence of fine-grained labels vs. coarse categories.
            4. **Threshold suggestions** — If you spot systematic over- or under-detection, suggest which threshold (tauBoundary, tauNone, tauCoarse, tauSvo) to adjust and in which direction.
            
            Respond in the SAME LANGUAGE as the analyzed text. Be concise and structured.
            
            --- RESULTS ---
        """.trimIndent())
        sb.appendLine()

        results.forEachIndexed { idx, sent ->
            sb.appendLine("### Sentence ${idx + 1}")
            sb.appendLine("**Text:** ${sent.text}")
            sb.appendLine()

            if (sent.entities.isNotEmpty()) {
                sb.appendLine("**NER entities (${sent.entities.size}):**")
                sent.entities.forEach { e ->
                    val coarse = e.metadata["coarse"] as? String ?: "?"
                    val score  = (e.metadata["score"] as? Float)?.let { "%.3f".format(it) } ?: "?"
                    val pBound = (e.metadata["pBoundary"] as? Float)?.let { "%.3f".format(it) } ?: "?"
                    sb.appendLine("  - \"${e.text}\" → coarse=$coarse fine=${e.type}  score=$score  p_boundary=$pBound  [${e.span?.start}:${e.span?.end}]")
                }
            } else {
                sb.appendLine("**NER entities:** (none)")
            }
            sb.appendLine()

            if (sent.svoSpans.isNotEmpty()) {
                sb.appendLine("**SVO spans (${sent.svoSpans.size}):**")
                sent.svoSpans.forEach { s ->
                    val override = if (s.nerOverride != null) " [ner_override=${s.nerOverride} score=${"%.3f".format(s.nerOverrideScore ?: 0f)}]" else ""
                    val synthetic = if (s.fromNer) " [synthetic]" else ""
                    sb.appendLine("  - [${s.role}] \"${s.text}\"  voice=${s.voice}  p_boundary=${"%.3f".format(s.svoBoundaryProb)}  p_role=${"%.3f".format(s.roleProb)}$override$synthetic")
                }
            } else {
                sb.appendLine("**SVO spans:** (none)")
            }
            sb.appendLine()
        }

        return sb.toString().trim()
    }

    // ── Call ──────────────────────────────────────────────────────────────────

    fun judge(cfg: LlmJudgeConfig, results: List<AnnotatedSentence>): String {
        require(cfg.apiKey.isNotBlank()) { "API key is required" }
        require(results.isNotEmpty())   { "No results to judge" }

        val prompt  = buildPrompt(results)
        val payload = mapper.writeValueAsString(mapOf(
            "model"       to cfg.model,
            "temperature" to 0.2,
            "messages"    to listOf(mapOf("role" to "user", "content" to prompt)),
        ))

        val url = cfg.baseUrl.trimEnd('/') + "/chat/completions"
        log.info("LLM judge → {} (model={})", url, cfg.model)

        val req = HttpRequest.newBuilder(URI.create(url))
            .timeout(Duration.ofSeconds(120))
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer ${cfg.apiKey}")
            .POST(HttpRequest.BodyPublishers.ofString(payload))
            .build()

        val resp = http.send(req, HttpResponse.BodyHandlers.ofString())
        if (resp.statusCode() !in 200..299) {
            log.error("LLM error {}: {}", resp.statusCode(), resp.body())
            error("HTTP ${resp.statusCode()}: ${resp.body().take(300)}")
        }

        // OpenAI-style response : choices[0].message.content
        val json = mapper.readValue<Map<String, Any>>(resp.body())
        @Suppress("UNCHECKED_CAST")
        val choices = json["choices"] as? List<Map<String, Any>>
            ?: error("Unexpected response format")
        @Suppress("UNCHECKED_CAST")
        val message = choices.first()["message"] as? Map<String, Any>
            ?: error("Unexpected message format")
        return message["content"] as? String ?: error("Empty content")
    }
}

