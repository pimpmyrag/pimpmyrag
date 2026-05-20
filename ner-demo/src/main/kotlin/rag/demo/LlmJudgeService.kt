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
    val baseUrl:   String  = "https://api.openai.com/v1",
    val apiKey:    String  = "",
    val model:     String  = "gpt-4o-mini",
    val agentMode: Boolean = false,
)

@Service
class LlmJudgeService(
    private val mapper: ObjectMapper,
    private val mcpTools: NerMcpTools,
) {

    private val log  = LoggerFactory.getLogger(LlmJudgeService::class.java)
    private val http = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(30))
        .build()

    // ── Tool definitions (OpenAI function-calling format) ─────────────────────

    private val toolDefinitions: List<Map<String, Any>> = listOf(
        tool("getConfig",
            "Returns the current NER thresholds (tauBoundary, tauNone, tauCoarse, batchSize). " +
            "tauBoundary is the PRIMARY lever (recall/precision). " +
            "tauNone and tauCoarse are DEBUG/TUNING parameters. " +
            "tauSvoBoundary / tauSvoAnchoredBoundary — SVO thresholds for syntactic role detection.",
            emptyMap()),
        tool("analyzeText",
            "Run entity extraction on a text. Returns entities with fine labels, raw scores, " +
            "SVO syntactic roles, morphology (gender/number/person), and eventlets. " +
            "svoSpans and syntacticRole/gender/number fields enrich entities for graph-based coreference. " +
            "report them briefly as bonus observations after the NER evaluation. " +
            "Fields: " +
            "  coarse = INDICATIVE family only (PER/LOC/ORG/TIME/EVENT/OBJECT/VALUE/ABSTRACT) — not evaluated directly; " +
            "  fine   = THE ACTUAL SEMANTIC LABEL to evaluate (32 values): " +
            "    PER→ hint_person_name, hint_person_role, hint_norp, hint_group_role; " +
            "    LOC→ hint_gpe, hint_fac_name, hint_loc_generic, hint_infra; " +
            "    ORG→ hint_org_name; " +
            "    TIME→ hint_time_date, hint_time_clock, hint_time_duration; " +
            "    EVENT→ hint_event_nominal, hint_event_named; " +
            "    OBJECT→ hint_weapon, hint_vehicle, hint_substance, hint_food, hint_tool, hint_object_generic, hint_object_name; " +
            "    VALUE→ hint_quantity, hint_measure, hint_percentage, hint_count, hint_money, hint_rate; " +
            "    ABSTRACT→ hint_law, hint_work_of_art, hint_concept, hint_disease, hint_language. " +
            "  score = COMPOSITE quality signal (boundary×coarse×fine) — PRIMARY. " +
            "  pBoundary = [DEBUG] raw boundary prob. " +
            "  pCoarse   = [DEBUG] coarse family confidence. " +
            "  pFine     = [DEBUG] fine label confidence — low = ambiguity. " +
            "  syntacticRole = subject|object|oblique|appos on the entity; svoSpans = full argument list.",
            mapOf("text" to (mapOf("type" to "string",
                "description" to "The text to analyze (one sentence or short paragraph)") to true))),
        tool("setThreshold",
            "Update one NER threshold by name. " +
            "tauBoundary is the PRIMARY lever — start here. " +
            "tauNone and tauCoarse are DEBUG/TUNING parameters, rarely need changing. " +
            "Valid: tauBoundary | tauNone | tauCoarse | tauSvoBoundary | tauSvoAnchoredBoundary. " +
            "Takes effect immediately for all subsequent analyzeText calls.",
            mapOf(
                "name"  to (mapOf("type" to "string",
                    "description" to "tauBoundary (primary) | tauNone | tauCoarse | tauSvoBoundary | tauSvoAnchoredBoundary") to true),
                "value" to (mapOf("type" to "number", "description" to "New float value") to true),
            )),
        tool("scanThreshold",
            "Sweep tauBoundary (or tauNone/tauCoarse) over a range on a reference text. " +
            "Reports entity counts and coarse-family breakdown (indicative) at each step. " +
            "Look for the 'elbow' — where entity count stabilises — to find the optimal tauBoundary. " +
            "Config is automatically restored after the sweep.",
            mapOf(
                "text"      to (mapOf("type" to "string",  "description" to "Reference text") to true),
                "threshold" to (mapOf("type" to "string",  "description" to "tauBoundary (primary) | tauNone | tauCoarse") to true),
                "from"      to (mapOf("type" to "number",  "description" to "Start of range e.g. 0.30") to true),
                "to"        to (mapOf("type" to "number",  "description" to "End of range e.g. 0.90") to true),
                "step"      to (mapOf("type" to "number",  "description" to "Step size e.g. 0.05") to true),
            )),
        tool("analyzeBatch",
            "Analyze multiple texts (max 30) and return aggregated NER stats per coarse family (indicative). " +
            "Useful to detect which fine-label families have low average confidence across a corpus. " +
            "lowConfidenceEntities (score < 0.70) are the most likely false positives. " +
            "svoRoleDistribution and svoEntityCoverage show syntactic role coverage across the corpus.",
            mapOf("texts" to (mapOf(
                "type" to "array", "items" to mapOf("type" to "string"),
                "description" to "List of texts (max 30)") to true))),
        tool("evaluateSvo",
            "Run inference and return SVO-focused output for a qualitative assessment. " +
            "Use this after analyzeText to get a detailed view of detected syntactic arguments, " +
            "morphology, eventlets (structured subject-verb-object tuples), and voice/certainty. " +
            "Returns: argumentSpans (subject/object/iobj with entity association), verbSpans, summary. " +
            "Mention briefly: nb correct roles, any suspicious role, entity association quality.",
            mapOf("text" to (mapOf("type" to "string",
                "description" to "The text to analyze") to true))),
    )

    private fun tool(
        name: String, description: String,
        params: Map<String, Pair<Map<String, Any>, Boolean>>,
    ): Map<String, Any> = mapOf(
        "type" to "function",
        "function" to mapOf(
            "name"        to name,
            "description" to description,
            "parameters"  to mapOf(
                "type"       to "object",
                "properties" to params.mapValues { it.value.first },
                "required"   to params.filter { it.value.second }.keys.toList(),
            ),
        ),
    )

    // ── Tool dispatcher (direct JVM calls — no HTTP round-trip) ──────────────

    @Suppress("UNCHECKED_CAST")
    private fun dispatchTool(name: String, argsJson: String): String {
        val args: Map<String, Any> = if (argsJson.isBlank()) emptyMap()
                                     else mapper.readValue(argsJson)
        val result: Any = when (name) {
            "getConfig"     -> mcpTools.getConfig()
            "analyzeText"   -> mcpTools.analyzeText(args["text"] as String)
            "setThreshold"  -> mcpTools.setThreshold(
                args["name"] as String,
                (args["value"] as Number).toFloat(),
            )
            "scanThreshold" -> mcpTools.scanThreshold(
                text      = args["text"] as String,
                threshold = args["threshold"] as String,
                from      = (args["from"] as Number).toFloat(),
                to        = (args["to"] as Number).toFloat(),
                step      = (args["step"] as Number).toFloat(),
            )
            "analyzeBatch"  -> mcpTools.analyzeBatch(
                (args["texts"] as List<*>).filterIsInstance<String>()
            )
            "evaluateSvo"        -> mcpTools.evaluateSvoPreview(args["text"] as String)
            "evaluateSvoPreview" -> mcpTools.evaluateSvoPreview(args["text"] as String) // compat
            else -> mapOf("error" to "Unknown tool: $name")
        }
        return mapper.writeValueAsString(result)
    }

    // ── Entry points ──────────────────────────────────────────────────────────

    fun judge(cfg: LlmJudgeConfig, results: List<AnnotatedSentence>): String {
        require(cfg.apiKey.isNotBlank()) { "API key is required" }
        require(results.isNotEmpty())    { "No results to judge" }
        return if (cfg.agentMode) agentJudge(cfg, results) else staticJudge(cfg, results)
    }

    /**
     * Streaming version — [onProgress] est appelé depuis le thread HTTP à chaque chunk.
     * - mode static  : onProgress(texte accumulé, isTrace=false)
     * - mode agent   : onProgress(ligne de trace d'outil, isTrace=true) puis tokens finaux
     * Retourne le verdict final (même format que [judge]).
     */
    fun judgeStream(
        cfg: LlmJudgeConfig,
        results: List<AnnotatedSentence>,
        onProgress: (text: String, isTrace: Boolean) -> Unit,
    ): String {
        require(cfg.apiKey.isNotBlank()) { "API key is required" }
        require(results.isNotEmpty())    { "No results to judge" }
        return if (cfg.agentMode) agentJudge(cfg, results, onProgress)
               else staticJudgeStream(cfg, results, onProgress)
    }

    // ── Static judge (one-shot flat prompt) ───────────────────────────────────

    fun buildPrompt(results: List<AnnotatedSentence>): String {
        val sb = StringBuilder()
        sb.appendLine("""
You are an expert NLP evaluator specialised in structured entity extraction for RAG systems.

═══════════════════════════════════════════════════════
⚠  THIS IS NOT CLASSICAL NER — READ CAREFULLY
═══════════════════════════════════════════════════════
This model is an entity extractor designed for a Retrieval-Augmented Generation (RAG) pipeline.
It is NOT a standard CoNLL-style NER system. Key differences:

1) NESTED / OVERLAPPING SPANS ARE INTENTIONAL AND DESIRED.
   The model deliberately emits overlapping entities when a larger span contains
   a semantically distinct sub-entity. This is a FEATURE, not an error.
   Examples of useful nesting:
     • "bataille de Stalingrad" → EVENT parent + GPE child "Stalingrad" (location info inside event)
     • "hôpital Necker" → FAC parent + ORG child (institution name within facility)
     • "président de la République française" → PERSON_ROLE parent + GPE child "République française"
     • "ministère de la Défense" → INST parent + FIELD child "Défense" (domain info)
     • "attentat du 13 novembre" → EVENT parent + TIME_DATE child "13 novembre"
     • "base militaire de Djibouti" → FAC parent + GPE child "Djibouti"
   The PARENT (wider span) is generally the most important entity for RAG indexing.
   The CHILDREN bring additional structured info (location, date, org inside an event, etc.).
   Do NOT count nested children as false positives or duplicates — they are bonus metadata.
   When evaluating, assess parent and children INDEPENDENTLY. Both can be correct simultaneously.

2) BROADER SCOPE THAN CLASSICAL NER.
   The 38 fine labels cover events, abstract concepts, values, objects, works, diseases, etc.
   Classical NER benchmarks (CoNLL, OntoNotes) cover only PER/LOC/ORG/MISC — comparisons
   should acknowledge this much wider scope. An entity like "la crise économique" (EVENT)
   or "3 km" (MEASURE) would never be annotated in CoNLL but is critical for RAG retrieval.

3) THE GOAL IS RECALL-ORIENTED.
   For a RAG candidate generator, a missed entity is permanently lost downstream (no second chance).
   A false positive can be filtered later by the retriever/reranker.
   Prefer high recall over high precision — FN are costlier than FP.

SVO & MORPHOLOGY — GRAPH-BASED COREFERENCE (non-destructive, additive):
The model detects syntactic roles (SUBJECT/OBJECT/OBLIQUE…), pronouns, and morphology
(gender/number/person) in the SAME forward pass as entity extraction.
These features feed an asynchronous GRAPH-BASED coreference pipeline (not ML-based):
  - Pronouns + gender/number/person → deterministic pronoun-to-entity resolution via graph matching
  - SVO roles → participant tracking across sentences (who did what to whom)
  - Eventlets = structured (subject, verb, object) tuples with voice and certainty
  - The graph approach is non-destructive: it enriches entities without modifying predictions.
Evaluate SVO quality alongside entity extraction — it is a full feature, not a preview.
Note any SVO roles / morpho fields per sentence.

═══════════════════════════════════════════════════════
MODEL ARCHITECTURE
═══════════════════════════════════════════════════════
Single DeBERTa-v3 model — multi-head span-based NER (ONE forward pass, no two-stage pipeline).
For each candidate span the model runs FOUR decoding heads simultaneously:
  • boundary head → "is this span an entity?"           prob = pBoundary
  • coarse head   → "which broad family (9 + NONE)?"   prob = pCoarse
  • fine head     → "which of 38 fine labels?"          prob = pFine
  • SVO heads     → syntactic role / voice / gender / number  [graph-based coref]

score = pBoundary × pCoarse × pFine  (harsh composite — all three heads must agree)

⚠ COARSE = indicative family only (display colour + structural masking via COARSE_TO_FINE).
  It is NOT an evaluated field. Do NOT judge quality on coarse alone.
  pCoarse and tauCoarse are debug/tuning params, not primary quality signals.

═══════════════════════════════════════════════════════
FINE LABELS (38) — the actual predictions to evaluate
═══════════════════════════════════════════════════════
PER  → hint_person_name (proper name), hint_person_role (title/function),
        hint_norp (nationality/ethnic/religious group), hint_group_role (collective human group)
LOC  → hint_gpe (geopolitical entity), hint_fac_name (named built place),
        hint_loc_generic (unnamed geographic feature), hint_infra (named infrastructure)
ORG  → hint_org_name (named formal organisation), hint_inst_name (named public institution),
        hint_inst_role (generic institution without qualifier)
TIME → hint_time_date (calendar date), hint_time_clock (time of day), hint_time_duration (interval)
EVENT→ hint_event_nominal (unnamed event noun), hint_event_named (named event)
OBJECT→ hint_weapon, hint_vehicle, hint_substance, hint_food, hint_tool,
         hint_object_generic, hint_object_name (named product)
VALUE → hint_measure (qty+unit), hint_percentage, hint_count,
         hint_money (monetary amount), hint_rate (ratio/index)
WORK  → hint_work_of_art (named work), hint_law (legal text),
         hint_document (report, contract…), hint_work_generic (generic cultural production)
ABSTRACT→ hint_disease, hint_language, hint_doctrine (ideology/doctrine),
           hint_state (abstract state/condition), hint_notion (pure abstract concept),
           hint_field (domain/sector of activity)

═══════════════════════════════════════════════════════
SCORES — semantics
═══════════════════════════════════════════════════════
  score     → COMPOSITE (boundary×coarse×fine). PRIMARY quality signal. Kept iff >= tauBoundary.
  pBoundary → [DEBUG/TUNING] Raw BILOU boundary probability. Use to diagnose FP/FN at threshold.
  pCoarse   → [DEBUG/TUNING] BILOU coarse family confidence. Only relevant for coarse mis-routing.
  pFine     → [DEBUG/TUNING] SpanClassifier fine label confidence. Low = fine-label ambiguity.
              When altFine is present (pFine < 0.60), it shows the runner-up fine label — useful
              to understand model hesitation between two labels in the same family.

═══════════════════════════════════════════════════════
EVALUATION TASKS
═══════════════════════════════════════════════════════
1) Per-sentence entity quality: FP, FN, wrong fine label, boundary errors.
   ⚠ Nested spans (marked "nested=true" with parentText/parentFine) are EXPECTED.
   Evaluate parent and child independently — both being correct is the ideal outcome.
   The parent (wider) span carries the main semantic value ; the child adds structured detail.
2) Overall precision/recall estimation and score distribution analysis.
   Remember: this system prioritises recall over precision for RAG retrieval.
3) Threshold recommendations (tauBoundary primarily) with concrete numeric values.
4) MANDATORY CONCLUSION — Comparative market score /10 vs:
   spaCy (fr_dep_news_trf / en_core_web_trf), CamemBERT-NER, Flair, Stanza,
   Azure Text Analytics, AWS Comprehend, Google Natural Language API.
   Reference publicly known F1 benchmarks (CoNLL-2003, WikiNER, FQuAD-NER…).
   ⚠ Remember that this model covers 38 fine labels (not 4) and produces nested spans —
   a direct F1 comparison with flat-NER systems is structurally unfair. Acknowledge this.

Respond in the SAME LANGUAGE as the analyzed text.

--- ENTITY EXTRACTION RESULTS ---
        """.trimIndent())
        results.forEachIndexed { idx, sent ->
            sb.appendLine("\n### Sentence ${idx + 1}\n**Text:** ${sent.text}")
            if (sent.entities.isNotEmpty()) {
                sb.appendLine("**NER (${sent.entities.size}):**")
                sent.entities.forEach { e ->
                    val coarse = e.metadata["coarse"] as? String ?: "?"
                    val score  = (e.metadata["score"] as? Float)?.let { "%.3f".format(it) } ?: "?"
                    val pBound = (e.metadata["pBoundary"] as? Float)?.let { "%.3f".format(it) } ?: "?"
                    val pCoarse= (e.metadata["pCoarse"] as? Float)?.let { "%.3f".format(it) } ?: "?"
                    val pFine  = (e.metadata["pFine"] as? Float)?.let { "%.3f".format(it) } ?: "?"
                    val svoExtra = buildString {
                        (e.metadata["syntacticRole"] as? String)?.let { append(" role=$it") }
                        (e.metadata["gender"] as? String)?.let { append(" gender=$it") }
                        (e.metadata["number"] as? String)?.let { append(" number=$it") }
                        (e.metadata["altFine"] as? String)?.let { alt ->
                            val altP = (e.metadata["altPFine"] as? Float)?.let { "%.3f".format(it) } ?: "?"
                            append(" altFine=$alt($altP)")
                        }
                    }
                    sb.appendLine("  - \"${e.text}\" fine=${e.type} coarse(indicative)=$coarse score=$score pBoundary(debug)=$pBound pCoarse(debug)=$pCoarse pFine(debug)=$pFine$svoExtra")
                }
            } else sb.appendLine("**NER:** (none)")
            // SVO section
            if (sent.svoSpans.isNotEmpty()) {
                sb.appendLine("**SVO (${sent.svoSpans.size} spans):**")
                sent.svoSpans.forEach { s ->
                    val entity = s.entity?.let { " → entity:\"${it.text}\"(${it.type})" } ?: ""
                    sb.appendLine("  - [${s.role}] \"${s.text}\" p_bnd=${"%.2f".format(s.svoBoundaryProb)} p_role=${"%.2f".format(s.roleProb)} voice=${s.voice}$entity")
                }
            }
        }
        return sb.toString().trim()
    }

    private fun staticJudge(cfg: LlmJudgeConfig, results: List<AnnotatedSentence>): String {
        val payload = mapper.writeValueAsString(mapOf(
            "model"       to cfg.model,
            "temperature" to 0.2,
            "messages"    to listOf(mapOf("role" to "user", "content" to buildPrompt(results))),
        ))
        return extractContent(chatCompletions(cfg, payload))
    }

    private fun staticJudgeStream(
        cfg: LlmJudgeConfig,
        results: List<AnnotatedSentence>,
        onProgress: (String, Boolean) -> Unit,
    ): String {
        val payload = mapper.writeValueAsString(mapOf(
            "model"       to cfg.model,
            "temperature" to 0.2,
            "stream"      to true,
            "messages"    to listOf(mapOf("role" to "user", "content" to buildPrompt(results))),
        ))
        val sb = StringBuilder()
        chatCompletionsStream(cfg, payload) { chunk ->
            sb.append(chunk)
            onProgress(sb.toString(), false)
        }
        return sb.toString()
    }

    // ── Agent judge (tool-calling loop) ──────────────────────────────────────

    private fun agentJudge(
        cfg: LlmJudgeConfig,
        results: List<AnnotatedSentence>,
        onProgress: (String, Boolean) -> Unit = { _, _ -> },
    ): String {
        val systemPrompt = """
            You are an expert NLP evaluator specialised in structured entity extraction for RAG systems.
            You MUST use the provided tools to explore the live model before writing any verdict.
            Never give a final answer without first calling at least getConfig() and analyzeText().

            ═══════════════════════════════════════════════════════
            ⚠  THIS IS NOT CLASSICAL NER — READ CAREFULLY
            ═══════════════════════════════════════════════════════
            This model is an entity extractor for a Retrieval-Augmented Generation (RAG) pipeline.
            It is NOT a standard CoNLL-style NER. Key differences:

            1) NESTED / OVERLAPPING SPANS ARE INTENTIONAL AND DESIRED.
               The model deliberately emits overlapping entities when a larger span contains
               a semantically distinct sub-entity. This is a FEATURE, not an error.
               Examples of useful nesting:
                 • "bataille de Stalingrad" → EVENT parent + GPE child "Stalingrad"
                 • "hôpital Necker" → FAC parent + ORG child
                 • "président de la République française" → PERSON_ROLE parent + GPE child
                 • "ministère de la Défense" → INST parent + FIELD child "Défense"
                 • "attentat du 13 novembre" → EVENT parent + TIME_DATE child
                 • "base militaire de Djibouti" → FAC parent + GPE child "Djibouti"
               The PARENT (wider span) is the most important entity for RAG indexing.
               CHILDREN add structured detail (loc, date, org inside event, etc.).
               Do NOT count nested children as false positives or duplicates.
               Assess parent and child INDEPENDENTLY — both correct simultaneously is ideal.

            2) BROADER SCOPE THAN CLASSICAL NER.
               38 fine labels covering events, abstract concepts, values, objects, works, diseases…
               Classical NER (CoNLL/OntoNotes) covers only PER/LOC/ORG/MISC — acknowledge this
               much wider scope when comparing. "la crise" (EVENT) or "3 km" (MEASURE) are valid here.

            3) RECALL-ORIENTED DESIGN.
               For a RAG candidate generator, a missed entity is permanently lost downstream.
               A false positive can be filtered later. FN >> FP in cost.

            SVO & MORPHOLOGY — GRAPH-BASED COREFERENCE (non-destructive, additive):
            The model detects syntactic argument roles (SUBJECT/OBJECT/OBLIQUE…), pronouns,
            and morphology (gender/number/person) in the SAME forward pass as entity extraction.
            These features feed an asynchronous GRAPH-BASED coreference pipeline (not ML-based):
              - Pronouns + gender/number/person → deterministic pronoun-to-entity resolution via graph matching
              - SVO roles → participant tracking across sentences (who did what to whom)
              - Eventlets = structured (subject, verb, object) tuples with voice and certainty
              - The graph approach is non-destructive: it enriches entities without modifying predictions.
            SVO is a full feature — evaluate it alongside entity extraction.
            After each analyzeText() call:
              - Note syntacticRole / gender / number fields found on entities
              - Note svoSpans detected (role, p_svo_bnd, entity association if present)
              - Note eventlet quality (correct subject-verb-object associations?)

            ═══════════════════════════════════════════════════════
            MODEL ARCHITECTURE
            ═══════════════════════════════════════════════════════
            Single DeBERTa-v3 model — multi-head span-based NER (ONE forward pass, no two-stage pipeline).
            For each candidate span the model runs FOUR decoding heads simultaneously:
              • boundary head → "is this span an entity?"           prob = pBoundary
              • coarse head   → "which broad family (9 + NONE)?"   prob = pCoarse
              • fine head     → "which of 38 fine labels?"          prob = pFine
              • SVO heads     → syntactic role / voice / gender / number  [graph-based coref]

            score = pBoundary × pCoarse × pFine  (harsh composite — all three heads must agree).
            A score of 0.50 does NOT mean "50% confidence" — it is the product of three independently
            confident heads (e.g. 0.85³ ≈ 0.61). Always inspect pBoundary and pFine individually.

            ═══════════════════════════════════════════════════════
            TAXONOMY — what you evaluate
            ═══════════════════════════════════════════════════════
            ⚠ COARSE = indicative family only (display colour + structural masking).
              It is NOT an evaluated field. Do NOT judge quality on coarse alone.
              pCoarse and tauCoarse are debug/tuning tools, not quality signals per se.

            FINE labels (38) — THE actual semantic prediction to evaluate:

              PER family — persons & human groups
                hint_person_name   : proper name of a physical person (first/last name, alias)
                hint_person_role   : role, title, function (president, general, CEO, nurse…)
                hint_norp          : nationality, religious/ethnic/political group (French, Catholics…)
                hint_group_role    : collective human designation (team, jury, delegation, staff…)

              LOC family — places
                hint_gpe           : geopolitical named entity: country, city, region (France, Paris…)
                hint_fac_name      : named built place: monument, stadium, hospital (Eiffel Tower…)
                hint_loc_generic   : generic / unnamed geographic feature (mountain, river, coast…)
                hint_infra         : named infrastructure: road, transport line, network (A6, line 4…)

              ORG family — organisations
                hint_org_name      : named formal organisation: company, party (LVMH, Greenpeace…)
                hint_inst_name     : named public institution, acronym or qualified name (UN, NATO, European Commission…)
                hint_inst_role     : generic institution without qualifier (government, police, army, court…)

              TIME family — temporal expressions
                hint_time_date     : date or calendar reference (12 March, 2024, next Monday)
                hint_time_clock    : precise time (14:30, at midnight, around 8am)
                hint_time_duration : duration or time interval (3 years, for 2 months, for an hour)

              EVENT family — events
                hint_event_nominal : event described by a common noun (the war, the trial, the crisis)
                hint_event_named   : properly named event (COP28, French Revolution, 2024 Olympics)

              OBJECT family — physical objects
                hint_weapon        : weapon or ammunition (missile, AK-47, bomb…)
                hint_vehicle       : vehicle (aircraft, ship, tank, car)
                hint_substance     : material or substance (oil, gas, uranium)
                hint_food          : food or beverage (wheat, wine, meat)
                hint_tool          : tool or equipment (medical device, construction machinery…)
                hint_object_generic: generic physical object
                hint_object_name   : properly named physical object (iPhone 15, Boeing 737…)

              VALUE family — numerical values
                hint_measure       : physical quantity with unit (3 km, 500 kg, 20 MW)
                hint_percentage    : percentage (12%, a quarter)
                hint_count         : integer count (3 dead, 12,000 soldiers)
                hint_money         : monetary amount (€200, 3 billion dollars)
                hint_rate          : rate, ratio, index (7% unemployment, CAC at 8000)

              WORK family — intellectual productions
                hint_work_of_art   : named work: book, film, song, painting (Mona Lisa, Avatar)
                hint_law           : legal text, law, treaty, decree (El Khomri law, Treaty of Rome)
                hint_document      : report, letter, press release, contract, data
                hint_work_generic  : generic cultural production without title (film, press, media)

              ABSTRACT family — concepts & abstract states
                hint_disease       : disease or pathology (Covid-19, lung cancer)
                hint_language      : human or programming language (French, Python, Arabic)
                hint_doctrine      : doctrine, ideology, school of thought (liberalism, Marxism)
                hint_state         : abstract state or condition (poverty, crisis, peace, war)
                hint_notion        : pure abstract concept, value, principle (democracy, freedom)
                hint_field         : domain or sector of activity (health, education, finance)
                hint_disease       : disease or pathology (Covid-19, lung cancer)
                hint_language      : human or programming language (French, Python, Arabic)

            ═══════════════════════════════════════════════════════
            SCORES — what each one means
            ═══════════════════════════════════════════════════════
              score      → COMPOSITE SCORE (boundary × coarse × fine). PRIMARY quality signal.
                           Entities are kept iff score >= tauBoundary.
                           Use this as the main signal when assessing detection quality.

              pBoundary  → [DEBUG/TUNING] Raw BILOU boundary probability.
                           Low pBoundary on a true entity → it was borderline; lower tauBoundary.
                           High pBoundary on a false positive → raise tauBoundary.

              pCoarse    → [DEBUG/TUNING] BILOU confidence in the coarse family.
                           Only relevant when diagnosing coarse mis-routing
                           (e.g. PER routed as ORG → wrong fine labels due to COARSE_TO_FINE mask).

              pFine      → [DEBUG/TUNING] SpanClassifier confidence in the chosen fine label.
                           Low pFine = genuine ambiguity between two fine labels in the same family.
                           When altFine is present (pFine < 0.60), it shows the runner-up label.

            ═══════════════════════════════════════════════════════
            MANDATORY WORKFLOW
            ═══════════════════════════════════════════════════════
            1. Call getConfig() — read current thresholds.
            2. Call analyzeText() on EACH provided sentence — inspect fine labels and scores.
               Focus on: wrong fine label, missed entity (FN), spurious entity (FP).
               ⚠ Nested spans (nested=true with parentText/parentFine) are EXPECTED — not FP.
               Evaluate parent and child independently.
               Also note: syntacticRole / gender / number on entities, svoSpans, eventlets.
            3. For sentences with borderline scores, call scanThreshold() to find the elbow.
            4. Write structured verdict:
               - Per-sentence: FP / FN / wrong fine label / correct detections
               - Note quality of nested spans: does the parent carry the main semantic value?
               - SVO: roles detected, eventlet quality, entity association, morphology correctness
               - Threshold recommendations with numeric values
            5. MANDATORY CONCLUSION — Comparative market score:
               - Rate this entity extraction model on a /10 scale.
               - Compare explicitly vs: spaCy (fr_dep_news_trf / en_core_web_trf),
                 CamemBERT-NER, Flair, Stanza, Azure Text Analytics, AWS Comprehend,
                 Google Natural Language API.
               - Reference publicly known F1 benchmarks (CoNLL-2003, WikiNER, FQuAD-NER…).
               - ⚠ This model has 38 fine labels + nested spans — a direct F1 comparison
                 with flat 4-label NER is structurally unfair. Acknowledge this.
               - Be honest about strengths and weaknesses relative to these solutions.

            Respond in the SAME LANGUAGE as the analyzed texts.
        """.trimIndent()

        val userContent = buildString {
            appendLine("Judge the entity extraction and SVO quality on these sentences.")
            appendLine("Evaluate entities (fine labels, nested spans), SVO roles, morphology, and eventlets.")
            results.forEachIndexed { i, s -> appendLine("  ${i + 1}. ${s.text}") }
            appendLine()
            appendLine("Start NOW by calling getConfig(), then analyzeText() on each sentence above.")
        }

        val messages = mutableListOf<MutableMap<String, Any>>(
            mutableMapOf("role" to "system", "content" to systemPrompt),
            mutableMapOf("role" to "user",   "content" to userContent),
        )

        val toolTrace = StringBuilder()
        var iteration = 0
        val maxIter = 12

        while (iteration++ < maxIter) {
            // Force at least one tool call on the first turn to prevent the model from
            // skipping straight to a text answer ("required" → Mistral/OpenAI both support it)
            val toolChoice: Any = if (iteration == 1) "required" else "auto"

            val payload = mapper.writeValueAsString(mapOf(
                "model"       to cfg.model,
                "temperature" to 0.1,
                "tools"       to toolDefinitions,
                "tool_choice" to toolChoice,
                "messages"    to messages,
            ))

            val responseJson = chatCompletions(cfg, payload)
            @Suppress("UNCHECKED_CAST")
            val choice       = (responseJson["choices"] as List<Map<String, Any>>).first()
            val message      = choice["message"] as Map<String, Any>
            val finishReason = choice["finish_reason"] as? String ?: "stop"

            // Persist assistant message
            val assistantMsg = mutableMapOf<String, Any>("role" to "assistant")
            message["content"]?.let { assistantMsg["content"] = it }
            @Suppress("UNCHECKED_CAST")
            (message["tool_calls"] as? List<Map<String, Any>>)?.let { assistantMsg["tool_calls"] = it }
            messages += assistantMsg

            if (finishReason == "stop" || finishReason == "end_turn") {
                val final = message["content"] as? String ?: ""
                return if (toolTrace.isNotEmpty()) "$toolTrace\n---\n$final" else final
            }

            // Execute tool calls
            @Suppress("UNCHECKED_CAST")
            val toolCalls = message["tool_calls"] as? List<Map<String, Any>> ?: break
            for (tc in toolCalls) {
                val id       = tc["id"] as String
                @Suppress("UNCHECKED_CAST")
                val fn       = tc["function"] as Map<String, Any>
                val toolName = fn["name"] as String
                val argsStr  = fn["arguments"] as? String ?: "{}"

                log.info("Agent → {}({})", toolName, argsStr.take(120))
                toolTrace.appendLine("🔧 **$toolName** `${argsStr.take(80)}${if (argsStr.length > 80) "…" else ""}`")
                onProgress(toolTrace.toString(), true)

                val toolResult = try {
                    dispatchTool(toolName, argsStr)
                } catch (e: Exception) {
                    mapper.writeValueAsString(mapOf("error" to (e.message ?: "unknown")))
                }
                messages += mutableMapOf(
                    "role"         to "tool",
                    "tool_call_id" to id,
                    "content"      to toolResult,
                )
            }
        }

        return "${toolTrace}\n⚠ Max iterations ($maxIter) reached."
    }

    // ── HTTP helpers ──────────────────────────────────────────────────────────

    /** Lit la réponse en SSE (stream:true) et appelle [onChunk] pour chaque token de contenu. */
    private fun chatCompletionsStream(cfg: LlmJudgeConfig, payload: String, onChunk: (String) -> Unit) {
        val url = cfg.baseUrl.trimEnd('/') + "/chat/completions"
        log.info("LLM stream → {} model={}", url, cfg.model)
        val req = HttpRequest.newBuilder(URI.create(url))
            .timeout(Duration.ofSeconds(180))
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer ${cfg.apiKey}")
            .POST(HttpRequest.BodyPublishers.ofString(payload))
            .build()
        val resp = http.send(req, HttpResponse.BodyHandlers.ofLines())
        if (resp.statusCode() !in 200..299) error("HTTP ${resp.statusCode()}")
        resp.body().use { lines ->
            lines.forEach { line ->
                if (!line.startsWith("data: ")) return@forEach
                val data = line.removePrefix("data: ").trim()
                if (data == "[DONE]") return@forEach
                try {
                    @Suppress("UNCHECKED_CAST")
                    val json  = mapper.readValue<Map<String, Any>>(data)
                    val delta = ((json["choices"] as? List<Map<String, Any>>)
                        ?.firstOrNull()?.get("delta") as? Map<String, Any>) ?: return@forEach
                    val tok = delta["content"] as? String ?: return@forEach
                    if (tok.isNotEmpty()) onChunk(tok)
                } catch (_: Exception) { /* skip malformed SSE lines */ }
            }
        }
    }

    private fun chatCompletions(cfg: LlmJudgeConfig, payload: String): Map<String, Any> {
        val url = cfg.baseUrl.trimEnd('/') + "/chat/completions"
        log.info("LLM → {} model={}", url, cfg.model)
        val req = HttpRequest.newBuilder(URI.create(url))
            .timeout(Duration.ofSeconds(120))
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer ${cfg.apiKey}")
            .POST(HttpRequest.BodyPublishers.ofString(payload))
            .build()
        val resp = http.send(req, HttpResponse.BodyHandlers.ofString())
        if (resp.statusCode() !in 200..299) {
            log.error("LLM {} {}", resp.statusCode(), resp.body())
            error("HTTP ${resp.statusCode()}: ${resp.body().take(300)}")
        }
        return mapper.readValue(resp.body())
    }

    private fun extractContent(json: Map<String, Any>): String {
        @Suppress("UNCHECKED_CAST")
        val choices = json["choices"] as? List<Map<String, Any>> ?: error("Unexpected response")
        @Suppress("UNCHECKED_CAST")
        val message = choices.first()["message"] as? Map<String, Any> ?: error("No message")
        return message["content"] as? String ?: error("Empty content")
    }
}
