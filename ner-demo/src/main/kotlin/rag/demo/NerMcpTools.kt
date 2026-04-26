package rag.demo

import org.springframework.ai.tool.annotation.Tool
import org.springframework.ai.tool.annotation.ToolParam
import org.springframework.stereotype.Component
import kotlin.math.roundToInt

/**
 * Outils MCP exposés au serveur pour permettre à un agent IA de :
 *   1. lire la configuration courante                   → getConfig
 *   2. modifier un seuil précis                         → setThreshold
 *   3. lancer une inférence et voir les scores bruts    → analyzeText
 *   4. balayer une plage de seuils sur un texte-test    → scanThreshold
 *   5. analyser un corpus et obtenir des stats agrégées → analyzeBatch
 *
 * Workflow type de calibration :
 *   getConfig() → analyzeText(sample) → [observer les pBoundary/score borderline]
 *   → setThreshold("tauBoundary", 0.65) → analyzeText(sample) → comparer
 *   → scanThreshold("tauBoundary", sample, 0.40, 0.90, 0.05) → trouver le coude
 */
@Component
class NerMcpTools(private val nerService: NerService) {

    // ── 1. Lire la config ────────────────────────────────────────────────────

    @Tool(description = """
        Returns the current NER+SVO thresholds and settings.
        Threshold semantics:
          tauBoundary   — minimum pBoundary to keep a span as an entity     (↓ = more recall, more FP)
          tauNone       — minimum pNone to assign NONE / discard a candidate (↑ = stricter rejection)
          tauCoarse     — minimum pCoarse for coarse type assignment          (↓ = more typed entities)
          tauSvoBoundary— minimum pBoundary for SVO span detection            (↓ = more SVO spans)
          batchSize     — sentences per ONNX inference call
    """)
    fun getConfig(): Map<String, Any> {
        val c = nerService.config
        return mapOf(
            "tauBoundary"    to c.tauBoundary,
            "tauNone"        to c.tauNone,
            "tauCoarse"      to c.tauCoarse,
            "tauSvoBoundary" to c.tauSvoBoundary,
            "batchSize"      to c.batchSize,
        )
    }

    // ── 2. Modifier un seuil ─────────────────────────────────────────────────

    @Tool(description = """
        Update a single NER/SVO threshold by name.
        Valid names: tauBoundary | tauNone | tauCoarse | tauSvoBoundary
        The change takes effect immediately for all subsequent calls.
        Returns the full updated config.
    """)
    fun setThreshold(
        @ToolParam(description = "Threshold name: tauBoundary | tauNone | tauCoarse | tauSvoBoundary")
        name: String,
        @ToolParam(description = "New value (float, clamped to the valid range for each threshold)")
        value: Float,
    ): Map<String, Any> {
        val c = nerService.config
        val updated = when (name) {
            "tauBoundary"    -> c.copy(tauBoundary    = value.coerceIn(0.05f, 0.99f))
            "tauNone"        -> c.copy(tauNone        = value.coerceIn(0.05f, 1.00f))
            "tauCoarse"      -> c.copy(tauCoarse      = value.coerceIn(0.00f, 0.99f))
            "tauSvoBoundary" -> c.copy(tauSvoBoundary = value.coerceIn(0.05f, 0.99f))
            else -> return mapOf("error" to "Unknown threshold name: $name. Use tauBoundary | tauNone | tauCoarse | tauSvoBoundary")
        }
        nerService.updateConfig(updated)
        return mapOf(
            "updated" to name,
            "oldValue" to when (name) {
                "tauBoundary"    -> c.tauBoundary
                "tauNone"        -> c.tauNone
                "tauCoarse"      -> c.tauCoarse
                else             -> c.tauSvoBoundary
            },
            "newValue" to value,
            "config"   to getConfig(),
        )
    }

    // ── 3. Analyser un texte ─────────────────────────────────────────────────

    @Tool(description = """
        Run NER+SVO inference on a text and return all detected entities with RAW MODEL SCORES.
        The raw scores are essential for threshold calibration:
          - pBoundary: how confident the model is that this span is an entity boundary
            → compare to tauBoundary to understand which entities are borderline
          - pNone: probability it is NOT an entity (inverse signal)
          - pCoarse: confidence in the coarse type label
          - score: combined score (used by the UI confidence indicator)
        Entities in the output were kept because score >= current tauBoundary.
        To see what you are MISSING, lower tauBoundary and re-run.
    """)
    fun analyzeText(
        @ToolParam(description = "The text to analyze (ideally one natural sentence or short paragraph)")
        text: String,
    ): Map<String, Any> {
        val result = nerService.analyse(text)
        return mapOf(
            "thresholdsUsed" to getConfig(),
            "entityCount"    to result.entities.size,
            "svoCount"       to result.svoSpans.size,
            "entities" to result.entities.map { e ->
                mapOf(
                    "text"      to e.text,
                    "coarse"    to (e.metadata["coarse"] ?: "NONE"),
                    "fine"      to e.type,
                    "score"     to fmt(e.metadata["score"]),
                    "pBoundary" to fmt(e.metadata["pBoundary"]),
                    "pCoarse"   to fmt(e.metadata["pCoarse"]),
                    "pFine"     to fmt(e.metadata["pFine"]),
                    "chars"     to "[${e.span?.start}:${e.span?.end}]",
                )
            },
            "svoSpans" to result.svoSpans.map { s ->
                mapOf(
                    "text"            to s.text,
                    "role"            to s.role,
                    "pBoundary"       to fmt(s.svoBoundaryProb),
                    "pRole"           to fmt(s.roleProb),
                    "voice"           to s.voice,
                )
            },
        )
    }

    // ── 4. Balayer un seuil ──────────────────────────────────────────────────

    @Tool(description = """
        Sweep a threshold over a range on a reference text and report entity counts at each step.
        Use this to find the optimal threshold value by looking for the 'elbow' in entity count:
          - too low → many spurious entities (noise)
          - too high → real entities are missed (under-recall)
        The config is restored to its original state after the sweep.
        Tip: call analyzeText first to understand which entities are at stake.
    """)
    fun scanThreshold(
        @ToolParam(description = "Reference text to use for the sweep")
        text: String,
        @ToolParam(description = "Threshold to sweep: tauBoundary | tauNone | tauCoarse | tauSvoBoundary")
        threshold: String,
        @ToolParam(description = "Start value of the sweep range (e.g. 0.30)")
        from: Float,
        @ToolParam(description = "End value of the sweep range (e.g. 0.90)")
        to: Float,
        @ToolParam(description = "Step size (e.g. 0.05 or 0.10)")
        step: Float,
    ): Map<String, Any> {
        if (threshold !in setOf("tauBoundary", "tauNone", "tauCoarse", "tauSvoBoundary"))
            return mapOf("error" to "Unknown threshold: $threshold")

        val originalCfg = nerService.config
        val steps = mutableListOf<Map<String, Any>>()
        val safeStep = step.coerceAtLeast(0.01f)
        var v = from
        while (v <= to + 1e-5f) {
            val testCfg = when (threshold) {
                "tauBoundary"    -> originalCfg.copy(tauBoundary    = v)
                "tauNone"        -> originalCfg.copy(tauNone        = v)
                "tauCoarse"      -> originalCfg.copy(tauCoarse      = v)
                else             -> originalCfg.copy(tauSvoBoundary = v)
            }
            nerService.updateConfig(testCfg)
            val result = nerService.analyse(text)
            steps += mapOf(
                "value"       to "%.3f".format(v),
                "entities"    to result.entities.size,
                "svoSpans"    to result.svoSpans.size,
                "byCoarse"    to result.entities.groupingBy {
                    it.metadata["coarse"] as? String ?: "NONE"
                }.eachCount(),
            )
            v = (v + safeStep).let {
                // round to avoid float drift
                (it * 1000).roundToInt() / 1000f
            }
        }

        nerService.updateConfig(originalCfg) // always restore
        return mapOf(
            "threshold"      to threshold,
            "range"          to "${from} → ${to} (step ${step})",
            "steps"          to steps,
            "configRestored" to true,
            "hint"           to "Look for the value where entity count stabilises — that is usually a good threshold.",
        )
    }

    // ── 5. Analyser un corpus ────────────────────────────────────────────────

    @Tool(description = """
        Analyze a batch of texts (max 30) and return aggregated statistics.
        Useful to understand the global distribution of entities and scores across a corpus,
        identify which coarse types are fragile (low average score) and need threshold tuning.
    """)
    fun analyzeBatch(
        @ToolParam(description = "List of texts (max 30 for performance)")
        texts: List<String>,
    ): Map<String, Any> {
        val limited = texts.take(30)
        val allEntities = mutableListOf<Map<String, Any>>()

        nerService.analyseStream(limited) { _, results ->
            results.forEach { r ->
                r.entities.forEach { e ->
                    allEntities += mapOf(
                        "text"      to e.text,
                        "coarse"    to (e.metadata["coarse"] as? String ?: "NONE"),
                        "fine"      to e.type,
                        "score"     to (e.metadata["score"] as? Float ?: 0f),
                        "pBoundary" to (e.metadata["pBoundary"] as? Float ?: 0f),
                    )
                }
            }
        }

        val byCoarse = allEntities.groupBy { it["coarse"] as String }
        val lowConf  = allEntities.filter { (it["score"] as Float) < 0.70f }
            .sortedBy { it["score"] as Float }.take(10)

        return mapOf(
            "textsAnalyzed"        to limited.size,
            "totalEntities"        to allEntities.size,
            "avgEntitiesPerText"   to if (limited.isEmpty()) 0.0 else
                                      "%.2f".format(allEntities.size.toDouble() / limited.size),
            "thresholdsUsed"       to getConfig(),
            "byCoarseType"         to byCoarse.mapValues { (_, ents) ->
                val scores = ents.map { it["score"] as Float }
                mapOf(
                    "count"    to ents.size,
                    "avgScore" to "%.3f".format(scores.average()),
                    "minScore" to "%.3f".format(scores.min()),
                    "maxScore" to "%.3f".format(scores.max()),
                )
            },
            "lowConfidenceEntities" to lowConf,
            "hint" to "Coarse types with low avgScore may benefit from a lower tauBoundary. " +
                      "Coarse types with minScore close to 0 may be spurious — consider raising tauCoarse.",
        )
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun fmt(v: Any?): String = when (v) {
        is Float  -> "%.4f".format(v)
        is Double -> "%.4f".format(v)
        null      -> "—"
        else      -> v.toString()
    }
}

