package rag.connectors.ner.onnx

/**
 * Main de test standalone — pas de Spring, pas de contexte.
 *
 * Lancer depuis le projet Gradle :
 *   ./gradlew :connectors:ner:onnx-ner:run -PmainClass=rag.connectors.ner.onnx.TestMultiHeadExtractorKt
 *
 * Ou directement depuis IntelliJ (Run gutter sur `fun main`).
 */

private const val MODEL_DIR = "/Users/simon_longuet/IdeaProjects/pimpmyrag/models/deberta/fine-tunning-21042026"
private const val MODEL_ONNX = "$MODEL_DIR/best_model_multitask.onnx"
private const val TOKENIZER_DIR = "/Users/simon_longuet/IdeaProjects/pimpmyrag/tokenizer_export_clean"

private val TEST_TEXTS = listOf(
    "Emmanuel Macron s'est rendu hier à Berlin pour rencontrer le chancelier Olaf Scholz.",
    "La Banque centrale européenne a relevé ses taux d'intérêt de 25 points de base mardi.",
    "Apple a annoncé le lancement de l'iPhone 17 le 15 septembre 2025 à Cupertino.",
    "Le tremblement de terre de magnitude 6,8 a touché la côte nord du Maroc.",
    "Le PSG a battu le Real Madrid 3-1 lors de la finale de la Ligue des champions.",
    "L'Assemblée nationale a adopté la loi sur le financement de la sécurité sociale.",
    "Le vaccin contre la grippe est disponible en pharmacie depuis le 1er octobre.",
    "Tesla a livré 500 000 véhicules électriques au troisième trimestre, un record.",
)

fun main() {
    println("═══════════════════════════════════════════════════════════")
    println("  Test OnnxMultiHeadEntityExtractor")
    println("  Modèle   : $MODEL_ONNX")
    println("  Tokenizer: $TOKENIZER_DIR")
    println("═══════════════════════════════════════════════════════════\n")

    val extractor = OnnxMultiHeadEntityExtractor(
        modelPath    = MODEL_ONNX,
        tokenizerDir = TOKENIZER_DIR,
        maxSeqLen    = 128,
        maxSpanLen   = 8,
        tauBoundary  = 0.70f,
        tauNone      = 0.50f,
        tauCoarse    = 0.45f,
    )

    extractor.use { ext ->
        // ── Test unitaire ────────────────────────────────────────────
        TEST_TEXTS.forEachIndexed { i, text ->
            val t0 = System.nanoTime()
            val entities = ext.extractFromText(text)
            val ms = (System.nanoTime() - t0) / 1_000_000L

            println("[$i] \"$text\"")
            println("    → ${entities.size} entité(s) en ${ms}ms")
            entities.forEach { e ->
                val coarse = e.metadata["coarse"]
                val score  = (e.metadata["score"] as? Float)?.let { "%.3f".format(it) } ?: "?"
                println("      • [${e.span.start}-${e.span.end}] \"${e.text}\"  type=${e.type}  coarse=$coarse  score=$score")
            }
            println()
        }

        // ── Test batch ───────────────────────────────────────────────
        println("─── Batch (${TEST_TEXTS.size} textes) ───────────────────────────")
        val t0 = System.nanoTime()
        val batchResults = ext.extractFromTexts(TEST_TEXTS)
        val ms = (System.nanoTime() - t0) / 1_000_000L
        val total = batchResults.sumOf { it.size }
        println("Batch terminé en ${ms}ms — $total entités extraites au total")
        println("Moyenne: ${"%.1f".format(ms.toDouble() / TEST_TEXTS.size)} ms/texte\n")

        // ── Récapitulatif par type ───────────────────────────────────
        println("─── Récapitulatif par type fine ─────────────────────────")
        batchResults.flatten()
            .groupBy { it.type }
            .entries
            .sortedByDescending { it.value.size }
            .forEach { (type, list) ->
                println("  %-30s : %d".format(type, list.size))
            }
    }

    println("\n✅ Test terminé.")
}

