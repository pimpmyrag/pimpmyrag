package io.axes

import com.fasterxml.jackson.annotation.JsonProperty
import io.axes.classifier.MultiClassEventClassifier
import io.axes.classifier.TrainingExample

data class TestCase(
    val id: String,
    val sentence: String,
    val expected: ExpectedScores
)

data class ExpectedScores(
    val category: String?,
    val subType: String?,
    val event: Double? = null,
    @JsonProperty("temporal_explicit") val temporalExplicit: Double? = null,

    // Lifecycle
    val birth: Double? = null,
    val marriage: Double? = null,
    val baptism: Double? = null,
    val death: Double? = null,
    val burial: Double? = null,
    val divorce: Double? = null,

    // Governance
    val dissolution: Double? = null,
    val coup: Double? = null,
    val proclamation: Double? = null,
    val oath: Double? = null,
    val decree: Double? = null,
    val election: Double? = null,

    // Conflict
    val battle: Double? = null,
    val attack: Double? = null,
    val siege: Double? = null,
    val bombardment: Double? = null,
    val skirmish: Double? = null,
    val conflict: Double? = null,  // ✅ Ajout du champ manquant

    // Legal
    @JsonProperty("trial_opening") val trialOpening: Double? = null,
    val conviction: Double? = null,
    val acquittal: Double? = null,
    val appeal: Double? = null,
    val dismissal: Double? = null,

    // Communication
    val speech: Double? = null,
    val announcement: Double? = null,
    @JsonProperty("press_conference") val pressConference: Double? = null,
    val statement: Double? = null,
    val broadcast: Double? = null,

    // Creation/Destruction
    val foundation: Double? = null,
    val inauguration: Double? = null,
    val fire: Double? = null,
    val earthquake: Double? = null,
    val demolition: Double? = null,

    // Général
    val opinion: Double? = null,
    val factual: Double? = null,
    @JsonProperty("future_prediction") val futurePrediction: Double? = null,
    val uncertainty: Double? = null,
    val incident: Double? = null,  // ✅ Ajout pour edge_02
    val impact: Double? = null      // ✅ Ajout pour edge_02
)


data class TestDataset(
    @JsonProperty("schema_version") val schemaVersion: String,
    @JsonProperty("created_at") val createdAt: String,
    @JsonProperty("embedding_model") val embeddingModel: String,
    val description: String,
    @JsonProperty("test_cases") val testCases: List<TestCase>
)

// EventTaxonomyTesterML.kt


data class TestResult(
    val id: String,
    val sentence: String,
    val expectedCategory: String,
    val predictedCategory: String,
    val confidence: Double,
    val passed: Boolean,
    val scores: Map<String, Double>
)

class EventTaxonomyTesterML(
    private val classifier: MultiClassEventClassifier,
    private val threshold: Double = 0.6
) {

    fun runTests(examples: List<TrainingExample>): List<TestResult> {
        val sentences = examples.map { it.text }
        val scoresMap = classifier.classifyBatch(sentences)

        return examples.mapIndexed { idx, example ->
            val scores = scoresMap[idx]
            val (predictedCategory, confidence) = scores.maxByOrNull { it.value }
                ?.let { it.key to it.value } ?: ("unknown" to 0.0)

            TestResult(
                id = "test_${idx + 1}",
                sentence = example.text,
                expectedCategory = example.label,
                predictedCategory = predictedCategory,
                confidence = confidence,
                passed = predictedCategory == example.label && confidence >= threshold,
                scores = scores
            )
        }
    }

    fun printResults(results: List<TestResult>) {
        val passed = results.count { it.passed }
        val total = results.size
        val passRate = (passed * 100.0 / total)

        println("\n========================================")
        println("📊 Résultats - Classification Supervisée")
        println("========================================")
        println("✅ Réussis : $passed / $total (${String.format("%.1f", passRate)}%)")
        println("❌ Échoués : ${total - passed} / $total")
        println()

        val byCategory = results.groupBy { it.expectedCategory }
        byCategory.forEach { (category, categoryResults) ->
            val categoryPassed = categoryResults.count { it.passed }
            val categoryTotal = categoryResults.size
            val categoryRate = (categoryPassed * 100.0 / categoryTotal)
            println("📁 $category: ${String.format("%.1f", categoryRate)}% ($categoryPassed/$categoryTotal)")
        }
        println()

        val failures = results.filter { !it.passed }
        if (failures.isNotEmpty()) {
            println("❌ Échecs détaillés:\n")
            failures.take(10).forEach { result ->
                println("ID: ${result.id}")
                println("Phrase: ${result.sentence}")
                println("Attendu: ${result.expectedCategory}")
                println("Prédit: ${result.predictedCategory} (${String.format("%.3f", result.confidence)})")
                println("Top 3 scores:")
                result.scores.toList().sortedByDescending { it.second }.take(3).forEach { (cat, score) ->
                    println("  - $cat: ${String.format("%.3f", score)}")
                }
                println()
            }
        }
    }
}


//data class TestResult(
//    val id: String,
//    val sentence: String,
//    val expectedCategory: String?,
//    val expectedSubType: String?,
//    val passed: Boolean,
//    val errors: List<String>
//)

//class EventTaxonomyTester(
//    private val threshold: Double = 0.05, // Seuil de tolérance pour les scores
//    private val classifier: MultiClassEventClassifier // ✅ Au lieu de Radar
//) {
//
//    fun runTests(testFile: File): List<TestResult> {
//        val dataset = jacksonObjectMapper().readValue<TestDataset>(testFile)
//        val sentences = dataset.testCases.map { it.sentence }
//
//        val scores = classifier.classifyBatch(dataset.testCases.map { it.sentence })
//
//
//        return dataset.testCases.mapIndexed { idx, testCase ->
//            val scores = scores[idx]
//            val errors = mutableListOf<String>()
//
//            // Vérification des axes généraux
//            checkScore(scores, "event", testCase.expected.event, errors)
//            checkScore(scores, "temporal_explicit", testCase.expected.temporalExplicit, errors)
//            checkScore(scores, "opinion", testCase.expected.opinion, errors)
//            checkScore(scores, "factual", testCase.expected.factual, errors)
//            checkScore(scores, "future_prediction", testCase.expected.futurePrediction, errors)
//            checkScore(scores, "uncertainty", testCase.expected.uncertainty, errors)
//
//            // Lifecycle
//            checkScore(scores, "birth", testCase.expected.birth, errors)
//            checkScore(scores, "marriage", testCase.expected.marriage, errors)
//            checkScore(scores, "baptism", testCase.expected.baptism, errors)
//            checkScore(scores, "death", testCase.expected.death, errors)
//            checkScore(scores, "burial", testCase.expected.burial, errors)
//            checkScore(scores, "divorce", testCase.expected.divorce, errors)
//
//            // Governance
//            checkScore(scores, "dissolution", testCase.expected.dissolution, errors)
//            checkScore(scores, "coup", testCase.expected.coup, errors)
//            checkScore(scores, "proclamation", testCase.expected.proclamation, errors)
//            checkScore(scores, "oath", testCase.expected.oath, errors)
//            checkScore(scores, "decree", testCase.expected.decree, errors)
//            checkScore(scores, "election", testCase.expected.election, errors)
//
//            // Conflict
//            checkScore(scores, "battle", testCase.expected.battle, errors)
//            checkScore(scores, "attack", testCase.expected.attack, errors)
//            checkScore(scores, "siege", testCase.expected.siege, errors)
//            checkScore(scores, "bombardment", testCase.expected.bombardment, errors)
//            checkScore(scores, "skirmish", testCase.expected.skirmish, errors)
//            checkScore(scores, "conflict", testCase.expected.conflict, errors)  // ✅ Ajout
//
//// Axes supplémentaires pour edge cases
//            checkScore(scores, "incident", testCase.expected.incident, errors)
//            checkScore(scores, "impact", testCase.expected.impact, errors)
//
//
//            // Legal
//            checkScore(scores, "trial_opening", testCase.expected.trialOpening, errors)
//            checkScore(scores, "conviction", testCase.expected.conviction, errors)
//            checkScore(scores, "acquittal", testCase.expected.acquittal, errors)
//            checkScore(scores, "appeal", testCase.expected.appeal, errors)
//            checkScore(scores, "dismissal", testCase.expected.dismissal, errors)
//
//            // Communication
//            checkScore(scores, "speech", testCase.expected.speech, errors)
//            checkScore(scores, "announcement", testCase.expected.announcement, errors)
//            checkScore(scores, "press_conference", testCase.expected.pressConference, errors)
//            checkScore(scores, "statement", testCase.expected.statement, errors)
//            checkScore(scores, "broadcast", testCase.expected.broadcast, errors)
//
//            // Creation/Destruction
//            checkScore(scores, "foundation", testCase.expected.foundation, errors)
//            checkScore(scores, "inauguration", testCase.expected.inauguration, errors)
//            checkScore(scores, "fire", testCase.expected.fire, errors)
//            checkScore(scores, "earthquake", testCase.expected.earthquake, errors)
//            checkScore(scores, "demolition", testCase.expected.demolition, errors)
//
//            TestResult(
//                id = testCase.id,
//                sentence = testCase.sentence,
//                expectedCategory = testCase.expected.category,
//                expectedSubType = testCase.expected.subType,
//                passed = errors.isEmpty(),
//                errors = errors
//            )
//        }
//    }

//    private fun checkScore(
//        scores: Map<String, Double>,
//        axisName: String,
//        expected: Double?,
//        errors: MutableList<String>
//    ) {
//        if (expected == null) return
//
//        val actual = scores[axisName]?.toDouble() ?: 0.5
//        val diff = abs(actual - expected)
//
//        if (diff > threshold) {
//            errors.add("$axisName: expected=${String.format("%.3f", expected)}, actual=${String.format("%.3f", actual)}, diff=${String.format("%.3f", diff)}")
//        }
//    }

//    fun printResults(results: List<TestResult>) {
//        val passed = results.count { it.passed }
//        val total = results.size
//        val passRate = (passed * 100.0 / total)
//
//        println("\n========================================")
//        println("📊 Résultats des tests - Taxonomie v3.0")
//        println("========================================")
//        println("✅ Réussis : $passed / $total (${String.format("%.1f", passRate)}%)")
//        println("❌ Échoués : ${total - passed} / $total")
//        println()
//
//        // Statistiques par catégorie
//        val byCategory = results.groupBy { it.expectedCategory }
//        byCategory.forEach { (category, categoryResults) ->
//            val categoryPassed = categoryResults.count { it.passed }
//            val categoryTotal = categoryResults.size
//            val categoryRate = (categoryPassed * 100.0 / categoryTotal)
//            println("📁 $category: ${String.format("%.1f", categoryRate)}% ($categoryPassed/$categoryTotal)")
//        }
//        println()
//
//        // Détails des échecs
//        results.filter { !it.passed }.forEach { result ->
//            println("❌ ${result.id} [${result.expectedCategory}/${result.expectedSubType}]")
//            println("   Phrase : ${result.sentence}")
//            result.errors.forEach { error ->
//                println("   - $error")
//            }
//            println()
//        }
//    }
//}


