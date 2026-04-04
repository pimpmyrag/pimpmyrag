package io.axes.classifier

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import org.springframework.stereotype.Service
import rag.engine.Embedder
import rag.model.RagDocument
import java.io.File
import java.util.UUID
import kotlin.math.sqrt

// ========== DTOs ==========

data class MultiAxisPrediction(
    val eventDetection: Map<String, Double>,
    val temporal: Map<String, Double>,
    val spatial: Map<String, Double>,
    val coreference: Map<String, Double>,
    val style: Map<String, Double>,
    val sentiment: Map<String, Double>,
    val contentType: Map<String, Double>,
    val domain: Map<String, Double>
)

// ========== Wrapper pour classification multi-axes ==========

class MultiAxisTextClassifier(
    private val embedder: Embedder,
    private val classifiers: Map<String, MultiClassEventClassifier>,
    private val l2NormalizeEmbeddings: Boolean = true
) {

//    fun classify(text: String): MultiAxisPrediction {
//        val embedding = embedder.embed(listOf(text).toRagDocuments()).first().let { emb ->
//            if (l2NormalizeEmbeddings) l2Normalize(emb) else emb
//        }
//
//        return MultiAxisPrediction(
//            eventDetection = classifiers["event_detection"]!!.classifyWithEmbedding(embedding),
//            temporal = classifiers["temporal"]!!.classifyWithEmbedding(embedding),
//            spatial = classifiers["spatial"]!!.classifyWithEmbedding(embedding),
//            coreference = classifiers["coreference"]!!.classifyWithEmbedding(embedding),
//            style = classifiers["style"]!!.classifyWithEmbedding(embedding),
//            sentiment = classifiers["sentiment"]!!.classifyWithEmbedding(embedding),
//            contentType = classifiers["content_type"]!!.classifyWithEmbedding(embedding),
//            domain = classifiers["domain"]!!.classifyWithEmbedding(embedding)
//        )
//    }

    private fun l2Normalize(vector: FloatArray): FloatArray {
        val norm = sqrt(vector.sumOf { (it * it).toDouble() }).toFloat()
        return if (norm > 0) vector.map { it / norm }.toFloatArray() else vector
    }

}

// ========== Service Spring ==========

@Service
class MultiAxisTrainingService(
    private val embedder: Embedder,
    private val syntheticGenerator: SyntheticDataGenerator,
    private val jacksonMapper: ObjectMapper = jacksonObjectMapper()
) {

    fun trainAllAxes(basePath: String = "data", axisName: String): Map<String, MultiClassEventClassifier> {
        // Scanner le dossier /data pour extraire les labels dynamiquement
        val dataDir = File(basePath)
        val jsonFiles = dataDir.listFiles { file -> file.isFile && file.name.endsWith(".json") } ?: emptyArray()

        val allLabels = mutableSetOf<String>()
        val allExamples = mutableListOf<TrainingExample>()
        jsonFiles.forEach { file ->
            try {
                val content = file.readText(Charsets.UTF_8)
                val examples = jacksonMapper.readValue<List<TrainingExample>>(content)
                examples.forEach { example ->
                    allLabels.add(example.label)
                    allExamples.add(example)
                }
            } catch (e: Exception) {
                try {
                    // Essayer avec ISO-8859-1 si UTF-8 échoue
                    val content = file.readText(Charsets.ISO_8859_1)
                    val examples = jacksonMapper.readValue<List<TrainingExample>>(content)
                    examples.forEach { example ->
                        allLabels.add(example.label)
                        allExamples.add(example)
                    }
                } catch (e2: Exception) {
                    println("⚠️ Erreur lecture ${file.name}: UTF-8: ${e.message}, ISO-8859-1: ${e2.message}")
                }
            }
        }

        println("📊 Labels trouvés: ${allLabels.joinToString(", ")}")
        println("📊 Total exemples réels: ${allExamples.size}")

        // Créer un axe unique "content_type" avec toutes les catégories trouvées
        val axes = mapOf(
            axisName to allLabels.toList()
        )

        return axes.mapValues { (axisName, categories) ->
            println("🚀 Entraînement axe: $axisName (${categories.size} catégories)")

            val dataFile = "$basePath/labels/synthetic_$axisName.json"

            // Sauvegarde des exemples réels si fichier absent
            if (!File(dataFile).exists()) {
                println("💾 Sauvegarde des exemples réels dans $dataFile")
                File(dataFile).writeText(jacksonMapper.writerWithDefaultPrettyPrinter().writeValueAsString(allExamples))
            } else {
                println("📁 Fichier $dataFile existe déjà, utilisation des données existantes")
            }

            val examples = syntheticGenerator.loadExamples(dataFile)

            val classifier = MultiClassEventClassifier(
                embedder = embedder,
                categories = categories,
                l2NormalizeEmbeddings = true
            )
            classifier.train(examples)
            classifier.save(path = "data/model-results_$axisName.json")
            println("Classifier entraîné et sauvegardé pour axe: $axisName avec ${examples.size} exemples")

            classifier
        }
    }
}

// ========== Utils ==========

fun String.toRagDocuments(): RagDocument = RagDocument(id = UUID.randomUUID().toString(), text = this)
fun List<String>.toRagDocuments(): List<RagDocument> = this.map { it.toRagDocuments() }
