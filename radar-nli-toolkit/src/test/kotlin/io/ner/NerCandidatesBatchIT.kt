package io.ner

import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import io.axes.classifier.MultiAxisTrainingService
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.condition.EnabledIfSystemProperty
import org.junit.jupiter.api.fail
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.mock.mockito.MockBean
import org.springframework.context.annotation.Import
import org.springframework.http.MediaType
import org.springframework.test.web.reactive.server.WebTestClient
import org.springframework.test.web.reactive.server.expectBody
import java.io.File
import java.text.Normalizer

/**
 * Test d'intégration NER — démarre le contexte Spring complet (modèles ONNX inclus)
 * et appelle `/api/classify/extract/candidates/batch` via WebTestClient injecté.
 *
 * Pré-requis : les modèles ONNX doivent être disponibles localement.
 * Le test est ignoré par défaut ; activez-le avec :
 *   -Dner.integration.enabled=true
 *
 * Configuration :
 *   -Dner.integration.enabled=true   (requis pour exécuter)
 *   -Dner.test.batchSize=16          (défaut)
 *   -Dner.test.threshold=70          (% minimum pour passer, défaut 70)
 *   -Dner.test.report=/path/to/report.md  (optionnel — rapport Markdown pour CI)
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Tag("integration")
@EnabledIfSystemProperty(
    named     = "ner.integration.enabled",
    matches   = "true",
    disabledReason = "Nécessite les modèles ONNX locaux. Activez avec -Dner.integration.enabled=true"
)
@Import(NerCiTestConfig::class)
class NerCandidatesBatchIT {

    /** Remplace MultiAxisTrainingService avant le démarrage du contexte,
     *  évitant l'instanciation de SyntheticDataGenerator (exige LLM_API_KEY). */
    @MockBean
    private lateinit var multiAxisTrainingService: MultiAxisTrainingService

    @Autowired
    private lateinit var webTestClient: WebTestClient

    // ── Configuration ────────────────────────────────────────────────────────
    private val batchSize  = System.getProperty("ner.test.batchSize", "16").toInt()
    private val threshold  = System.getProperty("ner.test.threshold", "70").toInt()
    private val reportFile = System.getProperty("ner.test.report")

    private val mapper = ObjectMapper().registerKotlinModule()

    // ── Modèles ──────────────────────────────────────────────────────────────

    @JsonIgnoreProperties(ignoreUnknown = true)
    data class ExpectedEntity(val type: String, val text: String?, val hint: String? = null)

    @JsonIgnoreProperties(ignoreUnknown = true)
    data class TestCase(val id: Int, val text: String, val expected: List<ExpectedEntity>,
                        val category: String, val note: String)

    @JsonIgnoreProperties(ignoreUnknown = true)
    data class Candidate(val nerType: String, val text: String,
                         val nerHint: String? = null, val isName: Boolean = false)

    @JsonIgnoreProperties(ignoreUnknown = true)
    data class TextResult(val candidates: List<Candidate> = emptyList())

    @JsonIgnoreProperties(ignoreUnknown = true)
    data class BatchResponse(val results: List<TextResult> = emptyList())

    // ── Normalisation unicode (miroir de Python _norm) ───────────────────────
    private val dashChars = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uff0d"
    private val nbSpaces  = "\u00a0\u202f\u2009\u2007"

    private fun norm(s: String): String {
        var r = Normalizer.normalize(s, Normalizer.Form.NFC).trim()
        dashChars.forEach { r = r.replace(it, '-') }
        nbSpaces.forEach  { r = r.replace(it, ' ') }
        return r
    }

    // ── Chargement du JSONL ──────────────────────────────────────────────────
    private fun loadTestCases(): List<TestCase> {
        val resource = javaClass.classLoader
            .getResourceAsStream("ner_candidates_tests.jsonl")
            ?: error("ner_candidates_tests.jsonl introuvable dans le classpath de test")
        return resource.bufferedReader(Charsets.UTF_8).useLines { lines ->
            lines.filter { it.isNotBlank() }.map { mapper.readValue<TestCase>(it) }.toList()
        }
    }

    // ── Appel API via WebTestClient injecté ──────────────────────────────────
    private fun callBatch(texts: List<String>): List<List<Candidate>> {
        val resp = webTestClient.post()
            .uri("/api/classify/extract/candidates/batch")
            .contentType(MediaType.APPLICATION_JSON)
            .bodyValue(mapOf("texts" to texts))
            .exchange()
            .expectStatus().isOk()
            .expectBody<BatchResponse>()
            .returnResult()
            .responseBody
            ?: return List(texts.size) { emptyList() }
        return List(texts.size) { i -> resp.results.getOrElse(i) { TextResult() }.candidates }
    }

    // ── Vérification d'un cas ────────────────────────────────────────────────
    private fun checkCase(case: TestCase, candidates: List<Candidate>): List<String> {
        val issues = mutableListOf<String>()
        for (exp in case.expected) {
            val normExpText = exp.text?.let { norm(it).lowercase() }
            fun matchesExact(c: Candidate) =
                c.nerType == exp.type &&
                (normExpText == null || norm(c.text).lowercase() == normExpText) &&
                (exp.hint    == null || c.nerHint.orEmpty().uppercase().contains(exp.hint.uppercase()))

            if (candidates.none { matchesExact(it) }) {
                val sameType = candidates.filter { it.nerType == exp.type }.map { "'${it.text}'" }.take(4)
                val ctx      = if (sameType.isEmpty()) "[aucun ${exp.type}]" else "[trouvé ${exp.type}: $sameType]"
                val hintStr  = exp.hint?.let { "/$it" } ?: ""
                val approx   = candidates.firstOrNull { c ->
                    c.nerType == exp.type &&
                    (exp.hint == null || c.nerHint.orEmpty().uppercase().contains(exp.hint.uppercase())) &&
                    normExpText != null &&
                    (norm(c.text).lowercase().contains(normExpText) || normExpText.contains(norm(c.text).lowercase()))
                }
                if (approx != null)
                    issues += "APPROX  attendu=${exp.type}$hintStr='${exp.text}' obtenu='${approx.text}' $ctx"
                else
                    issues += "MANQUANT ${exp.type}$hintStr='${exp.text}' $ctx"
            }
        }
        if (case.expected.isEmpty()) {
            val fps = candidates.filter { it.isName }
            if (fps.isNotEmpty())
                issues += "FAUX-POSITIF named: ${fps.joinToString { "${it.nerType}:${it.nerHint ?: "?"}='${it.text}'" }}"
        }
        return issues
    }

    // ── Rapport Markdown ─────────────────────────────────────────────────────
    private fun buildMarkdown(
        cases: List<TestCase>,
        failures: List<Triple<TestCase, List<String>, List<Candidate>>>,
        scorePct: Int,
    ) = buildString {
        val ok = cases.size - failures.size
        appendLine("## ${if (scorePct >= threshold) "✅" else "❌"} NER Candidates — $ok/${cases.size} OK ($scorePct%)")
        appendLine()
        appendLine("> **Seuil** : $threshold% &nbsp;|&nbsp; **Cas** : ${cases.size} &nbsp;|&nbsp; **Mode** : SpringBootTest (embedded)")
        appendLine()
        appendLine("### Par catégorie")
        appendLine()
        appendLine("| Catégorie | OK | Total | |")
        appendLine("|-----------|---:|------:|---|")
        cases.groupBy { it.category.substringBefore("/") }.toSortedMap().forEach { (cat, group) ->
            val catOk = group.count { c -> failures.none { it.first.id == c.id } }
            appendLine("| `$cat` | $catOk | ${group.size} | ${"🟩".repeat(catOk)}${"🟥".repeat(group.size - catOk)} |")
        }
        appendLine()
        if (failures.isEmpty()) {
            appendLine("### ✅ Tous les cas passent !")
        } else {
            appendLine("### ⚠️ Cas en échec (${failures.size})")
            appendLine()
            appendLine("<details><summary>Voir le détail (${failures.size} cas)</summary>")
            appendLine()
            failures.forEach { (case, issues, got) ->
                appendLine("**[${case.category}]** `${case.text.take(80)}`  ")
                appendLine("_${case.note}_  ")
                appendLine("Obtenu : `${got.joinToString { "${it.nerType}='${it.text}'" }}`  ")
                issues.forEach { appendLine("⚠️ $it  ") }
                appendLine()
            }
            appendLine("</details>")
        }
    }

    // ── Test principal ───────────────────────────────────────────────────────
    @Test
    fun `NER candidates batch - tous les cas du JSONL`() {
        val cases = loadTestCases()

        println("\n${"=".repeat(110)}")
        println("NER CANDIDATES BATCH  —  ${cases.size} cas  [SpringBootTest embedded  batchSize=$batchSize]")
        println("=".repeat(110))
        println("%4s  %-52s  %-20s  %s".format("#", "STATUT", "CATEGORIE", "PHRASE[:60]"))
        println("-".repeat(110))

        val allCandidates = ArrayList<List<Candidate>>(cases.size)
        cases.chunked(batchSize).forEach { chunk ->
            allCandidates += callBatch(chunk.map { it.text })
        }

        data class Failure(val case: TestCase, val issues: List<String>, val got: List<Candidate>)
        val failures = mutableListOf<Failure>()

        cases.forEachIndexed { i, case ->
            val candidates = allCandidates.getOrElse(i) { emptyList() }
            val issues     = checkCase(case, candidates)
            val label      = if (issues.isEmpty()) "OK" else "KO ${issues[0].take(44)}"
            println("%4d  %-52s  [%-18s]  %s".format(case.id, label, case.category.take(18), case.text.take(60)))
            if (issues.isNotEmpty()) failures += Failure(case, issues, candidates)
        }

        val byCategory = cases.groupBy { it.category.substringBefore("/") }
        println("\n-- PAR CATEGORIE " + "-".repeat(93))
        byCategory.toSortedMap().forEach { (cat, group) ->
            val ok = group.count { c -> failures.none { it.case.id == c.id } }
            println("  %-18s  %d/%d  %s".format(cat, ok, group.size, "#".repeat(ok) + ".".repeat(group.size - ok)))
        }

        val ok       = cases.size - failures.size
        val scorePct = ok * 100 / cases.size
        println("\n${"=".repeat(110)}")
        println("RESULTAT : $ok/${cases.size} OK  ($scorePct%)  -- seuil=$threshold%")
        println("=".repeat(110))

        val md = buildMarkdown(cases, failures.map { Triple(it.case, it.issues, it.got) }, scorePct)
        reportFile?.let { path ->
            File(path).also { it.parentFile?.mkdirs() }.writeText(md)
            println("Rapport Markdown ecrit : $path")
        }

        if (failures.isNotEmpty()) {
            val sb = StringBuilder("\n${failures.size} cas en echec :")
            failures.forEach { (case, issues, got) ->
                sb.appendLine("\n  [${case.category}] ${case.text.take(80)}")
                sb.appendLine("  Note   : ${case.note}")
                sb.appendLine("  Obtenu : ${got.map { Triple(it.nerType, it.nerHint, it.text) }}")
                issues.forEach { sb.appendLine("  !  $it") }
            }
            println(sb)
            if (scorePct < threshold)
                fail("Score NER $scorePct% < seuil $threshold% ($ok/${cases.size} OK)\n$sb")
            else
                println("!  $scorePct% >= seuil $threshold% -- test PASSE malgre ${failures.size} cas KO")
        }
    }
}
