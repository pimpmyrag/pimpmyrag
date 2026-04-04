package io.ner

import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import org.junit.jupiter.api.Assumptions
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.fail
import org.springframework.web.reactive.function.client.WebClient
import org.springframework.web.reactive.function.client.WebClientRequestException
import org.springframework.web.reactive.function.client.WebClientResponseException
import org.springframework.web.reactive.function.client.bodyToMono
import java.text.Normalizer

/**
 * Test d'intégration NER — lit `ner_candidates_tests.jsonl` depuis le classpath
 * et appelle `/api/classify/extract/candidates/batch` sur le serveur local.
 *
 * Pré-requis : le serveur doit être démarré (rag-app ou radar-nli-toolkit).
 * Si le serveur n'est pas disponible, le test est automatiquement ignoré (Assumption).
 *
 * Configuration :
 *   -Dner.test.url=http://localhost:8080   (défaut)
 *   -Dner.test.batchSize=16               (défaut)
 *   -Dner.test.threshold=70               (% minimum pour passer, défaut 70)
 */
@Tag("integration")
class NerCandidatesBatchIT {

    // ── Configuration ────────────────────────────────────────────────────────
    private val baseUrl   = System.getProperty("ner.test.url", "http://localhost:8080")
    private val batchSize = System.getProperty("ner.test.batchSize", "16").toInt()
    /** Score minimum (%) en dessous duquel le test fail. 0 = jamais fail. */
    private val threshold = System.getProperty("ner.test.threshold", "70").toInt()

    private val mapper = ObjectMapper().registerKotlinModule()
    private val client = WebClient.builder()
        .baseUrl(baseUrl)
        .codecs { it.defaultCodecs().maxInMemorySize(4 * 1024 * 1024) }
        .build()

    // ── Modèle JSONL (cas de test) ───────────────────────────────────────────

    /** Un attendu extrait du JSONL : type NER + texte du span + hint optionnel. */
    @JsonIgnoreProperties(ignoreUnknown = true)
    data class ExpectedEntity(
        val type: String,
        val text: String?,
        val hint: String? = null,
    )

    /** Un cas de test complet. */
    @JsonIgnoreProperties(ignoreUnknown = true)
    data class TestCase(
        val id:       Int,
        val text:     String,
        val expected: List<ExpectedEntity>,
        val category: String,
        val note:     String,
    )

    // ── Modèle de réponse API ────────────────────────────────────────────────

    /** Un candidat NER retourné par l'API. */
    @JsonIgnoreProperties(ignoreUnknown = true)
    data class Candidate(
        val nerType: String,
        val text:    String,
        val nerHint: String?  = null,
        val isName:  Boolean  = false,
    )

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
            lines.filter { it.isNotBlank() }
                 .map { mapper.readValue<TestCase>(it) }
                 .toList()
        }
    }

    // ── Disponibilité du serveur ─────────────────────────────────────────────
    private fun isServerAvailable(): Boolean = try {
        client.get().uri("/actuator/health")
            .retrieve()
            .toBodilessEntity()
            .block()
        true
    } catch (_: WebClientResponseException) {
        true   // serveur répond avec 4xx/5xx → disponible
    } catch (_: WebClientRequestException) {
        false  // connexion refusée / timeout → indisponible
    } catch (_: Exception) {
        false
    }

    // ── Appel API batch ──────────────────────────────────────────────────────
    private fun callBatch(texts: List<String>): List<List<Candidate>> {
        val body = mapOf("texts" to texts)
        val resp = client.post()
            .uri("/api/classify/extract/candidates/batch")
            .contentType(org.springframework.http.MediaType.APPLICATION_JSON)
            .bodyValue(body)
            .retrieve()
            .bodyToMono<BatchResponse>()
            .block()
            ?: return List(texts.size) { emptyList() }
        // padding défensif si la réponse est tronquée
        return List(texts.size) { i -> resp.results.getOrElse(i) { TextResult() }.candidates }
    }

    // ── Vérification d'un cas ────────────────────────────────────────────────
    private fun checkCase(case: TestCase, candidates: List<Candidate>): List<String> {
        val issues = mutableListOf<String>()

        for (exp in case.expected) {
            val normExpText = exp.text?.let { norm(it).lowercase() }

            fun matchesExact(c: Candidate): Boolean {
                if (c.nerType != exp.type) return false
                if (normExpText != null && norm(c.text).lowercase() != normExpText) return false
                if (exp.hint != null && !c.nerHint.orEmpty().uppercase().contains(exp.hint.uppercase())) return false
                return true
            }

            if (candidates.none { matchesExact(it) }) {
                val sameType = candidates.filter { it.nerType == exp.type }.map { "'${it.text}'" }.take(4)
                val ctx      = if (sameType.isEmpty()) "[aucun ${exp.type} détecté]"
                               else "[trouvé ${exp.type}: $sameType]"
                val hintStr  = exp.hint?.let { "/$it" } ?: ""

                // Correspondance approximative (inclusion de sous-chaîne)
                val approx = candidates.firstOrNull { c ->
                    c.nerType == exp.type &&
                    (exp.hint == null || c.nerHint.orEmpty().uppercase().contains(exp.hint.uppercase())) &&
                    normExpText != null &&
                    (norm(c.text).lowercase().contains(normExpText) ||
                     normExpText.contains(norm(c.text).lowercase()))
                }

                if (approx != null)
                    issues += "APPROX  attendu=${exp.type}$hintStr='${exp.text}' obtenu='${approx.text}' $ctx"
                else
                    issues += "MANQUANT ${exp.type}$hintStr='${exp.text}' $ctx"
            }
        }

        // Faux positifs si on attend zéro entité
        if (case.expected.isEmpty()) {
            val namedFps = candidates.filter { it.isName }
            if (namedFps.isNotEmpty()) {
                val fps = namedFps.joinToString(", ") { "${it.nerType}:${it.nerHint ?: "?"}='${it.text}'" }
                issues += "FAUX-POSITIF named: $fps"
            }
        }

        return issues
    }

    // ── Test principal ───────────────────────────────────────────────────────
    @Test
    fun `NER candidates batch — tous les cas du JSONL`() {
        Assumptions.assumeTrue(
            isServerAvailable(),
            "Serveur NER non disponible à $baseUrl — test d'intégration ignoré (lancez le serveur ou passez -Dner.test.url=…)"
        )

        val cases = loadTestCases()

        println("\n${"=".repeat(110)}")
        println("NER CANDIDATES BATCH  —  ${cases.size} cas  [url=$baseUrl  batchSize=$batchSize]")
        println("=".repeat(110))
        println("%4s  %-52s  %-20s  %s".format("#", "STATUT", "CATÉGORIE", "PHRASE[:60]"))
        println("-".repeat(110))

        // Appels API par chunks
        val allCandidates = ArrayList<List<Candidate>>(cases.size)
        cases.chunked(batchSize).forEach { chunk ->
            allCandidates += callBatch(chunk.map { it.text })
        }

        // Vérification case par case
        data class Failure(val case: TestCase, val issues: List<String>, val got: List<Candidate>)
        val failures = mutableListOf<Failure>()

        cases.forEachIndexed { i, case ->
            val candidates = allCandidates.getOrElse(i) { emptyList() }
            val issues     = checkCase(case, candidates)
            val label      = if (issues.isEmpty()) "✅ OK"
                             else "❌ ${issues[0].take(44)}"
            println("%4d  %-52s  [%-18s]  %s".format(
                case.id, label, case.category.take(18), case.text.take(60)
            ))
            if (issues.isNotEmpty()) failures += Failure(case, issues, candidates)
        }

        // ── Statistiques par catégorie ────────────────────────────────────
        val byCategory = cases.groupBy { it.category.substringBefore("/") }
        println("\n── PAR CATÉGORIE " + "─".repeat(93))
        byCategory.toSortedMap().forEach { (cat, group) ->
            val ok  = group.count { c -> failures.none { it.case.id == c.id } }
            val bar = "█".repeat(ok) + "░".repeat(group.size - ok)
            println("  %-18s  %d/%d  %s".format(cat, ok, group.size, bar))
        }

        // ── Résumé ────────────────────────────────────────────────────────
        val ok      = cases.size - failures.size
        val scorePct = ok * 100 / cases.size
        println("\n${"=".repeat(110)}")
        println("RÉSULTAT : $ok/${cases.size} OK  ($scorePct%)  — seuil=$threshold%")
        println("=".repeat(110))

        // ── Détail des échecs (affiché même si on passe le seuil) ────────
        if (failures.isNotEmpty()) {
            val sb = StringBuilder()
            sb.appendLine("\n${failures.size} cas en échec :")
            failures.forEach { (case, issues, got) ->
                sb.appendLine("\n  [${case.category}] ${case.text.take(80)}")
                sb.appendLine("  Note   : ${case.note}")
                sb.appendLine("  Obtenu : ${got.map { Triple(it.nerType, it.nerHint, it.text) }}")
                issues.forEach { sb.appendLine("  ⚠  $it") }
            }
            println(sb)   // toujours logué en console

            if (scorePct < threshold) {
                fail(
                    "Score NER $scorePct% < seuil $threshold% " +
                    "($ok/${cases.size} OK)\n$sb"
                )
            } else {
                println("⚠  $scorePct% ≥ seuil $threshold% — test PASSÉ malgré ${failures.size} cas KO")
            }
        }
    }
}

