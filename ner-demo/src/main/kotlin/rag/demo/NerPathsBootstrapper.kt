package rag.demo

import org.slf4j.LoggerFactory
import org.springframework.boot.SpringApplication
import org.springframework.boot.env.EnvironmentPostProcessor
import org.springframework.core.env.ConfigurableEnvironment
import org.springframework.core.env.MapPropertySource
import java.io.File
import java.net.URI

/**
 * Résout et télécharge automatiquement le modèle ONNX avant l'initialisation des beans.
 *
 * Priorité pour le modèle :
 *  1. `NER_MODEL_PATH` (env) ou `ner.model-path` (property) → fichier existant
 *  2. Cache utilisateur : `~/.pimpmyrag/model/best_model_multitask_full.onnx`
 *  3. Téléchargement depuis `MODEL_URL` (env) ou `ner.model-url` (property)
 *
 * Priorité pour le tokenizer :
 *  1. `NER_TOKENIZER_PATH` (env) ou `ner.tokenizer-path` (property) → dossier existant
 *  (Pas de téléchargement automatique : le tokenizer est bundlé dans l'installateur jpackage
 *   via le chemin `$APPDIR/tokenizer_export_clean`, ou présent dans le repo.)
 */
class NerPathsBootstrapper : EnvironmentPostProcessor {

    private val log = LoggerFactory.getLogger(NerPathsBootstrapper::class.java)

    override fun postProcessEnvironment(env: ConfigurableEnvironment, app: SpringApplication) {
        val overrides = mutableMapOf<String, Any>()

        resolveModelPath(env)?.let { overrides["ner.model-path"] = it }
        resolveTokenizerPath(env)?.let { overrides["ner.tokenizer-path"] = it }

        if (overrides.isNotEmpty()) {
            env.propertySources.addFirst(MapPropertySource("ner-bootstrap", overrides))
        }
    }

    // ── Model ──────────────────────────────────────────────────────────────────

    private fun resolveModelPath(env: ConfigurableEnvironment): String? {
        // 1. Explicit path already pointing to an existing file → keep as-is
        val explicit = System.getenv("NER_MODEL_PATH")
            ?: env.getProperty("ner.model-path")?.takeIf { it.isNotBlank() }
        if (explicit != null) {
            if (File(explicit).isFile) return explicit
            log.warn("[bootstrap] Modèle introuvable à '{}', tentative auto-download…", explicit)
        }

        // 2. User-local cache (re-used across versions)
        val cached = userModelFile()
        if (cached.isFile) {
            log.info("[bootstrap] Modèle en cache : {}", cached)
            return cached.absolutePath
        }

        // 3. Auto-download
        val modelUrl = System.getenv("MODEL_URL")
            ?: env.getProperty("ner.model-url")
            ?: return null  // no URL configured → let NerService fail with a clear message

        log.info("[bootstrap] Téléchargement du modèle depuis {} …", modelUrl)
        cached.parentFile.mkdirs()
        return try {
            downloadWithProgress(modelUrl, cached)
            cached.absolutePath
        } catch (e: Exception) {
            log.error("[bootstrap] Échec du téléchargement : {}", e.message)
            cached.deleteQuietly()
            null
        }
    }

    // ── Tokenizer ──────────────────────────────────────────────────────────────

    private fun resolveTokenizerPath(env: ConfigurableEnvironment): String? {
        val explicit = System.getenv("NER_TOKENIZER_PATH")
            ?: env.getProperty("ner.tokenizer-path")?.takeIf { it.isNotBlank() }
        if (explicit != null && File(explicit).isDirectory && File(explicit, "tokenizer.json").exists()) {
            return explicit  // already valid, keep
        }
        if (explicit != null && explicit.isNotBlank()) {
            log.warn("[bootstrap] Tokenizer introuvable à '{}', chemin conservé (Spring échouera si invalide)", explicit)
        }
        // No auto-download: tokenizer is either bundled (jpackage) or in the repo checkout
        return null
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private fun userModelFile() =
        File(System.getProperty("user.home"), ".pimpmyrag/model/best_model_multitask_full.onnx")

    private fun downloadWithProgress(url: String, dest: File) {
        val conn = URI.create(url).toURL().openConnection()
        conn.setRequestProperty("User-Agent", "PimpMyRAG-NerDemo/1.0")
        conn.connect()
        val total   = conn.contentLengthLong
        val totalMb = if (total > 0) total / 1_048_576L else -1L
        val label   = if (totalMb > 0) " (${totalMb} MB)" else ""
        println("[bootstrap] Téléchargement du modèle$label → ${dest.absolutePath}")

        conn.getInputStream().use { input ->
            dest.outputStream().use { output ->
                val buf      = ByteArray(131_072)
                var downloaded = 0L
                var lastPct    = -1
                var n: Int
                while (input.read(buf).also { n = it } != -1) {
                    output.write(buf, 0, n)
                    downloaded += n
                    if (total > 0) {
                        val pct = (downloaded * 100L / total).toInt()
                        if (pct / 5 > lastPct / 5) {
                            print("\r[bootstrap] ${pct}%  (${downloaded / 1_048_576} / ${totalMb} MB)  ")
                            System.out.flush()
                            lastPct = pct
                        }
                    }
                }
            }
        }
        println("\r[bootstrap] Téléchargement terminé — ${dest.length() / 1_048_576} MB   ")
    }

    private fun File.deleteQuietly() = runCatching { delete() }
}

