package io.probes

data class Probe(
    val name: String,
    val tags: List<String> = emptyList(),
    val weight: Float = 1.0f,
    val hypotheses: Map<String, String>,
    val notes: String? = null
) {
    fun hypothesis(locale: String = "en"): String =
        hypotheses[locale] ?: hypotheses["en"] ?: error("Aucune hypothèse pour $locale")
}

data class ProbesBundle(
    val schema_version: String,
    val created_at: String,
    val model: String,
    val default_locale: String = "en",
    val probes: List<Probe>,
    val locale_mapping: Map<String, List<String>>? = null,
    val notes: String? = null
)