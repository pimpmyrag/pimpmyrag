package io.ner

import org.springframework.boot.test.context.TestConfiguration
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Primary
import rag.engine.Embedder
import rag.engine.NerExtractor
import rag.engine.UDParser
import rag.model.Entity
import rag.model.RagDocument
import rag.model.UDDocument
import rag.model.UDSentence
import rag.model.UDToken
import rag.model.UPOS

/**
 * Configuration de test pour NerCandidatesBatchIT.
 *
 * Fournit des stubs pour les beans qui nécessitent des ressources indisponibles en CI :
 *  - UDParser      → stub whitespace tokenizer (pas de service Stanza sur localhost:8000)
 *  - NerExtractor  → stub vide (BILOU model XLM-RoBERTa absent du repo)
 *  - Embedder      → stub déterministe (BGE-M3 non requis pour le test NER)
 *
 * MultiAxisTrainingService est mocké via @MockBean dans NerCandidatesBatchIT
 * pour éviter l'instanciation de SyntheticDataGenerator (exige LLM_API_KEY).
 *
 * Le vrai NerExtractorFromUD (DeBERTa, best_model-v2.onnx) est chargé
 * normalement via OnnxNerAutoConfiguration — chemins fournis par ${ner.model.root}
 * défini via la propriété système (cf. build.gradle.kts) et
 * src/test/resources/application.yml.
 */
@TestConfiguration
class NerCiTestConfig {

    /**
     * Stub UDParser : tokenise sur les espaces blancs et produit des UDDocuments minimaux.
     * Permet au pipeline DeBERTa (NerExtractorFromUD) de recevoir de vrais tokens
     * sans appeler le service Stanza sur localhost:8000.
     */
    @Bean
    @Primary
    fun udParser(): UDParser = object : UDParser {
        override fun parse(documents: List<RagDocument>): List<UDDocument> =
            documents.map { doc ->
                val matches = Regex("\\S+").findAll(doc.text).toList()
                val tokens = matches.mapIndexed { i, m ->
                    UDToken(
                        id     = i + 1,
                        text   = m.value,
                        lemma  = m.value.lowercase(),
                        upos   = UPOS.NOUN,
                        xpos   = null,
                        head   = 0,
                        deprel = "root",
                        start  = m.range.first,
                        end    = m.range.last + 1,
                    )
                }
                val sentence = tokens.takeIf { it.isNotEmpty() }?.let {
                    UDSentence(id = 0, tokens = it, start = it.first().start, end = it.last().end)
                }
                UDDocument(text = doc.text, sentences = listOfNotNull(sentence))
            }
    }

    /**
     * Stub NerExtractor (BILOU — modèle XLM-RoBERTa absent du repo).
     * Retourne une liste vide pour chaque document.
     * Prend la place du bean OnnxLabelNerAutoConfiguration via @ConditionalOnMissingBean.
     */
    @Bean
    @Primary
    fun nerExtractor(): NerExtractor = object : NerExtractor {
        override fun extractNer(documents: List<RagDocument>): List<List<Entity>> =
            documents.map { emptyList() }
    }

    /**
     * Stub Embedder — vecteurs déterministes de dimension 8 basés sur le hash du texte.
     * Prend la place du bean OnnxEmbeddingAutoConfiguration via @ConditionalOnMissingBean.
     */
    @Bean
    @Primary
    fun embedder(): Embedder = object : Embedder {
        override fun embed(documents: List<RagDocument>): List<FloatArray> =
            documents.map { d ->
                FloatArray(8) { i -> ((d.text.hashCode() * (i + 1)) % 1000) / 1000f }
            }
    }
}
