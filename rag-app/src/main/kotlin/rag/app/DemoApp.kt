// src/main/kotlin/com/search/doc/ragkotlin/RagKotlinApplication.kt
package rag.app

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.context.properties.ConfigurationPropertiesScan
import org.springframework.boot.runApplication
import rag.connectors.stub.DefaultChunkerFactory
import rag.connectors.stub.StubDocReaderFactory
import rag.connectors.stub.StubDocWriterFactory
import rag.connectors.stub.StubEmbedderFactory
import rag.connectors.stub.StubFilterFactory
import rag.connectors.stub.StubGeneratorFactory
import rag.connectors.stub.StubRerankerFactory
import rag.connectors.stub.StubVectorReaderFactory
import rag.connectors.stub.StubVectorWriterFactory
import rag.dsl.staged.rag
import rag.model.RagDocument
import rag.model.RagUnitType
import rag.runner.PlanRunnerDag

@SpringBootApplication(scanBasePackages = ["rag.app", "com.acme.infinity"])
@ConfigurationPropertiesScan(basePackages = ["com.acme.infinity"])
class OpenRagIngestionApplication

fun main(args: Array<String>) {
    runApplication<OpenRagIngestionApplication>(*args)
    val plan = rag {
        fromRaw()
            .chunk { using = "default"; unit = RagUnitType.SENTENCE; maxTokens = 128 }
            .embed { model = "bge-m3"; store = "chroma" }
            .retrieveIds { store = "chroma"; topK = 5 }
            .hydrate { store = "mongo" }
            .rerank { strategy = "xnli"; topK = 3 }
            .generate { llm = "mistral" }
    }

    val runner = PlanRunnerDag(
        chunkers   = listOf(DefaultChunkerFactory()),
        embedders  = listOf(StubEmbedderFactory()),
        vsWriters  = listOf(StubVectorWriterFactory()),
        vsReaders  = listOf(StubVectorReaderFactory()),
        docReaders = listOf(StubDocReaderFactory()),
        filters    = listOf(StubFilterFactory()),
        rerankers  = listOf(StubRerankerFactory()),
        generators = listOf(StubGeneratorFactory())
    )

    val raw = RagDocument("doc1", "Jacques devint ingénieur. Puis il s'installa à Paris.")
    // Seed doc store for hydration
    DefaultChunkerFactory().create().chunk(raw, RagUnitType.SENTENCE, 128).forEach {
        StubDocWriterFactory().create().write(listOf(it))
    }

    val answer = runner.run(plan, query = "Quel fut son métier ?", raw = raw)
    println(" === ANSWER === ${answer.text}")
}