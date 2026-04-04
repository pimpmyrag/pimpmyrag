package rag.app

import com.example.ingest.repo.MongoProvider
import com.example.ingest.service.DocumentStore
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import rag.connectors.stub.DefaultChunkerFactory
import rag.connectors.stub.StubDocReaderFactory
import rag.connectors.stub.StubEmbedderFactory
import rag.connectors.stub.StubFilterFactory
import rag.connectors.stub.StubGeneratorFactory
import rag.connectors.stub.StubRerankerFactory
import rag.connectors.stub.StubVectorReaderFactory
import rag.connectors.stub.StubVectorWriterFactory
import rag.engine.DocumentStoreReader
import rag.engine.DocumentStoreWriter
import rag.engine.VectorStoreReader
import rag.engine.VectorStoreWriter
import rag.runner.PlanRunnerDag

@Configuration
class Config {

    @Bean
    fun sentenceProvider(mongoProvider: MongoProvider): DocumentStoreReader {
        return DocumentStore(mongoProvider)
    }

    @Bean
    fun sentenceWriterProvider(mongoProvider: MongoProvider): DocumentStoreWriter {
        return DocumentStore(mongoProvider)
    }

    @Bean
    fun mongoProvider(): MongoProvider {
        return MongoProvider(uri = "mongodb://root:password@localhost:27017/admin?authSource=admin", dbName = "rag-data-dev")
    }

    @Bean
    fun runner(
        storeReader: VectorStoreReader,
        storeWriter: VectorStoreWriter,
        documentStoreReader: DocumentStoreReader,
        documentStoreWriter: DocumentStoreWriter): PlanRunnerDag {
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
        return runner
    }

}