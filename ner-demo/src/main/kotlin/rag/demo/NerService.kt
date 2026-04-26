package rag.demo

import org.slf4j.LoggerFactory
import org.springframework.beans.factory.DisposableBean
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Service
import rag.connectors.ner.onnx.ExtractionResult
import rag.connectors.ner.onnx.OnnxMultiHeadEntityExtractor

@Service
class NerService(
    @Value("\${ner.model-path}")         modelPath: String,
    @Value("\${ner.tokenizer-path}")     tokenizerPath: String,
    @Value("\${ner.max-seq-len:128}")    maxSeqLen: Int,
    @Value("\${ner.max-span-len:12}")    maxSpanLen: Int,
    @Value("\${ner.tau-boundary:0.70}")  tauBoundary: Float,
    @Value("\${ner.tau-none:0.99}")      tauNone: Float,
    @Value("\${ner.tau-coarse:0.45}")    tauCoarse: Float,
    @Value("\${ner.tau-svo-boundary:0.50}") tauSvo: Float,
) : DisposableBean {

    private val log = LoggerFactory.getLogger(NerService::class.java)

    private val extractor = OnnxMultiHeadEntityExtractor(
        modelPath    = modelPath,
        tokenizerDir = tokenizerPath,
        maxSeqLen    = maxSeqLen,
        maxSpanLen   = maxSpanLen,
        tauBoundary  = tauBoundary,
        tauNone      = tauNone,
        tauCoarse    = tauCoarse,
        tauSvoBoundary = tauSvo,
    ).also { log.info("✅ Modèle NER chargé depuis {}", modelPath) }

    fun analyse(text: String): ExtractionResult {
        val t0 = System.currentTimeMillis()
        return extractor.extractWithSvo(text).also {
            log.info("Inférence : {} entités, {} SVO — {}ms",
                it.entities.size, it.svoSpans.size, System.currentTimeMillis() - t0)
        }
    }

    override fun destroy() = extractor.close()
}

