package rag.connectors.ner.onnx

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import java.nio.LongBuffer
import java.nio.file.Paths
import kotlin.math.exp

// Chemins configurables via propriétés système ou variables d'environnement :
//   -Ddiag.tokenizer.path=...   ou   DIAG_TOKENIZER_PATH=...
//   -Ddiag.model.path=...        ou   DIAG_MODEL_PATH=...
private val TOKENIZER_CLEAN: String =
    System.getProperty("diag.tokenizer.path")
        ?: System.getenv("DIAG_TOKENIZER_PATH")
        ?: System.getenv("NER_TOKENIZER_PATH")
        ?: "training/multi-head/tokenizer_export_clean"
private val MODEL_PATH: String =
    System.getProperty("diag.model.path")
        ?: System.getenv("DIAG_MODEL_PATH")
        ?: System.getenv("NER_MODEL_PATH")
        ?: error("Définir DIAG_MODEL_PATH ou NER_MODEL_PATH pour utiliser DiagTokenizer")
private const val TEXT = "Emmanuel Macron s'est rendu à Berlin pour rencontrer Olaf Scholz."

fun softmaxDiag(logits: FloatArray): FloatArray {
    val max = logits.max()
    val exps = FloatArray(logits.size) { exp((logits[it] - max).toDouble()).toFloat() }
    val sum = exps.sum()
    return FloatArray(exps.size) { exps[it] / sum }
}

fun main() {
    val tok = HuggingFaceTokenizer.newInstance(Paths.get(TOKENIZER_CLEAN))
    val enc = tok.encode(TEXT)

    val clsId     = 1L   // DeBERTa: CLS=1, SEP=2 (vérifiés via Python)
    val sepId     = 2L
    val rawLen    = enc.ids.size
    val seqLen    = rawLen  // DJL doit maintenant inclure CLS+SEP via post_processor
    val realCount = seqLen - 2  // real tokens without CLS/SEP

    val ids       = enc.ids   // L=17, CLS à [0], SEP à [16]
    val wIds      = enc.wordIds  // [-1, 0, 1, 2, 2, 2, 3, 3, 4, 5, …, -1]
    val charOffsets: List<Pair<Int, Int>?> =
        enc.charTokenSpans.map { s -> s?.let { it.start to it.end } }

    println("=== TOKENIZATION ===")
    println("L (avec CLS+SEP) = $seqLen   ids[0]=${ids.getOrNull(0)} (CLS=1?)  ids[last]=${ids.getOrNull(seqLen-1)} (SEP=2?)")
    println("tokens : ${enc.tokens.toList()}")
    println("wIds   : ${wIds.toList()}")

    val wordRanges = Regex("\\S+").findAll(TEXT).map { it.range.first to (it.range.last + 1) }.toList()
    println("\nwordRanges: ${wordRanges.mapIndexed { i, (s, e) -> "[$i]\"${TEXT.substring(s, e)}\"" }}")

    data class Word(val firstTok: Int, val lastTok: Int, val charStart: Int, val charEnd: Int, val text: String)
    val words = mutableListOf<Word>()
    var prevWid = Long.MIN_VALUE; var wFirstTok = -1; var wLastTok = -1

    fun flushWord() {
        if (wFirstTok < 0) return
        val wid = wIds[wFirstTok]
        if (wid < 0 || wid >= wordRanges.size) { wFirstTok = -1; return }
        val (cs, ce) = wordRanges[wid.toInt()]
        val wordText = TEXT.substring(cs, ce)
        val apoIdx = wordText.indexOfFirst { it == '\'' || it == '\u2019' }
        if (apoIdx in 1..3 && apoIdx < wordText.length - 1) {
            val truncStart = cs + apoIdx + 1
            val truncFirstTok = (wFirstTok..wLastTok).firstOrNull { i ->
                val off = charOffsets.getOrNull(i); off != null && off.first >= truncStart
            } ?: (wFirstTok + 1).coerceAtMost(wLastTok)
            words += Word(truncFirstTok, wLastTok, truncStart, ce, TEXT.substring(truncStart, ce))
        }
        words += Word(wFirstTok, wLastTok, cs, ce, wordText)
        wFirstTok = -1
    }

    for (i in 0 until seqLen) {
        val wid = wIds[i]
        if (wid < 0) { flushWord(); prevWid = wid; continue }
        if (wid != prevWid) { flushWord(); wFirstTok = i }
        wLastTok = i; prevWid = wid
    }
    flushWord()

    println("\n=== WORDS (${words.size}) ===")
    words.forEachIndexed { i, w -> println("  [$i] tok[${w.firstTok}:${w.lastTok}] char[${w.charStart}:${w.charEnd}] \"${w.text}\"") }

    val candidates = mutableListOf<Triple<Int, Int, String>>()
    for (si in words.indices) {
        for (ei in si until minOf(si + 8, words.size)) {
            val spanText = TEXT.substring(words[si].charStart, words[ei].charEnd).trim()
            if (spanText.length >= 2) candidates += Triple(words[si].firstTok, words[ei].lastTok, spanText)
        }
    }
    println("\n=== CANDIDATES (${candidates.size}) - premiers 15 ===")
    candidates.take(15).forEach { (s, e, t) -> println("  tok[$s:$e] \"$t\"") }

    println("\n=== ONNX INFERENCE ===")
    val env = OrtEnvironment.getEnvironment()
    val session = env.createSession(MODEL_PATH)
    val N = candidates.size
    val attMask  = LongArray(seqLen) { 1L }
    val starts   = LongArray(N) { candidates[it].first.toLong() }
    val ends     = LongArray(N) { candidates[it].second.toLong() }
    val batchIds = LongArray(N) { 0L }

    val tIds  = OnnxTensor.createTensor(env, LongBuffer.wrap(ids),     longArrayOf(1, seqLen.toLong()))
    val tMask = OnnxTensor.createTensor(env, LongBuffer.wrap(attMask), longArrayOf(1, seqLen.toLong()))
    val tS    = OnnxTensor.createTensor(env, LongBuffer.wrap(starts),   longArrayOf(N.toLong()))
    val tE    = OnnxTensor.createTensor(env, LongBuffer.wrap(ends),     longArrayOf(N.toLong()))
    val tB    = OnnxTensor.createTensor(env, LongBuffer.wrap(batchIds), longArrayOf(N.toLong()))

    session.run(mapOf("input_ids" to tIds, "attention_mask" to tMask,
        "span_starts" to tS, "span_ends" to tE, "span_batch_ids" to tB)).use { result ->
        @Suppress("UNCHECKED_CAST")
        val bndLogits    = result["boundary_logits"].get().value as Array<FloatArray>
        @Suppress("UNCHECKED_CAST")
        val coarseLogits = result["coarse_logits"].get().value as Array<FloatArray>
        val coarseLabels = listOf("PER","LOC","ORG","TIME","EVENT","OBJECT","VALUE","ABSTRACT","NONE")

        val probs = bndLogits.map { softmaxDiag(it)[1] }
        println("Max boundary prob : %.4f".format(probs.max()))
        println("Spans p>0.40      : ${probs.count { it > 0.40 }}")
        println("Spans p>0.70      : ${probs.count { it > 0.70 }}")
        println("\nTop 10 :")
        probs.indices.sortedByDescending { probs[it] }.take(10).forEach { k ->
            val (s, e, t) = candidates[k]
            val cp = softmaxDiag(coarseLogits[k])
            val ci = cp.indices.maxByOrNull { cp[it] }!!
            println("  p=%.4f  tok[$s:$e]  coarse=${coarseLabels[ci]}(%.3f)  \"$t\"".format(probs[k], cp[ci]))
        }
    }
    listOf(tIds, tMask, tS, tE, tB).forEach { it.close() }
    session.close()
    tok.close()
}
