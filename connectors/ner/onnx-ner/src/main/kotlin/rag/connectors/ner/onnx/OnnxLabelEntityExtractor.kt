package rag.connectors.ner.onnx

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import rag.engine.NerExtractor
import rag.model.Entity
import rag.model.RagDocument
import rag.model.Span
import java.nio.LongBuffer
import java.nio.file.Paths

class OnnxBilouEntityExtractor(
	modelPath: String,
	tokenizerDir: String,
	// Ordre exact du id2label du modèle fine-tuné (config.json) :
	// groupé par préfixe BILOU (tous les B- d'abord, puis I-, L-, U-)
	// et non par type d'entité comme dans l'ancienne version.
	private val labelNames: List<String> = listOf(
		"O",
		"B-PER", "B-LOC", "B-OBJECT", "B-ORG", "B-TIME", "B-EVENT",
		"I-PER", "I-LOC", "I-OBJECT", "I-ORG", "I-TIME", "I-EVENT",
		"L-PER", "L-LOC", "L-OBJECT", "L-ORG", "L-TIME", "L-EVENT",
		"U-PER", "U-LOC", "U-OBJECT", "U-ORG", "U-TIME", "U-EVENT"
	)
) : AutoCloseable, NerExtractor {

	private val env = OrtEnvironment.getEnvironment()
	private val session = env.createSession(modelPath, OrtSession.SessionOptions())
	private val tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerDir))

	override fun extractNer(documents: List<RagDocument>): List<List<Entity>> =
		documents.map { extractFromText(it.text) }

	fun extractFromText(text: String): List<Entity> {

		println("========== NER DEBUG ==========")
		println("INPUT TEXT:\n$text")
		println("================================")

		// 1. Découpage whitespace — identique à l'entraînement (is_split_into_words=True)
		val words = Regex("\\S+").findAll(text).map { it.value }.toList()
		val wordCharSpans = Regex("\\S+").findAll(text)
			.map { it.range.first to it.range.last + 1 }
			.toList()

		println("\nWORDS:")
		words.forEachIndexed { i, w -> println("  [$i] \"$w\" charSpan=${wordCharSpans[i]}") }

		// 2. Encodage pré-tokenisé : ajoute <s> et </s> UNE SEULE FOIS autour de la séquence
		//    (équivalent Python : tokenizer(words, is_split_into_words=True))
		//    L'ancienne version encodait chaque mot séparément → BOS/EOS autour de chaque mot → FAUX
		val enc      = tokenizer.encode(words.toTypedArray())
		val tokens   = enc.tokens
		val wordIds  = enc.wordIds   // long[] : index du mot source ; sentinel hors [0, words.size) pour BOS/EOS

		println("\n========== TOKENIZATION DETAILS ==========")
		tokens.forEachIndexed { i, tok ->
			println("  [$i] \"$tok\" id=${enc.ids[i]}  wordId=${wordIds[i]}")
		}

		// 3. Construction du charSpan par token :
		//    - Premier subtoken d'un mot → charSpan du mot
		//    - Subtoken de continuation   → null  (ignoré comme -100 à l'entraînement)
		//    - Tokens spéciaux (BOS/EOS)  → null
		val tokenCharSpans = buildList<Pair<Int, Int>?> {
			var prevWordId = -1L
			for (i in wordIds.indices) {
				val wid = wordIds[i]
				if (wid < 0 || wid >= words.size) {
					add(null)                              // token spécial
				} else if (wid != prevWordId) {
					add(wordCharSpans[wid.toInt()])        // premier subtoken du mot
				} else {
					add(null)                              // subtoken de continuation
				}
				prevWordId = wid
			}
		}

		println("\n========== ALIGNED CHAR SPANS ==========")
		tokenCharSpans.forEachIndexed { i, cs ->
			if (cs != null) println("  [$i] \"${tokens[i]}\" → charSpan=$cs")
		}
		println("seq_length = ${enc.ids.size}")
		println("=========================================\n")

		// 4. Construction des tenseurs
		val inputIdArr   = enc.ids
		val attentionArr = LongArray(enc.ids.size) { 1L }
		val inputIdsT    = tensor2d(inputIdArr)
		val attT         = tensor2d(attentionArr)

		println("Running ONNX inference...")
		val logits = session.run(
			mapOf("input_ids" to inputIdsT, "attention_mask" to attT)
		).use { it[0].value }

		inputIdsT.close()
		attT.close()

		val seq = to3D(logits)[0]

		println("\n========== MODEL LOGITS (first tokens) ==========")
		for (i in 0 until minOf(seq.size, 10)) {
			println("  token[$i] \"${tokens[i]}\" → ${labelNames[argmax(seq[i])]}")
		}
		println("=================================================")

		val rawEntities = decodeBilou(seq, tokenCharSpans, text)

		println("\n========== DECODED ENTITIES (raw) ==========")
		rawEntities.forEach { println(" - ${it.type}: \"${it.text}\" span=${it.span}") }

		// Fusion des entités consécutives de même type séparées uniquement par des espaces.
		val entities = mergeConsecutive(rawEntities, text)

		println("\n========== DECODED ENTITIES (merged) ==========")
		entities.forEach { println(" - ${it.type}: \"${it.text}\" span=${it.span}") }
		println("================================================\n")

		return entities
	}

	/** Fusionne les entités adjacentes du même type séparées par du blanc uniquement. */
	private fun mergeConsecutive(entities: List<Entity>, fullText: String): List<Entity> {
		if (entities.isEmpty()) return entities
		val result = mutableListOf<Entity>()
		var current = entities[0]
		for (next in entities.drop(1)) {
			val curEnd   = current.span?.end   ?: -1
			val nextStart = next.span?.start    ?: -1
			val nextEnd   = next.span?.end      ?: -1
			val curStart  = current.span?.start ?: -1
			val canMerge  = next.type == current.type
				&& curEnd >= 0 && nextStart >= curEnd
				&& fullText.substring(curEnd, nextStart).isBlank()
			if (canMerge) {
				val merged = fullText.substring(curStart, nextEnd)
				current = Entity(merged, current.type, Span(curStart, nextEnd, emptyList()))
			} else {
				result += current
				current = next
			}
		}
		result += current
		return result
	}

	private fun decodeBilou(
		seq: Array<FloatArray>,
		spans: List<Pair<Int, Int>?>,
		text: String
	): List<Entity> {

		val out = mutableListOf<Entity>()
		var i = 0

		while (i < seq.size) {

			val span = spans[i]
			if (span == null) { i++; continue }

			val raw = labelNames[argmax(seq[i])]
			if (raw == "O") { i++; continue }

			val (tag, type) = raw.split("-", limit = 2)

			when (tag) {
				"U" -> {
					val (cs, ce) = span
					if (ce <= text.length)
						out += Entity(text.substring(cs, ce), type.lowercase(), Span(cs, ce, emptyList()))
					i++
				}

				"B" -> {
					val start = i
					var end   = i
					i++

					while (i < seq.size) {
						val nextSpan = spans[i]
						// Subtoken de continuation ou token spécial → on saute sans briser l'entité
						if (nextSpan == null) { i++; continue }

						val nextLab = labelNames[argmax(seq[i])]
						if (nextLab == "I-$type") { end = i; i++; continue }  // BIO+BILOU : end suit chaque I-X
						if (nextLab == "L-$type") { end = i; i++; break }     // BILOU uniquement (rétrocompat)
						break
					}

					val s = spans[start]
					val e = spans[end]
					if (s != null && e != null) {
						val cs = s.first
						val ce = e.second
						if (ce <= text.length)
							out += Entity(text.substring(cs, ce), type.lowercase(), Span(cs, ce, emptyList()))
					}
				}

				else -> i++
			}
		}

		return out
	}

	private fun tensor2d(arr: LongArray) =
		OnnxTensor.createTensor(env, LongBuffer.wrap(arr), longArrayOf(1, arr.size.toLong()))

	private fun argmax(a: FloatArray): Int {
		var best = 0
		var max  = a[0]
		for (i in 1 until a.size) if (a[i] > max) { max = a[i]; best = i }
		return best
	}

	private fun to3D(obj: Any?): Array<Array<FloatArray>> =
		when (obj) {
			is Array<*> -> obj.map { lvl1 ->
				when (lvl1) {
					is Array<*>   -> lvl1.map { lvl2 ->
						when (lvl2) {
							is FloatArray  -> lvl2
							is DoubleArray -> lvl2.map { it.toFloat() }.toFloatArray()
							else           -> FloatArray(0)
						}
					}.toTypedArray()
					is FloatArray  -> arrayOf(lvl1)
					is DoubleArray -> arrayOf(lvl1.map { it.toFloat() }.toFloatArray())
					else           -> arrayOf()
				}
			}.toTypedArray()
			else -> arrayOf()
		}

	override fun close() {
		tokenizer.close()
		session.close()
	}
}