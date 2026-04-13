package rag.connectors.ner.onnx

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.slf4j.LoggerFactory
import rag.engine.NerExtractor
import rag.model.Entity
import rag.model.RagDocument
import rag.model.Span
import java.nio.LongBuffer
import java.nio.file.Paths

class OnnxBilouEntityExtractor(
	modelPath: String,
	tokenizerDir: String,
	// BIO — 13 labels (schéma du modèle RoBERTa v5, checkpoint-7800)
	// Ordre EXACT du id2label dans config.json — tout écart = mauvais type rendu.
	private val labelNames: List<String> = listOf(
		"O",
		"B-PER", "B-LOC", "B-ORG", "B-TIME", "B-EVENT", "B-OBJECT",
		"I-PER", "I-LOC", "I-ORG", "I-TIME", "I-EVENT", "I-OBJECT",
	),
	private val maxSeqLen: Int = 128,
	private val useCoreMl: Boolean = false,
	private val intraOpThreads: Int = Runtime.getRuntime().availableProcessors(),
	private val labelToId: Map<String, Int> = labelNames.withIndex().associate { it.value to it.index }
) : AutoCloseable, NerExtractor {

	private val log = LoggerFactory.getLogger(OnnxBilouEntityExtractor::class.java)

	private val env = OrtEnvironment.getEnvironment()
	private val session = env.createSession(modelPath, OrtSession.SessionOptions().apply {
		setIntraOpNumThreads(intraOpThreads)
		if (useCoreMl) {
			try {
				addCoreML()
				log.info("[BILOU] CoreML EP activé (Apple Neural Engine / GPU)")
			} catch (e: Exception) {
				log.warn("[BILOU] CoreML non disponible : {} → fallback CPU", e.message)
			}
		}
	})
	private val tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerDir))

	override fun extractNer(documents: List<RagDocument>): List<List<Entity>> =
		extractFromTexts(documents.map { it.text })

	fun extractFromText(text: String): List<Entity> =
		extractFromTexts(listOf(text)).first()

	fun extractFromTexts(texts: List<String>): List<List<Entity>> {
		if (texts.isEmpty()) return emptyList()
		val t0 = System.nanoTime()

		data class TextEncoding(
			val text: String,
			val tokens: Array<String>,
			val ids: LongArray,
			val seqLen: Int,
			val tokenCharSpans: List<Pair<Int, Int>?>
		)

		// 1) Tokenisation
		val tTok = System.nanoTime()
		val encodings = texts.map { text ->
			val words = Regex("\\S+").findAll(text).map { it.value }.toList()
			val wordCharSpans = Regex("\\S+").findAll(text)
				.map { it.range.first to it.range.last + 1 }
				.toList()

			val enc = tokenizer.encode(words.toTypedArray())
			val wordIds = enc.wordIds
			val seqLen = minOf(enc.ids.size, maxSeqLen)

			val tokenCharSpans = buildList<Pair<Int, Int>?> {
				var prevWordId = -1L
				for (i in 0 until seqLen) {
					val wid = wordIds[i]
					if (wid < 0 || wid >= words.size) {
						add(null)
					} else if (wid != prevWordId) {
						add(wordCharSpans[wid.toInt()])
					} else {
						add(null)
					}
					prevWordId = wid
				}
			}

			TextEncoding(
				text = text,
				tokens = enc.tokens,
				ids = enc.ids,
				seqLen = seqLen,
				tokenCharSpans = tokenCharSpans
			)
		}
		log.debug("[BILOU] tokenisation  batchSize={}  ms={}", texts.size, ms(tTok))

		// 2) Construction des tenseurs
		val tTensor = System.nanoTime()
		val maxLen = encodings.maxOf { it.seqLen }
		val batchSize = texts.size

		val inputIdsFlat = LongArray(batchSize * maxLen) { 0L }
		val attentionFlat = LongArray(batchSize * maxLen) { 0L }

		encodings.forEachIndexed { i, enc ->
			for (j in 0 until enc.seqLen) {
				inputIdsFlat[i * maxLen + j] = enc.ids[j]
				attentionFlat[i * maxLen + j] = 1L
			}
		}
		log.debug("[BILOU] tenseurs       maxLen={}  ms={}", maxLen, ms(tTensor))

		// 3) Inférence ONNX
		val tInfer = System.nanoTime()
		val inputIdsT = OnnxTensor.createTensor(
			env,
			LongBuffer.wrap(inputIdsFlat),
			longArrayOf(batchSize.toLong(), maxLen.toLong())
		)
		val attT = OnnxTensor.createTensor(
			env,
			LongBuffer.wrap(attentionFlat),
			longArrayOf(batchSize.toLong(), maxLen.toLong())
		)

		val logits = session.run(
			mapOf(
				"input_ids" to inputIdsT,
				"attention_mask" to attT
			)
		).use { it[0].value }

		inputIdsT.close()
		attT.close()
		log.debug("[BILOU] inférence ONNX ms={}", ms(tInfer))

		// 4) Décodage
		val tDec = System.nanoTime()
		val batchSeq = to3D(logits)

		// Vérification forte de la dimension des logits
		validateLabelDimension(batchSeq)

		val result = encodings.mapIndexed { i, enc ->
			val seq = Array(enc.seqLen) { j -> batchSeq[i][j] }

			val rawEntities = decodeBilou(seq, enc.tokenCharSpans, enc.text)

			val cleanedEntities = rawEntities.mapNotNull { sanitizeEntitySpan(it, enc.text) }

			if (log.isDebugEnabled) {
				val before = rawEntities.map { "${it.type}:${it.text}" }
				val after  = cleanedEntities.map { "${it.type}:${it.text}" }
				log.debug("[BILOU] doc={} beforeSanitize={} afterSanitize={}", i, before, after)
			}

			mergeConsecutive(cleanedEntities, enc.text)
		}

		log.debug("[BILOU] décodage       ms={}", ms(tDec))
		log.debug("[BILOU] total          batchSize={}  maxLen={}  ms={}", batchSize, maxLen, ms(t0))

		return result
	}

	/**
	 * Fusionne les entités adjacentes du même type séparées seulement par du blanc.
	 *
	 * IMPORTANT :
	 * - on NE FUSIONNE PAS les PER
	 *   car cela crée des spans ambigus du type :
	 *     "président Emmanuel Macron"
	 *     "députés européens"
	 *     "French president Emmanuel Macron"
	 *
	 * - on garde le merge pour LOC/ORG/TIME/EVENT/OBJECT,
	 *   ce qui reste utile dans certains cas.
	 */
	private fun mergeConsecutive(entities: List<Entity>, fullText: String): List<Entity> {
		if (entities.isEmpty()) return entities

		fun confOf(e: Entity): Float? = (e.metadata["confidence"] as? Number)?.toFloat()
		fun confMinOf(e: Entity): Float? = (e.metadata["confidenceMin"] as? Number)?.toFloat()

		val result = mutableListOf<Entity>()
		var current = entities[0]

		for (next in entities.drop(1)) {
			val curEnd = current.span?.end ?: -1
			val nextStart = next.span?.start ?: -1
			val nextEnd = next.span?.end ?: -1
			val curStart = current.span?.start ?: -1

			val sameType = next.type == current.type
			val gapIsBlank = curEnd >= 0 && nextStart >= curEnd &&
					fullText.substring(curEnd, nextStart).isBlank()

			val mergeAllowedType = current.type != "per"
			val canMerge = sameType && gapIsBlank && mergeAllowedType

			if (canMerge) {
				val mergedText = fullText.substring(curStart, nextEnd)

				val c1 = confOf(current)
				val c2 = confOf(next)
				val mergedConf = listOfNotNull(c1, c2).minOrNull() // conservateur

				val m1 = confMinOf(current)
				val m2 = confMinOf(next)
				val mergedConfMin = listOfNotNull(m1, m2).minOrNull() ?: mergedConf

				val mergedEntity = Entity(
					text = mergedText,
					type = current.type,
					span = Span(curStart, nextEnd, emptyList()),
					metadata = buildMap {
						if (mergedConf != null) put("confidence", mergedConf)
						if (mergedConfMin != null) put("confidenceMin", mergedConfMin)
					}
				)
				current = sanitizeEntitySpan(mergedEntity, fullText) ?: mergedEntity
			} else {
				result += current
				current = next
			}
		}

		result += current
		return result
	}

	/**
	 * Décodage BIO/BILOU coarse.
	 * Produit seulement les familles :
	 *   per / loc / org / time / event / object
	 */
	private fun decodeBilou(
		seq: Array<FloatArray>,
		spans: List<Pair<Int, Int>?>,
		text: String
	): List<Entity> {

		val out = mutableListOf<Entity>()
		var i = 0

		while (i < seq.size) {
			val span = spans[i]
			if (span == null) {
				i++
				continue
			}

			val (labelIdx, pLabel) = argmaxWithProb(seq[i])
			val raw = labelNames.getOrElse(labelIdx) { "O" }

			if (raw == "O") {
				i++
				continue
			}

			val parts = raw.split("-", limit = 2)
			if (parts.size != 2) {
				log.warn("[BILOU] label inattendu='{}' — skip", raw)
				i++
				continue
			}

			val (tag, type) = parts
			val typeLower = type.lowercase()

			when (tag) {
				// Si tu ajoutes un jour U-xxx dans labelNames, ça marchera
				"U" -> {
					val (cs, ce) = span
					if (cs in 0..text.length && ce in 0..text.length && cs < ce) {
						out += Entity(
							text = text.substring(cs, ce),
							type = typeLower,
							span = Span(cs, ce, emptyList()),
							metadata = mapOf(
								"confidence" to pLabel,
								"confidenceMin" to pLabel,
								"label" to raw
							)
						)
					}
					i++
				}

				"B" -> {
					val start = i
					var end = i

					val confs = mutableListOf<Float>()
					confs += pLabel // proba du B sur le premier token

					i++

					while (i < seq.size) {
						val nextSpan = spans[i]
						if (nextSpan == null) {
							i++
							continue
						}

						val (nextIdx, nextProb) = argmaxWithProb(seq[i])
						val nextLab = labelNames.getOrElse(nextIdx) { "O" }

						// BIO : on accepte I-type
						if (nextLab == "I-$type") {
							end = i
							confs += nextProb
							i++
							continue
						}

						// BILOU : si tu ajoutes L-type, ce bloc marchera aussi
						if (nextLab == "L-$type") {
							end = i
							confs += nextProb
							i++
							break
						}

						break
					}

					val s = spans[start]
					val e = spans[end]
					if (s != null && e != null) {
						val cs = s.first
						val ce = e.second
						if (cs in 0..text.length && ce in 0..text.length && cs < ce) {
							val confAvg = (confs.sum() / confs.size)
							val confMin = confs.minOrNull() ?: confAvg

							out += Entity(
								text = text.substring(cs, ce),
								type = typeLower,
								span = Span(cs, ce, emptyList()),
								metadata = mapOf(
									"confidence" to confAvg,
									"confidenceMin" to confMin,
									"labelsUsed" to listOf(raw) // optionnel, debug
								)
							)
						}
					}
				}

				else -> {
					// I / L sans B : on ignore (artefact)
					i++
				}
			}
		}

		return out
	}

	/**
	 * Trim léger des spans coarse :
	 * - espaces
	 * - ponctuation en bord
	 * - déterminants / prépositions parasites au début
	 *
	 * On reste conservateur :
	 * ce n'est PAS le raffinement final, juste un nettoyage avant merge/downstream.
	 */
	private fun sanitizeEntitySpan(entity: Entity, fullText: String): Entity? {
		val span = entity.span ?: return entity
		var start = span.start
		var end = span.end

		if (start !in 0..fullText.length || end !in 0..fullText.length || start >= end) return null

		// 1) trim espaces
		while (start < end && fullText[start].isWhitespace()) start++
		while (end > start && fullText[end - 1].isWhitespace()) end--

		// 2) trim ponctuation à droite
		while (end > start && fullText[end - 1] in setOf('.', ',', ':', ';', '!', '?', ')', ']', '"', '\'')) {
			end--
		}

		// 3) trim ponctuation à gauche
		while (start < end && fullText[start] in setOf('(', '[', '"', '\'')) {
			start++
		}

		if (start >= end) return null

		// 4) trim léger des déterminants / prépositions parasites au début
		val lowered = fullText.substring(start, end).lowercase()

		val prefixes = listOf(
			"le ", "la ", "les ", "l'",
			"un ", "une ", "des ",
			"de ", "du ", "d'",
			"à ", "au ", "aux ",
			"sur ", "dans ", "en "
		)

		for (p in prefixes) {
			if (lowered.startsWith(p)) {
				start += p.length
				break
			}
		}

		// re-trim espaces éventuels après rognage
		while (start < end && fullText[start].isWhitespace()) start++
		while (end > start && fullText[end - 1].isWhitespace()) end--

		if (start >= end) return null

		val cleanedText = fullText.substring(start, end)
		return entity.copy(
			text = cleanedText,
			span = Span(start, end, emptyList())
		)
	}

	/**
	 * Vérifie que la dimension de logits correspond bien à labelNames.size.
	 * Si ce n'est pas le cas, il y a probablement un mismatch id2label / export ONNX.
	 */
	private fun validateLabelDimension(batchSeq: Array<Array<FloatArray>>) {
		if (batchSeq.isEmpty()) return
		if (batchSeq[0].isEmpty()) return
		val dim = batchSeq[0][0].size
		if (dim != labelNames.size) {
			log.error(
				"[BILOU] MISMATCH logitsDim={} vs labelNames.size={} — probable décalage id2label/config",
				dim, labelNames.size
			)
		} else {
			log.debug("[BILOU] logitsDim OK = {}", dim)
		}
	}

	private fun ms(nanoStart: Long) = (System.nanoTime() - nanoStart) / 1_000_000L

	private fun argmax(a: FloatArray): Int {
		var best = 0
		var max = a[0]
		for (i in 1 until a.size) {
			if (a[i] > max) {
				max = a[i]
				best = i
			}
		}
		return best
	}

	private fun to3D(obj: Any?): Array<Array<FloatArray>> =
		when (obj) {
			is Array<*> -> obj.map { lvl1 ->
				when (lvl1) {
					is Array<*> -> lvl1.map { lvl2 ->
						when (lvl2) {
							is FloatArray -> lvl2
							is DoubleArray -> lvl2.map { it.toFloat() }.toFloatArray()
							else -> FloatArray(0)
						}
					}.toTypedArray()

					is FloatArray -> arrayOf(lvl1)
					is DoubleArray -> arrayOf(lvl1.map { it.toFloat() }.toFloatArray())
					else -> arrayOf()
				}
			}.toTypedArray()

			else -> arrayOf()
		}

	override fun close() {
		tokenizer.close()
		session.close()
	}

	private fun logSumExp(logits: FloatArray): Double {
		var max = logits[0].toDouble()
		for (i in 1 until logits.size) {
			val v = logits[i].toDouble()
			if (v > max) max = v
		}
		var sum = 0.0
		for (i in logits.indices) {
			sum += kotlin.math.exp(logits[i].toDouble() - max)
		}
		return max + kotlin.math.ln(sum)
	}

	private fun probOf(logits: FloatArray, idx: Int): Float {
		val lse = logSumExp(logits)
		return kotlin.math.exp(logits[idx].toDouble() - lse).toFloat()
	}

	private fun argmaxWithProb(logits: FloatArray): Pair<Int, Float> {
		var best = 0
		var max = logits[0]
		for (i in 1 until logits.size) {
			if (logits[i] > max) {
				max = logits[i]
				best = i
			}
		}
		return best to probOf(logits, best)
	}
}
