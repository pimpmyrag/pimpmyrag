
package com.acme.infinity.embed

data class IndexedText(val index: Int, val text: String)

fun packBatches(items: List<IndexedText>, batchSize: Int, maxBatchBytes: Int): List<List<IndexedText>> {
    val batches = mutableListOf<MutableList<IndexedText>>()
    var current = mutableListOf<IndexedText>()
    var currentBytes = 0

    fun flush() { if (current.isNotEmpty()) { batches += current; current = mutableListOf(); currentBytes = 0 } }

    for (it in items) {
        val estimated = it.text.toByteArray(Charsets.UTF_8).size + 50
        if (current.size >= batchSize || (currentBytes + estimated) > maxBatchBytes) flush()
        current += it
        currentBytes += estimated
    }
    flush()
    return batches
}
