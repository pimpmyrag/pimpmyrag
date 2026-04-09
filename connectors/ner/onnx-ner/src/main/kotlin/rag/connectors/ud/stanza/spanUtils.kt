package rag.connectors.ud.stanza

import rag.model.Span
import rag.model.UDToken
import rag.model.UPOS

fun reconstructSpan(tokens: List<UDToken>, headId: Int): Span {
    val head = tokens.firstOrNull { it.id == headId }
        ?: return Span(0, 0, emptyList())

    if (head.upos !in setOf(UPOS.NOUN, UPOS.PROPN, UPOS.PRON, UPOS.NUM)) {
        return Span(head.start, head.end, listOf(head))
    }

    val keep = linkedMapOf<Int, UDToken>()
    keep[head.id] = head

    fun base(d: String): String = d.lowercase().substringBefore(":")

    fun childrenOf(id: Int) = tokens.filter { it.head == id }

    fun shouldInclude(child: UDToken): Boolean {
        val rel = base(child.deprel)

        // exclusions dures
        if (rel in setOf("obl", "advmod", "advcl", "acl", "ccomp", "xcomp",
                "parataxis", "list", "conj", "punct")) {
            return false
        }

        // cas nominaux simples
        if (rel in setOf("amod", "compound", "flat", "name", "nummod")) {
            return true
        }

        // apposition nominale utile : NOUN + PROPN
        if (rel == "appos" && child.upos == UPOS.PROPN) {
            return true
        }

        // chaînes nominales type "rue de la Charité", "accord de Paris", "3 mars 2026"
        if (rel == "nmod" && child.upos in setOf(UPOS.NOUN, UPOS.PROPN, UPOS.NUM, UPOS.ADJ)) {
            return true
        }

        return false
    }

    fun visit(node: UDToken) {
        for (child in childrenOf(node.id)) {
            val rel = base(child.deprel)

            if (!shouldInclude(child)) continue

            if (child.id !in keep) {
                keep[child.id] = child
                visit(child)
            }

            // inclure aussi les "case" et éventuellement "det" attachés au child inclus
            if (rel == "nmod" || rel == "appos") {
                for (g in childrenOf(child.id)) {
                    val grel = base(g.deprel)
                    if (grel == "case" || grel == "det") {
                        keep[g.id] = g
                    }
                }
            }
        }
    }

    visit(head)

    val sorted = keep.values.sortedBy { it.start }
    return Span(sorted.first().start, sorted.last().end, sorted)
}
