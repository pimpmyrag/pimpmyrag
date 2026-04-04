package rag.connectors.ud.stanza

import rag.model.Span
import rag.model.UDToken
import rag.model.UPOS

fun reconstructSpan(tokens: List<UDToken>, headId: Int): Span {

    val head = tokens.firstOrNull { it.id == headId }
        ?: return Span(0, 0, emptyList())

    // On reconstruit seulement les vrais NPs
    if (head.upos !in setOf(UPOS.NOUN, UPOS.PROPN, UPOS.PRON, UPOS.NUM))
        return Span(head.start, head.end, listOf(head))

    val keep = linkedMapOf<Int, UDToken>()
    keep[head.id] = head

    // ✅ relations internes d’un syntagme nominal
    // 'det' intentionnellement exclu : les déterminants (la, le, l', les)
    // ne font pas partie du span NER. Pipeline A gère déjà ce cas via
    // le trim DET/ADP dans enrichOne.
    val include = setOf(
        "amod",
        "compound",
        "flat", "flat:name", "name",
        "nummod"
    )

    // ✅ relations à EXCLURE absolument
    val exclude = setOf(
        "obl",        // <- couvrira obl:mod lorsqu’on prend base()
        "advmod",
        "advcl",
        "acl",
        "ccomp",
        "xcomp",
        "parataxis",
        "list",
        "conj",
        "punct",
        "appos"
    )

    fun base(d: String): String = d.lowercase().substringBefore(":")

    fun visit(node: UDToken) {
        for (child in tokens.filter { it.head == node.id }) {
            val rel = base(child.deprel)

            // ❌ relations qui cassent le NP
            if (rel in exclude) continue

            // Les prépositions/articles contractés (case) ne font jamais partie du span NER.
            // On les exclut systématiquement (sinon "sur la Seine" → "sur" inclus, etc.)
            if (rel == "case") continue

            if (rel in include) {
                if (child.id !in keep) {   // garde anti-cycle : n'explorer qu'une fois
                    keep[child.id] = child
                    visit(child)
                }
            }
        }
    }

    visit(head)

    val sorted = keep.values.sortedBy { it.start }
    return Span(sorted.first().start, sorted.last().end, sorted)
}
