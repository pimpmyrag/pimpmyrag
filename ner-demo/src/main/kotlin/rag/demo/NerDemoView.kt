package rag.demo

import com.vaadin.flow.component.ClickEvent
import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.button.ButtonVariant
import com.vaadin.flow.component.html.*
import com.vaadin.flow.component.notification.Notification
import com.vaadin.flow.component.orderedlayout.FlexLayout
import com.vaadin.flow.component.orderedlayout.HorizontalLayout
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.component.textfield.TextArea
import com.vaadin.flow.router.PageTitle
import com.vaadin.flow.router.Route
import rag.connectors.ner.onnx.ExtractionResult
import rag.connectors.ner.onnx.SvoSpan
import rag.model.Entity

@Route("")
@PageTitle("NER + SVO — DeBERTa")
class NerDemoView(private val nerService: NerService) : VerticalLayout() {

    // ── Palette couleurs (miroir Python) ──────────────────────────────────────

    companion object {
        val COARSE_COLORS = mapOf(
            "PER"      to ("#dbeafe" to "#1d4ed8"),
            "LOC"      to ("#d1fae5" to "#065f46"),
            "ORG"      to ("#ede9fe" to "#5b21b6"),
            "TIME"     to ("#ffedd5" to "#9a3412"),
            "EVENT"    to ("#fee2e2" to "#991b1b"),
            "VALUE"    to ("#ccfbf1" to "#0f766e"),
            "OBJECT"   to ("#fef3c7" to "#92400e"),
            "ABSTRACT" to ("#f1f5f9" to "#334155"),
            "NONE"     to ("#f3f4f6" to "#6b7280"),
        )
        val SVO_COLORS = mapOf(
            "svo_verb"    to ("#e0f2fe" to "#0369a1"),
            "svo_subject" to ("#dcfce7" to "#15803d"),
            "svo_object"  to ("#fce7f3" to "#9d174d"),
            "svo_iobj"    to ("#fff7ed" to "#c2410c"),
            "pron_subj"   to ("#f0fdf4" to "#166534"),
            "pron_obj"    to ("#fdf2f8" to "#7e22ce"),
        )
        val COMPACT_LABEL = mapOf(
            "hint_person_name"    to "pers.name",
            "hint_person_role"    to "pers.role",
            "hint_norp"           to "norp",
            "hint_group_role"     to "group.role",
            "hint_org_name"       to "org",
            "hint_gpe"            to "gpe",
            "hint_fac_name"       to "facility",
            "hint_loc_generic"    to "loc",
            "hint_infra"          to "infra",
            "hint_weapon"         to "weapon",
            "hint_vehicle"        to "vehicle",
            "hint_substance"      to "substance",
            "hint_food"           to "food",
            "hint_tool"           to "tool",
            "hint_object_generic" to "object",
            "hint_object_name"    to "product",
            "hint_event_nominal"  to "evt.nominal",
            "hint_event_named"    to "evt.named",
            "hint_time_date"      to "date",
            "hint_time_clock"     to "clock",
            "hint_time_duration"  to "duration",
            "hint_quantity"       to "qty",
            "hint_measure"        to "measure",
            "hint_percentage"     to "pct",
            "hint_count"          to "count",
            "hint_money"          to "money",
            "hint_rate"           to "rate",
            "hint_law"            to "law",
            "hint_work_of_art"    to "art",
            "hint_concept"        to "concept",
            "hint_disease"        to "disease",
            "hint_language"       to "lang",
        )
    }

    // ── Composants persistants ────────────────────────────────────────────────

    private val resultArea  = Div()
    private val detailPanel = buildDetailPanel()

    // ── Construction de l'UI ──────────────────────────────────────────────────

    init {
        setSizeFull()
        isPadding = false
        style["background"] = "#f0f2f5"
        style["padding"]    = "24px"
        style["box-sizing"] = "border-box"

        add(buildHeader())
        add(buildInputSection())

        val main = HorizontalLayout(resultArea, detailPanel)
        main.setWidthFull()
        main.setFlexGrow(1.0, resultArea)
        main.alignItems = FlexLayout.Alignment.START
        main.style["gap"] = "16px"
        add(main)
        setFlexGrow(1.0, main)
    }

    private fun buildHeader(): Div {
        val h = H1("🔍 NER + SVO")
        h.style["font-family"] = "Inter, system-ui, sans-serif"
        h.style["margin"]      = "0 0 16px"
        h.style["font-size"]   = "1.8em"
        val sub = Span("DeBERTa multitête — annotation NER + arcs SVO")
        sub.style["color"]     = "#64748b"
        sub.style["font-size"] = "0.9em"
        return card(h, sub)
    }

    private fun buildInputSection(): Div {
        val textArea = TextArea()
        textArea.placeholder = "Collez votre texte ici… (dépêche, article, phrase)"
        textArea.setWidthFull()
        textArea.minRows = 3

        val btn = Button("Analyser") {
            val text = textArea.value.trim()
            if (text.isBlank()) return@Button
            try {
                val result = nerService.analyse(text)
                renderResult(text, result)
            } catch (e: Exception) {
                Notification.show("Erreur : ${e.message}", 5000, Notification.Position.MIDDLE)
            }
        }
        btn.addThemeVariants(ButtonVariant.LUMO_PRIMARY)

        val row = HorizontalLayout(btn)
        row.style["margin-top"] = "8px"

        return card(textArea, row)
    }

    private fun buildDetailPanel(): Div {
        val div = Div()
        div.style["width"]      = "310px"
        div.style["min-width"]  = "310px"
        div.style["min-height"] = "160px"
        div.style["background"] = "#fff"
        div.style["border-radius"] = "12px"
        div.style["padding"]    = "16px"
        div.style["box-shadow"] = "0 2px 12px rgba(0,0,0,.07)"
        div.style["font-family"] = "Inter, system-ui, sans-serif"
        div.style["font-size"]  = "0.84em"
        div.add(placeholder("Cliquez sur un span pour ses détails."))
        return div
    }

    // ── Rendu du résultat ─────────────────────────────────────────────────────

    private data class SpanInfo(
        val charStart: Int, val charEnd: Int,
        val displayText: String,
        val bg: String, val fg: String,
        val label: String,
        val isNer: Boolean,
        val entity: Entity?   = null,
        val svo: SvoSpan?     = null,
    )

    private fun renderResult(text: String, result: ExtractionResult) {
        resultArea.removeAll()
        detailPanel.removeAll()
        detailPanel.add(placeholder("Cliquez sur un span pour ses détails."))

        val spans = mutableListOf<SpanInfo>()

        // NER
        for (ent in result.entities) {
            val cs = ent.span?.start ?: continue
            val ce = ent.span.end
            if (cs < 0 || ce <= cs || ce > text.length) continue
            val coarse = ent.metadata["coarse"] as? String ?: "NONE"
            val (bg, fg) = COARSE_COLORS[coarse] ?: ("#f3f4f6" to "#6b7280")
            val label = COMPACT_LABEL[ent.type] ?: ent.type.removePrefix("hint_")
            spans += SpanInfo(cs, ce, ent.text, bg, fg, label, isNer = true, entity = ent)
        }

        // SVO
        for (svo in result.svoSpans) {
            val (bg, fg) = SVO_COLORS[svo.role] ?: ("#e5e7eb" to "#374151")
            val label = svo.role.replace("svo_", "").replace("pron_", "pron:")
            spans += SpanInfo(svo.charStart, svo.charEnd, svo.text, bg, fg, label, isNer = false, svo = svo)
        }

        // Tri + dé-overlap (NER prioritaire)
        val sorted  = spans.sortedWith(compareBy({ it.charStart }, { if (it.isNer) 0 else 1 }))
        val kept    = mutableListOf<SpanInfo>()
        var lastEnd = 0
        for (s in sorted) {
            if (s.charStart >= lastEnd) { kept += s; lastEnd = s.charEnd }
        }

        // Construction des composants inline
        val container = Div()
        container.style["font-family"]  = "Inter, system-ui, sans-serif"
        container.style["font-size"]    = "1.02em"
        container.style["line-height"]  = "2.2"

        var cursor = 0
        for (s in kept) {
            if (s.charStart > cursor) container.add(Span(text.substring(cursor, s.charStart)))
            container.add(buildPill(s) { showDetail(s) })
            cursor = s.charEnd
        }
        if (cursor < text.length) container.add(Span(text.substring(cursor)))

        // Légende
        val legend = buildLegend()

        // Stats
        val stats = Span("${result.entities.size} entité(s) · ${result.svoSpans.size} span(s) SVO")
        stats.style["font-size"] = "0.8em"
        stats.style["color"]     = "#94a3b8"
        stats.style["display"]   = "block"
        stats.style["margin-top"] = "8px"

        resultArea.add(legend, container, stats)
    }

    private fun buildPill(info: SpanInfo, onClick: () -> Unit): Span {
        val pill = Span()
        pill.style["background"]     = info.bg
        pill.style["color"]          = info.fg
        pill.style["border-radius"]  = "4px"
        pill.style["padding"]        = "2px 6px 2px 5px"
        pill.style["margin"]         = "0 2px"
        pill.style["cursor"]         = "pointer"
        pill.style["font-weight"]    = "500"
        pill.style["display"]        = "inline"
        pill.style["border"]         = "1.5px solid ${info.fg}28"

        val textNode  = Span(info.displayText)
        val labelNode = Span(" ${info.label}")
        labelNode.style["font-size"]   = "0.72em"
        labelNode.style["font-weight"] = "700"
        labelNode.style["margin-left"] = "3px"
        labelNode.style["opacity"]     = "0.85"

        pill.add(textNode, labelNode)
        pill.addClickListener { onClick() }
        return pill
    }

    // ── Panneau de détail ─────────────────────────────────────────────────────

    private fun showDetail(info: SpanInfo) {
        detailPanel.removeAll()

        if (info.isNer && info.entity != null) {
            val ent    = info.entity
            val coarse = ent.metadata["coarse"] as? String ?: "?"
            val label  = COMPACT_LABEL[ent.type] ?: ent.type.removePrefix("hint_")
            detailPanel.add(H4("🏷 $label"))
            detailPanel.add(detailDivider())
            addRow(detailPanel, "Texte",   ent.text)
            addRow(detailPanel, "Coarse",  coarse)
            addRow(detailPanel, "Fine",    ent.type)
            detailPanel.add(sectionHeader("SCORES"))
            addRow(detailPanel, "p_entity", fmt(ent.metadata["pBoundary"]))
            addRow(detailPanel, "p_coarse", fmt(ent.metadata["pCoarse"]))
            addRow(detailPanel, "p_fine",   fmt(ent.metadata["pFine"]))
            addRow(detailPanel, "score",    fmt(ent.metadata["score"]))

        } else if (!info.isNer && info.svo != null) {
            val svo   = info.svo
            val emoji = mapOf("svo_verb" to "🔵", "svo_subject" to "🟢",
                              "svo_object" to "🔴", "svo_iobj" to "🟠",
                              "pron_subj" to "🟢", "pron_obj" to "🔴")
            detailPanel.add(H4("${emoji[svo.role] ?: "⚪"} ${svo.role}"))
            detailPanel.add(detailDivider())
            addRow(detailPanel, "Texte",       svo.text)
            addRow(detailPanel, "Rôle",        svo.role)
            addRow(detailPanel, "Voice",       svo.voice)
            detailPanel.add(sectionHeader("SCORES"))
            addRow(detailPanel, "p_boundary",  "%.3f".format(svo.svoBoundaryProb))
            addRow(detailPanel, "p_role",      "%.3f".format(svo.roleProb))
            addRow(detailPanel, "voice conf",  "%.3f".format(svo.voiceProb))
            svo.gender?.let { addRow(detailPanel, "Genre",  it) }
            svo.number?.let { addRow(detailPanel, "Nombre", it) }
        }
    }

    // ── Helpers UI ────────────────────────────────────────────────────────────

    private fun card(vararg children: com.vaadin.flow.component.Component): Div {
        val div = Div(*children)
        div.setWidthFull()
        div.style["background"]    = "#fff"
        div.style["border-radius"] = "12px"
        div.style["padding"]       = "20px"
        div.style["margin-bottom"] = "16px"
        div.style["box-shadow"]    = "0 2px 12px rgba(0,0,0,.07)"
        return div
    }

    private fun buildLegend(): Div {
        val div = Div()
        div.style["display"]     = "flex"
        div.style["flex-wrap"]   = "wrap"
        div.style["gap"]         = "6px"
        div.style["margin-bottom"] = "12px"

        val nerEntries = COARSE_COLORS.entries.filter { it.key != "NONE" }
        for ((coarse, colors) in nerEntries) {
            val chip = Span(coarse)
            chip.style["background"]    = colors.first
            chip.style["color"]         = colors.second
            chip.style["border-radius"] = "6px"
            chip.style["padding"]       = "2px 8px"
            chip.style["font-size"]     = "0.75em"
            chip.style["font-weight"]   = "600"
            div.add(chip)
        }
        // SVO
        for ((role, colors) in SVO_COLORS) {
            val short = role.replace("svo_", "").replace("pron_", "pron:")
            val chip = Span(short)
            chip.style["background"]    = colors.first
            chip.style["color"]         = colors.second
            chip.style["border-radius"] = "6px"
            chip.style["padding"]       = "2px 8px"
            chip.style["font-size"]     = "0.75em"
            chip.style["font-weight"]   = "600"
            chip.style["border"]        = "1px dashed ${colors.second}60"
            div.add(chip)
        }
        return div
    }

    private fun addRow(container: Div, key: String, value: String) {
        val row = HorizontalLayout()
        row.setWidthFull()
        row.isPadding  = false
        row.isSpacing  = false
        row.style["padding"]       = "4px 0"
        row.style["border-bottom"] = "1px solid #f1f5f9"
        row.style["align-items"]   = "baseline"

        val k = Span(key)
        k.style["color"]       = "#64748b"
        k.style["font-weight"] = "600"
        k.style["font-size"]   = "0.85em"
        k.style["min-width"]   = "80px"
        k.style["width"]       = "80px"

        val v = Span(value)
        v.style["font-size"]    = "0.88em"
        v.style["font-family"]  = "monospace"
        v.style["word-break"]   = "break-all"
        v.style["flex"]         = "1"

        row.add(k, v)
        container.add(row)
    }

    private fun sectionHeader(text: String): Div {
        val div = Div()
        div.add(Span(text))
        div.style["font-weight"]   = "700"
        div.style["font-size"]     = "0.78em"
        div.style["color"]         = "#64748b"
        div.style["letter-spacing"] = "0.06em"
        div.style["padding-top"]   = "8px"
        return div
    }

    private fun detailDivider(): Hr {
        val hr = Hr()
        hr.style["border"]     = "none"
        hr.style["border-top"] = "1px solid #f1f5f9"
        hr.style["margin"]     = "6px 0"
        return hr
    }

    private fun placeholder(text: String): Span {
        val s = Span(text)
        s.style["color"]     = "#94a3b8"
        s.style["font-size"] = "0.88em"
        return s
    }

    private fun fmt(v: Any?): String = when (v) {
        is Float  -> "%.3f".format(v)
        is Double -> "%.3f".format(v)
        null      -> "—"
        else      -> v.toString()
    }
}

