package rag.demo

import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.button.ButtonVariant
import com.vaadin.flow.component.dialog.Dialog
import com.vaadin.flow.component.html.Div
import com.vaadin.flow.component.html.Paragraph
import com.vaadin.flow.component.html.Pre
import com.vaadin.flow.component.html.Span
import com.vaadin.flow.component.notification.Notification
import com.vaadin.flow.component.orderedlayout.FlexComponent
import com.vaadin.flow.component.orderedlayout.HorizontalLayout
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.component.progressbar.ProgressBar
import com.vaadin.flow.component.textfield.PasswordField
import com.vaadin.flow.component.textfield.TextField
import com.vaadin.flow.component.UI
import java.util.concurrent.Executors

/**
 * Dialog "LLM Judge" :
 *  - champs endpoint / API key / modèle (mémorisés en session)
 *  - bouton Envoyer → appel async à [LlmJudgeService]
 *  - affichage du verdict dans une zone scrollable
 */
class LlmJudgeDialog(
    private val judgeService: LlmJudgeService,
    private val i18n: I18n,
    private val getResults: () -> List<AnnotatedSentence>,
) : Dialog() {

    companion object {
        // Mémorise la config pour la session (champs pré-remplis)
        private var savedConfig = LlmJudgeConfig()
        private val bgExec = Executors.newCachedThreadPool { r ->
            Thread(r, "llm-judge").also { it.isDaemon = true }
        }
    }

    // ── Fields ────────────────────────────────────────────────────────────────
    private val tfUrl   = TextField(i18n.judgeUrlLabel, savedConfig.baseUrl, "")
        .also { it.setWidthFull() }
    private val pfKey   = PasswordField(i18n.judgeKeyLabel)
        .also { it.value = savedConfig.apiKey; it.setWidthFull() }
    private val tfModel = TextField(i18n.judgeModelLabel, savedConfig.model, "")
        .also { it.width = "180px" }

    // ── Result area ───────────────────────────────────────────────────────────
    private val resultArea = Div().also { d ->
        d.style["white-space"]  = "pre-wrap"
        d.style["font-family"]  = "Inter, system-ui, sans-serif"
        d.style["font-size"]    = "0.85em"
        d.style["line-height"]  = "1.65"
        d.style["color"]        = "#0f172a"
        d.style["min-height"]   = "80px"
        d.style["padding"]      = "12px 16px"
        d.style["background"]   = "#f8fafc"
        d.style["border"]       = "1px solid #e2e8f0"
        d.style["border-radius"]= "6px"
        d.style["overflow-y"]   = "auto"
        d.style["max-height"]   = "52vh"
    }

    private val spinner = ProgressBar().also {
        it.isIndeterminate = true; it.isVisible = false; it.setWidthFull()
    }

    private val btnSend = Button(i18n.judgeSendBtn) { doJudge() }.also {
        it.addThemeVariants(ButtonVariant.LUMO_PRIMARY)
    }

    // ── Layout ────────────────────────────────────────────────────────────────
    init {
        headerTitle = i18n.judgeDialogTitle
        width  = "780px"
        isResizable = true

        val configRow = HorizontalLayout(tfModel).also {
            it.setWidthFull(); it.isPadding = false; it.isSpacing = true
            it.alignItems = FlexComponent.Alignment.BASELINE
        }
        val closeBtn = Button(i18n.judgeCloseBtn) { close() }

        val content = VerticalLayout(
            note(i18n.judgeNote),
            tfUrl, pfKey, configRow,
            spinner,
            HorizontalLayout(btnSend, closeBtn).also {
                it.isPadding = false; it.isSpacing = true
            },
            resultArea,
        ).also { it.isPadding = false; it.isSpacing = true; it.setWidthFull() }

        add(content)
    }

    // ── Action ────────────────────────────────────────────────────────────────
    private fun doJudge() {
        val results = getResults()
        if (results.isEmpty()) {
            Notification.show(i18n.noResults, 2500, Notification.Position.BOTTOM_START)
            return
        }
        // Save config
        savedConfig = LlmJudgeConfig(
            baseUrl = tfUrl.value.trim().ifBlank { "https://api.openai.com/v1" },
            apiKey  = pfKey.value.trim(),
            model   = tfModel.value.trim().ifBlank { "gpt-4o-mini" },
        )
        if (savedConfig.apiKey.isBlank()) {
            Notification.show(i18n.judgeKeyRequired, 3000, Notification.Position.MIDDLE)
            return
        }

        resultArea.removeAll()
        resultArea.add(Span(i18n.judgeWaiting))
        spinner.isVisible = true
        btnSend.isEnabled = false

        val ui = UI.getCurrent()
        val cfg = savedConfig
        bgExec.submit {
            val verdict = try {
                judgeService.judge(cfg, results)
            } catch (e: Exception) {
                "❌ ${e.message}"
            }
            ui.access {
                spinner.isVisible = false
                btnSend.isEnabled = true
                resultArea.removeAll()
                renderMarkdown(verdict)
            }
        }
    }

    /**
     * Rendu basique du markdown LLM :
     *  - ### → titre h3
     *  - **gras** → <strong>
     *  - lignes normales → paragraphe
     */
    private fun renderMarkdown(text: String) {
        val lines = text.lines()
        lines.forEach { raw ->
            val line = raw.trimEnd()
            when {
                line.startsWith("### ") -> {
                    val h = Span(line.removePrefix("### "))
                    h.style["font-weight"] = "700"
                    h.style["font-size"]   = "0.92em"
                    h.style["color"]       = "#1e3a5f"
                    h.style["display"]     = "block"
                    h.style["margin-top"]  = "10px"
                    resultArea.add(h)
                }
                line.startsWith("## ") -> {
                    val h = Span(line.removePrefix("## "))
                    h.style["font-weight"] = "700"
                    h.style["font-size"]   = "1.0em"
                    h.style["color"]       = "#1e3a5f"
                    h.style["display"]     = "block"
                    h.style["margin-top"]  = "12px"
                    resultArea.add(h)
                }
                line.startsWith("- ") || line.startsWith("* ") -> {
                    val p = Div()
                    p.style["padding-left"] = "14px"
                    p.style["display"]      = "block"
                    p.add(Span("• " + parseBold(line.drop(2))))
                    resultArea.add(p)
                }
                line.isBlank() -> {
                    val sep = Div(); sep.style["height"] = "4px"; resultArea.add(sep)
                }
                else -> {
                    val p = Div()
                    p.style["display"] = "block"
                    p.add(Span(parseBold(line)))
                    resultArea.add(p)
                }
            }
        }
    }

    /** Transforme **gras** en texte simple (Vaadin Span ne supporte pas HTML interne facilement). */
    private fun parseBold(line: String): String =
        line.replace(Regex("""\*\*(.+?)\*\*"""), "$1")
            .replace(Regex("""`(.+?)`"""), "«$1»")

    // ── Helpers ───────────────────────────────────────────────────────────────
    private fun note(text: String) = Paragraph(text).also { p ->
        p.style["font-size"] = "0.78em"
        p.style["color"]     = "#64748b"
        p.style["margin"]    = "0"
    }
}

