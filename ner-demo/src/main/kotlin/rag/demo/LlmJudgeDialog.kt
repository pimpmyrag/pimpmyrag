package rag.demo

import com.vaadin.flow.component.UI
import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.button.ButtonVariant
import com.vaadin.flow.component.combobox.ComboBox
import com.vaadin.flow.component.Html
import com.vaadin.flow.component.html.Div
import com.vaadin.flow.component.html.Span
import com.vaadin.flow.component.notification.Notification
import com.vaadin.flow.component.orderedlayout.FlexComponent
import com.vaadin.flow.component.orderedlayout.HorizontalLayout
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.component.progressbar.ProgressBar
import com.vaadin.flow.component.textfield.PasswordField
import com.vaadin.flow.component.textfield.TextField
import org.commonmark.ext.gfm.strikethrough.StrikethroughExtension
import org.commonmark.ext.gfm.tables.TablesExtension
import org.commonmark.parser.Parser
import org.commonmark.renderer.html.HtmlRenderer
import java.util.concurrent.Executors

/**
 * Panneau LLM Judge — embarqué directement dans la vue (plus de Dialog popup).
 * Toggle via [NerDemoView.toggleJudge] ou raccourci Alt+J.
 *
 *  ┌─────────────────────────────────────────────────────┐
 *  │  🤖 LLM Judge                               [✕]    │  ← panelHeader
 *  ├──────────────┬──────────────────────────────────────┤
 *  │  Config       │  Résultats                          │
 *  │  • URL/key    │  • trace outils (agent)             │
 *  │  • modèle     │  • verdict structuré (markdown)     │
 *  │  • mode       │                                     │
 *  └──────────────┴──────────────────────────────────────┘
 */
class LlmJudgePanel(
    private val judgeService: LlmJudgeService,
    private val i18n: I18n,
    private val getResults: () -> List<AnnotatedSentence>,
    private val onClose: () -> Unit = {},
) : Div() {

    companion object {
        private var savedConfig = LlmJudgeConfig()
        private val bgExec = Executors.newCachedThreadPool { r ->
            Thread(r, "llm-judge").also { it.isDaemon = true }
        }

        // ── Catalogues modèles par provider ───────────────────────────────────
        private val MODELS_OPENAI = listOf(
            "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            "o4-mini", "o3", "o3-mini",
            "gpt-4o", "gpt-4o-mini",
            "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
        )
        private val MODELS_MISTRAL = listOf(
            "mistral-large-latest", "mistral-medium-3",
            "mistral-small-latest", "mistral-saba-latest",
            "open-mistral-nemo", "open-mistral-7b",
            "codestral-latest", "pixtral-large-latest",
            "magistral-medium-2506", "magistral-small-2506",
        )
        private val MODELS_ANTHROPIC = listOf(
            "claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-3-5",
            "claude-3-7-sonnet-latest", "claude-3-5-haiku-latest",
            "claude-3-opus-latest", "claude-3-sonnet-20240229",
        )
        private val MODELS_GOOGLE = listOf(
            "gemini-2.5-pro", "gemini-2.5-flash",
            "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash",
        )
        private val MODELS_OLLAMA = listOf(
            "llama3.3", "llama3.1", "mistral", "mixtral",
            "phi4", "qwen2.5", "deepseek-r1", "gemma3",
        )

        /** Détecte le provider depuis l'URL et retourne la liste de modèles + le modèle par défaut. */
        fun providerModels(url: String): Pair<List<String>, String> = when {
            url.contains("openai.com",    ignoreCase = true) -> MODELS_OPENAI    to "gpt-4o-mini"
            url.contains("mistral.ai",    ignoreCase = true) -> MODELS_MISTRAL   to "mistral-small-latest"
            url.contains("anthropic.com", ignoreCase = true) -> MODELS_ANTHROPIC to "claude-haiku-3-5"
            url.contains("googleapis.com",ignoreCase = true) ||
            url.contains("generativelanguage", ignoreCase = true) -> MODELS_GOOGLE to "gemini-2.5-flash"
            url.contains("ollama",        ignoreCase = true) ||
            url.contains("localhost",     ignoreCase = true) ||
            url.contains("127.0.0.1",     ignoreCase = true) -> MODELS_OLLAMA   to "llama3.3"
            else -> (MODELS_OPENAI + MODELS_MISTRAL + MODELS_ANTHROPIC + MODELS_GOOGLE + MODELS_OLLAMA) to "gpt-4o-mini"
        }

        // ── Markdown (commonmark) ─────────────────────────────────────────────
        private val mdExtensions = listOf(TablesExtension.create(), StrikethroughExtension.create())
        private val mdParser   = Parser.builder().extensions(mdExtensions).build()
        private val mdRenderer = HtmlRenderer.builder().extensions(mdExtensions).sanitizeUrls(true).build()

        private val MD_CSS = """
            <style>
            .llm-md{font-family:Inter,system-ui,sans-serif;font-size:0.85em;line-height:1.7;color:#0f172a}
            .llm-md h1,.llm-md h2,.llm-md h3{font-weight:700;color:#1e3a5f;margin:14px 0 4px}
            .llm-md h1{font-size:1.1em}.llm-md h2{font-size:1.0em}.llm-md h3{font-size:0.90em}
            .llm-md p{margin:4px 0}
            .llm-md ul,.llm-md ol{padding-left:18px;margin:4px 0}
            .llm-md li{margin:2px 0}
            .llm-md strong{font-weight:700}
            .llm-md em{font-style:italic}
            .llm-md del{text-decoration:line-through;color:#94a3b8}
            .llm-md code{background:#f1f5f9;border-radius:3px;padding:1px 5px;font-family:monospace,monospace;font-size:0.88em;color:#0f172a}
            .llm-md pre{background:#f1f5f9;border-radius:6px;padding:10px 14px;overflow-x:auto;margin:6px 0}
            .llm-md pre code{background:none;padding:0}
            .llm-md hr{border:none;border-top:1px solid #e2e8f0;margin:10px 0}
            .llm-md blockquote{border-left:3px solid #bae6fd;margin:4px 0;padding:2px 12px;color:#475569;background:#f0f9ff;border-radius:0 4px 4px 0}
            .llm-md table{border-collapse:collapse;width:100%;margin:8px 0;font-size:0.95em}
            .llm-md th,.llm-md td{border:1px solid #e2e8f0;padding:5px 10px;text-align:left}
            .llm-md th{background:#f8fafc;font-weight:700}
            .llm-md tr:nth-child(even){background:#f8fafc}
            .llm-md a{color:#1d4ed8;text-decoration:underline}
            </style>
        """.trimIndent()
    }

    // ── Config fields ─────────────────────────────────────────────────────────
    // Note: captured here to avoid shadowing by TextField.i18n inside apply{}
    private val ui18n = i18n
    private val tfUrl = TextField().apply {
        label = ui18n.judgeUrlLabel; value = savedConfig.baseUrl
        setWidthFull(); placeholder = "https://api.openai.com/v1"
    }
    private val pfKey = PasswordField().apply {
        label = ui18n.judgeKeyLabel; value = savedConfig.apiKey; setWidthFull()
    }
    private val cbModel = ComboBox<String>().apply {
        label = ui18n.judgeModelLabel; setWidthFull()
        isAllowCustomValue = true
        // Initialise avec le provider détecté depuis l'URL sauvegardée
        val (models, default) = providerModels(savedConfig.baseUrl)
        setItems(models)
        value = savedConfig.model.ifBlank { default }
        // Quand l'utilisateur saisit une valeur libre
        addCustomValueSetListener { e -> value = e.detail }
    }

    // ── Mode toggle ───────────────────────────────────────────────────────────
    private var agentMode = savedConfig.agentMode
    private val cardStatic = modeCard("📋", i18n.judgeModeStaticTitle, i18n.judgeModeStaticDesc, !agentMode)
    private val cardAgent  = modeCard("🤖", i18n.judgeModeAgentTitle,  i18n.judgeModeAgentDesc,  agentMode)

    // ── Right panel: trace + verdict ──────────────────────────────────────────
    private val traceArea = Div().apply {
        style["font-family"]   = "monospace, monospace"
        style["font-size"]     = "0.76em"
        style["color"]         = "#0369a1"
        style["background"]    = "#f0f9ff"
        style["border"]        = "1px solid #bae6fd"
        style["border-radius"] = "6px"
        style["padding"]       = "8px 12px"
        style["overflow-y"]    = "auto"
        style["max-height"]    = "80px"
        style["flex-shrink"]   = "0"
        isVisible = false
    }
    private val verdictArea = Div().apply {
        style["font-family"] = "Inter, system-ui, sans-serif"
        style["font-size"]   = "0.85em"
        style["line-height"] = "1.7"
        style["color"]       = "#0f172a"
        style["overflow-y"]  = "auto"
        style["flex"]        = "1"
        style["min-height"]  = "0"
        add(emptyState())
    }
    private val spinner = ProgressBar().apply { isIndeterminate = true; isVisible = false; setWidthFull() }

    private val btnSend = Button(i18n.judgeSendBtn) { doJudge() }.apply {
        addThemeVariants(ButtonVariant.LUMO_PRIMARY); setWidthFull()
    }

    private var traceLabel: Div = Div()

    // ── Layout ────────────────────────────────────────────────────────────────
    init {
        // Bande du bas, pleine largeur, hauteur fixe
        setWidthFull()
        style["height"]         = "340px"
        style["min-height"]     = "340px"
        style["max-height"]     = "340px"
        style["display"]        = "flex"
        style["flex-direction"] = "row"
        style["border-top"]     = "2px solid #e2e8f0"
        style["background"]     = "#ffffff"
        style["overflow"]       = "hidden"
        style["flex-shrink"]    = "0"

        element.appendChild(Html(MD_CSS).element)

        // ── Drag-to-resize handle (JS injecté à l'attach) ────────────────────
        addAttachListener {
            element.executeJs("""
                (function() {
                    const panel = this;
                    if (panel.querySelector('#jdh')) return;
                    const h = document.createElement('div');
                    h.id = 'jdh';
                    h.title = 'Drag to resize';
                    h.style.cssText = 'height:6px;background:linear-gradient(transparent 30%,#cbd5e1 50%,transparent 70%);cursor:ns-resize;user-select:none;flex-shrink:0;transition:background .15s;position:relative;z-index:1;';
                    h.addEventListener('mouseenter', () => h.style.background = 'linear-gradient(transparent 30%,#6366f1 50%,transparent 70%)');
                    h.addEventListener('mouseleave', () => h.style.background = 'linear-gradient(transparent 30%,#cbd5e1 50%,transparent 70%)');
                    let sY=0,sH=0;
                    h.addEventListener('mousedown', function(e) {
                        sY=e.clientY; sH=panel.offsetHeight; e.preventDefault();
                        const mv = e2 => {
                            const n = Math.max(160, Math.min(window.innerHeight*0.85, sH+(sY-e2.clientY)));
                            panel.style.height=n+'px'; panel.style.minHeight=n+'px'; panel.style.maxHeight=n+'px';
                        };
                        const up = () => { document.removeEventListener('mousemove',mv); document.removeEventListener('mouseup',up); };
                        document.addEventListener('mousemove',mv); document.addEventListener('mouseup',up);
                    });
                    panel.insertBefore(h, panel.firstChild);
                }).call(this);
            """.trimIndent())
        }

        tfUrl.addValueChangeListener { e ->
            val (models, default) = providerModels(e.value)
            cbModel.setItems(models)
            if (cbModel.value.isNullOrBlank() || !models.contains(cbModel.value))
                cbModel.value = default
        }
        cardStatic.addClickListener { selectMode(agent = false) }
        cardAgent.addClickListener  { selectMode(agent = true)  }

        // ── Colonne gauche : config (largeur fixe) ────────────────────────────
        val panelTitle = Span("🤖 ${i18n.judgeDialogTitle}").apply {
            style["font-size"]   = "0.78em"
            style["font-weight"] = "700"
            style["color"]       = "#1e3a5f"
        }
        val btnClose = Button("✕") { onClose() }.apply {
            element.setAttribute("theme", "tertiary")
            style["color"] = "#94a3b8"; style["min-width"] = "unset"; style["padding"] = "0 6px"
        }
        val colHeader = HorizontalLayout(panelTitle, btnClose).apply {
            setWidthFull(); isPadding = false; isSpacing = false
            alignItems = FlexComponent.Alignment.CENTER
            setFlexGrow(1.0, panelTitle)
            style["padding"]       = "6px 10px"
            style["border-bottom"] = "1px solid #e2e8f0"
            style["background"]    = "#f8fafc"
            style["flex-shrink"]   = "0"
        }
        val modeRow = HorizontalLayout(cardStatic, cardAgent).apply {
            setWidthFull(); isPadding = false; isSpacing = true; style["gap"] = "5px"
        }
        val leftCol = Div().apply {
            style["display"]        = "flex"
            style["flex-direction"] = "column"
            style["width"]          = "300px"
            style["min-width"]      = "300px"
            style["height"]         = "100%"
            style["overflow-y"]     = "auto"
            style["border-right"]   = "1px solid #e2e8f0"
            style["flex-shrink"]    = "0"
        }
        val configBody = VerticalLayout(
            sectionLabel(i18n.judgeConfigSection),
            tfUrl, pfKey, cbModel,
            sectionLabel(i18n.judgeModeSection),
            modeRow,
            spinner,
            btnSend,
        ).apply {
            setWidthFull(); isPadding = true; isSpacing = false
            style["gap"] = "4px"
            setHorizontalComponentAlignment(FlexComponent.Alignment.STRETCH, btnSend)
        }
        leftCol.add(colHeader, configBody)

        // ── Colonne droite : verdict (flexible) ───────────────────────────────
        val traceLabel   = sectionLabel("🔧 ${i18n.judgeTraceSection}").apply { isVisible = false }
        val verdictLabel = sectionLabel("📝 ${i18n.judgeVerdictSection}")
        traceArea.addAttachListener { if (traceArea.isVisible) traceLabel.isVisible = true }
        traceArea.addAttachListener { traceLabel.isVisible = traceArea.isVisible }
        this.traceLabel = traceLabel

        val rightCol = Div().apply {
            style["display"]        = "flex"
            style["flex-direction"] = "column"
            style["flex"]           = "1"
            style["min-width"]      = "0"
            style["height"]         = "100%"
            style["overflow"]       = "hidden"
            style["padding"]        = "8px 12px"
            style["box-sizing"]     = "border-box"
        }
        // sectionLabels comme Divs directs
        rightCol.add(traceLabel, traceArea, verdictLabel, verdictArea)
        // verdictArea doit remplir l'espace restant — appliqué via style flex
        verdictArea.style["flex"] = "1"
        verdictArea.style["min-height"] = "0"

        add(leftCol, rightCol)
    }


    // ── Mode card builder ─────────────────────────────────────────────────────
    private fun modeCard(emoji: String, title: String, desc: String, selected: Boolean): Div {
        val titleSpan = Span("$emoji $title").apply {
            style["font-weight"] = "700"; style["font-size"] = "0.82em"
        }
        val descSpan = Span(desc).apply {
            style["font-size"]   = "0.72em"
            style["color"]       = "#64748b"
            style["margin-top"]  = "2px"
            style["line-height"] = "1.3"
        }
        return Div(titleSpan, descSpan).apply {
            style["display"]        = "flex"
            style["flex-direction"] = "column"
            style["padding"]        = "8px 10px"
            style["border-radius"]  = "8px"
            style["cursor"]         = "pointer"
            style["flex"]           = "1"
            style["transition"]     = "all .15s"
            applyCardStyle(selected)
        }
    }

    private fun Div.applyCardStyle(selected: Boolean) {
        if (selected) { style["border"] = "2px solid #1d4ed8"; style["background"] = "#eff6ff" }
        else          { style["border"] = "2px solid #e2e8f0"; style["background"] = "#f8fafc" }
    }

    private fun selectMode(agent: Boolean) {
        agentMode = agent
        cardStatic.applyCardStyle(!agent)
        cardAgent.applyCardStyle(agent)
    }

    // ── Action ────────────────────────────────────────────────────────────────
    private fun doJudge() {
        val results = getResults()
        if (results.isEmpty()) {
            Notification.show(i18n.noResults, 2500, Notification.Position.BOTTOM_START)
            return
        }
        savedConfig = LlmJudgeConfig(
            baseUrl   = tfUrl.value.trim().ifBlank { "https://api.openai.com/v1" },
            apiKey    = pfKey.value.trim(),
            model     = (cbModel.value ?: "").trim().ifBlank { "gpt-4o-mini" },
            agentMode = agentMode,
        )
        if (savedConfig.apiKey.isBlank()) {
            Notification.show(i18n.judgeKeyRequired, 3000, Notification.Position.MIDDLE)
            return
        }

        // Reset UI
        verdictArea.removeAll()
        verdictArea.add(streamPlaceholder(i18n.judgeWaiting))
        traceArea.removeAll(); traceArea.isVisible = false
        traceLabel.isVisible = false
        spinner.isVisible = true; btnSend.isEnabled = false

        val ui    = UI.getCurrent()
        val cfg   = savedConfig
        var lastUpdate = 0L

        bgExec.submit {
            val verdict = try {
                judgeService.judgeStream(cfg, results) { partial, isTrace ->
                    // Throttle UI updates to ~15 fps (66ms) to avoid flooding Vaadin push
                    val now = System.currentTimeMillis()
                    if (now - lastUpdate < 66) return@judgeStream
                    lastUpdate = now
                    ui.access {
                        if (isTrace) {
                            // Mode agent : afficher la trace d'outils dans la traceArea
                            traceArea.removeAll()
                            traceArea.add(Span(partial).apply { style["white-space"] = "pre-wrap" })
                            traceArea.isVisible = true; traceLabel.isVisible = true
                        } else {
                            // Mode static : afficher le texte brut qui s'accumule
                            verdictArea.removeAll()
                            verdictArea.add(streamText(partial))
                        }
                    }
                }
            } catch (e: Exception) { "❌ ${e.message}" }

            ui.access {
                spinner.isVisible = false; btnSend.isEnabled = true
                verdictArea.removeAll(); traceArea.removeAll()

                val separatorIdx = verdict.indexOf("\n---\n")
                if (agentMode && separatorIdx >= 0) {
                    val trace   = verdict.substring(0, separatorIdx).trim()
                    val content = verdict.substring(separatorIdx + 5).trim()
                    traceArea.add(Span(trace).apply { style["white-space"] = "pre-wrap" })
                    traceArea.isVisible = true; traceLabel.isVisible = true
                    renderMarkdown(content, verdictArea)
                } else {
                    renderMarkdown(verdict, verdictArea)
                }
            }
        }
    }

    /** Zone de texte brut pendant le streaming (police mono, wrap). */
    private fun streamText(text: String) = Div().apply {
        style["white-space"]  = "pre-wrap"
        style["font-size"]    = "0.82em"
        style["line-height"]  = "1.6"
        style["color"]        = "#334155"
        style["font-family"]  = "ui-monospace,monospace"
        style["overflow-y"]   = "auto"
        style["flex"]         = "1"
        style["min-height"]   = "0"
        add(Span(text))
    }

    private fun streamPlaceholder(msg: String) = Span(msg).apply {
        style["color"]     = "#94a3b8"
        style["font-size"] = "0.86em"
    }

    // ── Markdown renderer (commonmark) ────────────────────────────────────────
    private fun renderMarkdown(text: String, target: Div) {
        val html = mdRenderer.render(mdParser.parse(text))
        target.removeAll()
        target.add(Html("""<div class="llm-md">$html</div>"""))
    }

    // ── Helpers ───────────────────────────────────────────────────────────────
    private fun sectionLabel(text: String) = Div(Span(text)).apply {
        style["font-size"]      = "0.67em"
        style["font-weight"]    = "700"
        style["letter-spacing"] = "0.08em"
        style["color"]          = "#94a3b8"
        style["text-transform"] = "uppercase"
        style["padding-bottom"] = "2px"
    }

    private fun emptyState() = Div(Span(i18n.detailPlaceholder)).apply {
        style["color"]       = "#94a3b8"
        style["font-size"]   = "0.86em"
        style["padding-top"] = "20px"
        style["text-align"]  = "center"
    }
}
