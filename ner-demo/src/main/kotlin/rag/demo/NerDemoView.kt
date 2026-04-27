package rag.demo

import com.fasterxml.jackson.databind.ObjectMapper
import com.vaadin.flow.component.UI
import com.vaadin.flow.component.accordion.Accordion
import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.button.ButtonVariant
import com.vaadin.flow.component.checkbox.Checkbox
import com.vaadin.flow.component.dialog.Dialog
import com.vaadin.flow.component.html.*
import com.vaadin.flow.component.Html
import com.vaadin.flow.component.notification.Notification
import com.vaadin.flow.component.orderedlayout.FlexComponent
import com.vaadin.flow.component.orderedlayout.HorizontalLayout
import com.vaadin.flow.component.orderedlayout.VerticalLayout
import com.vaadin.flow.component.progressbar.ProgressBar
import com.vaadin.flow.component.tabs.Tab
import com.vaadin.flow.component.tabs.TabSheet
import com.vaadin.flow.component.textfield.NumberField
import com.vaadin.flow.component.textfield.TextArea
import com.vaadin.flow.component.upload.Upload
import com.vaadin.flow.component.upload.receivers.MemoryBuffer
import com.vaadin.flow.router.PageTitle
import com.vaadin.flow.router.Route
import com.vaadin.flow.server.InputStreamFactory
import com.vaadin.flow.server.StreamResource
import rag.model.Entity
import java.io.ByteArrayInputStream
import java.util.concurrent.Executors

@Route("")
@PageTitle("NER + SVO — DeBERTa")
class NerDemoView(
    private val nerService: NerService,
    private val mapper: ObjectMapper,
    private val llmJudgeService: LlmJudgeService,
) : VerticalLayout() {

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
            "hint_person_name"    to "pers",
            "hint_person_role"    to "role",
            "hint_norp"           to "norp",
            "hint_group_role"     to "grp",
            "hint_org_name"       to "org",
            "hint_gpe"            to "gpe",
            "hint_fac_name"       to "fac",
            "hint_loc_generic"    to "loc",
            "hint_infra"          to "infra",
            "hint_weapon"         to "wpn",
            "hint_vehicle"        to "veh",
            "hint_substance"      to "subs",
            "hint_food"           to "food",
            "hint_tool"           to "tool",
            "hint_object_generic" to "obj",
            "hint_object_name"    to "prod",
            "hint_event_nominal"  to "evt",
            "hint_event_named"    to "evt✦",
            "hint_time_date"      to "date",
            "hint_time_clock"     to "time",
            "hint_time_duration"  to "dur",
            "hint_quantity"       to "qty",
            "hint_measure"        to "meas",
            "hint_percentage"     to "pct",
            "hint_count"          to "cnt",
            "hint_money"          to "€",
            "hint_rate"           to "rate",
            "hint_law"            to "law",
            "hint_work_of_art"    to "art",
            "hint_concept"        to "cpt",
            "hint_disease"        to "dis",
            "hint_language"       to "lang",
        )
        val ALL_COARSE = COARSE_COLORS.keys.filter { it != "NONE" }
        private val SVO_EMOJI = mapOf(
            "svo_verb" to "🔵", "svo_subject" to "🟢", "svo_object" to "🔴",
            "svo_iobj" to "🟠", "pron_subj" to "🟢", "pron_obj" to "🔴",
        )
        /** Couleurs des rôles syntaxiques UD (cohérentes avec SVO_COLORS). */
        val SYNTACTIC_ROLE_COLORS = mapOf(
            "nsubj" to ("#dcfce7" to "#15803d"),
            "obj"   to ("#fce7f3" to "#9d174d"),
            "iobj"  to ("#fff7ed" to "#c2410c"),
        )
        private val executor = Executors.newCachedThreadPool { r ->
            Thread(r, "ner-stream").also { it.isDaemon = true }
        }
    }

    // ── State ─────────────────────────────────────────────────────────────────
    private var lastResults: List<AnnotatedSentence> = emptyList()
    private val sentenceSlots = mutableListOf<Span>()

    // ── Panneaux collapsibles ─────────────────────────────────────────────────
    private var sidebarDiv: Div? = null
    private var rightColDiv: Div? = null

    // ── i18n (langue détectée depuis le navigateur) ───────────────────────────
    private val i18n: I18n = I18n.forLanguage(UI.getCurrent().locale.language)

    // ── LLM Judge panel (embarqué, visible/caché via Alt+J) ──────────────────
    private val judgePanel = LlmJudgePanel(llmJudgeService, i18n, { lastResults }) { toggleJudge() }
        .also { it.isVisible = true }
    private val btnJudge = Button("✕ Judge") { toggleJudge() }
        .also { it.style["font-size"] = "0.82em" }

    private fun toggleJudge() {
        val open = !judgePanel.isVisible
        judgePanel.isVisible = open
        btnJudge.text = if (open) "✕ Judge" else i18n.btnJudge
    }

    // ── Input ─────────────────────────────────────────────────────────────────
    private val inputArea = TextArea().apply {
        // placeholder assigné dans init{} pour éviter la collision avec TextArea.i18n
        setWidthFull()
        minRows = 7
        style["font-size"] = "0.86em"
        style["font-family"] = "Inter, system-ui, sans-serif"
        style["resize"] = "none"
    }

    // ── Param widgets ─────────────────────────────────────────────────────────
    private val cbShowNer       = Checkbox("NER", true)
    private val cbShowSvo       = Checkbox("SVO", true)
    private val cbShowArcs      = Checkbox(i18n.cbArcs, false)
    private val cbAutoSplit     = Checkbox(i18n.cbSplitAuto, true)
    private val cbReconcile     = Checkbox(i18n.cbReconcileLabel, true)
    private val cbFineForCoarse = ALL_COARSE.associateWith { Checkbox(it, true) }
    private val nfTauBoundary   = nf("Sensibilité entités",  0.20, 0.95, 0.70,
        helper = "Seuil de détection d'une entité. Baisser = plus d'entités, monter = plus précis.")
    private val nfTauNone       = nf("Rejet non-entité",     0.50, 1.00, 0.99,
        helper = "Score minimum pour qu'un token soit ignoré (non-entité). Proche de 1 = strict.")
    private val nfTauCoarse     = nf("Confiance catégorie",  0.00, 0.90, 0.45,
        helper = "Confiance minimum pour assigner une catégorie générale (PER, LOC, ORG…).")
    private val nfTauSvo        = nf("Sensibilité SVO",      0.20, 0.95, 0.50,
        helper = "Seuil de détection des rôles syntaxiques (sujet, verbe, objet).")
    private val nfTauSvoAnchored= nf("SVO ancré sur entité", 0.10, 0.70, 0.40,
        helper = "Seuil SVO appliqué quand le span coïncide avec une entité NER connue.")
    private val nfBatchSize     = nf("Taille du lot",        1.0,  32.0, 8.0,  step = 1.0,
        helper = "Nombre de phrases envoyées simultanément au modèle. Plus élevé = plus rapide.")
    private val nfMinNerRec   = nf("Score min réconciliation", 0.10, 0.95, 0.50,
        helper = "Score NER minimum pour qu'une entité participe à la réconciliation NER↔SVO.")
    private val nfMinNerFill  = nf("Score min remplissage",    0.10, 0.95, 0.60,
        helper = "Score NER minimum pour compléter un span SVO par une entité adjacente.")
    private val nfMaxGap      = nf("Écart max entre spans",    20.0, 400.0, 120.0, step = 10.0,
        helper = "Distance maximale en caractères entre deux spans pour les fusionner.")

    /** Seuil de score minimum par label coarse (null = seuil global). */
    private val nfScoreByCoarse: Map<String, NumberField> = ALL_COARSE.associateWith { coarse ->
        nf(coarse, 0.00, 1.00, 0.00, helper = "0 = utiliser le seuil global").also { it.style["width"] = "96px" }
    }

    // ── Results ───────────────────────────────────────────────────────────────
    private val textFlow    = Div()
    private val progressBar = ProgressBar().apply { isIndeterminate = true; isVisible = false }
    private val detailPanel = buildDetailPanel()

    // ── Layout ────────────────────────────────────────────────────────────────
    init {
        inputArea.placeholder = i18n.placeholder
        setSizeFull()
        isPadding = false
        isSpacing = false
        style["background"] = "#f8fafc"
        style["font-family"] = "Inter, system-ui, sans-serif"

        loadConfigToWidgets(nerService.config)

        // ── CSS responsive + resize handles injecté dynamiquement ───────────
        UI.getCurrent().page.executeJs("""
            (function(){
              var s = document.createElement('style');
              s.textContent = [
                '@media (max-width: 900px) {',
                '  .ner-sidebar { display: none !important; }',
                '  .ner-sidebar.ner-open { display: flex !important; position: fixed; left: 0; top: 0; bottom: 0; z-index: 200; box-shadow: 4px 0 16px rgba(0,0,0,.18); }',
                '  .ner-rightcol { display: none !important; }',
                '}',
                '@media (max-width: 1200px) and (min-width: 901px) {',
                '  .ner-rightcol { width: 220px !important; min-width: 220px !important; }',
                '}',
                '.ner-resize-x { position: absolute; right: -4px; top: 0; bottom: 0; width: 8px; cursor: col-resize; z-index: 100; background: transparent; }',
                '.ner-resize-x:hover, .ner-resize-x.dragging { background: rgba(99,102,241,.35); border-radius: 4px; }',
                '.ner-resize-y { position: absolute; top: -4px; left: 0; right: 0; height: 8px; cursor: row-resize; z-index: 100; background: transparent; }',
                '.ner-resize-y:hover, .ner-resize-y.dragging { background: rgba(99,102,241,.35); border-radius: 4px; }',
              ].join('');
              document.head.appendChild(s);

              // ── Generic drag-resize helper ───────────────────────────────────
              window._nerMakeResizableX = function(handle, target, minW, maxW) {
                handle.addEventListener('mousedown', function(e) {
                  e.preventDefault();
                  handle.classList.add('dragging');
                  var startX = e.clientX;
                  var startW = target.getBoundingClientRect().width;
                  function onMove(ev) {
                    var w = Math.min(maxW, Math.max(minW, startW + ev.clientX - startX));
                    target.style.width = w + 'px';
                    target.style.minWidth = w + 'px';
                  }
                  function onUp() {
                    handle.classList.remove('dragging');
                    document.removeEventListener('mousemove', onMove);
                    document.removeEventListener('mouseup', onUp);
                  }
                  document.addEventListener('mousemove', onMove);
                  document.addEventListener('mouseup', onUp);
                });
              };
              window._nerMakeResizableY = function(handle, target, minH, maxH) {
                handle.addEventListener('mousedown', function(e) {
                  e.preventDefault();
                  handle.classList.add('dragging');
                  var startY = e.clientY;
                  var startH = target.getBoundingClientRect().height;
                  function onMove(ev) {
                    var h = Math.min(maxH, Math.max(minH, startH - (ev.clientY - startY)));
                    target.style.height = h + 'px';
                    target.style.flex = 'none';
                  }
                  function onUp() {
                    handle.classList.remove('dragging');
                    document.removeEventListener('mousemove', onMove);
                    document.removeEventListener('mouseup', onUp);
                  }
                  document.addEventListener('mousemove', onMove);
                  document.addEventListener('mouseup', onUp);
                });
              };
            })();
        """.trimIndent())

        add(buildHeader())

        val body = HorizontalLayout()
        body.setSizeFull()
        body.isPadding = false
        body.isSpacing = false
        body.style["overflow"] = "hidden"
        body.style["min-height"] = "0"

        val resultsPane = buildResultsPane()

        // ── Colonne droite : détail uniquement ────────────────────────────────
        val rightCol = Div().apply {
            style["display"]        = "flex"
            style["flex-direction"] = "column"
            style["width"]          = "270px"
            style["min-width"]      = "270px"
            style["height"]         = "100%"
            style["overflow"]       = "hidden"
            style["flex-shrink"]    = "0"
            element.classList.add("ner-rightcol")
        }
        rightColDiv = rightCol
        detailPanel.style["flex"]       = "1"
        detailPanel.style["height"]     = "auto"
        detailPanel.style["min-height"] = "0"
        rightCol.add(detailPanel)

        val sidebar = buildSidebar()

        // ── Colonne gauche unifiée : sidebar + Judge, avec UN seul handle ─────
        val leftWrapper = Div().apply {
            style["display"]        = "flex"
            style["flex-direction"] = "column"
            style["width"]          = "360px"
            style["min-width"]      = "200px"
            style["height"]         = "100%"
            style["overflow"]       = "hidden"
            style["flex-shrink"]    = "0"
            style["position"]       = "relative"
            element.classList.add("ner-sidebar")
        }
        // Retire la classe ner-sidebar du sidebar intérieur (évite double toggle mobile)
        sidebar.element.classList.remove("ner-sidebar")
        // Le sidebar intérieur prend tout l'espace vertical restant
        sidebar.style["flex"]           = "1"
        sidebar.style["width"]          = "100%"
        sidebar.style["min-width"]      = "unset"
        sidebar.style["height"]         = "auto"
        sidebar.style["min-height"]     = "0"
        sidebar.style["border-right"]   = "none"

        leftWrapper.add(sidebar, judgePanel)
        sidebarDiv = leftWrapper
        val resizeHandleX = Div().apply {
            element.classList.add("ner-resize-x")
            element.setAttribute("title", "Redimensionner la colonne gauche")
        }
        leftWrapper.add(resizeHandleX)
        UI.getCurrent().page.executeJs(
            "window._nerMakeResizableX(\$0, \$1, 200, 750);",
            resizeHandleX.element, leftWrapper.element
        )
        // Bordure droite sur le wrapper
        leftWrapper.style["border-right"] = "1px solid #e2e8f0"

        body.add(leftWrapper, resultsPane, rightCol)
        body.setFlexGrow(1.0, resultsPane)
        body.alignItems = FlexComponent.Alignment.STRETCH

        add(body)
        setFlexGrow(1.0, body)


        // ── Alt+J : toggle LLM Judge panel ───────────────────────────────────
        UI.getCurrent().page.executeJs(
            "window.addEventListener('keydown', e => { if(e.altKey && e.key.toLowerCase()==='j') { e.preventDefault(); \$0.click(); } });",
            btnJudge.element
        )
    }

    // ── Header ────────────────────────────────────────────────────────────────
    private fun buildHeader(): Div {
        // ── Titre principal ───────────────────────────────────────────────────
        val title = Span(i18n.headerTitle)
        title.style["font-size"]      = "1.0em"
        title.style["font-weight"]    = "800"
        title.style["color"]          = "white"
        title.style["letter-spacing"] = "-0.02em"

        // ── Badges tech ───────────────────────────────────────────────────────
        fun techBadge(label: String, bg: String, fg: String = "white") = Span(label).also { b ->
            b.style["background"]      = bg
            b.style["color"]           = fg
            b.style["font-size"]       = "0.60em"
            b.style["font-weight"]     = "700"
            b.style["letter-spacing"]  = "0.06em"
            b.style["padding"]         = "2px 7px"
            b.style["border-radius"]   = "4px"
            b.style["vertical-align"]  = "middle"
            b.style["white-space"]     = "nowrap"
        }
        val badgeNer = techBadge(i18n.headerBadgeNer, "rgba(99,102,241,.75)")
        val badgeSvo = techBadge(i18n.headerBadgeSvo, "rgba(3,105,161,.75)")
        val badgeLlm = techBadge(i18n.headerBadgeLlm, "rgba(124,58,237,.75)")

        val titleRow = HorizontalLayout(title, badgeNer, badgeSvo, badgeLlm).also {
            it.isPadding = false; it.isSpacing = false
            it.alignItems = FlexComponent.Alignment.CENTER
            it.style["gap"] = "7px"
        }

        // ── Sous-titre ────────────────────────────────────────────────────────
        val sub = Span(i18n.headerSub)
        sub.style["color"]     = "rgba(255,255,255,.52)"
        sub.style["font-size"] = "0.70em"
        sub.style["display"]   = "block"
        sub.style["margin-top"]= "1px"

        val left = Div(titleRow, sub).also { d ->
            d.style["display"] = "flex"; d.style["flex-direction"] = "column"
        }

        // ── Boutons header ────────────────────────────────────────────────────
        fun hBtn(label: String, action: () -> Unit) = Button(label) { action() }.also { b ->
            b.style["background"]    = "rgba(255,255,255,.12)"
            b.style["border"]        = "1px solid rgba(255,255,255,.22)"
            b.style["color"]         = "white"
            b.style["border-radius"] = "6px"
            b.style["font-size"]     = "0.76em"
            b.style["padding"]       = "3px 9px"
            b.style["cursor"]        = "pointer"
            b.style["height"]        = "28px"
        }

        val actions = HorizontalLayout(
            hBtn("☰") { sidebarDiv?.element?.executeJs("this.classList.toggle('ner-open')") },
            hBtn(i18n.btnImportConfig) { showImportConfigDialog() },
            hBtn(i18n.btnExportConfig) { exportConfig() },
            hBtn("🌙") { toggleDark() },
        ).also { it.isPadding = false; it.isSpacing = true; it.alignItems = FlexComponent.Alignment.CENTER }

        val row = HorizontalLayout(left, actions).also {
            it.setWidthFull(); it.isPadding = false
            it.alignItems = FlexComponent.Alignment.CENTER
            it.setFlexGrow(1.0, left)
        }
        return Div(row).also { d ->
            d.setWidthFull()
            d.style["background"]  = "linear-gradient(135deg,#1e3a5f,#1d4ed8 60%,#6366f1)"
            d.style["padding"]     = "10px 20px"
            d.style["box-sizing"]  = "border-box"
            d.style["flex-shrink"] = "0"
        }
    }

    // ── Left sidebar ──────────────────────────────────────────────────────────
    private fun buildSidebar(): Div {
        val inputLabel = Span(i18n.inputLabel)
        inputLabel.style["font-size"] = "0.68em"
        inputLabel.style["font-weight"] = "700"
        inputLabel.style["color"] = "#94a3b8"
        inputLabel.style["letter-spacing"] = "0.08em"

        val btnAnalyse = Button(i18n.btnAnalyse) {
            val text = inputArea.value.trim()
            if (text.isBlank()) return@Button
            saveWidgetsToConfig(); launchStream(text)
        }.also {
            it.addThemeVariants(ButtonVariant.LUMO_PRIMARY)
            it.style["font-size"] = "0.82em"
        }

        val btnClear = Button(i18n.btnClear) {
            inputArea.value = ""
            textFlow.removeAll(); sentenceSlots.clear()
            detailPanel.removeAll()
            detailPanel.add(placeholder(i18n.detailPlaceholder))
            lastResults = emptyList()
        }.also { it.style["font-size"] = "0.82em" }

        val btnExport = Button(i18n.btnJson) {
            if (lastResults.isEmpty()) Notification.show(i18n.noResults, 2500, Notification.Position.BOTTOM_START)
            else exportResultsJson()
        }.also { it.style["font-size"] = "0.82em" }

        val btnRow = HorizontalLayout(btnAnalyse, btnClear, btnExport, btnJudge).also {
            it.isPadding = false; it.isSpacing = true
        }

        progressBar.setWidthFull()

        val sidebar = Div()
        sidebar.style["width"] = "100%"
        sidebar.style["min-width"] = "unset"
        sidebar.style["height"] = "auto"
        sidebar.style["overflow-y"] = "auto"
        sidebar.style["background"] = "#ffffff"
        sidebar.style["padding"] = "16px"
        sidebar.style["box-sizing"] = "border-box"
        sidebar.style["display"] = "flex"
        sidebar.style["flex-direction"] = "column"
        sidebar.style["gap"] = "10px"
        sidebar.style["flex-shrink"] = "0"
        sidebar.style["position"] = "relative"

        sidebar.add(buildOnboardingCard(), inputLabel, inputArea, btnRow, progressBar, buildParamsAccordion())
        return sidebar
    }

    // ── Onboarding card ───────────────────────────────────────────────────────
    private fun buildOnboardingCard(): Div {
        val examples = listOf(
            "🏛 Politique" to "Emmanuel Macron a rencontré Angela Merkel à Berlin le 12 mars 2024 pour discuter du budget de l'Union européenne.",
            "🏢 Économie"  to "Apple a annoncé l'acquisition de la startup française Mistral AI pour 500 millions d'euros lors du salon VivaTech à Paris.",
            "⚽ Sport"     to "Le Paris Saint-Germain a battu l'Olympique de Marseille 3 à 1 lors du Classique disputé au Parc des Princes dimanche soir.",
        )

        val card = Div()
        card.style["background"]     = "#eff6ff"
        card.style["border"]         = "1px solid #bfdbfe"
        card.style["border-radius"]  = "10px"
        card.style["padding"]        = "14px"
        card.style["font-size"]      = "0.83em"
        card.style["color"]          = "#1e3a5f"
        card.style["flex-shrink"]    = "0"

        val titleRow = HorizontalLayout()
        titleRow.isPadding = false; titleRow.isSpacing = false
        titleRow.style["align-items"] = "center"
        titleRow.setWidthFull()

        val title = Span("💡 Comment utiliser cette démo")
        title.style["font-weight"] = "700"
        title.style["font-size"]   = "0.95em"

        val btnDismiss = Button("✕") {
            card.isVisible = false
            UI.getCurrent().page.executeJs("localStorage.setItem('ner-onboarding-dismissed','1')")
        }.also { b ->
            b.style["background"]  = "none"
            b.style["border"]      = "none"
            b.style["color"]       = "#64748b"
            b.style["cursor"]      = "pointer"
            b.style["padding"]     = "0 4px"
            b.style["min-width"]   = "unset"
            b.style["font-size"]   = "0.85em"
        }

        titleRow.add(title)
        titleRow.setFlexGrow(1.0, title)
        titleRow.add(btnDismiss)

        val desc = Div(Span("Choisissez un exemple ci-dessous ou collez votre texte, puis cliquez sur Analyser. " +
            "Chaque mot coloré est une entité détectée — cliquez dessus pour voir ses détails dans le panneau de droite."))
        desc.style["color"]         = "#475569"
        desc.style["margin"]        = "8px 0 10px 0"
        desc.style["line-height"]   = "1.5"
        desc.style["font-size"]     = "0.90em"

        val legendHelp = Div()
        legendHelp.style["font-size"]  = "0.82em"
        legendHelp.style["color"]      = "#475569"
        legendHelp.style["margin-bottom"] = "10px"
        legendHelp.style["line-height"] = "1.6"
        legendHelp.add(Html("""<span>
            <b>Fond coloré</b> = entité nommée (qui/quoi/où/quand…)<br/>
            <b>Soulignement</b> = rôle syntaxique (<span style="color:#15803d">sujet</span>,
            <span style="color:#0369a1">verbe</span>,
            <span style="color:#9d174d">objet</span>)<br/>
            Passez la souris ou cliquez pour les détails.
        </span>"""))

        val examplesDiv = Div()
        examplesDiv.style["display"]        = "flex"
        examplesDiv.style["flex-direction"] = "column"
        examplesDiv.style["gap"]            = "5px"

        for ((label, text) in examples) {
            val btn = Div()
            btn.style["background"]    = "white"
            btn.style["border"]        = "1px solid #bfdbfe"
            btn.style["border-radius"] = "7px"
            btn.style["padding"]       = "8px 10px"
            btn.style["cursor"]        = "pointer"
            val lbl = Span(label)
            lbl.style["font-weight"]  = "700"
            lbl.style["margin-right"] = "6px"
            lbl.style["font-size"]    = "0.88em"
            val preview = Span(text.take(65) + "…")
            preview.style["color"]     = "#475569"
            preview.style["font-size"] = "0.84em"
            btn.add(lbl, preview)
            btn.addClickListener {
                inputArea.value = text
                saveWidgetsToConfig()
                launchStream(text)
                card.isVisible = false
                UI.getCurrent().page.executeJs("localStorage.setItem('ner-onboarding-dismissed','1')")
            }
            examplesDiv.add(btn)
        }

        card.add(titleRow, desc, legendHelp, examplesDiv)

        // Masquer si déjà ignoré
        UI.getCurrent().page.executeJs(
            "if(localStorage.getItem('ner-onboarding-dismissed')==='1') \$0.setAttribute('hidden','');",
            card.element
        )
        return card
    }

    // ── Results pane ──────────────────────────────────────────────────────────
    private fun buildResultsPane(): Div {
        textFlow.style["font-size"] = "1.06em"
        textFlow.style["line-height"] = "2.8"
        textFlow.style["font-family"] = "Inter, system-ui, sans-serif"
        textFlow.style["color"] = "#0f172a"
        textFlow.style["word-break"] = "break-word"

        val inner = Div(buildLegend(), textFlow)
        inner.style["padding"] = "28px 36px"
        inner.style["box-sizing"] = "border-box"
        inner.style["overflow-y"] = "auto"
        inner.style["height"] = "100%"

        val pane = Div(inner)
        pane.style["display"] = "flex"
        pane.style["flex-direction"] = "column"
        pane.style["overflow"] = "hidden"
        pane.style["min-width"] = "0"
        pane.style["height"] = "100%"
        return pane
    }

    // ── Detail panel ──────────────────────────────────────────────────────────
    private fun buildDetailPanel(): Div {
        val div = Div()
        div.style["width"] = "270px"
        div.style["min-width"] = "270px"
        div.style["height"] = "100%"
        div.style["overflow-y"] = "auto"
        div.style["border-left"] = "1px solid #e2e8f0"
        div.style["background"] = "#ffffff"
        div.style["padding"] = "16px"
        div.style["box-sizing"] = "border-box"
        div.style["font-family"] = "Inter, system-ui, sans-serif"
        div.style["font-size"] = "0.82em"
        div.style["flex-shrink"] = "0"
        div.add(placeholder(i18n.detailPlaceholder))
        return div
    }

    // ── Params accordion ──────────────────────────────────────────────────────

    /** Mini-titre de section dans l'accordéon. */
    private fun tabSection(text: String) = Div(Span(text)).also { d ->
        d.style["font-size"]      = "0.67em"
        d.style["font-weight"]    = "700"
        d.style["letter-spacing"] = "0.09em"
        d.style["color"]          = "#94a3b8"
        d.style["text-transform"] = "uppercase"
        d.style["padding"]        = "10px 0 5px 0"
    }

    /**
     * Bouton toggle générique : clique pour cocher/décocher le [cb] backing.
     * [activeColor]/[activeBg] définissent la teinte quand actif.
     */
    private fun buildToggleBtn(
        label: String, cb: Checkbox,
        activeColor: String = "#1d4ed8", activeBg: String = "#eff6ff",
    ): Div {
        val chip = Div(Span(label))
        fun sync(active: Boolean) {
            chip.style["background"]  = if (active) activeBg else "#f8fafc"
            chip.style["color"]       = if (active) activeColor else "#94a3b8"
            chip.style["border"]      = "2px solid ${if (active) activeColor + "88" else "#e2e8f0"}"
            chip.style["font-weight"] = if (active) "700" else "500"
        }
        chip.style["padding"]      = "5px 13px"
        chip.style["border-radius"]= "8px"
        chip.style["cursor"]       = "pointer"
        chip.style["font-size"]    = "0.80em"
        chip.style["transition"]   = "all .12s"
        chip.style["user-select"]  = "none"
        sync(cb.value)
        cb.addValueChangeListener { sync(it.value) }
        chip.addClickListener { cb.value = !cb.value }
        return chip
    }

    /**
     * Chip coloré pour une catégorie NER : clique = toggle du [cb] backing.
     * Quand actif → fond coloré + label fin visible ; inactif → grisé.
     */
    private fun buildCoarseChip(coarse: String, cb: Checkbox): Div {
        val (bg, fg) = COARSE_COLORS[coarse] ?: ("#f3f4f6" to "#6b7280")
        val chip = Div(Span(coarse))
        fun sync(active: Boolean) {
            chip.style["background"] = if (active) bg  else "#f1f5f9"
            chip.style["color"]      = if (active) fg  else "#c4c8d0"
            chip.style["border"]     = if (active) "2px solid ${fg}55" else "2px solid #e2e8f0"
        }
        chip.style["padding"]       = "4px 11px"
        chip.style["border-radius"] = "6px"
        chip.style["cursor"]        = "pointer"
        chip.style["font-size"]     = "0.72em"
        chip.style["font-weight"]   = "800"
        chip.style["letter-spacing"]= "0.06em"
        chip.style["transition"]    = "all .12s"
        chip.style["user-select"]   = "none"
        chip.element.setAttribute("title", "Labels fins pour $coarse")
        sync(cb.value)
        cb.addValueChangeListener { sync(it.value) }
        chip.addClickListener { cb.value = !cb.value }
        return chip
    }

    private fun buildParamsAccordion(): Accordion {
        val accordion = Accordion()
        accordion.setWidthFull()

        // ── Onglet Affichage ──────────────────────────────────────────────────
        val annotRow = Div().apply {
            style["display"] = "flex"; style["flex-wrap"] = "wrap"; style["gap"] = "6px"
            add(buildToggleBtn("NER",        cbShowNer,  "#1d4ed8", "#eff6ff"))
            add(buildToggleBtn("SVO",        cbShowSvo,  "#0369a1", "#e0f2fe"))
            add(buildToggleBtn("Arcs UD",    cbShowArcs, "#6366f1", "#eef2ff"))
        }
        val procRow = Div().apply {
            style["display"] = "flex"; style["flex-wrap"] = "wrap"; style["gap"] = "6px"
            add(buildToggleBtn("✂ Split auto",     cbAutoSplit, "#059669", "#d1fae5"))
            add(buildToggleBtn("⚡ Réconcilier NER↔SVO", cbReconcile, "#7c3aed", "#ede9fe"))
        }
        val catRow = Div().apply {
            style["display"] = "flex"; style["flex-wrap"] = "wrap"; style["gap"] = "5px"
            ALL_COARSE.forEach { coarse -> add(buildCoarseChip(coarse, cbFineForCoarse[coarse]!!)) }
        }
        val catHelp = Span("Activer = affiche le label fin (ex: \"pers\") plutôt que la catégorie (\"PER\")").also {
            it.style["font-size"] = "0.70em"; it.style["color"] = "#94a3b8"; it.style["display"] = "block"
            it.style["margin-bottom"] = "4px"
        }

        val tabAff = VerticalLayout().apply {
            isPadding = false; isSpacing = false; setWidthFull()
            style["padding"] = "0 0 8px 0"
            add(tabSection("Annotations"), annotRow)
            add(tabSection("Traitement automatique"), procRow)
            add(tabSection("Labels fins par catégorie"), catHelp, catRow)
        }

        val tabSeuils = VerticalLayout().apply { isPadding = false; isSpacing = false }
        tabSeuils.add(HorizontalLayout(nfTauBoundary, nfTauNone, nfTauCoarse, nfTauSvo, nfTauSvoAnchored, nfBatchSize).also {
            it.isSpacing = true; it.isPadding = false; it.style["flex-wrap"] = "wrap"
        })

        val tabRec = VerticalLayout().apply { isPadding = false; isSpacing = false }
        tabRec.add(HorizontalLayout(nfMinNerRec, nfMinNerFill, nfMaxGap).also {
            it.isSpacing = true; it.isPadding = false; it.style["flex-wrap"] = "wrap"
        })

        val lbl0 = Span("Score minimum par catégorie (0 = utiliser le seuil global)")
        lbl0.style["font-size"] = "0.75em"; lbl0.style["color"] = "#64748b"
        lbl0.style["display"] = "block"; lbl0.style["margin-bottom"] = "8px"
        val tabType = VerticalLayout().apply { isPadding = false; isSpacing = false }
        tabType.add(lbl0)
        tabType.add(HorizontalLayout(*ALL_COARSE.map { nfScoreByCoarse[it]!! }.toTypedArray()).also {
            it.isSpacing = true; it.isPadding = false; it.style["flex-wrap"] = "wrap"
        })

        val ts = TabSheet(); ts.setWidthFull()
        ts.add(Tab(i18n.tabDisplay),          tabAff)
        ts.add(Tab("Détection (seuils)"),     tabSeuils)
        ts.add(Tab("Réconciliation NER↔SVO"), tabRec)
        ts.add(Tab("Seuils par catégorie"),   tabType)

        accordion.add(i18n.paramsTitle, ts)
        // Ouvert par défaut pour que les réglages soient découvrables
        accordion.open(0)
        return accordion
    }

    // ── Legend ────────────────────────────────────────────────────────────────
    private fun buildLegend(): Div {
        val div = Div()
        div.style["display"] = "flex"
        div.style["flex-wrap"] = "wrap"
        div.style["gap"] = "5px"
        div.style["margin-bottom"] = "20px"
        div.style["padding-bottom"] = "14px"
        div.style["border-bottom"] = "1px solid #f1f5f9"
        COARSE_COLORS.entries.filter { it.key != "NONE" }.forEach { (c, cols) ->
            div.add(legendChip(c, cols.first, cols.second, false))
        }
        SVO_COLORS.entries.forEach { (role, cols) ->
            val short = role.replace("svo_", "").replace("pron_", "pron:")
            div.add(legendChip(short, cols.first, cols.second, true))
        }
        return div
    }

    private fun legendChip(label: String, bg: String, fg: String, dashed: Boolean) = Span(label).also { c ->
        c.style["background"] = bg; c.style["color"] = fg
        c.style["border-radius"] = "4px"; c.style["padding"] = "2px 7px"
        c.style["font-size"] = "0.70em"; c.style["font-weight"] = "700"
        c.style["letter-spacing"] = "0.05em"
        if (dashed) c.style["border"] = "1px dashed ${fg}55"
    }

    // ── Streaming ─────────────────────────────────────────────────────────────

    /** Séparateur avant chaque phrase : "" | " " | "\n" | "\n\n" */
    private fun buildEntries(text: String): List<Pair<String, String>> {
        return if (cbAutoSplit.value) {
            val result = mutableListOf<Pair<String, String>>()
            for ((pIdx, para) in text.trim().split(Regex("""\n{2,}""")).withIndex()) {
                val normalized = para.lines().map { it.trim() }.filter { it.isNotBlank() }.joinToString(" ")
                if (normalized.isBlank()) continue
                for ((sIdx, sent) in nerService.splitSentences(normalized).withIndex()) {
                    val sep = when {
                        result.isEmpty() -> ""
                        sIdx == 0        -> "\n\n"   // nouveau paragraphe
                        else             -> " "      // suite de phrase
                    }
                    result += sent to sep
                }
            }
            result
        } else {
            text.lines().map { it.trim() }.filter { it.isNotBlank() }
                .mapIndexed { i, s -> s to (if (i == 0) "" else "\n") }
        }
    }

    private fun launchStream(text: String) {
        val entries = buildEntries(text)
        val sentences = entries.map { it.first }
        if (sentences.isEmpty()) return

        lastResults = emptyList()
        sentenceSlots.clear()
        textFlow.removeAll()
        detailPanel.removeAll()
        detailPanel.add(placeholder(i18n.detailPlaceholder))
        progressBar.isVisible = true

        // Pré-remplir le flux texte en respectant les séparateurs d'origine
        for ((sent, sep) in entries) {
            when (sep) {
                "\n\n" -> repeat(2) { textFlow.add(Html("<br/>")) }
                "\n"   -> textFlow.add(Html("<br/>"))
                " "    -> textFlow.add(Span(" "))
            }
            val slot = Span()
            slot.style["display"] = "inline"
            slot.style["color"] = "#94a3b8"
            slot.add(Span(sent))
            sentenceSlots.add(slot)
            textFlow.add(slot)
        }

        val collected = arrayOfNulls<AnnotatedSentence>(sentences.size)
        val ui = UI.getCurrent()

        executor.submit {
            try {
                nerService.analyseStream(sentences) { startIdx, batchResults ->
                    batchResults.forEachIndexed { bi, r -> collected[startIdx + bi] = r }
                    ui.access {
                        batchResults.forEachIndexed { bi, r ->
                            val slot = sentenceSlots[startIdx + bi]
                            slot.removeAll()
                            slot.style.remove("color")
                            renderIntoSlot(slot, r)
                        }
                    }
                }
            } catch (e: Exception) {
                ui.access { Notification.show("${i18n.errorPrefix}${e.message}", 5000, Notification.Position.MIDDLE) }
            } finally {
                ui.access {
                    progressBar.isVisible = false
                    lastResults = collected.filterNotNull()
                }
            }
        }
    }

    // ── spaCy-style inline rendering ──────────────────────────────────────────
    private data class SpanInfo(
        val charStart: Int, val charEnd: Int, val displayText: String,
        val bg: String, val fg: String, val label: String, val isNer: Boolean,
        val entity: Entity? = null, val svo: EnrichedSvoSpan? = null,
    )

    /**
     * Dans une couche, garde les spans les plus longs sans chevauchement PARTIEL.
     *
     * [allowCompound] = true (NER) : les spans entièrement contenus dans un parent
     *   sont conservés comme entités imbriquées (compound). Permet d'afficher
     *   "Organisation Mondiale de la Santé" (ORG) à l'intérieur de
     *   "secrétaire général de l'OMS" (PER_ROLE).
     *
     * [allowCompound] = false (SVO) : NMS strict — tout span qui chevauche OU
     *   est contenu dans un span déjà gardé est éliminé. Évite les doublons
     *   "Le tunnel sous la manche" / "tunnel sous la manche" en nsubj.
     */
    private fun keepLongest(spans: List<SpanInfo>, allowCompound: Boolean = false): List<SpanInfo> {
        val sorted = spans.sortedByDescending { it.charEnd - it.charStart }
        val kept = mutableListOf<SpanInfo>()
        for (s in sorted) {
            val overlapping = kept.filter { it.charStart < s.charEnd && it.charEnd > s.charStart }
            when {
                overlapping.isEmpty() -> kept += s
                allowCompound && overlapping.all { k -> k.charStart <= s.charStart && k.charEnd >= s.charEnd } -> kept += s
                // chevauchement partiel ou mode strict → éliminé
            }
        }
        return kept.sortedBy { it.charStart }
    }

    private fun renderIntoSlot(slot: Span, result: AnnotatedSentence) {
        val sentText = result.text
        val fineSet  = cbFineForCoarse.entries.filter { it.value.value }.map { it.key }.toSet()

        // ── Couche NER ─────────────────────────────────────────────────────────
        val nerLayer: List<SpanInfo> = if (cbShowNer.value) {
            result.entities.mapNotNull { ent ->
                val ms = ent.span ?: return@mapNotNull null
                val cs = ms.start; val ce = ms.end
                if (cs < 0 || ce <= cs || ce > sentText.length) return@mapNotNull null
                val coarse = ent.metadata["coarse"] as? String ?: "NONE"
                val (bg, fg) = COARSE_COLORS[coarse] ?: ("#f3f4f6" to "#6b7280")
                val label = if (coarse in fineSet) COMPACT_LABEL[ent.type] ?: ent.type.removePrefix("hint_") else coarse
                SpanInfo(cs, ce, ent.text, bg, fg, label, isNer = true, entity = ent)
            }.let { keepLongest(it, allowCompound = true) }   // compound NER : nested OK
        } else emptyList()

        // ── Couche SVO ─────────────────────────────────────────────────────────
        val svoLayer: List<SpanInfo> = if (cbShowSvo.value) {
            result.svoSpans.map { svo ->
                val (bg, fg) = SVO_COLORS[svo.role] ?: ("#e5e7eb" to "#374151")
                val lbl = svo.role.replace("svo_", "").replace("pron_", "pron:")
                // entity = svo.entity : le span SVO porte déjà l'entité NER fusionnée
                // par reconcile() → le panneau détail peut afficher NER + SVO d'un seul clic.
                SpanInfo(svo.charStart, svo.charEnd, svo.text, bg, fg, lbl, isNer = false,
                    entity = svo.entity, svo = svo)
            }.let { keepLongest(it, allowCompound = false) }  // SVO strict : un seul span par position
        } else emptyList()

        // ── Décomposition aux frontières des deux couches ──────────────────────
        // NER et SVO peuvent se chevaucher / s'imbriquer librement entre couches.
        val pts = sortedSetOf(0, sentText.length)
        (nerLayer + svoLayer).forEach { pts += it.charStart; pts += it.charEnd }
        val ptList = pts.filter { it in 0..sentText.length }

        for (i in 0 until ptList.size - 1) {
            val s = ptList[i]; val e = ptList[i + 1]
            if (s >= e || s >= sentText.length) continue
            val txt = sentText.substring(s, minOf(e, sentText.length))
            if (txt.isEmpty()) continue

            // Span NER/SVO le plus précis (innermost) qui couvre ENTIÈREMENT cet intervalle.
            // Pour les entités imbriquées, l'entité la plus courte (la plus spécifique)
            // qui couvre [s,e] est préférée, ce qui permet d'afficher l'ORG nested
            // plutôt que le PER_ROLE parent sur la sous-zone de l'ORG.
            val ner = nerLayer.filter { it.charStart <= s && it.charEnd >= e }
                              .minByOrNull { it.charEnd - it.charStart }
            val svo = svoLayer.filter { it.charStart <= s && it.charEnd >= e }
                              .minByOrNull { it.charEnd - it.charStart }

            if (ner == null && svo == null) { slot.add(Span(txt)); continue }

            val isNerFirst = ner != null && s == ner.charStart
            val isNerLast  = ner != null && e == ner.charEnd
            val isSvoFirst = svo != null && s == svo.charStart
            val isSvoLast  = svo != null && e == svo.charEnd

            // SpanInfo unifié pour le panneau détail :
            // - Si NER est présent : on y attache le svo éventuel → showDetail voit les deux.
            // - Si SVO seul         : le svo.entity (de reconcile) est déjà dans info.entity → idem.
            val detailInfo = if (ner != null) ner.copy(svo = svo?.svo) else svo!!
            slot.add(buildSegment(txt, ner, svo, isNerFirst, isNerLast, isSvoFirst, isSvoLast) {
                showDetail(detailInfo, result.entities)
            })
        }
    }

    // ── Segment combiné NER + SVO ──────────────────────────────────────────────
    private fun buildSegment(
        text: String,
        ner: SpanInfo?, svo: SpanInfo?,
        isNerFirst: Boolean, isNerLast: Boolean,
        isSvoFirst: Boolean, isSvoLast: Boolean,
        onClick: () -> Unit,
    ): Span {
        val seg = Span()
        seg.style["display"] = "inline"
        seg.style["cursor"] = if (ner != null || svo != null) "pointer" else "default"

        // NER → fond plein, border-radius seulement aux vraies extrémités du span
        if (ner != null) {
            seg.style["background"] = ner.bg
            val radL = if (isNerFirst) "0.3em" else "0"
            val radR = if (isNerLast)  "0.3em" else "0"
            seg.style["border-radius"] = "$radL $radR $radR $radL"
            val padL = if (isNerFirst) "0.4em" else "0.05em"
            val padR = if (isNerLast)  "0.1em" else "0.05em"
            seg.style["padding"] = "0.18em $padR 0.18em $padL"
        }

        // SVO → soulignement bas continu sur toute l'étendue du span SVO
        if (svo != null) {
            val lowConf = (svo.svo?.svoBoundaryProb ?: 1f) < 0.60f
            seg.style["border-bottom"] = "${if (lowConf) "2px dashed" else "2.5px solid"} ${svo.fg}"
            if (ner == null) {
                val padL = if (isSvoFirst) "0.15em" else "0"
                val padR = if (isSvoLast)  "0.15em" else "0"
                seg.style["padding"] = "0.1em $padR 0.05em $padL"
            }
        }

        val textNode = Span(text)
        textNode.style["color"] = if (ner != null) "#0f172a" else (svo?.fg ?: "#0f172a")
        textNode.style["font-weight"] = "500"
        seg.add(textNode)

        // Labels : affichés uniquement à la dernière fraction du span
        if (isNerLast && ner != null) {
            seg.add(buildLabelBadge(ner.label.uppercase(), ner.fg, filled = true))
            // Badge rôle syntaxique (nsubj / obj / iobj) — si la tête SVO a reconcilié ce span
            val syntRole = ner.entity?.metadata?.get("syntacticRole") as? String
            if (syntRole != null) {
                val (_, roleFg) = SYNTACTIC_ROLE_COLORS[syntRole] ?: ("#f3f4f6" to "#6b7280")
                seg.add(buildSyntaxRoleBadge(syntRole, roleFg))
            }
        }
        if (isSvoLast && svo != null) seg.add(buildLabelBadge(svo.label.uppercase(), svo.fg, filled = false))

        seg.addClickListener { onClick() }
        return seg
    }

    private fun buildLabelBadge(text: String, fg: String, filled: Boolean): Span {
        val lbl = Span(text)
        lbl.style["font-size"]      = "0.68em"
        lbl.style["font-weight"]    = "800"
        lbl.style["margin-left"]    = "0.35em"
        lbl.style["color"]          = fg
        lbl.style["vertical-align"] = "middle"
        lbl.style["letter-spacing"] = "0.06em"
        lbl.style["white-space"]    = "nowrap"
        if (filled) {
            lbl.style["background"]    = "rgba(0,0,0,0.09)"
            lbl.style["padding"]       = "0.08em 0.28em"
            lbl.style["border-radius"] = "0.22em"
        }
        return lbl
    }

    /**
     * Badge rôle syntaxique UD (nsubj / obj / iobj) affiché en superscript coloré
     * après le label NER, quand la tête SVO a reconcilié ce span inline.
     */
    private fun buildSyntaxRoleBadge(role: String, fg: String): Span {
        val (bg, _) = SYNTACTIC_ROLE_COLORS[role] ?: ("#f3f4f6" to "#6b7280")
        return Span(role).also { lbl ->
            lbl.style["font-size"]      = "0.60em"
            lbl.style["font-weight"]    = "700"
            lbl.style["margin-left"]    = "0.22em"
            lbl.style["color"]          = fg
            lbl.style["background"]     = bg
            lbl.style["border"]         = "1px solid ${fg}66"
            lbl.style["padding"]        = "0.05em 0.25em"
            lbl.style["border-radius"]  = "0.3em"
            lbl.style["vertical-align"] = "middle"
            lbl.style["letter-spacing"] = "0.04em"
            lbl.style["white-space"]    = "nowrap"
        }
    }

    // ── Detail panel content ──────────────────────────────────────────────────
    /**
     * Panneau de détail unifié.
     *
     * La logique d'affichage dépend du contenu de [info] :
     *  - [info.entity] non-null  → section NER complète (que [info.isNer] soit true ou false)
     *  - [info.svo] non-null     → section SVO annexée APRÈS la section NER
     *                              (ou seule si entity == null : span pur SVO sans entité)
     *
     * Ce design couvre trois cas :
     *  1. NER pur          : entity != null, svo == null
     *  2. Merged NER + SVO : entity != null, svo != null  (même offset ou snap)
     *  3. SVO pur          : entity == null, svo != null  (verbe, pronom, arg sans entité)
     */
    private fun showDetail(info: SpanInfo, allEntities: List<rag.model.Entity> = emptyList()) {
        detailPanel.removeAll()
        val ent = info.entity

        if (ent != null) {
            // ── Section NER ────────────────────────────────────────────────────
            val coarse = ent.metadata["coarse"] as? String ?: "?"
            val label  = COMPACT_LABEL[ent.type] ?: ent.type.removePrefix("hint_")
            detailPanel.add(sectionTitle("🏷 ${label.uppercase()}"), detailDivider())
            addRow(detailPanel, i18n.rowText,   ent.text)
            addRow(detailPanel, i18n.rowCoarse, coarse)
            addRow(detailPanel, i18n.rowFine,   ent.type)
            addRow(detailPanel, i18n.rowChars,  "[${ent.span?.start}:${ent.span?.end}]")
            detailPanel.add(sectionHeader(i18n.scoresSection))
            addRow(detailPanel, "p_entity", fmt(ent.metadata["pBoundary"]))
            addRow(detailPanel, "p_coarse", fmt(ent.metadata["pCoarse"]))
            addRow(detailPanel, "p_fine",   fmt(ent.metadata["pFine"]))
            addRow(detailPanel, "score",    fmt(ent.metadata["score"]))
            if (ent.metadata["svoAnchored"] == true) {
                val badge = Span("⚡ SVO-anchored — confiance NER réduite")
                badge.style["font-size"]   = "0.72em"
                badge.style["color"]       = "#c2410c"
                badge.style["font-weight"] = "600"
                badge.style["display"]     = "block"
                badge.style["margin-top"]  = "4px"
                detailPanel.add(badge)
            }
            if (ent.metadata["nested"] == true) {
                val parentText   = ent.metadata["parentText"]   as? String ?: "?"
                val parentFine   = ent.metadata["parentFine"]   as? String ?: "?"
                val parentCoarse = ent.metadata["parentCoarse"] as? String ?: "?"
                val parentStart  = ent.metadata["parentStart"]
                val parentEnd    = ent.metadata["parentEnd"]
                detailPanel.add(sectionHeader("🔼 PARENT"))
                addRow(detailPanel, i18n.rowText,   parentText)
                addRow(detailPanel, i18n.rowCoarse, parentCoarse)
                addRow(detailPanel, i18n.rowFine,   parentFine)
                addRow(detailPanel, i18n.rowChars,  "[$parentStart:$parentEnd]")
            }
            val children = allEntities.filter { child ->
                child.metadata["nested"] == true &&
                child.metadata["parentStart"] == ent.span?.start &&
                child.metadata["parentEnd"]   == ent.span?.end
            }
            if (children.isNotEmpty()) {
                detailPanel.add(sectionHeader("🔽 IMBRIQUÉS (${children.size})"))
                children.forEach { child ->
                    val childLabel = COMPACT_LABEL[child.type] ?: child.type.removePrefix("hint_")
                    addRow(detailPanel, childLabel.uppercase(),
                        "\"${child.text}\"  [${child.span?.start}:${child.span?.end}]  ${fmt(child.metadata["score"])}")
                }
            }
            // Rôle syntaxique depuis metadata (inline forward pass) — affiché si pas de SVO annexé
            // (si svo != null il sera affiché plus bas avec les métriques complètes)
            if (info.svo == null) {
                val syntRole = ent.metadata["syntacticRole"] as? String
                if (syntRole != null) {
                    val (roleBg, roleFg) = SYNTACTIC_ROLE_COLORS[syntRole] ?: ("#f3f4f6" to "#6b7280")
                    val roleTitle = Span("🔗 RÔLE SVO — $syntRole")
                    roleTitle.style["font-size"] = "0.80em"; roleTitle.style["font-weight"] = "700"
                    roleTitle.style["color"] = roleFg; roleTitle.style["background"] = roleBg
                    roleTitle.style["padding"] = "2px 8px"; roleTitle.style["border-radius"] = "4px"
                    roleTitle.style["display"] = "inline-block"; roleTitle.style["margin-top"] = "10px"
                    detailPanel.add(roleTitle)
                    addRow(detailPanel, "svo_role",  ent.metadata["svoRole"] as? String ?: "—")
                    addRow(detailPanel, "p_role",    fmt(ent.metadata["svoRoleProb"]))
                    addRow(detailPanel, "p_svo_bnd", fmt(ent.metadata["svoBoundaryScore"]))
                }
            }
            val gender = ent.metadata["gender"] as? String
            val number = ent.metadata["number"] as? String
            if (gender != null || number != null) {
                detailPanel.add(sectionHeader("🔤 MORPHOLOGIE"))
                gender?.let { addRow(detailPanel, "genre",  it) }
                number?.let { addRow(detailPanel, "nombre", it) }
            }

            // ── Section SVO annexée (cas merged NER+SVO) ───────────────────────
            info.svo?.let { appendSvoSection(it) }

        } else if (info.svo != null) {
            // ── Section SVO pur (verbe, pronom, argument sans entité NER) ───────
            val svo = info.svo; val emoji = SVO_EMOJI[svo.role] ?: "⚪"
            detailPanel.add(sectionTitle("$emoji ${svo.role}"), detailDivider())
            addRow(detailPanel, i18n.rowText,   svo.text)
            addRow(detailPanel, i18n.rowRole,   svo.role)
            addRow(detailPanel, i18n.rowVoice,  svo.voice)
            addRow(detailPanel, i18n.rowChars,  "[${svo.charStart}:${svo.charEnd}]")
            if (svo.fromNer) addRow(detailPanel, i18n.rowSource, i18n.syntheticNer)
            svo.nerOverride?.let { addRow(detailPanel, "🔗 override", "$it (${fmt(svo.nerOverrideScore)})") }
            detailPanel.add(sectionHeader(i18n.scoresSection))
            addRow(detailPanel, "p_boundary", "%.3f".format(svo.svoBoundaryProb))
            addRow(detailPanel, "p_role",     "%.3f".format(svo.roleProb))
            addRow(detailPanel, "voice conf", "%.3f".format(svo.voiceProb))
            svo.gender?.let { addRow(detailPanel, i18n.rowGender, it) }
            svo.number?.let { addRow(detailPanel, i18n.rowNumber, it) }
        }
    }

    /**
     * Section SVO annexée en bas du panneau NER (cas merged).
     * Affiche les métriques de la tête SVO : boundary, rôle, voice.
     * Le rôle syntaxique est mis en avant visuellement (header coloré).
     */
    private fun appendSvoSection(svo: EnrichedSvoSpan) {
        val emoji  = SVO_EMOJI[svo.role] ?: "⚪"
        val (svoBg, svoFg) = SVO_COLORS[svo.role] ?: ("#e5e7eb" to "#374151")
        val hdr = Span("$emoji SVO — ${svo.role}")
        hdr.style["font-size"]     = "0.80em"
        hdr.style["font-weight"]   = "700"
        hdr.style["color"]         = svoFg
        hdr.style["background"]    = svoBg
        hdr.style["padding"]       = "2px 8px"
        hdr.style["border-radius"] = "4px"
        hdr.style["display"]       = "inline-block"
        hdr.style["margin-top"]    = "10px"
        detailPanel.add(hdr)
        addRow(detailPanel, "p_svo_bnd", "%.3f".format(svo.svoBoundaryProb))
        addRow(detailPanel, "p_role",    "%.3f".format(svo.roleProb))
        addRow(detailPanel, "voice",     "${svo.voice} (${"%.2f".format(svo.voiceProb)})")
        svo.nerOverride?.let { addRow(detailPanel, "🔗 override", "$it (${fmt(svo.nerOverrideScore)})") }
        svo.gender?.let { addRow(detailPanel, i18n.rowGender, it) }
        svo.number?.let { addRow(detailPanel, i18n.rowNumber, it) }
        if (svo.fromNer) addRow(detailPanel, i18n.rowSource, i18n.syntheticNer)
    }

    // ── Export / Import ───────────────────────────────────────────────────────
    private fun triggerDownload(filename: String, bytes: ByteArray) {
        val factory = InputStreamFactory { ByteArrayInputStream(bytes) }
        val res = StreamResource(filename, factory)
        val anchor = Anchor(res, "")
        anchor.element.setAttribute("download", "")
        anchor.style["display"] = "none"
        element.appendChild(anchor.element)
        anchor.element.executeJs("this.click(); setTimeout(() => this.remove(), 200);")
    }

    private fun exportConfig() {
        triggerDownload("ner-demo-config.json",
            mapper.writerWithDefaultPrettyPrinter().writeValueAsBytes(nerService.config))
    }

    private fun showImportConfigDialog() {
        val dialog = Dialog(); dialog.headerTitle = i18n.configDialogTitle
        val buffer = MemoryBuffer()
        val upload = Upload(buffer)
        upload.setAcceptedFileTypes("application/json", ".json")
        upload.addSucceededListener {
            try {
                val cfg = mapper.readValue(buffer.inputStream, DemoConfig::class.java)
                nerService.updateConfig(cfg); loadConfigToWidgets(cfg)
                Notification.show(i18n.configImported, 2500, Notification.Position.BOTTOM_START)
                dialog.close()
            } catch (e: Exception) {
                Notification.show("${i18n.errorPrefix}${e.message}", 5000, Notification.Position.MIDDLE)
            }
        }
        dialog.add(upload); dialog.open()
    }

    private fun exportResultsJson() {
        val payload = lastResults.mapIndexed { i, r ->
            mapOf(
                "idx"  to (i + 1),
                "text" to r.text,
                "ner"  to r.entities.map { e -> mapOf(
                    "text" to e.text, "coarse" to e.metadata["coarse"],
                    "fine" to e.type, "char_start" to e.span?.start, "char_end" to e.span?.end,
                    "score" to e.metadata["score"],
                    "syntactic_role" to e.metadata["syntacticRole"],
                    "svo_role"       to e.metadata["svoRole"],
                    "svo_role_prob"  to e.metadata["svoRoleProb"],
                    "gender"         to e.metadata["gender"],
                    "number"         to e.metadata["number"],
                )},
                "svo"  to r.svoSpans.map { s -> mapOf(
                    "text" to s.text, "role" to s.role,
                    "char_start" to s.charStart, "char_end" to s.charEnd,
                    "voice" to s.voice, "gender" to s.gender, "number" to s.number,
                    "p_boundary" to s.svoBoundaryProb, "p_role" to s.roleProb,
                    "ner_override" to s.nerOverride, "from_ner" to s.fromNer,
                )},
            )
        }
        triggerDownload("ner-results.json",
            mapper.writerWithDefaultPrettyPrinter().writeValueAsBytes(payload))
    }

    // ── Config sync ───────────────────────────────────────────────────────────
    private fun saveWidgetsToConfig() {
        nerService.updateConfig(DemoConfig(
            tauBoundary             = nfTauBoundary.value?.toFloat()    ?: 0.70f,
            tauNone                 = nfTauNone.value?.toFloat()         ?: 0.99f,
            tauCoarse               = nfTauCoarse.value?.toFloat()       ?: 0.45f,
            tauSvoBoundary          = nfTauSvo.value?.toFloat()          ?: 0.50f,
            tauSvoAnchoredBoundary  = nfTauSvoAnchored.value?.toFloat()  ?: 0.40f,
            batchSize               = nfBatchSize.value?.toInt()         ?: 8,
            showNer              = cbShowNer.value,
            showSvo              = cbShowSvo.value,
            showArcs             = cbShowArcs.value,
            autoSplit            = cbAutoSplit.value,
            doReconcile          = cbReconcile.value,
            fineForCoarse        = cbFineForCoarse.entries.filter { it.value.value }.map { it.key }.toSet(),
            minNerScoreReconcile = nfMinNerRec.value?.toFloat()      ?: 0.50f,
            minNerScoreFill      = nfMinNerFill.value?.toFloat()     ?: 0.60f,
            maxGapChars          = nfMaxGap.value?.toInt()           ?: 120,
            scoreByCoarse        = nfScoreByCoarse
                .mapValues { it.value.value?.toFloat() ?: 0f }
                .filterValues { it > 0f },
        ))
    }

    private fun loadConfigToWidgets(cfg: DemoConfig) {
        nfTauBoundary.value  = cfg.tauBoundary.toDouble()
        nfTauNone.value      = cfg.tauNone.toDouble()
        nfTauCoarse.value    = cfg.tauCoarse.toDouble()
        nfTauSvo.value       = cfg.tauSvoBoundary.toDouble()
        nfTauSvoAnchored.value = cfg.tauSvoAnchoredBoundary.toDouble()
        nfBatchSize.value    = cfg.batchSize.toDouble()
        cbShowNer.value     = cfg.showNer
        cbShowSvo.value     = cfg.showSvo
        cbShowArcs.value    = cfg.showArcs
        cbAutoSplit.value   = cfg.autoSplit
        cbReconcile.value   = cfg.doReconcile
        cbFineForCoarse.forEach { (c, cb) -> cb.value = c in cfg.fineForCoarse }
        nfMinNerRec.value   = cfg.minNerScoreReconcile.toDouble()
        nfMinNerFill.value  = cfg.minNerScoreFill.toDouble()
        nfMaxGap.value      = cfg.maxGapChars.toDouble()
        nfScoreByCoarse.forEach { (coarse, nf) ->
            nf.value = cfg.scoreByCoarse[coarse]?.toDouble() ?: 0.0
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────
    private fun addRow(container: Div, key: String, value: String) {
        val row = HorizontalLayout()
        row.setWidthFull(); row.isPadding = false; row.isSpacing = false
        row.style["padding"] = "3px 0"
        row.style["border-bottom"] = "1px solid #f1f5f9"
        row.style["align-items"] = "baseline"
        val k = Span(key)
        k.style["color"] = "#64748b"; k.style["font-weight"] = "600"
        k.style["font-size"] = "0.82em"; k.style["min-width"] = "78px"; k.style["width"] = "78px"
        val v = Span(value)
        v.style["font-size"] = "0.84em"; v.style["font-family"] = "monospace"
        v.style["word-break"] = "break-all"; v.style["flex"] = "1"
        row.add(k, v); container.add(row)
    }

    private fun sectionTitle(text: String) = H4(text).also {
        it.style["margin"] = "0 0 2px 0"; it.style["font-size"] = "0.95em"
    }

    private fun sectionHeader(text: String) = Div(Span(text)).also { d ->
        d.style["font-weight"] = "700"; d.style["font-size"] = "0.70em"
        d.style["color"] = "#94a3b8"; d.style["letter-spacing"] = "0.09em"
        d.style["padding-top"] = "10px"; d.style["padding-bottom"] = "2px"
    }

    private fun detailDivider() = Hr().also {
        it.style["border"] = "none"
        it.style["border-top"] = "1px solid #f1f5f9"
        it.style["margin"] = "4px 0"
    }

    private fun placeholder(text: String) = Span(text).also {
        it.style["color"] = "#94a3b8"; it.style["font-size"] = "0.86em"
    }

    private fun fmt(v: Any?): String = when (v) {
        is Float  -> "%.3f".format(v)
        is Double -> "%.3f".format(v)
        null      -> "—"
        else      -> v.toString()
    }

    private fun nf(
        label: String, min: Double, max: Double, default: Double,
        step: Double = 0.05, helper: String? = null,
    ) = NumberField(label).also { f ->
        f.min = min; f.max = max; f.value = default; f.step = step
        f.style["width"] = "108px"
        if (helper != null) f.helperText = helper
    }

    private fun toggleDark() {
        UI.getCurrent().page.executeJs("""
            var html = document.documentElement;
            var dark = html.getAttribute('data-theme') !== 'dark';
            html.setAttribute('data-theme', dark ? 'dark' : 'light');
            localStorage.setItem('ner-dark', dark ? '1' : '0');
        """.trimIndent())
    }
}
