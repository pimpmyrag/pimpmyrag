package rag.demo

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import com.vaadin.flow.component.UI
import com.vaadin.flow.component.button.Button
import com.vaadin.flow.component.button.ButtonVariant
import com.vaadin.flow.component.combobox.ComboBox
import com.vaadin.flow.component.dialog.Dialog
import com.vaadin.flow.component.Html
import com.vaadin.flow.component.html.Anchor
import com.vaadin.flow.component.html.Div
import com.vaadin.flow.component.html.Image
import com.vaadin.flow.component.html.Span
import com.vaadin.flow.data.renderer.ComponentRenderer
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
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration
import java.util.concurrent.Executors

/**
 * Panneau LLM Judge — embarqué dans la colonne gauche (sous la sidebar).
 * Toggle via [NerDemoView.toggleJudge] ou raccourci Alt+J.
 *
 *  ┌─────────────────────────────────┐
 *  │ ░░░ (drag pour redimensionner)  │  ← handle vertical
 *  ├─────────────────────────────────┤
 *  │ 🤖 LLM Judge            [✕]    │  ← header
 *  ├─────────────────────────────────┤
 *  │ Provider  [OpenAI ▾]            │
 *  │ API Key   [••••••••]            │
 *  │ Modèle    [gpt-4o-mini ▾]       │
 *  │ Mode      [📋 Static][🤖 Agent] │
 *  │           [▶ Analyser]          │
 *  ├─────────────────────────────────┤
 *  │ 📝 Verdict (scrollable)         │
 *  └─────────────────────────────────┘
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
        private val httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(15)).build()
        private val jsonMapper = ObjectMapper()

        // ── Presets providers ─────────────────────────────────────────────────
        data class ProviderPreset(
            val name: String,
            val url: String,
            val emoji: String,
            val isGithub: Boolean = false,
            val isAzure: Boolean = false,
            val urlNote: String? = null,
        )
        val PROVIDER_PRESETS = listOf(
            ProviderPreset("OpenAI",          "https://api.openai.com/v1",                                    "🟢"),
            ProviderPreset("GitHub Copilot",  "https://api.githubcopilot.com",                               "🐙", isGithub = true),
            ProviderPreset("GitHub Models",   "https://models.inference.ai.azure.com",                       "🐙", isGithub = true),
            ProviderPreset("Azure OpenAI",    "https://<resource>.openai.azure.com/openai",                  "☁️", isAzure = true,
                urlNote = "Remplacez <resource> par le nom de votre ressource Azure. Le modèle = votre nom de déploiement."),
            ProviderPreset("Mistral",         "https://api.mistral.ai/v1",                                    "🟠"),
            ProviderPreset("Anthropic",       "https://api.anthropic.com/v1",                                 "🟣"),
            ProviderPreset("Google",          "https://generativelanguage.googleapis.com/v1beta/openai",      "🔵"),
            ProviderPreset("Groq",            "https://api.groq.com/openai/v1",                               "⚡"),
            ProviderPreset("Together AI",     "https://api.together.xyz/v1",                                  "🤝"),
            ProviderPreset("Cohere",          "https://api.cohere.com/compatibility/v1",                      "🌀"),
            ProviderPreset("DeepSeek",        "https://api.deepseek.com/v1",                                  "🔮"),
            ProviderPreset("Ollama",          "http://localhost:11434/v1",                                     "🦙"),
            ProviderPreset("LM Studio",       "http://localhost:1234/v1",                                     "🖥️"),
        )

        // ── GitHub Device Flow ────────────────────────────────────────────────
        /** Démarre le device flow OAuth GitHub et retourne (deviceCode, userCode, verificationUri, interval). */
        data class DeviceCodeResponse(val deviceCode: String, val userCode: String, val verificationUri: String, val interval: Int)
        internal fun startDeviceFlow(clientId: String): DeviceCodeResponse {
            val req = HttpRequest.newBuilder(URI("https://github.com/login/device/code"))
                .POST(HttpRequest.BodyPublishers.ofString("client_id=$clientId&scope=copilot+read%3Auser"))
                .header("Accept", "application/json")
                .header("Content-Type", "application/x-www-form-urlencoded")
                .build()
            val resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString())
            val json = jsonMapper.readValue<Map<String, Any>>(resp.body())
            return DeviceCodeResponse(
                deviceCode      = json["device_code"] as? String ?: error("no device_code"),
                userCode        = json["user_code"]        as? String ?: error("no user_code"),
                verificationUri = json["verification_uri"] as? String ?: "https://github.com/login/device",
                interval        = (json["interval"] as? Int) ?: 5,
            )
        }

        /** Sonde jusqu'à obtenir le token OAuth (lève une exception si expiré/refusé). */
        internal fun pollOAuthToken(clientId: String, deviceCode: String, intervalSec: Int): String {
            val body = "client_id=$clientId&device_code=$deviceCode&grant_type=urn:ietf:params:oauth:grant-type:device_code"
            repeat(60) { // max ~5 min
                Thread.sleep(intervalSec * 1000L)
                val req = HttpRequest.newBuilder(URI("https://github.com/login/oauth/access_token"))
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .header("Accept", "application/json")
                    .header("Content-Type", "application/x-www-form-urlencoded")
                    .build()
                val resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString())
                val json = jsonMapper.readValue<Map<String, Any>>(resp.body())
                val token = json["access_token"] as? String
                if (!token.isNullOrBlank()) return token
                val err = json["error"] as? String ?: ""
                if (err == "access_denied" || err == "expired_token") error("GitHub OAuth: $err")
                // authorization_pending ou slow_down → continuer
            }
            error("Timeout: l'autorisation GitHub n'a pas été accordée dans les délais.")
        }

        /** Échange le token OAuth GitHub contre un token Copilot éphémère.
         *  Retourne null si GitHub Copilot n'est pas disponible sur ce compte. */
        internal fun exchangeForCopilotToken(oauthToken: String): String? {
            val req = HttpRequest.newBuilder(URI("https://api.github.com/copilot_internal/v2/token"))
                .GET()
                .header("Authorization", "token $oauthToken")
                .header("Accept", "application/json")
                .header("User-Agent", "NerDemo/1.0")
                .build()
            return try {
                val resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString())
                if (resp.statusCode() != 200) return null
                val json = jsonMapper.readValue<Map<String, Any>>(resp.body())
                json["token"] as? String
            } catch (_: Exception) { null }
        }

        // ── Azure AD Device Flow ──────────────────────────────────────────────
        /**
         * Démarre le device flow Azure AD (compatible Microsoft 365 / Azure OpenAI).
         * tenantId : "common" pour les comptes perso/org mixtes, ou l'ID du tenant.
         * clientId : App ID de l'Azure App Registration.
         * scope    : "https://cognitiveservices.azure.com/.default" pour Azure OpenAI.
         */
        data class AzureDeviceCodeResponse(
            val deviceCode: String, val userCode: String,
            val verificationUri: String, val interval: Int, val expiresIn: Int,
        )
        internal fun startAzureDeviceCodeFlow(tenantId: String, clientId: String, scope: String): AzureDeviceCodeResponse {
            val body = "client_id=${clientId}&scope=${java.net.URLEncoder.encode(scope, "UTF-8")}"
            val req = HttpRequest.newBuilder(URI("https://login.microsoftonline.com/$tenantId/oauth2/v2.0/devicecode"))
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .header("Content-Type", "application/x-www-form-urlencoded")
                .build()
            val resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString())
            val json = jsonMapper.readValue<Map<String, Any>>(resp.body())
            if (json.containsKey("error")) error("Azure Device Flow : ${json["error_description"] ?: json["error"]}")
            return AzureDeviceCodeResponse(
                deviceCode      = json["device_code"]      as? String ?: error("no device_code"),
                userCode        = json["user_code"]         as? String ?: error("no user_code"),
                verificationUri = json["verification_uri"]  as? String ?: "https://microsoft.com/devicelogin",
                interval        = (json["interval"] as? Int) ?: 5,
                expiresIn       = (json["expires_in"] as? Int) ?: 900,
            )
        }

        internal fun pollAzureAccessToken(tenantId: String, clientId: String, deviceCode: String, intervalSec: Int): String {
            val body = "grant_type=urn:ietf:params:oauth:grant-type:device_code" +
                "&client_id=$clientId&device_code=$deviceCode"
            val maxTries = 120
            repeat(maxTries) {
                Thread.sleep(intervalSec * 1000L)
                val req = HttpRequest.newBuilder(URI("https://login.microsoftonline.com/$tenantId/oauth2/v2.0/token"))
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .header("Content-Type", "application/x-www-form-urlencoded")
                    .build()
                val resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString())
                val json = jsonMapper.readValue<Map<String, Any>>(resp.body())
                val token = json["access_token"] as? String
                if (!token.isNullOrBlank()) return token
                val err = json["error"] as? String ?: ""
                if (err == "authorization_declined" || err == "expired_token") error("Azure OAuth: $err")
                // authorization_pending ou slow_down → continuer
            }
            error("Timeout: l'autorisation Azure n'a pas été accordée dans les délais.")
        }

        // ── Catalogues modèles par provider ───────────────────────────────────
        private val MODELS_OPENAI = listOf(
            "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            "o4-mini", "o3", "o3-mini",
            "gpt-4o", "gpt-4o-mini",
            "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
        )
        private val MODELS_GH_COPILOT = listOf(
            "gpt-4o", "gpt-4o-mini",
            "claude-3.5-sonnet", "claude-3.7-sonnet",
            "o1-mini", "o3-mini",
            "gemini-1.5-pro", "gemini-2.0-flash",
        )
        private val MODELS_GH_MODELS = listOf(
            "gpt-4o", "gpt-4o-mini",
            "Meta-Llama-3.3-70B-Instruct",
            "mistral-large-2411",
            "Phi-4", "Phi-3.5-MoE-instruct",
            "AI21-Jamba-1.5-Large",
            "cohere-command-r-plus-08-2024",
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
        private val MODELS_GROQ = listOf(
            "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
            "mixtral-8x7b-32768", "gemma2-9b-it",
        )
        private val MODELS_TOGETHER = listOf(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
        )
        private val MODELS_OLLAMA = listOf(
            "llama3.3", "llama3.1", "mistral", "mixtral",
            "phi4", "qwen2.5", "deepseek-r1", "gemma3",
        )
        private val MODELS_DEEPSEEK = listOf(
            "deepseek-chat", "deepseek-reasoner",
        )

        fun providerLogoUrl(preset: ProviderPreset): String? = when (preset.name) {
            "OpenAI"                       -> "https://logo.clearbit.com/openai.com"
            "GitHub Copilot", "GitHub Models" -> "https://logo.clearbit.com/github.com"
            "Azure OpenAI"                 -> "https://logo.clearbit.com/microsoft.com"
            "Mistral"                      -> "https://logo.clearbit.com/mistral.ai"
            "Anthropic"                    -> "https://logo.clearbit.com/anthropic.com"
            "Google"                       -> "https://logo.clearbit.com/google.com"
            "Groq"                         -> "https://logo.clearbit.com/groq.com"
            "Together AI"                  -> "https://logo.clearbit.com/together.ai"
            "Cohere"                       -> "https://logo.clearbit.com/cohere.com"
            "DeepSeek"                     -> "https://logo.clearbit.com/deepseek.com"
            "Ollama"                       -> "https://ollama.com/public/ollama.png"
            "LM Studio"                    -> "https://logo.clearbit.com/lmstudio.ai"
            else                           -> null
        }

        fun providerModels(url: String): Pair<List<String>, String> = when {
            url.contains("inference.ai.azure",   ignoreCase = true) -> MODELS_GH_MODELS  to "gpt-4o"
            url.contains("openai.com",           ignoreCase = true) -> MODELS_OPENAI     to "gpt-4o-mini"
            url.contains("mistral.ai",           ignoreCase = true) -> MODELS_MISTRAL    to "mistral-small-latest"
            url.contains("anthropic.com",        ignoreCase = true) -> MODELS_ANTHROPIC  to "claude-haiku-3-5"
            url.contains("googleapis.com",       ignoreCase = true) ||
            url.contains("generativelanguage",   ignoreCase = true) -> MODELS_GOOGLE     to "gemini-2.5-flash"
            url.contains("groq.com",             ignoreCase = true) -> MODELS_GROQ       to "llama-3.3-70b-versatile"
            url.contains("together.xyz",         ignoreCase = true) -> MODELS_TOGETHER   to "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            url.contains("deepseek.com",         ignoreCase = true) -> MODELS_DEEPSEEK   to "deepseek-chat"
            url.contains("ollama",               ignoreCase = true) ||
            url.contains("localhost",            ignoreCase = true) ||
            url.contains("127.0.0.1",            ignoreCase = true) -> MODELS_OLLAMA    to "llama3.3"
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
    private val ui18n = i18n

    /** ComboBox de presets provider — logo réel + nom */
    private val cbProvider = ComboBox<ProviderPreset>().apply {
        label = "Provider"
        setWidthFull()
        setItems(PROVIDER_PRESETS)
        // Renderer avec logo
        setRenderer(ComponentRenderer { preset ->
            val row = HorizontalLayout().apply {
                isPadding = false; isSpacing = false
                alignItems = FlexComponent.Alignment.CENTER
                style["gap"] = "8px"
            }
            val logoUrl = LlmJudgePanel.providerLogoUrl(preset)
            if (logoUrl != null) {
                val img = Image(logoUrl, preset.name)
                img.style["width"]        = "18px"
                img.style["height"]       = "18px"
                img.style["object-fit"]   = "contain"
                img.style["border-radius"]= "3px"
                img.style["flex-shrink"]  = "0"
                row.add(img)
            } else {
                // Fallback : cercle coloré avec emoji
                row.add(Span(preset.emoji).apply { style["font-size"] = "14px" })
            }
            row.add(Span(preset.name).apply { style["font-size"] = "0.88em" })
            row
        })
        // Label dans le champ sélectionné (texte seul, logo affiché dans la liste)
        setItemLabelGenerator { "${it.emoji} ${it.name}" }
        isAllowCustomValue = false
        placeholder = "Choisir un provider…"
        val saved = savedConfig.baseUrl
        value = PROVIDER_PRESETS.firstOrNull { saved.contains(it.url, ignoreCase = true) || it.url == saved }
    }

    private val tfUrl = TextField().apply {
        label = ui18n.judgeUrlLabel
        value = savedConfig.baseUrl
        setWidthFull()
        placeholder = "https://api.openai.com/v1"
        style["font-size"] = "0.82em"
    }

    // ── Section GitHub OAuth ──────────────────────────────────────────────────
    /** Client ID de l'OAuth App GitHub enregistrée par le développeur. */
    private val tfGhClientId = TextField().apply {
        label = "GitHub OAuth App — Client ID"
        placeholder = "Iv1.xxxxxxxxxxxx"
        setWidthFull()
        helperText = "Créez une OAuth App sur github.com/settings/developers (callback = http://localhost)"
        style["font-size"] = "0.80em"
        isVisible = false
    }

    private val btnGithubOAuth = Button("🔐 Se connecter avec GitHub").apply {
        addClickListener { this@LlmJudgePanel.startGithubDeviceFlow() }
        setWidthFull()
        isVisible = false
    }

    // ── Section Azure AD OAuth ────────────────────────────────────────────────
    private val ghPatNote = Div().apply {
        val link = Anchor("https://github.com/settings/tokens", "Créer un PAT GitHub →")
        link.setTarget("_blank")
        link.style["color"] = "#1d4ed8"; link.style["font-size"] = "0.78em"
        add(Html("""<span style="font-size:0.78em;color:#475569">
            Utilisez votre <b>GitHub PAT</b> (Personal Access Token) comme clé API.
            Les scopes <code>read:user</code> + <code>copilot</code> suffisent.<br/>
        </span>"""), link)
        style["background"]    = "#f0f9ff"
        style["border"]        = "1px solid #bae6fd"
        style["border-radius"] = "6px"
        style["padding"]       = "6px 10px"
        style["line-height"]   = "1.6"
        isVisible = false
    }

    // ── Section Azure AD OAuth ────────────────────────────────────────────────
    private val azureNote = Div().apply {
        val link = Anchor("https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity", "Docs Azure OpenAI Auth →")
        link.setTarget("_blank")
        link.style["color"] = "#1d4ed8"; link.style["font-size"] = "0.78em"
        add(Html("""<span style="font-size:0.78em;color:#475569">
            <b>Azure OpenAI</b> — deux options :<br/>
            • <b>Clé API</b> : copiez-la depuis le portail Azure (Cognitive Services).<br/>
            • <b>Azure AD</b> : utilisez le Device Flow ci-dessous (compte M365/Entra ID).<br/>
            L'URL doit être : <code>https://&lt;resource&gt;.openai.azure.com/openai</code><br/>
            Le modèle = <b>nom de votre déploiement</b> Azure.<br/>
        </span>"""), link)
        style["background"]    = "#f0f9ff"
        style["border"]        = "1px solid #bae6fd"
        style["border-radius"] = "6px"
        style["padding"]       = "6px 10px"
        style["line-height"]   = "1.6"
        isVisible = false
    }

    private val tfAzureTenantId = TextField().apply {
        label = "Tenant ID (ou \"common\")"
        placeholder = "common  — ou  xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        value = "common"
        setWidthFull()
        helperText = "Trouvez-le dans Azure Portal → Azure Active Directory → Vue d'ensemble"
        style["font-size"] = "0.80em"
        isVisible = false
    }

    private val tfAzureClientId = TextField().apply {
        label = "Azure App Registration — Client ID"
        placeholder = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        setWidthFull()
        helperText = "App Registration avec permission Cognitive Services (ou Azure OpenAI)"
        style["font-size"] = "0.80em"
        isVisible = false
    }

    private val btnAzureOAuth = Button("☁️ Se connecter avec Microsoft").apply {
        addClickListener { this@LlmJudgePanel.startAzureDeviceFlow() }
        setWidthFull()
        isVisible = false
    }

    private val pfKey = PasswordField().apply {
        label = ui18n.judgeKeyLabel; value = savedConfig.apiKey; setWidthFull()
    }
    private val cbModel = ComboBox<String>().apply {
        label = ui18n.judgeModelLabel; setWidthFull()
        isAllowCustomValue = true
        val (models, default) = providerModels(savedConfig.baseUrl)
        setItems(models)
        value = savedConfig.model.ifBlank { default }
        addCustomValueSetListener { e -> value = e.detail }
    }

    // ── Mode toggle ───────────────────────────────────────────────────────────
    private var agentMode = savedConfig.agentMode
    private val cardStatic = modeCard("📋", i18n.judgeModeStaticTitle, i18n.judgeModeStaticDesc, !agentMode)
    private val cardAgent  = modeCard("🤖", i18n.judgeModeAgentTitle,  i18n.judgeModeAgentDesc,  agentMode)

    // ── Results area ──────────────────────────────────────────────────────────
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
        // Panneau vertical dans la colonne gauche — hauteur initiale 300px, resizable
        setWidthFull()
        style["height"]         = "300px"
        style["display"]        = "flex"
        style["flex-direction"] = "column"
        style["border-top"]     = "2px solid #e2e8f0"
        style["background"]     = "#ffffff"
        style["overflow"]       = "hidden"
        style["flex-shrink"]    = "0"
        style["position"]       = "relative"

        element.appendChild(Html(MD_CSS).element)

        // ── Style des boutons OAuth ───────────────────────────────────────────
        applyOAuthBtnStyle(btnGithubOAuth, "#24292f", "#ffffff")
        applyOAuthBtnStyle(btnAzureOAuth,  "#0078d4", "#ffffff")

        // ── Drag-to-resize handle (barre horizontale en haut du panneau) ──────
        addAttachListener {
            element.executeJs("""
                (function() {
                    const panel = this;
                    if (panel.querySelector('.jdh-resize')) return;
                    const h = document.createElement('div');
                    h.className = 'jdh-resize';
                    h.title = 'Drag pour redimensionner';
                    h.style.cssText = [
                        'width:100%',
                        'height:7px',
                        'cursor:ns-resize',
                        'user-select:none',
                        'flex-shrink:0',
                        'background:linear-gradient(transparent 40%,#cbd5e1 50%,transparent 60%)',
                        'transition:background .15s',
                        'z-index:10',
                    ].join(';');
                    h.addEventListener('mouseenter', () => h.style.background = 'linear-gradient(transparent 30%,#6366f1 50%,transparent 70%)');
                    h.addEventListener('mouseleave', () => { if(!h._dragging) h.style.background = 'linear-gradient(transparent 40%,#cbd5e1 50%,transparent 60%)'; });
                    let sY=0, sH=0;
                    h.addEventListener('mousedown', function(e) {
                        h._dragging = true;
                        sY = e.clientY; sH = panel.getBoundingClientRect().height;
                        e.preventDefault();
                        h.style.background = 'linear-gradient(transparent 30%,#6366f1 50%,transparent 70%)';
                        const mv = e2 => {
                            const n = Math.max(120, Math.min(window.innerHeight * 0.75, sH + (sY - e2.clientY)));
                            panel.style.height = n + 'px';
                            panel.style.flex = 'none';
                        };
                        const up = () => {
                            h._dragging = false;
                            h.style.background = 'linear-gradient(transparent 40%,#cbd5e1 50%,transparent 60%)';
                            document.removeEventListener('mousemove', mv);
                            document.removeEventListener('mouseup', up);
                        };
                        document.addEventListener('mousemove', mv);
                        document.addEventListener('mouseup', up);
                    });
                    panel.insertBefore(h, panel.firstChild);
                }).call(this);
            """.trimIndent())
        }

        // ── Wiring provider preset → URL + modèles + sections auth ──────────
        fun applyProviderPreset(preset: ProviderPreset) {
            // Ne pas écraser une URL déjà personnalisée pour Azure
            if (!preset.isAzure || tfUrl.value.isBlank() || tfUrl.value == preset.url)
                tfUrl.value = preset.url
            val (models, default) = providerModels(preset.url)
            cbModel.setItems(models)
            if (cbModel.value.isNullOrBlank() || !models.contains(cbModel.value))
                cbModel.value = default
            val isGhCopilot = preset.name == "GitHub Copilot"
            val isGhModels  = preset.name == "GitHub Models"
            val isAzure     = preset.isAzure
            // GitHub
            tfGhClientId.isVisible   = isGhCopilot
            btnGithubOAuth.isVisible  = isGhCopilot
            ghPatNote.isVisible       = isGhCopilot || isGhModels
            // Azure
            azureNote.isVisible       = isAzure
            tfAzureTenantId.isVisible = isAzure
            tfAzureClientId.isVisible = isAzure
            btnAzureOAuth.isVisible   = isAzure
            // Label clé
            pfKey.label = when {
                isGhCopilot || isGhModels -> "GitHub Token"
                isAzure -> "Azure API Key (ou laisser vide pour AD)"
                else    -> ui18n.judgeKeyLabel
            }
        }
        cbProvider.addValueChangeListener { e ->
            val preset = e.value ?: return@addValueChangeListener
            applyProviderPreset(preset)
        }
        // Initialise la section GitHub si un preset GitHub est déjà sauvegardé
        cbProvider.value?.let { applyProviderPreset(it) }

        tfUrl.addValueChangeListener { e ->
            val (models, default) = providerModels(e.value)
            cbModel.setItems(models)
            if (cbModel.value.isNullOrBlank() || !models.contains(cbModel.value))
                cbModel.value = default
        }
        cardStatic.addClickListener { selectMode(agent = false) }
        cardAgent.addClickListener  { selectMode(agent = true)  }

        // ── Header ────────────────────────────────────────────────────────────
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
            style["padding"]       = "5px 10px"
            style["border-bottom"] = "1px solid #e2e8f0"
            style["background"]    = "#f8fafc"
            style["flex-shrink"]   = "0"
        }

        // ── Config compacte ───────────────────────────────────────────────────
        val modeRow = HorizontalLayout(cardStatic, cardAgent).apply {
            setWidthFull(); isPadding = false; isSpacing = true; style["gap"] = "5px"
        }
        val configBody = VerticalLayout(
            cbProvider, tfUrl,
            ghPatNote,
            tfGhClientId, btnGithubOAuth,
            azureNote,
            tfAzureTenantId, tfAzureClientId, btnAzureOAuth,
            pfKey, cbModel,
            sectionLabel(i18n.judgeModeSection),
            modeRow,
            spinner,
            btnSend,
        ).apply {
            setWidthFull(); isPadding = true; isSpacing = false
            style["gap"] = "4px"
            style["padding"] = "8px 10px"
            style["flex-shrink"] = "0"
            style["overflow-y"] = "auto"
            style["max-height"] = "55%"
            setHorizontalComponentAlignment(FlexComponent.Alignment.STRETCH, btnSend)
        }

        // ── Verdict (prend le reste de la hauteur) ────────────────────────────
        val traceLabel   = sectionLabel("🔧 ${i18n.judgeTraceSection}").apply { isVisible = false }
        val verdictLabel = sectionLabel("📝 ${i18n.judgeVerdictSection}")
        this.traceLabel  = traceLabel

        val verdictWrapper = Div().apply {
            style["display"]        = "flex"
            style["flex-direction"] = "column"
            style["flex"]           = "1"
            style["min-height"]     = "0"
            style["overflow"]       = "hidden"
            style["padding"]        = "4px 10px 8px"
            style["box-sizing"]     = "border-box"
        }
        verdictWrapper.add(traceLabel, traceArea, verdictLabel, verdictArea)

        add(colHeader, configBody, verdictWrapper)
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

    // ── GitHub Device Flow ────────────────────────────────────────────────────
    private fun startGithubDeviceFlow() {
        val clientId = tfGhClientId.value.trim()
        if (clientId.isBlank()) {
            Notification.show("Renseignez le Client ID de votre GitHub OAuth App.", 4000, Notification.Position.MIDDLE)
            return
        }
        val ui = UI.getCurrent()

        // Dialog avec code utilisateur
        val dialog = Dialog()
        dialog.headerTitle = "🐙 Connexion GitHub — Code de vérification"
        dialog.isCloseOnOutsideClick = false

        val codeSpan = Span("…").apply {
            style["font-size"]      = "2.2em"
            style["font-weight"]    = "900"
            style["letter-spacing"] = "0.18em"
            style["color"]          = "#1d4ed8"
            style["font-family"]    = "monospace"
        }
        val linkDiv = Div()
        val statusSpan = Span("En attente de l'autorisation…").apply {
            style["color"]     = "#64748b"
            style["font-size"] = "0.85em"
        }
        val progressDlg = ProgressBar().apply { isIndeterminate = true; setWidthFull() }
        dialog.add(VerticalLayout(
            Div(Span("Rendez-vous sur "), linkDiv, Span(" et saisissez ce code :")),
            codeSpan, progressDlg, statusSpan,
        ).apply { isPadding = true; isSpacing = true; alignItems = FlexComponent.Alignment.CENTER })
        val btnCancel = Button("Annuler") { dialog.close() }
        dialog.footer.add(btnCancel)
        dialog.open()

        bgExec.submit {
            try {
                val dc = startDeviceFlow(clientId)
                ui.access {
                    codeSpan.text = dc.userCode
                    val link = Anchor(dc.verificationUri, dc.verificationUri)
                    link.setTarget("_blank")
                    link.style["color"] = "#1d4ed8"; link.style["font-weight"] = "700"
                    linkDiv.removeAll(); linkDiv.add(link)
                }
                // Poll pour le token OAuth
                val oauthToken = pollOAuthToken(clientId, dc.deviceCode, dc.interval)

                // Tenter l'échange Copilot
                ui.access { statusSpan.text = "✅ Connecté — récupération du token Copilot…" }
                val copilotToken = exchangeForCopilotToken(oauthToken)
                val finalToken = copilotToken ?: oauthToken

                ui.access {
                    pfKey.value = finalToken
                    dialog.close()
                    val msg = if (copilotToken != null)
                        "✅ Token Copilot obtenu et renseigné !"
                    else
                        "✅ Token GitHub renseigné (Copilot non disponible ou PAT direct)."
                    Notification.show(msg, 4000, Notification.Position.BOTTOM_START)
                }
            } catch (e: Exception) {
                ui.access {
                    statusSpan.text = "❌ ${e.message}"
                    progressDlg.isVisible = false
                    Notification.show("Erreur GitHub OAuth : ${e.message}", 5000, Notification.Position.MIDDLE)
                }
            }
        }
    }

    // ── Azure AD Device Flow ──────────────────────────────────────────────────
    private fun startAzureDeviceFlow() {
        val tenantId  = tfAzureTenantId.value.trim().ifBlank { "common" }
        val clientId  = tfAzureClientId.value.trim()
        if (clientId.isBlank()) {
            Notification.show("Renseignez le Client ID de votre Azure App Registration.", 4000, Notification.Position.MIDDLE)
            return
        }
        // Scope Azure Cognitive Services (couvre Azure OpenAI)
        val scope = "https://cognitiveservices.azure.com/.default"
        val ui = UI.getCurrent()

        val dialog = Dialog()
        dialog.headerTitle = "☁️ Connexion Microsoft — Code de vérification"
        dialog.isCloseOnOutsideClick = false

        val codeSpan = Span("…").apply {
            style["font-size"]      = "2.2em"
            style["font-weight"]    = "900"
            style["letter-spacing"] = "0.18em"
            style["color"]          = "#0078d4"
            style["font-family"]    = "monospace"
        }
        val linkDiv  = Div()
        val statusSpan = Span("En attente de l'autorisation…").apply {
            style["color"] = "#64748b"; style["font-size"] = "0.85em"
        }
        val progressDlg = ProgressBar().apply { isIndeterminate = true; setWidthFull() }
        dialog.add(VerticalLayout(
            Div(Span("Rendez-vous sur "), linkDiv, Span(" et saisissez ce code :")),
            codeSpan, progressDlg, statusSpan,
        ).apply { isPadding = true; isSpacing = true; alignItems = FlexComponent.Alignment.CENTER })
        dialog.footer.add(Button("Annuler") { dialog.close() })
        dialog.open()

        bgExec.submit {
            try {
                val dc = startAzureDeviceCodeFlow(tenantId, clientId, scope)
                ui.access {
                    codeSpan.text = dc.userCode
                    val link = Anchor(dc.verificationUri, dc.verificationUri)
                    link.setTarget("_blank")
                    link.style["color"] = "#0078d4"; link.style["font-weight"] = "700"
                    linkDiv.removeAll(); linkDiv.add(link)
                }
                val token = pollAzureAccessToken(tenantId, clientId, dc.deviceCode, dc.interval)
                ui.access {
                    pfKey.value = token
                    dialog.close()
                    Notification.show("✅ Token Azure AD obtenu et renseigné !", 4000, Notification.Position.BOTTOM_START)
                }
            } catch (e: Exception) {
                ui.access {
                    statusSpan.text = "❌ ${e.message}"
                    progressDlg.isVisible = false
                    Notification.show("Erreur Azure OAuth : ${e.message}", 5000, Notification.Position.MIDDLE)
                }
            }
        }
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
                    val now = System.currentTimeMillis()
                    if (now - lastUpdate < 66) return@judgeStream
                    lastUpdate = now
                    ui.access {
                        if (isTrace) {
                            traceArea.removeAll()
                            traceArea.add(Span(partial).apply { style["white-space"] = "pre-wrap" })
                            traceArea.isVisible = true; traceLabel.isVisible = true
                        } else {
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

    private fun renderMarkdown(text: String, target: Div) {
        val html = mdRenderer.render(mdParser.parse(text))
        target.removeAll()
        target.add(Html("""<div class="llm-md">$html</div>"""))
    }

    private fun sectionLabel(text: String) = Div(Span(text)).apply {
        style["font-size"]      = "0.67em"
        style["font-weight"]    = "700"
        style["letter-spacing"] = "0.08em"
        style["color"]          = "#94a3b8"
        style["text-transform"] = "uppercase"
        style["padding-bottom"] = "2px"
    }

    /** Applique un style unifié aux boutons d'authentification OAuth. */
    private fun applyOAuthBtnStyle(btn: Button, bg: String, fg: String) {
        btn.style["background"]     = bg
        btn.style["color"]          = fg
        btn.style["border"]         = "none"
        btn.style["border-radius"]  = "7px"
        btn.style["font-size"]      = "0.82em"
        btn.style["font-weight"]    = "600"
        btn.style["height"]         = "36px"
        btn.style["cursor"]         = "pointer"
        btn.style["transition"]     = "opacity .15s"
    }

    private fun emptyState() = Div(Span(i18n.detailPlaceholder)).apply {
        style["color"]       = "#94a3b8"
        style["font-size"]   = "0.86em"
        style["padding-top"] = "20px"
        style["text-align"]  = "center"
    }
}
