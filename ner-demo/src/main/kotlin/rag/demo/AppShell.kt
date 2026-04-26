package rag.demo

import com.vaadin.flow.component.page.AppShellConfigurator
import com.vaadin.flow.component.page.Push
import com.vaadin.flow.server.AppShellSettings

/** Active le server-push WebSocket pour le streaming des résultats NER. */
@Push
class AppShell : AppShellConfigurator {
    override fun configurePage(settings: AppShellSettings) {
        settings.addMetaTag("viewport", "width=device-width, initial-scale=1")
    }
}

