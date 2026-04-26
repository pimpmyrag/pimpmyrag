package rag.demo

import org.springframework.ai.tool.ToolCallbackProvider
import org.springframework.ai.tool.method.MethodToolCallbackProvider
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

/**
 * Enregistre les outils MCP auprès du serveur Spring AI.
 * Le starter spring-ai-mcp-server-webmvc-spring-boot-starter expose automatiquement
 * les ToolCallback via SSE sur /mcp/sse (configurable dans application.properties).
 */
@Configuration
class McpConfig(private val nerMcpTools: NerMcpTools) {

    @Bean
    fun nerToolCallbackProvider(): ToolCallbackProvider =
        MethodToolCallbackProvider.builder()
            .toolObjects(nerMcpTools)
            .build()
}

