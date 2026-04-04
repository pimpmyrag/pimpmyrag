package rag.connectors.ud.stanza.config

import org.springframework.boot.context.properties.ConfigurationProperties

@ConfigurationProperties(prefix = "ud.client")
data class UdClientProperties(
    /** Host (default localhost) */
    val host: String = "127.0.0.1",
    /** Port (default 8000) */
    val port: Int = 8000,
    /** Base path on the server (default /ud) */
    val basePath: String = "/ud",
    /** HTTP connect timeout in ms */
    val connectTimeoutMs: Long = 10_000,
    /** HTTP socket/request timeout in ms */
    val socketTimeoutMs: Long = 60_000,
    /** Batch size used by client when using batch endpoints */
    val batchSize: Int = 16
)
