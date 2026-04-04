package com.acme.infinity.client

/**
 * Interface for the Infinity client to allow for different implementations (e.g., Spring WebClient, Ktor-client).
 */
interface IInfinityClient {
    suspend fun embed(texts: List<String>): List<FloatArray>
}
