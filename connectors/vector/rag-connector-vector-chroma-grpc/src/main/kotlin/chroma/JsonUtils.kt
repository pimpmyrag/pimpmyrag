
package chroma

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper

object JsonUtils {
    private val mapper = jacksonObjectMapper()

    fun toJson(m: Map<String, Any>): String = mapper.writeValueAsString(m)

    fun parse(json: String): Map<String, Any> =
        mapper.readValue(json, Map::class.java) as Map<String, Any>
}
