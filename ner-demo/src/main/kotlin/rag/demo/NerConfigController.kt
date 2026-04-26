package rag.demo

import com.fasterxml.jackson.databind.ObjectMapper
import jakarta.servlet.http.HttpServletResponse
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.*
import org.springframework.web.multipart.MultipartFile

/**
 * REST API pour la config de la démo NER.
 *
 *  GET  /api/config          → config courante (JSON)
 *  PUT  /api/config          → met à jour la config (JSON body)
 *  GET  /api/config/export   → télécharge la config (JSON file)
 *  POST /api/config/import   → importe depuis un fichier JSON multipart
 */
@RestController
@RequestMapping("/api/config")
class NerConfigController(
    private val nerService: NerService,
    private val mapper: ObjectMapper,
) {

    @GetMapping(produces = [MediaType.APPLICATION_JSON_VALUE])
    fun getConfig(): DemoConfig = nerService.config

    @PutMapping(consumes = [MediaType.APPLICATION_JSON_VALUE],
                produces = [MediaType.APPLICATION_JSON_VALUE])
    fun updateConfig(@RequestBody cfg: DemoConfig): DemoConfig {
        nerService.updateConfig(cfg)
        return nerService.config
    }

    @GetMapping("/export")
    fun exportConfig(response: HttpServletResponse) {
        response.contentType = "application/json"
        response.setHeader("Content-Disposition", "attachment; filename=\"ner-demo-config.json\"")
        mapper.writerWithDefaultPrettyPrinter()
              .writeValue(response.outputStream, nerService.config)
    }

    @PostMapping("/import", consumes = [MediaType.MULTIPART_FORM_DATA_VALUE],
                             produces = [MediaType.APPLICATION_JSON_VALUE])
    fun importConfig(@RequestParam("file") file: MultipartFile): DemoConfig {
        val cfg = mapper.readValue(file.inputStream, DemoConfig::class.java)
        nerService.updateConfig(cfg)
        return nerService.config
    }
}

