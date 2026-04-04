package io

import org.springframework.boot.CommandLineRunner
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.context.properties.ConfigurationPropertiesScan
import org.springframework.boot.runApplication


@SpringBootApplication(scanBasePackages = ["com", "io"])
@ConfigurationPropertiesScan(basePackages = ["com", "io"])
class OpenRagIngestionApplication: CommandLineRunner {

    override fun run(vararg args: String?) {
        println("OpenRagIngestionApplication started...")
    }
}


fun main(args: Array<String>) {
    runApplication<OpenRagIngestionApplication>(*args)
}

