package rag.demo

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
open class NerDemoApp

fun main(args: Array<String>) {
    runApplication<NerDemoApp>(*args)
}

