import org.gradle.api.*
import org.gradle.api.tasks.*

abstract class GenerateToml : DefaultTask() {

    @TaskAction
    fun generate() {
        val libs = mutableMapOf<String, String>()

        project.allprojects.forEach { sub ->
            sub.configurations.forEach config@{ cfg ->
                cfg.dependencies.forEach { dep ->
                    if (dep.version != null && dep.group != null) {
                        libs["${dep.group}:${dep.name}"] = dep.version!!
                    }
                }
            }
        }

        val toml = buildString {
            appendLine("[versions]")
            libs.values.toSet().sorted().forEach {
                appendLine("""v$it = "$it"""")
            }
            appendLine()
            appendLine("[libraries]")
            libs.entries.sortedBy { it.key }.forEach { (g, v) ->
                val (group, name) = g.split(":")
                appendLine("""${name.replace("-", "_")} = { group = "$group", name = "$name", version = "v$v" }""")
            }
        }

        val file = project.file("gradle/libs.generated.toml")
        file.writeText(toml)

        println("Generated → gradle/libs.generated.toml")
    }
}