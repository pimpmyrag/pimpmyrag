plugins {
    kotlin("jvm")
    kotlin("plugin.spring")
    id("org.springframework.boot")
    id("io.spring.dependency-management")
    // Le plugin Vaadin est incompatible avec Gradle 9 (ResolvedConfiguration.getFiles() supprimé).
    // Pour le build Docker, on utilise temporairement Gradle 8.13 (voir ner-demo/Dockerfile).
    id("com.vaadin") version "24.7.2"
}

dependencyManagement {
    imports {
        mavenBom("org.springframework.ai:spring-ai-bom:1.0.0")
        mavenBom("com.vaadin:vaadin-bom:24.7.2")
    }
}

dependencies {
    implementation("com.vaadin:vaadin-spring-boot-starter")
    implementation(project(":connectors:ner:onnx-ner"))
    implementation(project(":rag-model"))
    implementation(project(":rag-engine"))
    implementation(libs.onnx.runtime)
    implementation(libs.djl.tokenizers)
    implementation(libs.icu4j)
    implementation(libs.spring.boot.autoconfigure)
    implementation("org.springframework.boot:spring-boot-starter-web")
    implementation(libs.jackson.module.kotlin)
    implementation(kotlin("stdlib"))
    implementation(kotlin("reflect"))
    // ── MCP server (SSE transport, Spring MVC) ────────────────────────────────
    implementation("org.springframework.ai:spring-ai-starter-mcp-server-webmvc")
    // ── Markdown rendering (LLM Judge verdict) ────────────────────────────────
    implementation("org.commonmark:commonmark:0.22.0")
    implementation("org.commonmark:commonmark-ext-gfm-tables:0.22.0")
    implementation("org.commonmark:commonmark-ext-gfm-strikethrough:0.22.0")
}

tasks.test {
    useJUnitPlatform()
}
