plugins {
    kotlin("jvm")
    kotlin("plugin.spring")
    id("org.springframework.boot")
    id("io.spring.dependency-management")
    // NB : le plugin com.vaadin est intentionnellement absent ici (incompatible Gradle 9).
    // Il est injecté via sed dans le Dockerfile UNIQUEMENT lors du build Docker avec Gradle 8.13.
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
