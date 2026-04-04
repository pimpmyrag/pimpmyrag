plugins {
    kotlin("jvm")
    application
    id("org.springframework.boot")
    id("io.spring.dependency-management")
    kotlin("plugin.spring")
}


dependencies {
    implementation(libs.jackson.module.kotlin)
    implementation(libs.spring.boot.starter.webflux)
    implementation(libs.kotlinx.coroutines.core)
//    implementation(libs.slf4j.simple)
//    implementation(projects.connectors.embed.infinity)
//    implementation(projects.connectors.rerank.onnxCe)
    implementation(projects.connectors.ud.msUd)
    implementation(projects.connectors.ner.onnxNer)
    implementation(projects.connectors.embed.onnxEmb)
    implementation(projects.ragEngine)
    implementation(projects.ragModel)

    runtimeOnly(libs.onnx.runtime)
//    runtimeOnly(libs.onnxruntime.native)

    testImplementation(kotlin("test"))
    testImplementation(libs.junit.jupiter)
    testImplementation(libs.spring.boot.starter.test)
}

tasks.withType<Jar> {
    archiveBaseName.set(project.path.substring(1).replace(":", "-"))
}