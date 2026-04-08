plugins {
    kotlin("jvm")
    kotlin("plugin.spring")
}


dependencies {
    implementation(libs.onnx.runtime)
    testCompileOnly(libs.onnx.runtime)   // helper.kt imports OnnxTensor in compile scope
    implementation(libs.djl.tokenizers)
    implementation(projects.ragEngine)
    implementation(projects.ragModel)
    api(libs.spring.boot.autoconfigure)
    implementation(libs.spring.boot.starter.validation)
    implementation(kotlin("stdlib"))
    testImplementation(kotlin("test"))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}


tasks.test {
    useJUnitPlatform()
}
