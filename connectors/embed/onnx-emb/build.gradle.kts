plugins {
    id("io.spring.dependency-management")
    `java-library`
}
dependencies {
    compileOnly(libs.onnx.runtime)
    implementation(libs.djl.tokenizers)
    implementation(projects.ragEngine)
    implementation(projects.ragModel)
    api(libs.spring.boot.autoconfigure)
    implementation(libs.spring.boot.starter.validation)
    testImplementation(kotlin("test"))
}


tasks.test {
    useJUnitPlatform()
}