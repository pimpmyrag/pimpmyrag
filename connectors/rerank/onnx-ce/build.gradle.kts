plugins {
    kotlin("jvm")
}


dependencies {
    compileOnly(libs.onnx.runtime)
    implementation(libs.djl.tokenizers)
    implementation(projects.ragEngine)
    implementation(projects.ragModel)
    testImplementation(kotlin("test"))
    testImplementation(kotlin("test"))
}


tasks.test {
    useJUnitPlatform()
}