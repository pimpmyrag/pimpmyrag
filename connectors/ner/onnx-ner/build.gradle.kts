plugins {
    kotlin("jvm")
    kotlin("plugin.spring")
}


dependencies {
    compileOnly(libs.onnx.runtime)
    implementation(libs.djl.tokenizers)
    implementation(projects.ragEngine)
    implementation(projects.ragModel)
    api(libs.spring.boot.autoconfigure)
    implementation(libs.spring.boot.starter.validation)
    testImplementation(kotlin("test"))
    testImplementation(kotlin("test"))
    implementation(kotlin("stdlib"))
}


tasks.test {
    useJUnitPlatform()
}
repositories {
    mavenCentral()
}