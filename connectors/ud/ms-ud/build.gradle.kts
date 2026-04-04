plugins {
    kotlin("jvm")
    kotlin("plugin.spring")
}


dependencies {
    implementation(libs.jackson.module.kotlin)
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.spring.boot.starter.webflux)
    implementation(libs.kotlinx.coroutines.reactor)
    implementation(libs.reactor.kotlin.extensions)
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