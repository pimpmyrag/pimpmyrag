plugins {
    id("io.spring.dependency-management")
    `java-library`
}


dependencies {
    implementation(libs.reactor.kotlin.extensions)
    implementation(libs.kotlin.reflect)
    api(libs.spring.boot.autoconfigure)
    api(libs.spring.boot.starter.webflux)
    implementation(libs.jackson.module.kotlin)
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.coroutines.reactor)
    implementation(libs.reactor.kotlin.extensions)
    implementation(projects.ragModel)
    implementation(projects.ragEngine)

    // Validation for @ConfigurationProperties
    implementation(libs.spring.boot.starter.validation)


    // Validation for @ConfigurationProperties

    // Tests
    testImplementation(libs.spring.boot.starter.test)
    testImplementation(libs.junit.jupiter)
}

tasks.test { useJUnitPlatform() }
