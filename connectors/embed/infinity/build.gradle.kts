plugins {
    id("io.spring.dependency-management")
    `java-library`
}

version = "0.1.0"
java.sourceCompatibility = JavaVersion.VERSION_21

repositories { mavenCentral() }

//dependencyManagement {
//    imports {
//        mavenBom("org.springframework.boot:spring-boot-dependencies:3.3.5")
//    }
//}

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

    // Tests
    testImplementation(libs.spring.boot.starter.test)
    testImplementation(libs.junit.jupiter)
}


tasks.test { useJUnitPlatform() }
