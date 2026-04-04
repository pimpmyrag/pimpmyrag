
java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

repositories { mavenCentral() }

dependencies {
    implementation(libs.kotlin.stdlib)
    implementation(libs.kmongo)
    implementation(libs.kmongo.property)
    implementation(libs.kmongo.coroutine)
    implementation(libs.jackson.module.kotlin)
    implementation(libs.slf4j.simple)
    implementation(projects.ragModel)
    implementation(projects.ragDsl)
    implementation(projects.ragEngine)
}
