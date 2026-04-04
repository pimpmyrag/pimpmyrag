plugins {
    kotlin("jvm")
}

group = "com.pimpmyrag"
version = "1.0.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation("io.qdrant:client:1.16.2")
    implementation(project(":rag-engine"))
    implementation(project(":rag-model"))
    testImplementation(kotlin("test"))
}

kotlin {
    jvmToolchain(21)
}

tasks.test {
    useJUnitPlatform()
}