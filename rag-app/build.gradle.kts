plugins {
    id("org.springframework.boot")
    id("io.spring.dependency-management")
    kotlin("plugin.spring")
}

dependencyManagement {
    imports {
        mavenBom("org.springframework.boot:spring-boot-dependencies:3.3.5")
    }
}

dependencies {
    implementation(libs.kotlin.stdlib)
    implementation(projects.ragDsl)
    implementation(projects.ragModel)
    implementation(projects.ragEngine)
    implementation(projects.ragPlanner)
    implementation(projects.ragDslStaged)
    implementation(projects.ragRunner)
    implementation(projects.connectors.ragConnectorsStub)
    implementation(projects.connectors.documentStore.mongodb)
    implementation(projects.connectors.vector.qadrant)
    implementation(projects.connectors.embed.infinity)
    implementation(projects.connectors.rerank.infinity)
}
repositories {
    mavenCentral()
}