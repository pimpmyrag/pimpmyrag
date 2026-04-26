plugins {
    kotlin("jvm")
    kotlin("plugin.spring")
    id("org.springframework.boot")
    id("io.spring.dependency-management")
    id("com.vaadin")
}

dependencyManagement {
    imports {
        mavenBom("com.vaadin:vaadin-bom:24.6.3")
    }
}

dependencies {
    implementation("com.vaadin:vaadin-spring-boot-starter")
    implementation(project(":connectors:ner:onnx-ner"))
    implementation(libs.onnx.runtime)
    implementation(libs.djl.tokenizers)
    implementation(libs.spring.boot.autoconfigure)
    implementation("org.springframework.boot:spring-boot-starter")
    implementation(kotlin("stdlib"))
    implementation(kotlin("reflect"))
}

tasks.test {
    useJUnitPlatform()
}

