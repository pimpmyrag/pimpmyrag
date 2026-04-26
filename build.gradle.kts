plugins {
    kotlin("jvm") version "1.9.24" apply false
    kotlin("plugin.spring") version "1.9.24" apply false
    id("com.google.protobuf") version "0.9.4" apply false
    id("org.springframework.boot") version "3.3.5" apply false
    id("io.spring.dependency-management") version "1.1.6" apply false
    id("com.vaadin") version "24.7.2" apply false
}

//configurations.all {
//    resolutionStrategy {
//        force("org.springframework.boot:spring-boot-starter-webflux:3.2.5")
//    }
//}
//subprojects {
//    if(project.name == "rag-app") {
//        val copyModels = tasks.register<Copy>("copyModels") {
//            from("docker-resources/onnx_converter_bundle/models") {
//                include("**/*")
//            }
//            into("src/main/resources/models")
//            doLast {
//                println("✅ Modèles ONNX copiés vers src/main/resources/models")
//            }
//        }
//
//        pluginManager.withPlugin("java") {
//            tasks.named("processResources") {
//                dependsOn(copyModels)
//            }
//        }
//    }
//}



//tasks.named("processResources") {
//    dependsOn(copyModels)
//}

tasks.register("generateToml", GenerateToml::class)

allprojects {
    apply(plugin = "org.jetbrains.kotlin.jvm")

    group = "com.pimpmyrag"
    version = "1.0.0"

    extensions.configure<org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension> {
        jvmToolchain(21)
    }

    tasks.withType<Jar> {
        archiveBaseName.set(project.path.substring(1).replace(":", "-"))
    }
    repositories {
        mavenCentral()
    }

    tasks.withType<Test> {
        useJUnitPlatform()
    }

    dependencies {
        "testRuntimeOnly"("org.junit.platform:junit-platform-launcher:1.10.2")
    }
}