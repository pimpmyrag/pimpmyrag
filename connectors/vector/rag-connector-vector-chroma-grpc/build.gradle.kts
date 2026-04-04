plugins {
    kotlin("jvm")
    id("com.google.protobuf")
}

repositories {
    mavenCentral()
}


dependencies {
    implementation("io.grpc:grpc-netty-shaded:1.62.2")
    implementation("io.grpc:grpc-protobuf:1.62.2")
    implementation("io.grpc:grpc-stub:1.62.2")
//    implementation("io.grpc:grpc-kotlin-stub:1.5.0")
    implementation("com.google.protobuf:protobuf-kotlin:3.25.1")
    implementation("com.fasterxml.jackson.module:jackson-module-kotlin:2.21.0")
//    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    implementation("javax.annotation:javax.annotation-api:1.3.2")
}
protobuf {
    protoc {
        artifact = "com.google.protobuf:protoc:3.25.1"
    }
    plugins {
        create("grpc") {
            artifact = "io.grpc:protoc-gen-grpc-java:1.62.2"
        }
//        create("grpckt") {
//            artifact = "io.grpc:protoc-gen-grpc-kotlin:1.5.0:jdk8@jar"
//        }
    }
    generateProtoTasks {
        all().forEach { task ->
            task.plugins {
                create("grpc")
//                create("grpckt")
            }
        }
    }
}

