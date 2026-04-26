enableFeaturePreview("TYPESAFE_PROJECT_ACCESSORS")
pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
        google()
    }
    plugins {
        kotlin("jvm")
    }
}

dependencyResolutionManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
        google()
    }
}


rootProject.name = "pimpmyrag"
include(
    ":rag-dsl",
    ":rag-model",
    ":rag-engine",
    ":rag-planner",
    ":rag-app",
    ":rag-runner",
    ":rag-dsl-staged",
    "connectors:rag-connectors-stub",
    "connectors:ner:onnx-ner",
    "connectors:rerank:infinity",
    "connectors:rerank:onnx-ce",
    "connectors:ud:ms-ud",
    "connectors:embed:infinity",
    "connectors:embed:onnx-emb",
    "connectors:document-store:mongodb",
    "connectors:vector:qadrant",
    ":radar-nli-toolkit",
    ":ner-demo",

)