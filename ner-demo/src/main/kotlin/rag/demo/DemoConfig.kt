package rag.demo

import com.fasterxml.jackson.annotation.JsonIgnoreProperties

/** Configuration dynamique de la démo NER — exportable/importable en JSON. */
@JsonIgnoreProperties(ignoreUnknown = true)
data class DemoConfig(

    // ── Seuils d'inférence NER ──────────────────────────────────────────────
    val tauBoundary:    Float = 0.70f,
    val tauNone:        Float = 0.99f,
    val tauCoarse:      Float = 0.45f,
    val tauSvoBoundary: Float = 0.50f,

    // ── Batch + modèle ──────────────────────────────────────────────────────
    val batchSize: Int = 8,

    // ── Affichage ────────────────────────────────────────────────────────────
    val showSvo:       Boolean = true,
    val showArcs:      Boolean = true,
    val autoSplit:     Boolean = true,
    /** Catégories coarse pour lesquelles on affiche le label FIN (pas juste coarse). */
    val fineForCoarse: Set<String> = setOf("PER","LOC","ORG","TIME","EVENT","VALUE","OBJECT","ABSTRACT"),

    // ── Réconciliation NER ↔ SVO ────────────────────────────────────────────
    val doReconcile:            Boolean = true,
    val minNerScoreReconcile:   Float   = 0.50f,
    val minNerScoreFill:        Float   = 0.60f,
    val maxGapChars:            Int     = 120,
)

