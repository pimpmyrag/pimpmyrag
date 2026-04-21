package rag.connectors.ner.onnx

/**
 * Main de test standalone — pas de Spring, pas de contexte.
 *
 * Lancer depuis le projet Gradle :
 *   ./gradlew :connectors:ner:onnx-ner:run -PmainClass=rag.connectors.ner.onnx.TestMultiHeadExtractorKt
 *
 * Ou directement depuis IntelliJ (Run gutter sur `fun main`).
 */

private const val MODEL_DIR = "/Users/simon_longuet/IdeaProjects/pimpmyrag/models/deberta/fine-tunning-21042026"
private const val MODEL_ONNX = "$MODEL_DIR/best_model_multitask.onnx"
private const val TOKENIZER_DIR = "/Users/simon_longuet/IdeaProjects/pimpmyrag/deberta/tokenizer_export"

private val TEST_TEXTS = listOf(
    // ── Phrases originales ───────────────────────────────────────────────────
    "Emmanuel Macron s'est rendu hier à Berlin pour rencontrer le chancelier Olaf Scholz.",
    "La Banque centrale européenne a relevé ses taux d'intérêt de 25 points de base mardi.",
    "Apple a annoncé le lancement de l'iPhone 17 le 15 septembre 2025 à Cupertino.",
    "Le tremblement de terre de magnitude 6,8 a touché la côte nord du Maroc.",
    "Le PSG a battu le Real Madrid 3-1 lors de la finale de la Ligue des champions.",
    "L'Assemblée nationale a adopté la loi sur le financement de la sécurité sociale.",
    "Le vaccin contre la grippe est disponible en pharmacie depuis le 1er octobre.",
    "Tesla a livré 500 000 véhicules électriques au troisième trimestre, un record.",

    // ── Personnes & rôles ────────────────────────────────────────────────────
    "Le président Joe Biden a signé le décret le 3 novembre 2024 à Washington.",
    "La chercheuse Marie Curie a reçu deux prix Nobel, en physique et en chimie.",
    "Le général de Gaulle a lancé son appel depuis Londres le 18 juin 1940.",
    "Ursula von der Leyen, présidente de la Commission européenne, a pris la parole à Bruxelles.",
    "Le PDG d'Airbus, Guillaume Faury, a annoncé 2 000 recrutements supplémentaires.",
    "L'actrice Juliette Binoche a reçu la Palme d'or à Cannes en 1997.",
    "Le Premier ministre britannique Rishi Sunak a démissionné lundi matin.",

    // ── Organisations ────────────────────────────────────────────────────────
    "L'ONU a convoqué une réunion d'urgence du Conseil de sécurité vendredi.",
    "L'Agence européenne des médicaments a approuvé un nouveau traitement contre le cancer du poumon.",
    "Le Fonds monétaire international prévoit une croissance mondiale de 3,2 % en 2026.",
    "Google a racheté la start-up française Nabla pour 400 millions d'euros.",
    "La SNCF a annoncé la suppression de 200 trains ce week-end en raison d'une grève.",
    "Interpol a arrêté le chef d'un réseau de cybercriminalité à Bucarest.",
    "L'OTAN a déployé 5 000 soldats supplémentaires en Pologne.",

    // ── Lieux ───────────────────────────────────────────────────────────────
    "L'éruption du Vésuve a détruit la ville de Pompéi en l'an 79.",
    "Les négociations se sont tenues au palais de l'Élysée.",
    "Le tunnel sous la Manche relie Folkestone à Coquelles.",
    "Les inondations ont touché la vallée de la Loire et le département de la Vendée.",
    "Le sommet du G20 se tiendra à Rio de Janeiro en novembre.",
    "Un incendie s'est déclaré dans une usine chimique près de Lyon lundi soir.",

    // ── Dates & durées ──────────────────────────────────────────────────────
    "La conférence est prévue du 12 au 15 mars 2026 à Genève.",
    "Les travaux devraient durer trois ans et demi avant la livraison du bâtiment.",
    "Le couvre-feu a été levé à 6 heures du matin par les autorités locales.",
    "L'accord a été signé le vendredi 7 février 2025 à 14h30.",
    "Depuis le début de l'année, plus de 10 000 demandes d'asile ont été enregistrées.",

    // ── Valeurs & quantités ──────────────────────────────────────────────────
    "Le gouvernement prévoit un déficit de 5,4 % du PIB pour 2025.",
    "Le baril de pétrole Brent a dépassé les 90 dollars jeudi.",
    "La facture d'électricité moyenne a augmenté de 15 % en un an.",
    "L'entreprise a dégagé un bénéfice net de 2,3 milliards d'euros au premier semestre.",
    "La vitesse moyenne relevée sur l'autoroute était de 142 km/h.",

    // ── Événements ──────────────────────────────────────────────────────────
    "Les Jeux olympiques de Paris 2024 ont réuni 206 nations.",
    "L'explosion s'est produite dans le port de Beyrouth le 4 août 2020.",
    "La guerre en Ukraine a éclaté le 24 février 2022 après l'invasion russe.",
    "La finale de la Coupe du monde de rugby a opposé l'Afrique du Sud à la Nouvelle-Zélande.",
    "Le Championnat d'Europe de football se tiendra en Allemagne en 2028.",

    // ── Droit & abstractions ─────────────────────────────────────────────────
    "Le Conseil constitutionnel a censuré plusieurs articles de la loi immigration.",
    "La directive européenne sur l'intelligence artificielle entre en vigueur le 1er août.",
    "L'article 49-3 a été utilisé pour faire passer le budget sans vote.",
    "Le règlement général sur la protection des données impose des sanctions pouvant atteindre 4 % du chiffre d'affaires mondial.",

    // ── Santé & sciences ─────────────────────────────────────────────────────
    "Les chercheurs de l'Institut Pasteur ont identifié un nouveau variant du virus Ebola.",
    "L'Inserm recommande une dose de rappel du vaccin contre la grippe tous les ans.",
    "Un essai clinique de phase III a montré une efficacité de 87 % contre le paludisme.",
    "L'épidémie de rougeole a touché plus de 3 000 enfants dans la région des Grands Lacs.",

    // ── Contractions & cas limites ───────────────────────────────────────────
    "L'aéroport Charles-de-Gaulle enregistre 70 millions de passagers par an.",
    "D'après le ministre de l'Économie, l'inflation devrait redescendre sous 2 % d'ici juin.",
    "L'hôpital Necker accueille chaque année plus de 50 000 enfants en urgence.",
    "C'est à l'université de Bordeaux que le Prix Nobel de chimie a été annoncé.",
)

fun main() {
    println("═══════════════════════════════════════════════════════════")
    println("  Test OnnxMultiHeadEntityExtractor")
    println("  Modèle   : $MODEL_ONNX")
    println("  Tokenizer: $TOKENIZER_DIR")
    println("═══════════════════════════════════════════════════════════\n")

    // Seuils de production — élimine la majorité des faux positifs (verbes, adjectifs)
    val extractor = OnnxMultiHeadEntityExtractor(
        modelPath    = MODEL_ONNX,
        tokenizerDir = TOKENIZER_DIR,
        maxSeqLen    = 128,
        maxSpanLen   = 8,
        tauBoundary  = 0.40f,
        tauNone      = 0.99f,
        tauCoarse    = 0.20f,
        minScore     = 0.35f,
    )

    extractor.use { ext ->
        // ── Test unitaire ────────────────────────────────────────────
        TEST_TEXTS.forEachIndexed { i, text ->
            val t0 = System.nanoTime()
            val entities = ext.extractFromText(text)
            val ms = (System.nanoTime() - t0) / 1_000_000L

            println("[$i] \"$text\"")
            println("    → ${entities.size} entité(s) en ${ms}ms")
            entities.forEach { e ->
                val coarse = e.metadata["coarse"]
                val kind   = e.metadata["kind"]
                val score  = (e.metadata["score"] as? Float)?.let { "%.3f".format(it) } ?: "?"
                println("      • [${e.span?.start}-${e.span?.end}] \"${e.text}\"  type=${e.type}  kind=$kind  coarse=$coarse  score=$score")
            }
            println()
        }

        // ── Test batch ───────────────────────────────────────────────
        println("─── Batch (${TEST_TEXTS.size} textes) ───────────────────────────")
        val t0 = System.nanoTime()
        val batchResults = ext.extractFromTexts(TEST_TEXTS)
        val ms = (System.nanoTime() - t0) / 1_000_000L
        val total = batchResults.sumOf { it.size }
        println("Batch terminé en ${ms}ms — $total entités extraites au total")
        println("Moyenne: ${"%.1f".format(ms.toDouble() / TEST_TEXTS.size)} ms/texte\n")

        // ── Récapitulatif par type ───────────────────────────────────
        println("─── Récapitulatif par type fine ─────────────────────────")
        batchResults.flatten()
            .groupBy { it.type }
            .entries
            .sortedByDescending { it.value.size }
            .forEach { (type, list) ->
                println("  %-30s : %d".format(type, list.size))
            }
    }

    println("\n✅ Test terminé.")
}

