package rag.connectors.ner.onnx

/**
 * Main de test standalone — pas de Spring, pas de contexte.
 *
 * Lancer depuis le projet Gradle :
 *   ./gradlew :connectors:ner:onnx-ner:run -PmainClass=rag.connectors.ner.onnx.TestMultiHeadExtractorKt
 *
 * Ou directement depuis IntelliJ (Run gutter sur `fun main`).
 */

// Chemins relatifs à la racine du repo — override via propriétés système si besoin :
//   -Dner.model.dir=...  -Dner.tokenizer.dir=...
private val REPO_ROOT: String = System.getProperty("user.dir").let { cwd ->
    // Remonte jusqu'à la racine du repo (contient gradlew)
    java.io.File(cwd).let { f ->
        generateSequence(f) { it.parentFile }
            .firstOrNull { it.resolve("gradlew").exists() }?.absolutePath ?: cwd
    }
}
private val MODEL_DIR = System.getProperty("ner.model.dir", "$REPO_ROOT/models/deberta/fine-tunning-23042026")
private val MODEL_ONNX = "$MODEL_DIR/best_model_multitask_full.onnx"
private val TOKENIZER_DIR = System.getProperty("ner.tokenizer.dir", "$REPO_ROOT/training/multi-head/tokenizer_export_clean")

// Sources JSONL pour le bench 1000 phrases (même corpus que bench Python)
private val BENCH_SOURCES = listOf(
    "$REPO_ROOT/training/multi-head/data/abstract_sentences.jsonl",
    "$REPO_ROOT/training/multi-head/data/abstract_sentences_extra.jsonl",
    "$REPO_ROOT/training/multi-head/data/converted_no_coarse_1000.jsonl",
)

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

// ─────────────────────────────────────────────────────────────────────────────
// Utilitaires benchmark
// ─────────────────────────────────────────────────────────────────────────────

/** Charge les phrases depuis les fichiers JSONL + le fichier txt de test. */
private fun loadBenchPhrases(n: Int = 1000, seed: Long = 42): List<String> {
    val sentences = mutableListOf<String>()

    // Fichier txt de test manuel
    val richFile = java.io.File("$REPO_ROOT/training/multi-head/test_phrases_rich.txt")
    if (richFile.exists()) sentences += richFile.readLines().filter { it.isNotBlank() }

    // Fichiers JSONL — extraction du champ "text" par regex (pas de dépendance JSON)
    val textRegex = Regex(""""text"\s*:\s*"((?:[^"\\]|\\.)*)"""")
    for (path in BENCH_SOURCES) {
        val f = java.io.File(path)
        if (!f.exists()) continue
        f.forEachLine { line ->
            val match = textRegex.find(line) ?: return@forEachLine
            val text = match.groupValues[1]
                .replace("\\n", " ").replace("\\\"", "\"").replace("\\\\", "\\").trim()
            if (text.length > 20) sentences += text
        }
    }

    // Déduplique
    val unique = sentences.distinct()
    println("📂 ${unique.size} phrases sources chargées")

    // Sample avec remplacement si besoin
    val rng = java.util.Random(seed)
    return if (unique.size >= n) {
        unique.shuffled(rng).take(n)
    } else {
        val result = unique.toMutableList()
        while (result.size < n) result += unique[rng.nextInt(unique.size)]
        result.take(n)
    }
}

/** Affiche un triplet SVO joliment. */
private fun formatTriplet(t: SvoTriplet): String {
    val s = t.subject?.let { "\"${it.text}\"[${it.gender ?: "?"}/${it.number ?: "?"}]" } ?: "∅"
    val v = "\"${t.verb.text}\"[${t.verb.voice}]"
    val o = t.obj?.let { "\"${it.text}\"" } ?: "∅"
    return "$s →$v→ $o"
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fun main() {
    println("═══════════════════════════════════════════════════════════")
    println("  Test OnnxMultiHeadEntityExtractor — 8 têtes NER+SVO")
    println("  Modèle   : $MODEL_ONNX")
    println("  Tokenizer: $TOKENIZER_DIR")
    println("═══════════════════════════════════════════════════════════\n")

    val extractor = OnnxMultiHeadEntityExtractor(
        modelPath       = MODEL_ONNX,
        tokenizerDir    = TOKENIZER_DIR,
        maxSeqLen       = 128,
        maxSpanLen      = 8,
        tauBoundary     = 0.40f,
        tauNone         = 0.99f,
        tauCoarse       = 0.20f,
        minScore        = 0.35f,
        tauSvoBoundary  = 0.50f,
    )

    extractor.use { ext ->

        // ════════════════════════════════════════════════════════
        // 1. TEST UNITAIRE — phrases manuelles avec NER + SVO
        // ════════════════════════════════════════════════════════
        println("━━━ 1. TEST UNITAIRE (${TEST_TEXTS.size} phrases) ━━━━━━━━━━━━━━━━━━")
        TEST_TEXTS.forEachIndexed { i, text ->
            val t0  = System.nanoTime()
            val res = ext.extractWithSvo(text)
            val ms  = (System.nanoTime() - t0) / 1_000_000L

            println("\n[$i] \"$text\"")
            println("    ⏱  ${ms}ms  |  NER: ${res.entities.size}  SVO: ${res.svoSpans.size}")

            // NER
            res.entities.take(5).forEach { e ->
                val score = (e.metadata["score"] as? Float)?.let { "%.3f".format(it) } ?: "?"
                println("    🏷  [${e.span?.start}-${e.span?.end}] \"${e.text}\"  ${e.type}  ${e.metadata["coarse"]}  score=$score")
            }
            if (res.entities.size > 5) println("    🏷  … +${res.entities.size - 5} autres")

            // Triplets SVO
            val triplets = res.svoTriplets()
            if (triplets.isNotEmpty()) {
                triplets.take(2).forEach { println("    🔺  ${formatTriplet(it)}") }
            }
        }

        // ════════════════════════════════════════════════════════
        // 2. BATCH des phrases de test — batch_size interne = tout
        // ════════════════════════════════════════════════════════
        println("\n━━━ 2. BATCH ${TEST_TEXTS.size} phrases (un seul appel) ━━━━━━━━━━━━")
        val t0Batch = System.nanoTime()
        val batchRes = ext.extractWithSvoFromTexts(TEST_TEXTS)
        val msBatch  = (System.nanoTime() - t0Batch) / 1_000_000L
        val nerTotal = batchRes.sumOf { it.entities.size }
        val svoTotal = batchRes.sumOf { it.svoSpans.size }
        println("Terminé en ${msBatch}ms — NER: $nerTotal  SVO: $svoTotal")
        println("Moyenne: ${"%.1f".format(msBatch.toDouble() / TEST_TEXTS.size)} ms/texte")

        println("\n  Distribution NER (fine) :")
        batchRes.flatMap { it.entities }
            .groupBy { it.type }
            .entries.sortedByDescending { it.value.size }.take(12)
            .forEach { (type, list) -> println("    %-30s : %d".format(type, list.size)) }

        println("\n  Distribution SVO (rôles) :")
        batchRes.flatMap { it.svoSpans }
            .groupBy { it.role }
            .entries.sortedByDescending { it.value.size }
            .forEach { (role, list) -> println("    %-16s : %d".format(role, list.size)) }

        // ════════════════════════════════════════════════════════
        // 3. BENCHMARK 1000 PHRASES — warmup + mesures
        // ════════════════════════════════════════════════════════
        println("\n━━━ 3. BENCHMARK 1000 PHRASES ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        val phrases = loadBenchPhrases(n = 1000)
        println("🎲 ${phrases.size} phrases sélectionnées\n")

        val batchSize  = 32
        val nBatches   = (phrases.size + batchSize - 1) / batchSize
        val warmupBatches = 2

        // Warmup
        print("🔥 Warmup ($warmupBatches batches)… ")
        repeat(warmupBatches) { b ->
            val start = b * batchSize
            ext.extractWithSvoFromTexts(phrases.subList(start, minOf(start + batchSize, phrases.size)))
        }
        println("✅")

        // Mesure
        val batchTimes = mutableListOf<Long>()
        var totalNer = 0; var totalSvo = 0

        println("🚀 Inférence : ${phrases.size} phrases, batch_size=$batchSize → $nBatches batches")
        val tTotal = System.nanoTime()

        for (bIdx in 0 until nBatches) {
            val start = bIdx * batchSize
            val batch = phrases.subList(start, minOf(start + batchSize, phrases.size))
            val t0    = System.nanoTime()
            val res   = ext.extractWithSvoFromTexts(batch)
            val elapsed = (System.nanoTime() - t0) / 1_000_000L
            batchTimes += elapsed
            totalNer += res.sumOf { it.entities.size }
            totalSvo += res.sumOf { it.svoSpans.size }

            val step = maxOf(1, nBatches / 10)
            if ((bIdx + 1) % step == 0 || bIdx + 1 == nBatches) {
                val done = start + batch.size
                println("  batch %3d/%d  (%4d/%d phrases)  %dms  (%.1fms/phrase)".format(
                    bIdx + 1, nBatches, done, phrases.size, elapsed, elapsed.toDouble() / batch.size))
            }
        }

        val totalMs = (System.nanoTime() - tTotal) / 1_000_000L
        val sortedTimes = batchTimes.sorted()
        val p50 = sortedTimes[sortedTimes.size / 2]
        val p95 = sortedTimes[(sortedTimes.size * 0.95).toInt()]

        val sep = "═".repeat(65)
        println("\n$sep")
        println("  RÉSULTATS BENCHMARK — ${phrases.size} phrases / batch_size=$batchSize")
        println(sep)
        println("  Temps total            : ${totalMs}ms")
        println("  Latence moyenne/phrase : ${"%.2f".format(totalMs.toDouble() / phrases.size)}ms")
        println("  Throughput             : ${"%.1f".format(phrases.size * 1000.0 / totalMs)} phrases/s")
        println("  Temps/batch moyen      : ${"%.1f".format(batchTimes.average())}ms")
        println("  Temps/batch min        : ${batchTimes.min()}ms")
        println("  Temps/batch max        : ${batchTimes.max()}ms")
        println("  P50 batch              : ${p50}ms")
        println("  P95 batch              : ${p95}ms")
        println("  NER spans total        : $totalNer  (moy ${"%.1f".format(totalNer.toDouble() / phrases.size)}/phrase)")
        println("  SVO spans total        : $totalSvo  (moy ${"%.1f".format(totalSvo.toDouble() / phrases.size)}/phrase)")
        println(sep)
    }

    println("\n✅ Test terminé.")
}

