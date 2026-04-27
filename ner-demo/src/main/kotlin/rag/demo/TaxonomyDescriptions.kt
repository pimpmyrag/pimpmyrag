package rag.demo

/**
 * Descriptions localisées de la taxonomie NER (32 labels fins + 8 familles coarse).
 * Utilisées pour les tooltips (survol) et le panneau de détail.
 *
 * Langues complètes : FR, EN.  DE / ES / IT : descriptions coarse complètes, fine → fallback EN.
 */
object TaxonomyDescriptions {

    // ── Fine labels ──────────────────────────────────────────────────────────────

    private val FINE_FR = mapOf(
        "hint_person_name"    to "Nom propre d'une personne physique (prénom, nom, alias).\nEx : Emmanuel Macron, Marie Curie, « le footballeur ».",
        "hint_person_role"    to "Rôle, titre ou fonction d'une personne.\nEx : président, PDG, général, secrétaire général.",
        "hint_norp"           to "Nationalité, groupe ethnique, religieux ou politique.\nEx : Français, Chiites, Républicains, Kurdes.",
        "hint_group_role"     to "Désignation collective humaine.\nEx : équipe, jury, délégation, gouvernement, milice.",
        "hint_org_name"       to "Organisation formelle nommée : entreprise, institution, parti.\nEx : ONU, LVMH, PS, Apple, Banque de France.",
        "hint_gpe"            to "Entité géopolitique nommée : pays, ville, région.\nEx : France, Paris, Bretagne, Union européenne.",
        "hint_fac_name"       to "Lieu bâti nommé : monument, stade, hôpital.\nEx : Tour Eiffel, Stade de France, CHU Lariboisière.",
        "hint_loc_generic"    to "Lieu géographique générique non nommé.\nEx : montagne, fleuve, côte, désert, plage, vallée.",
        "hint_infra"          to "Infrastructure nommée : route, ligne, réseau.\nEx : A6, ligne 4, RER B, autoroute du Soleil.",
        "hint_time_date"      to "Date ou référence calendaire.\nEx : 12 mars, 2024, lundi prochain, hier, l'an passé.",
        "hint_time_clock"     to "Heure précise ou approximative.\nEx : 14h30, à minuit, vers 8h, en fin d'après-midi.",
        "hint_time_duration"  to "Durée ou intervalle temporel.\nEx : 3 ans, depuis 2 mois, pendant une semaine, un trimestre.",
        "hint_event_nominal"  to "Événement décrit nominalement, sans nom propre.\nEx : la guerre, le procès, la crise, l'élection.",
        "hint_event_named"    to "Événement proprement nommé.\nEx : COP28, Révolution française, JO 2024, Seconde Guerre mondiale.",
        "hint_weapon"         to "Arme ou munition.\nEx : missile, AK-47, bombe, obus, drone militaire.",
        "hint_vehicle"        to "Véhicule.\nEx : avion, navire, tank, camion, porte-avions.",
        "hint_substance"      to "Matière ou substance.\nEx : pétrole, gaz, uranium, lithium, plutonium.",
        "hint_food"           to "Aliment ou boisson.\nEx : blé, vin rouge, viande, café, produits laitiers.",
        "hint_tool"           to "Outil ou équipement technique.\nEx : matériel médical, engin de chantier, équipement industriel.",
        "hint_object_generic" to "Objet physique générique non nommé.\nEx : bâtiment, document, prototype, colis.",
        "hint_object_name"    to "Objet physique proprement nommé.\nEx : iPhone 15, Boeing 737, Airbus A320, Tesla Model 3.",
        "hint_quantity"       to "Quantité physique avec unité.\nEx : 3 km, 500 kg, 20 MW, 10°C, 45 000 ha.",
        "hint_measure"        to "Mesure scientifique ou technique sans quantité explicite.\nEx : température, pression, fréquence, débit.",
        "hint_percentage"     to "Pourcentage ou fraction.\nEx : 12%, un quart, 80% des cas, la moitié.",
        "hint_count"          to "Dénombrement entier.\nEx : 3 morts, 12 000 soldats, 40 résidents, centaines de milliers.",
        "hint_money"          to "Montant monétaire.\nEx : 200€, 3 milliards de dollars, plusieurs millions, un budget de 50M€.",
        "hint_rate"           to "Taux, ratio ou indice.\nEx : taux de chômage 7%, CAC à 8000, PIB +2,3%, inflation 3,1%.",
        "hint_law"            to "Texte juridique, loi, traité ou décret.\nEx : loi El Khomri, traité de Rome, RGPD, décret n°2024-11.",
        "hint_work_of_art"    to "Œuvre nommée : livre, film, chanson, tableau.\nEx : La Joconde, Avatar, Notre-Dame de Paris (roman), L'Express.",
        "hint_concept"        to "Concept abstrait nommé.\nEx : libéralisme, intelligence artificielle, démocratie, blockchain.",
        "hint_disease"        to "Maladie ou pathologie.\nEx : Covid-19, cancer du poumon, diabète de type 2, Ebola.",
        "hint_language"       to "Langue humaine ou informatique.\nEx : français, Python, arabe, Java, anglais, Kotlin.",
    )

    private val FINE_EN = mapOf(
        "hint_person_name"    to "Proper name of a person (first name, last name, alias).\nE.g. Emmanuel Macron, Marie Curie, 'the footballer'.",
        "hint_person_role"    to "Role, title or function of a person.\nE.g. president, CEO, general, secretary-general.",
        "hint_norp"           to "Nationality, ethnic, religious or political group.\nE.g. French, Shiites, Republicans, Kurds.",
        "hint_group_role"     to "Collective human designation.\nE.g. team, jury, delegation, government, militia.",
        "hint_org_name"       to "Named formal organisation: company, institution, party.\nE.g. UN, LVMH, Labour, Apple, Bank of England.",
        "hint_gpe"            to "Named geopolitical entity: country, city, region.\nE.g. France, Paris, Brittany, European Union.",
        "hint_fac_name"       to "Named built place: monument, stadium, hospital.\nE.g. Eiffel Tower, Stade de France, St Thomas' Hospital.",
        "hint_loc_generic"    to "Generic unnamed geographic location.\nE.g. mountain, river, coast, desert, beach, valley.",
        "hint_infra"          to "Named infrastructure: road, line, network.\nE.g. A6 motorway, Line 4, RER B, Channel Tunnel.",
        "hint_time_date"      to "Date or calendar reference.\nE.g. March 12, 2024, next Monday, yesterday, last year.",
        "hint_time_clock"     to "Precise or approximate time.\nE.g. 2:30 PM, at midnight, around 8 AM, late afternoon.",
        "hint_time_duration"  to "Duration or time interval.\nE.g. 3 years, for 2 months, during a week, one quarter.",
        "hint_event_nominal"  to "Nominally described event without a proper name.\nE.g. the war, the trial, the crisis, the election.",
        "hint_event_named"    to "Properly named event.\nE.g. COP28, French Revolution, 2024 Olympics, WWII.",
        "hint_weapon"         to "Weapon or ammunition.\nE.g. missile, AK-47, bomb, shell, military drone.",
        "hint_vehicle"        to "Vehicle.\nE.g. aircraft, ship, tank, truck, aircraft carrier.",
        "hint_substance"      to "Material or substance.\nE.g. oil, gas, uranium, lithium, plutonium.",
        "hint_food"           to "Food or beverage.\nE.g. wheat, red wine, meat, coffee, dairy products.",
        "hint_tool"           to "Tool or technical equipment.\nE.g. medical device, construction machinery, industrial equipment.",
        "hint_object_generic" to "Generic unnamed physical object.\nE.g. building, document, prototype, parcel.",
        "hint_object_name"    to "Properly named physical object.\nE.g. iPhone 15, Boeing 737, Airbus A320, Tesla Model 3.",
        "hint_quantity"       to "Physical quantity with unit.\nE.g. 3 km, 500 kg, 20 MW, 10°C, 45,000 ha.",
        "hint_measure"        to "Scientific or technical measure without explicit quantity.\nE.g. temperature, pressure, frequency, flow rate.",
        "hint_percentage"     to "Percentage or fraction.\nE.g. 12%, a quarter, 80% of cases, half.",
        "hint_count"          to "Integer count.\nE.g. 3 dead, 12,000 soldiers, 40 residents, hundreds of thousands.",
        "hint_money"          to "Monetary amount.\nE.g. €200, 3 billion dollars, several million, a €50M budget.",
        "hint_rate"           to "Rate, ratio or index.\nE.g. unemployment 7%, CAC 8000, GDP +2.3%, inflation 3.1%.",
        "hint_law"            to "Legal text, law, treaty or decree.\nE.g. GDPR, Treaty of Rome, Patriot Act, Decree 2024-11.",
        "hint_work_of_art"    to "Named work: book, film, song, painting.\nE.g. Mona Lisa, Avatar, Notre-Dame de Paris (novel), The Times.",
        "hint_concept"        to "Named abstract concept.\nE.g. liberalism, artificial intelligence, democracy, blockchain.",
        "hint_disease"        to "Disease or pathology.\nE.g. Covid-19, lung cancer, type-2 diabetes, Ebola.",
        "hint_language"       to "Human or programming language.\nE.g. French, Python, Arabic, Java, English, Kotlin.",
    )

    // ── Coarse families ──────────────────────────────────────────────────────────

    private val COARSE_FR = mapOf(
        "PER"      to "👤 Personnes & groupes humains\nnoms propres, rôles/titres, nationalités/groupes, collectifs",
        "LOC"      to "📍 Lieux\nentités géopolitiques, lieux bâtis, lieux génériques, infrastructures",
        "ORG"      to "🏢 Organisations\nentreprises, institutions, partis, associations",
        "TIME"     to "🕐 Temps\ndates, heures, durées et intervalles",
        "EVENT"    to "⚡ Événements\nnommés (COP28, JO…) ou nominaux (la guerre, la crise…)",
        "OBJECT"   to "📦 Objets physiques\narmes, véhicules, substances, aliments, outils, objets nommés",
        "VALUE"    to "🔢 Valeurs numériques\nquantités, mesures, pourcentages, comptes, montants, taux",
        "ABSTRACT" to "💡 Concepts & œuvres\nlois, œuvres d'art, concepts abstraits, maladies, langues",
    )

    private val COARSE_EN = mapOf(
        "PER"      to "👤 Persons & human groups\nproper names, roles/titles, nationalities/groups, collectives",
        "LOC"      to "📍 Locations\ngeopolitical entities, landmarks, generic places, infrastructure",
        "ORG"      to "🏢 Organizations\ncompanies, institutions, parties, associations",
        "TIME"     to "🕐 Time\ndates, times, durations and intervals",
        "EVENT"    to "⚡ Events\nnamed (COP28, Olympics…) or nominal (the war, the crisis…)",
        "OBJECT"   to "📦 Physical objects\nweapons, vehicles, substances, foods, tools, named objects",
        "VALUE"    to "🔢 Numeric values\nquantities, measures, percentages, counts, amounts, rates",
        "ABSTRACT" to "💡 Concepts & works\nlaws, artworks, abstract concepts, diseases, languages",
    )

    private val COARSE_DE = mapOf(
        "PER"      to "👤 Personen & Menschengruppen\nEigennamen, Rollen/Titel, Nationalitäten, Kollektive",
        "LOC"      to "📍 Orte\nGeopolitische Entitäten, Gebäude, generische Orte, Infrastruktur",
        "ORG"      to "🏢 Organisationen\nUnternehmen, Institutionen, Parteien, Verbände",
        "TIME"     to "🕐 Zeit\nDaten, Uhrzeiten, Dauern und Zeitintervalle",
        "EVENT"    to "⚡ Ereignisse\nbenannt (COP28, Olympia…) oder nominal (der Krieg, die Krise…)",
        "OBJECT"   to "📦 Physische Objekte\nWaffen, Fahrzeuge, Substanzen, Lebensmittel, Werkzeuge",
        "VALUE"    to "🔢 Numerische Werte\nMengen, Maße, Prozentsätze, Zählungen, Beträge, Raten",
        "ABSTRACT" to "💡 Konzepte & Werke\nGesetze, Kunstwerke, abstrakte Konzepte, Krankheiten, Sprachen",
    )

    private val COARSE_ES = mapOf(
        "PER"      to "👤 Personas y grupos humanos\nnombres propios, roles/títulos, nacionalidades, colectivos",
        "LOC"      to "📍 Lugares\nentidades geopolíticas, monumentos, lugares genéricos, infraestructuras",
        "ORG"      to "🏢 Organizaciones\nempresas, instituciones, partidos, asociaciones",
        "TIME"     to "🕐 Tiempo\nfechas, horas, duraciones e intervalos",
        "EVENT"    to "⚡ Eventos\nnombrados (COP28, JJ.OO…) o nominales (la guerra, la crisis…)",
        "OBJECT"   to "📦 Objetos físicos\narmas, vehículos, sustancias, alimentos, herramientas",
        "VALUE"    to "🔢 Valores numéricos\ncantidades, medidas, porcentajes, conteos, importes, tasas",
        "ABSTRACT" to "💡 Conceptos y obras\nleyes, obras de arte, conceptos abstractos, enfermedades, idiomas",
    )

    private val COARSE_IT = mapOf(
        "PER"      to "👤 Persone e gruppi umani\nnomi propri, ruoli/titoli, nazionalità, collettivi",
        "LOC"      to "📍 Luoghi\nentità geopolitiche, monumenti, luoghi generici, infrastrutture",
        "ORG"      to "🏢 Organizzazioni\naziende, istituzioni, partiti, associazioni",
        "TIME"     to "🕐 Tempo\ndate, ore, durate e intervalli",
        "EVENT"    to "⚡ Eventi\nnominati (COP28, Olimpiadi…) o nominali (la guerra, la crisi…)",
        "OBJECT"   to "📦 Oggetti fisici\narmi, veicoli, sostanze, alimenti, strumenti",
        "VALUE"    to "🔢 Valori numerici\nquantità, misure, percentuali, conteggi, importi, tassi",
        "ABSTRACT" to "💡 Concetti e opere\nleggi, opere d'arte, concetti astratti, malattie, lingue",
    )

    // ── Public API ───────────────────────────────────────────────────────────────

    fun fine(label: String, lang: String): String =
        when (lang.take(2)) {
            "fr" -> FINE_FR[label]
            "en" -> FINE_EN[label]
            else  -> FINE_EN[label]  // DE/ES/IT: fallback EN pour les labels fins
        } ?: label.removePrefix("hint_").replace("_", " ")

    fun coarse(family: String, lang: String): String =
        when (lang.take(2)) {
            "fr" -> COARSE_FR[family]
            "de" -> COARSE_DE[family]
            "es" -> COARSE_ES[family]
            "it" -> COARSE_IT[family]
            else  -> COARSE_EN[family]
        } ?: family

    /** Fine labels appartenant à une famille coarse, dans l'ordre affiché. */
    val COARSE_TO_FINE = mapOf(
        "PER"      to listOf("hint_person_name","hint_person_role","hint_norp","hint_group_role"),
        "LOC"      to listOf("hint_gpe","hint_fac_name","hint_loc_generic","hint_infra"),
        "ORG"      to listOf("hint_org_name"),
        "TIME"     to listOf("hint_time_date","hint_time_clock","hint_time_duration"),
        "EVENT"    to listOf("hint_event_nominal","hint_event_named"),
        "OBJECT"   to listOf("hint_weapon","hint_vehicle","hint_substance","hint_food","hint_tool","hint_object_generic","hint_object_name"),
        "VALUE"    to listOf("hint_quantity","hint_measure","hint_percentage","hint_count","hint_money","hint_rate"),
        "ABSTRACT" to listOf("hint_law","hint_work_of_art","hint_concept","hint_disease","hint_language"),
    )

    /** Tooltip complet pour un chip coarse : description + liste des fine labels. */
    fun coarseChipTooltip(family: String, lang: String): String {
        val fines = COARSE_TO_FINE[family] ?: return coarse(family, lang)
        val fineList = fines.joinToString(", ") { it.removePrefix("hint_").replace("_", " ") }
        return "${coarse(family, lang)}\n──\n$fineList"
    }
}

