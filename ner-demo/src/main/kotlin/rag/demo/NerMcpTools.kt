package rag.demo

import org.springframework.ai.tool.annotation.Tool
import org.springframework.ai.tool.annotation.ToolParam
import org.springframework.stereotype.Component
import kotlin.math.roundToInt

/**
 * Outils MCP exposés au serveur pour permettre à un agent IA de :
 *   1. lire la configuration courante                   → getConfig
 *   2. modifier un seuil global                         → setThreshold
 *   3. modifier le seuil de score d'un label coarse     → setCoarseScore
 *   4. lancer une inférence et voir les scores bruts    → analyzeText
 *   5. balayer une plage de seuils sur un texte-test    → scanThreshold
 *   6. analyser un corpus et obtenir des stats agrégées → analyzeBatch
 *   7. appliquer la meilleure config estimée + ré-analyser → applyAndAnalyze
 *   8. mesurer les performances d'inférence             → probePerformance
 *
 * ═══════════════════════════════════════════════════════════════════════
 * TAXONOMIE NER — modèle DeBERTa-v3 multitête span-based
 * ═══════════════════════════════════════════════════════════════════════
 *
 * ARCHITECTURE — modèle UNIQUE multi-tête :
 *   Un seul modèle DeBERTa-v3 avec plusieurs têtes de décodage sur chaque span candidat :
 *     • tête boundary  → "ce span est-il une entité ?" (prob pBoundary)
 *     • tête coarse    → "quelle grande famille ?" (prob pCoarse, 9 classes + NONE)
 *     • tête fine      → "quel label parmi 32 ?" (prob pFine)
 *     • têtes SVO      → rôle syntaxique / voix / genre / nombre  [feature preview — voir ci-dessous]
 *
 *   Il N'Y A PAS deux modèles distincts (pas de XLM-RoBERTa, pas de BILOU séparé).
 *   Les trois probabilités pBoundary / pCoarse / pFine sont produites en une seule
 *   inférence sur le même encodeur DeBERTa partagé.
 *
 * ⚠ COARSE = famille indicative seulement.
 *   La tête coarse conditionne le masque structurel COARSE_TO_FINE (réduit les candidats
 *   fins à ceux compatibles avec la famille) et sert à l'affichage de couleur dans l'UI.
 *   Elle N'est PAS évaluée directement ; seule la valeur FINE compte sémantiquement.
 *   pCoarse / tauCoarse servent uniquement au débogage et au paramétrage du masque.
 *
 * FAMILLES COARSE (8 + NONE) :
 *   PER, LOC, ORG, TIME, EVENT, OBJECT, VALUE, ABSTRACT, NONE
 *
 * LABELS FINE-GRAINED (32) — seules valeurs sémantiques réelles :
 *
 *   Famille PER (personnes & groupes humains)
 *     hint_person_name   — nom propre d'une personne physique (prénom, nom, alias)
 *     hint_person_role   — rôle, titre ou fonction (président, général, PDG…)
 *     hint_norp          — nationalité, groupe religieux/ethnique/politique (Français, Chiites…)
 *     hint_group_role    — désignation collective humaine (équipe, jury, délégation…)
 *
 *   Famille LOC (lieux)
 *     hint_gpe           — entité géopolitique nommée : pays, ville, région (France, Paris…)
 *     hint_fac_name      — lieu bâti nommé : monument, stade, hôpital (Tour Eiffel, Bercy…)
 *     hint_loc_generic   — lieu géographique générique non nommé (montagne, fleuve, côte…)
 *     hint_infra         — infrastructure nommée : route, ligne, réseau (A6, ligne 4…)
 *
 *   Famille ORG (organisations)
 *     hint_org_name      — organisation formelle nommée : entreprise, institution, parti (ONU, LVMH…)
 *
 *   Famille TIME (expressions temporelles)
 *     hint_time_date     — date ou référence calendaire (12 mars, 2024, lundi prochain)
 *     hint_time_clock    — heure précise (14h30, à minuit, vers 8h)
 *     hint_time_duration — durée ou intervalle temporel (3 ans, depuis 2 mois…)
 *
 *   Famille EVENT (événements)
 *     hint_event_nominal — événement décrit nominalement (la guerre, le procès, la crise)
 *     hint_event_named   — événement proprement nommé (COP28, Révolution française, JO 2024)
 *
 *   Famille OBJECT (objets physiques)
 *     hint_weapon        — arme ou munition (missile, AK-47, bombe…)
 *     hint_vehicle       — véhicule (avion, navire, tank, voiture)
 *     hint_substance     — matière ou substance (pétrole, gaz, uranium)
 *     hint_food          — aliment ou boisson (blé, vin, viande)
 *     hint_tool          — outil ou équipement (matériel médical, engin de chantier…)
 *     hint_object_generic — objet physique générique
 *     hint_object_name   — objet physique proprement nommé (iPhone 15, Boeing 737…)
 *
 *   Famille VALUE (valeurs numériques)
 *     hint_quantity      — quantité physique avec unité (3 km, 500 kg, 20 MW)
 *     hint_measure       — mesure scientifique ou technique (température, pression)
 *     hint_percentage    — pourcentage (12%, un quart)
 *     hint_count         — dénombrement entier (3 morts, 12 000 soldats)
 *     hint_money         — montant monétaire (200€, 3 milliards de dollars)
 *     hint_rate          — taux, ratio, indice (taux de chômage à 7%, CAC à 8000)
 *
 *   Famille ABSTRACT (concepts & œuvres)
 *     hint_law           — texte juridique, loi, traité, décret (loi El Khomri, traité de Rome)
 *     hint_work_of_art   — œuvre nommée : livre, film, chanson (La Joconde, Avatar)
 *     hint_concept       — concept abstrait nommé (libéralisme, intelligence artificielle)
 *     hint_disease       — maladie ou pathologie (Covid-19, cancer du poumon)
 *     hint_language      — langue humaine ou informatique (français, Python, arabe)
 *
 * CHAMPS SCORES — nature et usage :
 *
 *   ⚠ COMPRENDRE LE SCORE COMPOSITE — lecture indispensable avant toute calibration ⚠
 *
 *   score = pBoundary × pCoarse × pFine
 *
 *   C'est un PRODUIT de trois probabilités indépendantes issues de trois têtes distinctes.
 *   Cela signifie qu'un score composite peut être "faible" même quand chacune des trois
 *   têtes est individuellement très confiante. Exemples :
 *
 *     pBoundary=0.90 × pCoarse=0.90 × pFine=0.85 → score = 0.688  ← entité solide
 *     pBoundary=0.85 × pCoarse=0.80 × pFine=0.78 → score = 0.530  ← entité correcte
 *     pBoundary=0.80 × pCoarse=0.75 × pFine=0.72 → score = 0.432  ← encore valide
 *
 *   Un score de 0.50 NE veut PAS dire "50% de confiance" — c'est l'effet multiplicatif.
 *   La valeur absolue du score est donc peu parlante ; c'est sa distribution relative qui
 *   compte (trouver le "coude" via scanThreshold).
 *
 *   Pour évaluer la VRAIE confiance du modèle sur une entité, regarder les têtes :
 *
 *   pBoundary  → tête boundary : "est-ce bien une entité ?"
 *                < 0.75 → span frontière incertaine → peut être du bruit
 *                ≥ 0.90 → frontière bien détectée, le score bas vient d'ailleurs
 *
 *   pCoarse    → tête coarse : "quelle grande famille ?" (PER/LOC/ORG/…)
 *                RÔLE INDICATIF SEULEMENT — pas évalué en tant que tel.
 *                Sert au masque structurel COARSE_TO_FINE (réduit le nb de candidats fins).
 *                Sélection par argmax(pCoarse) parmi familles ≥ tauCoarse — pas de
 *                compétition via score composite (biais fort sinon vers familles
 *                à peu de labels fins comme ORG qui aurait pFine≈1.0 par construction).
 *                < 0.60 → confusion de famille → peut engendrer un mauvais label fin
 *                ≥ 0.80 → famille bien identifiée, score bas = pFine seul en cause
 *
 *   pFine      → tête fine : "quel label parmi les 32 ?" — LA valeur sémantique réelle
 *                < 0.60 → ambiguïté entre deux labels fins voisins (ex: event_nominal
 *                          vs concept) → le label retenu est le plus probable mais fragile
 *                ≥ 0.85 → label fin confiant, score bas = produit des deux autres
 *
 *   Règle de lecture :
 *     Si pBoundary ≥ 0.80 ET pFine ≥ 0.75 → l'entité est bien détectée et bien labellisée,
 *     quel que soit le score composite. Ne pas l'écarter sur la base du score seul.
 *     → Utiliser setCoarseScore pour adapter le seuil famille par famille si besoin.
 *
 * Workflow type de calibration :
 *   getConfig() → analyzeText(sample) → [observer les scores borderline]
 *   → setThreshold("tauBoundary", 0.65) → analyzeText(sample) → comparer
 *   → scanThreshold("tauBoundary", sample, 0.40, 0.90, 0.05) → trouver le coude
 *   → analyzeBatch(texts) → repérer quelle famille coarse a un avgScore faible
 *   → setCoarseScore("EVENT", 0.80) → resserrer uniquement la famille EVENT
 *   → applyAndAnalyze(sample, tauBoundary=0.65, scoreByCoarse={"EVENT":0.80})
 *      → commit définitif + vérification finale en un seul appel
 *
 * STRATÉGIE DE CHOIX DES PHRASES DE TEST :
 *
 *   ① PRIORITÉ ABSOLUE — phrases fournies par l'utilisateur
 *     Ce sont les seules qui reflètent le domaine, le style et les ambiguïtés réels.
 *     Toute calibration finale (applyAndAnalyze) DOIT être validée sur ces phrases.
 *     Ne jamais conclure que "la config est bonne" sans avoir testé sur des données user.
 *
 *   ② EXPLORATION COMPLÉMENTAIRE — phrases auto-générées
 *     Utile pour couvrir des types d'entités absents des exemples user, ou pour
 *     comprendre les comportements limites :
 *       • Générer 2–3 phrases par famille coarse non représentée (PER, LOC, EVENT…)
 *       • Varier le style : titre de presse / corps d'article / note informelle / chiffres
 *       • Inclure des cas ambigus (ex : "Le Monde" = ORG ou œuvre ?)
 *     ⚠ Les phrases auto-générées sont "propres" et peuvent masquer le bruit réel.
 *        Un seuil optimal sur phrase synthétique peut être trop permissif sur données réelles.
 *
 *   ③ VALIDATION CROISÉE recommandée
 *     1. Explorer avec phrases auto-générées → estimer une fourchette de seuils
 *     2. Vérifier sur phrases user → ajuster si les résultats diffèrent
 *     3. Si aucune phrase user disponible → signaler explicitement dans la réponse
 *        que la calibration est provisoire et devra être confirmée sur données réelles.
 *
 * SPANS IMBRIQUÉS (compound NMS) :
 *   Quand un span est entièrement contenu dans un autre span déjà conservé :
 *   - Si les deux ont le MÊME label fine → le sous-span est supprimé (doublon, typique
 *     pour les EVENT_NAMED imbriqués dans un autre EVENT_NAMED).
 *   - Si les labels fine sont DIFFÉRENTS → le sous-span est conservé comme nested=true.
 *     Exemple : "secrétaire général des Nations Unies" (hint_person_role)
 *               contient "Nations Unies" (hint_org_name) → le sous-span est remontré.
 *   Les spans nested portent les champs additionnels :
 *     nested=true, parentText, parentFine, parentCoarse, parentStart, parentEnd
 *
 * ═══════════════════════════════════════════════════════════════════════
 * FEATURE PREVIEW — TÊTES SVO (syntaxe + morphologie)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Le modèle embarque trois têtes supplémentaires évaluées sur le MÊME forward pass NER,
 * sans surcoût d'inférence. Elles sont ADDITIONNELLES et NON DESTRUCTIVES :
 * elles n'affectent pas les entités NER, elles les enrichissent.
 *
 * TÊTE SVO — rôle syntaxique argumental :
 *   Pour chaque span candidat, la tête SVO prédit:
 *     • svo_boundary  → "ce span est-il un argument SVO?" (prob pSvoBoundary)
 *     • svo_role      → svo_subject | svo_object | svo_iobj | svo_verb
 *                       pron_subj | pron_obj  (pronoms)
 *   Réconciliation NER↔SVO :
 *     Phase 1 (inline) : si une entité NER passe aussi le seuil SVO sur le MÊME span k,
 *       elle reçoit svoRole / syntacticRole en metadata (nsubj | obj | iobj).
 *     Phase 2 (snap positionnel) : un span SVO brut sans entité NER est snapé sur
 *       la meilleure entité voisine par recouvrement ≥ 60%.
 *
 * TÊTE SVO-ANCHORED (promotion d'entités borderline) :
 *   Si la tête SVO détecte un argument non-pronominal (svo_subject/object/iobj)
 *   avec pBoundary ∈ [tauSvoAnchoredBoundary, tauBoundary[, l'entité est promue
 *   avec des seuils NER assouplis (×0.85 fine, ×0.60 score). Taguée svoAnchored=true.
 *   → Utile pour récupérer des entités "borderline" que NER n'aurait pas gardées seul.
 *
 * TÊTES MORPHO — genre / nombre :
 *   Lus sur TOUTES les entités NER (indépendamment du SVO boundary) :
 *     gender : Masc | Fem  (absent si indéterminé)
 *     number : Sing | Plur (absent si indéterminé)
 *
 * COMMENT ÉVALUER LES SVO EN PREVIEW :
 *   1. analyzeText(texte) → observer le champ "svoSpans" :
 *        - svoRole     : le rôle argumental de chaque span
 *        - entityText  : l'entité NER fusionnée (si reconcile() a matchée)
 *        - p_svo_bnd   : confiance de la tête SVO boundary
 *        - p_role      : confiance du label de rôle
 *   2. Les entités affichent maintenant syntacticRole (nsubj|obj|iobj), gender, number.
 *   3. tauSvoBoundary (défaut 0.50) : lever pour réduire le bruit SVO.
 *      tauSvoAnchoredBoundary (défaut 0.40) : lever pour moins de promotions borderline.
 *   4. La calibration NER (tauBoundary, etc.) est INDÉPENDANTE des seuils SVO.
 */
@Component
class NerMcpTools(private val nerService: NerService) {

    // ── 1. Lire la config ────────────────────────────────────────────────────

    @Tool(description = """
        Returns the current NER inference thresholds and settings.

        ⚠ SCORE SEMANTICS — read before interpreting thresholds:
          score = pBoundary × pCoarse × pFine  (product of 3 independent head probabilities)
          This is a HARSH composite: even a high-quality entity (all heads ≥ 0.85) yields
          score ≈ 0.61. A score of 0.50 does NOT mean "50% confidence" — it is an artefact
          of the multiplicative formula. Always look at pBoundary and pFine individually
          to assess true head-level confidence.

        Threshold semantics (all floats in [0,1]):
          tauBoundary    — minimum composite score to KEEP a span.
                           Applied to the harsh product score (see above).
                           Recommended range: 0.35–0.65. Default 0.70 is conservative.
                           ↓ lower = more recall (accepting lower-product entities)
                           ↑ higher = more precision (rejects entities where one head is weaker)

          tauNone        — hard rejection gate (pNone threshold). Rarely needs tuning.

          tauCoarse      — coarse family gate. Indicative only (masking & display).
                           The coarse family is selected by argmax(pCoarse) among families
                           with pCoarse ≥ tauCoarse — NOT by composite score competition.
                           This avoids a bias that would favour families with few fine labels
                           (e.g. ORG with 1 label always gets pFine≈1.0 after masked softmax).
                           Tune only when pCoarse analysis reveals systematic family misrouting.

          scoreByCoarse  — per-family composite score floor.
                           Useful when one family (e.g. EVENT) produces many FP with low pFine
                           while other families are fine at global tauBoundary.
                           Empty = global tauBoundary applies to all.
    """)
    fun getConfig(): Map<String, Any> {
        val c = nerService.config
        return mapOf(
            "tauBoundary"            to c.tauBoundary,
            "tauNone"                to c.tauNone,
            "tauCoarse"              to c.tauCoarse,
            "tauSvoBoundary"         to c.tauSvoBoundary,
            "tauSvoAnchoredBoundary" to c.tauSvoAnchoredBoundary,
            "batchSize"              to c.batchSize,
            "scoreByCoarse"          to c.scoreByCoarse.ifEmpty { "(none — global tauBoundary applies)" },
            "runtime"                to nerService.runtimeInfo(),
        )
    }

    // ── 2. Modifier un seuil ─────────────────────────────────────────────────

    @Tool(description = """
        Update a single NER threshold by name. The change takes effect immediately for all
        subsequent analyzeText / scanThreshold / analyzeBatch calls.
        Returns the full updated config.

        ⚠ tauBoundary applies to score = pBoundary × pCoarse × pFine (harsh product).
          A value of 0.50 lets through entities where all three heads are around 0.79.
          A value of 0.70 requires all three heads to average ~0.89 — very strict.
          Recommended starting range: 0.40–0.65. Use scanThreshold to find the elbow.

        Valid names and their purpose:
          tauBoundary              — PRIMARY lever. Controls recall/precision for ALL families.
                                     Lower first (try 0.50) before touching anything else.
          tauNone                  — Rejection gate for near-zero candidates. Rarely needs tuning.
          tauCoarse                — Coarse family confidence gate (indicative, structural masking only).
                                     Tune only when pCoarse shows systematic mis-routing.
          tauSvoBoundary           — [SVO PREVIEW] Minimum pSvoBoundary to detect a syntactic argument span.
                                     Default 0.50. Raise to reduce SVO noise (fewer but surer roles).
                                     Does NOT affect NER entity detection.
          tauSvoAnchoredBoundary   — [SVO PREVIEW] Relaxed NER boundary for spans where SVO is confident
                                     but NER pBoundary is between this value and tauBoundary.
                                     Default 0.40. Raise to reduce SVO-promoted entities (svoAnchored=true).
    """)
    fun setThreshold(
        @ToolParam(description = "Threshold name: tauBoundary | tauNone | tauCoarse | tauSvoBoundary | tauSvoAnchoredBoundary")
        name: String,
        @ToolParam(description = "New value (float, automatically clamped to valid range)")
        value: Float,
    ): Map<String, Any> {
        val c = nerService.config
        val updated = when (name) {
            "tauBoundary"            -> c.copy(tauBoundary            = value.coerceIn(0.05f, 0.99f))
            "tauNone"                -> c.copy(tauNone                = value.coerceIn(0.05f, 1.00f))
            "tauCoarse"              -> c.copy(tauCoarse              = value.coerceIn(0.00f, 0.99f))
            "tauSvoBoundary"         -> c.copy(tauSvoBoundary         = value.coerceIn(0.05f, 0.99f))
            "tauSvoAnchoredBoundary" -> c.copy(tauSvoAnchoredBoundary = value.coerceIn(0.05f, 0.99f))
            else -> return mapOf("error" to "Unknown threshold name: $name. Use tauBoundary | tauNone | tauCoarse | tauSvoBoundary | tauSvoAnchoredBoundary")
        }
        nerService.updateConfig(updated)
        val oldValue = when (name) {
            "tauBoundary"            -> c.tauBoundary
            "tauNone"                -> c.tauNone
            "tauCoarse"              -> c.tauCoarse
            "tauSvoBoundary"         -> c.tauSvoBoundary
            else                     -> c.tauSvoAnchoredBoundary
        }
        return mapOf(
            "updated"  to name,
            "oldValue" to oldValue,
            "newValue" to value,
            "config"   to getConfig(),
        )
    }

    // ── 3. Seuil de score par famille coarse ────────────────────────────────

    @Tool(description = """
        Set (or clear) a per-coarse minimum composite score threshold.

        By default all coarse families share the same tauBoundary.
        Use this tool to make a specific family STRICTER or MORE PERMISSIVE independently.

        Valid coarse names: PER | LOC | ORG | TIME | EVENT | VALUE | OBJECT | ABSTRACT

        Examples:
          setCoarseScore("EVENT", 0.85)  → only EVENT entities with score ≥ 0.85 are kept
          setCoarseScore("PER",   0.60)  → accept PER entities with lower confidence
          setCoarseScore("EVENT", 0.0)   → reset EVENT to global tauBoundary

        Compound/nested behaviour reminder:
          A nested span (fully contained in a longer parent) is only kept when its fine label
          DIFFERS from the parent's fine label. Same-fine nested spans are always discarded.
          The coarse threshold does NOT affect this NMS logic.

        Returns the updated scoreByCoarse map.
    """)
    fun setCoarseScore(
        @ToolParam(description = "Coarse family name: PER | LOC | ORG | TIME | EVENT | VALUE | OBJECT | ABSTRACT")
        coarse: String,
        @ToolParam(description = "Minimum composite score for this family (0 = reset to global tauBoundary, max 1.0)")
        value: Float,
    ): Map<String, Any> {
        val validCoarse = setOf("PER", "LOC", "ORG", "TIME", "EVENT", "VALUE", "OBJECT", "ABSTRACT")
        if (coarse !in validCoarse)
            return mapOf("error" to "Unknown coarse: $coarse. Valid: ${validCoarse.joinToString("|")}")

        val c = nerService.config
        val clamped = value.coerceIn(0f, 1f)
        val newMap  = c.scoreByCoarse.toMutableMap()
        if (clamped <= 0f) newMap.remove(coarse) else newMap[coarse] = clamped
        nerService.updateConfig(c.copy(scoreByCoarse = newMap))
        return mapOf(
            "updated"       to coarse,
            "newValue"      to if (clamped <= 0f) "reset to global tauBoundary (${c.tauBoundary})" else clamped,
            "scoreByCoarse" to nerService.config.scoreByCoarse.ifEmpty { "(none)" },
        )
    }

    // ── 4. Analyser un texte ─────────────────────────────────────────────────

    @Tool(description = """
        Run NER inference on a text and return all detected entities with their labels, RAW SCORES,
        and — as a PREVIEW — SVO syntactic roles reconciled from the multi-head model.

        SVO IS NON-DESTRUCTIVE: the SVO heads run on the same forward pass as NER at zero extra cost.
        They enrich entities with syntactic roles (nsubj/obj/iobj), gender, number, and voice.
        Calibrate NER thresholds (tauBoundary etc.) independently — SVO does NOT affect NER detection.

        ══════════════════════════════════════════════════════════════
        ⚠ INSTRUCTION D'ÉVALUATION — OBLIGATOIRE POUR LE LLM JUDGE ⚠
        Après chaque appel à analyzeText, tu DOIS impérativement :
          1. Lister et commenter TOUS les spans présents dans "svoSpans" :
             - le texte du span, son rôle SVO, sa confiance p_svo_bnd et p_role
             - l'entité NER associée (entityText/entityFine) si présente, ou indiquer "aucune entité matchée"
             - si la voix (voice) est pertinente, la mentionner
          2. Pour chaque entité NER ayant un champ "syntacticRole", noter le rôle détecté.
          3. Émettre un avis qualitatif : les rôles semblent-ils corrects pour la phrase ?
          4. Signaler les cas suspects : faible p_svo_bnd (< 0.60), rôle incohérent avec la sémantique.
          Ne pas résumer en une ligne les SVO — les commenter explicitement un par un.
        ══════════════════════════════════════════════════════════════

        CHOICE OF TEST TEXT — important for calibration quality:
          PREFER user-provided sentences: they reflect the real domain, style, and noise level.
          Self-generated sentences are clean and may over-estimate model quality.
          If no user text is available, flag calibration as PROVISIONAL pending real data.

        ENTITY FIELDS:
          text           — surface form of the detected entity span
          coarse         — indicative family (PER/LOC/ORG/…), used for display and fine masking
          fine           — THE ACTUAL SEMANTIC LABEL (32 values). This is what you evaluate.
          score / pBoundary / pCoarse / pFine — see class KDoc for score formula details
          nested         — true if fully contained in a parent span with different fine label
          [SVO PREVIEW]
          syntacticRole  — "nsubj" | "obj" | "iobj" if SVO head fired on this entity's span (inline)
          svoRole        — raw SVO label: svo_subject | svo_object | svo_iobj | pron_subj | pron_obj
          svoRoleProb    — confidence of the SVO role label
          svoBoundaryScore — confidence of the SVO boundary head on this span
          gender         — "Masc" | "Fem" if detected (morphology head)
          number         — "Sing" | "Plur" if detected (morphology head)
          svoAnchored    — true if entity was promoted by SVO confidence (pBoundary < tauBoundary
                           but ≥ tauSvoAnchoredBoundary). These have reduced NER confidence.

        SVO SPAN FIELDS (reconciled — includes entity association):
          text           — surface form of the SVO span
          role           — svo_verb | svo_subject | svo_object | svo_iobj | pron_subj | pron_obj
          p_svo_bnd      — SVO boundary confidence
          p_role         — role label confidence
          voice          — ACTIVE | PASSIVE (with confidence)
          entityText     — text of the merged NER entity (null if no entity matched)
          entityFine     — fine label of the matched entity (null if no entity matched)
          entityCoarse   — coarse family of the matched entity
          chars          — character offsets [start:end]

        [SVO CALIBRATION GUIDE]
          If too many SVO spans appear → raise tauSvoBoundary (default 0.50).
          If too many svoAnchored entities appear → raise tauSvoAnchoredBoundary (default 0.40).
          If NER entities miss their role → lower tauSvoBoundary slightly.
          NER calibration (tauBoundary etc.) is fully independent.
    """)
    fun analyzeText(
        @ToolParam(description = "The text to analyze (ideally one natural sentence or short paragraph)")
        text: String,
    ): Map<String, Any> {
        val t0 = System.currentTimeMillis()
        val result = nerService.analyseSingle(text)
        val inferenceMs = System.currentTimeMillis() - t0

        // Résumé SVO pré-formaté pour guider le LLM judge
        val svoArgSpans = result.svoSpans.filter {
            it.role in setOf("svo_subject", "svo_object", "svo_iobj", "pron_subj", "pron_obj")
        }
        val svoSummaryLines = if (result.svoSpans.isEmpty()) {
            listOf("Aucun span SVO détecté avec les seuils actuels (tauSvoBoundary=${nerService.config.tauSvoBoundary}).")
        } else {
            result.svoSpans.map { s ->
                val entityPart = if (s.entity != null)
                    "→ entité NER : \"${s.entity!!.text}\" (${s.entity!!.type})"
                else "→ aucune entité NER matchée"
                "• [${s.role}] \"${s.text}\"  p_bnd=%.3f  p_role=%.3f  voix=${s.voice}  $entityPart".format(
                    s.svoBoundaryProb, s.roleProb)
            }
        }

        return mapOf(
            "⚠_SVO_PREVIEW_A_EVALUER" to mapOf(
                "instruction" to "Commenter OBLIGATOIREMENT chaque entrée de svoSpansSummary ci-dessous. " +
                    "Évaluer si les rôles sont corrects pour la phrase analysée. " +
                    "Indiquer les associations entité NER↔rôle SVO. Signaler tout rôle suspect.",
                "svoCount"         to result.svoSpans.size,
                "svoArgCount"      to svoArgSpans.size,
                "svoSpansSummary"  to svoSummaryLines,
                "nerEntitiesWithRole" to result.entities
                    .filter { it.metadata["syntacticRole"] != null }
                    .map { e -> "\"${e.text}\" (${e.type}) → rôle=${e.metadata["syntacticRole"]} gender=${e.metadata["gender"] ?: "?"} number=${e.metadata["number"] ?: "?"}" },
            ),
            "thresholdsUsed" to getConfig(),
            "inferenceMs"    to inferenceMs,
            "entityCount"    to result.entities.size,
            "svoCount"       to result.svoSpans.size,
            "entities" to result.entities.map { e ->
                buildMap {
                    put("text",      e.text)
                    put("coarse",    e.metadata["coarse"] ?: "NONE")
                    put("fine",      e.type)
                    put("score",     fmt(e.metadata["score"]))
                    put("pBoundary", fmt(e.metadata["pBoundary"]))
                    put("pCoarse",   fmt(e.metadata["pCoarse"]))
                    put("pFine",     fmt(e.metadata["pFine"]))
                    put("chars",     "[${e.span?.start}:${e.span?.end}]")
                    if (e.metadata["nested"] == true) {
                        put("nested",       true)
                        put("parentText",   e.metadata["parentText"])
                        put("parentFine",   e.metadata["parentFine"])
                        put("parentCoarse", e.metadata["parentCoarse"])
                    }
                    // ── SVO PREVIEW ──────────────────────────────────────────
                    (e.metadata["syntacticRole"] as? String)?.let { put("syntacticRole", it) }
                    (e.metadata["svoRole"]       as? String)?.let { put("svoRole",       it) }
                    (e.metadata["svoRoleProb"]   as? Float )?.let { put("svoRoleProb",   fmt(it)) }
                    (e.metadata["svoBoundaryScore"] as? Float)?.let { put("svoBoundaryScore", fmt(it)) }
                    (e.metadata["gender"]        as? String)?.let { put("gender",        it) }
                    (e.metadata["number"]        as? String)?.let { put("number",        it) }
                    if (e.metadata["svoAnchored"] == true) put("svoAnchored", true)
                }
            },
            "svoSpans" to result.svoSpans.map { s ->
                buildMap {
                    put("text",        s.text)
                    put("role",        s.role)
                    put("p_svo_bnd",   "%.3f".format(s.svoBoundaryProb))
                    put("p_role",      "%.3f".format(s.roleProb))
                    put("voice",       "${s.voice} (%.2f)".format(s.voiceProb))
                    put("chars",       "[${s.charStart}:${s.charEnd}]")
                    // Entité NER fusionnée par reconcile() (null si aucun match)
                    s.entity?.let {
                        put("entityText",   it.text)
                        put("entityFine",   it.type)
                        put("entityCoarse", it.metadata["coarse"] ?: "NONE")
                    }
                    if (s.nerOverride != null && s.entity == null)
                        put("nerOverride", "${s.nerOverride} (${fmt(s.nerOverrideScore)})")
                    if (s.fromNer) put("fromNer", true)
                    s.gender?.let { put("gender", it) }
                    s.number?.let { put("number", it) }
                }
            },
        )
    }

    // ── 5. Balayer un seuil ──────────────────────────────────────────────────

    @Tool(description = """
        Sweep a NER threshold over a range on a reference text and report entity counts at each step.
        The config is automatically restored to its original state after the sweep.

        ⚠ Score reminder: score = pBoundary × pCoarse × pFine (harsh product).
          Entity counts will drop sharply around the noise floor, not linearly.
          Even well-detected entities have scores in the 0.45–0.70 range.
          Recommended sweep: tauBoundary from 0.30 to 0.80, step 0.05.

        Purpose: find the optimal tauBoundary by identifying the 'elbow' —
          the value where entity count stabilises, i.e. further lowering the threshold
          adds only noise (low-pBoundary or low-pFine spans) without meaningful new entities.

        How to read the output:
          - Entity count drops sharply below a certain value → that is the noise floor.
          - 'byCoarse' shows which families lose/gain entities at each step.
          - A good tauBoundary sits just above the noise-floor elbow.
          - If one family disappears much earlier than others → candidate for setCoarseScore.

        Tip: call analyzeText first to see the actual score distribution, then sweep.
    """)
    fun scanThreshold(
        @ToolParam(description = "Reference text for the sweep")
        text: String,
        @ToolParam(description = "Threshold to sweep: tauBoundary | tauNone | tauCoarse  (primary: tauBoundary)")
        threshold: String,
        @ToolParam(description = "Start value of the sweep range (e.g. 0.30)")
        from: Float,
        @ToolParam(description = "End value of the sweep range (e.g. 0.90)")
        to: Float,
        @ToolParam(description = "Step size (e.g. 0.05 or 0.10)")
        step: Float,
    ): Map<String, Any> {
        if (threshold !in setOf("tauBoundary", "tauNone", "tauCoarse", "tauSvoBoundary", "tauSvoAnchoredBoundary"))
            return mapOf("error" to "Unknown threshold: $threshold")

        val originalCfg = nerService.config
        val steps = mutableListOf<Map<String, Any>>()
        val safeStep = step.coerceAtLeast(0.01f)
        var v = from
        while (v <= to + 1e-5f) {
            val testCfg = when (threshold) {
                "tauBoundary"            -> originalCfg.copy(tauBoundary            = v)
                "tauNone"                -> originalCfg.copy(tauNone                = v)
                "tauCoarse"              -> originalCfg.copy(tauCoarse              = v)
                "tauSvoBoundary"         -> originalCfg.copy(tauSvoBoundary         = v)
                else                     -> originalCfg.copy(tauSvoAnchoredBoundary = v)
            }
            nerService.updateConfig(testCfg)
            val result = nerService.analyse(text)
            steps += mapOf(
                "value"       to "%.3f".format(v),
                "entities"    to result.entities.size,
                "svoSpans"    to result.svoSpans.size,
                "byCoarse"    to result.entities.groupingBy {
                    it.metadata["coarse"] as? String ?: "NONE"
                }.eachCount(),
            )
            v = (v + safeStep).let {
                // round to avoid float drift
                (it * 1000).roundToInt() / 1000f
            }
        }

        nerService.updateConfig(originalCfg) // always restore
        return mapOf(
            "threshold"      to threshold,
            "range"          to "${from} → ${to} (step ${step})",
            "steps"          to steps,
            "configRestored" to true,
            "hint"           to "Look for the value where entity count stabilises — that is usually a good threshold.",
        )
    }

    // ── 6. Analyser un corpus ────────────────────────────────────────────────

    @Tool(description = """
        Analyze a batch of texts (max 30) and return aggregated NER statistics.
        SVO role counts are included as a PREVIEW — non-destructive, independent of NER calibration.

        CHOICE OF TEXTS:
          Best: real sentences from the user's target domain (news, reports, notes…).
          If user provided no texts, self-generate diverse examples covering all 8 coarse
          families in different styles (headline, body paragraph, informal note).
          ⚠ Self-generated texts are "clean" — they will show higher scores and fewer
          borderline cases than real data. Always tell the user if you used synthetic texts
          and recommend they re-run analyzeBatch with their own data before finalizing config.

        ⚠ Score reminder: avgScore / minScore are computed on the COMPOSITE score
          (pBoundary × pCoarse × pFine). A family with avgScore=0.55 does NOT mean
          the model is poorly confident — it means the product of three heads averages 0.55,
          which is perfectly normal for a well-functioning model.
          Use byFine-level analysis (via analyzeText on representative sentences) to
          check actual pBoundary / pFine values before concluding a family is noisy.

        Use this to:
          - Compare avgScore across families to identify relatively weaker ones
          - Spot low-confidence entities (score < 0.50) that MAY be false positives
            (but check their pBoundary + pFine before discarding)
          - Evaluate fine-label distribution against domain expectations
          - [SVO PREVIEW] Check svoRoleDistribution — counts of nsubj/obj/iobj/verb per corpus

        Output fields:
          byCoarseType   — per-family count + avgScore / minScore / maxScore.
          lowConfidenceEntities — up to 10 entities with score < 0.50, sorted ascending.
          svoRoleDistribution — [SVO PREVIEW] count of each SVO role across the corpus.
          svoEntityCoverage   — [SVO PREVIEW] fraction of SVO argument spans that matched a NER entity.
          hint           — actionable recommendation based on score distribution.
    """)
    fun analyzeBatch(
        @ToolParam(description = "List of texts (max 30 for performance)")
        texts: List<String>,
    ): Map<String, Any> {
        val limited = texts.take(30)
        val allEntities = mutableListOf<Map<String, Any>>()
        val svoRoleCounts = mutableMapOf<String, Int>()
        var svoTotal = 0; var svoWithEntity = 0
        val t0 = System.currentTimeMillis()

        nerService.analyseStream(limited) { _, results ->
            results.forEach { r ->
                r.entities.forEach { e ->
                    allEntities += mapOf(
                        "text"      to e.text,
                        "coarse"    to (e.metadata["coarse"] as? String ?: "NONE"),
                        "fine"      to e.type,
                        "score"     to (e.metadata["score"] as? Float ?: 0f),
                        "pBoundary" to (e.metadata["pBoundary"] as? Float ?: 0f),
                    )
                }
                r.svoSpans.forEach { s ->
                    svoRoleCounts[s.role] = (svoRoleCounts[s.role] ?: 0) + 1
                    val isArg = s.role in setOf("svo_subject","svo_object","svo_iobj","pron_subj","pron_obj")
                    if (isArg) { svoTotal++; if (s.entity != null) svoWithEntity++ }
                }
            }
        }
        val totalMs = System.currentTimeMillis() - t0

        val byCoarse = allEntities.groupBy { it["coarse"] as String }
        val lowConf  = allEntities.filter { (it["score"] as Float) < 0.50f }
            .sortedBy { it["score"] as Float }.take(10)

        return mapOf(
            "textsAnalyzed"        to limited.size,
            "totalEntities"        to allEntities.size,
            "totalMs"              to totalMs,
            "msPerSentence"        to if (limited.isEmpty()) 0 else totalMs / limited.size,
            "avgEntitiesPerText"   to if (limited.isEmpty()) 0.0 else
                                      "%.2f".format(allEntities.size.toDouble() / limited.size),
            "thresholdsUsed"       to getConfig(),
            "byCoarseType"         to byCoarse.mapValues { (_, ents) ->
                val scores = ents.map { it["score"] as Float }
                mapOf(
                    "count"    to ents.size,
                    "avgScore" to "%.3f".format(scores.average()),
                    "minScore" to "%.3f".format(scores.min()),
                    "maxScore" to "%.3f".format(scores.max()),
                )
            },
            "lowConfidenceEntities" to lowConf,
            // ── SVO PREVIEW ────────────────────────────────────────────────────
            "svoRoleDistribution"  to svoRoleCounts,
            "svoEntityCoverage"    to if (svoTotal == 0) "n/a"
                                      else "%.1f%%".format(svoWithEntity * 100.0 / svoTotal) +
                                           " ($svoWithEntity/$svoTotal argument spans matched a NER entity)",
            "hint" to "Low avgScore on a family does NOT imply poor detection — score = pBoundary×pCoarse×pFine " +
                      "is a harsh product (e.g. 0.85³ ≈ 0.61). " +
                      "Use analyzeText on representative sentences to inspect individual head values. " +
                      "Only raise tauBoundary / setCoarseScore if pBoundary or pFine on low-score " +
                      "entities are genuinely weak (< 0.70), not just because the composite is low.",
        )
    }

    // ── 7. Appliquer la meilleure config estimée et ré-analyser ─────────────

    @Tool(description = """
        Apply a complete config estimated by the agent and immediately re-run NER inference
        on a reference text. The config is saved PERMANENTLY (subsequent analyzeText calls
        will use it). Returns inference results + a delta comparing the new vs old config.

        Intended workflow:
          1. Call getConfig + analyzeText to inspect baseline scores (remember: product formula).
          2. Call scanThreshold to find the elbow in entity count vs tauBoundary.
          3. Use analyzeBatch on a small corpus; check head values (pBoundary, pFine) on
             low-score entities before deciding they are noise.
          4. Call applyAndAnalyze with your best-guess config to commit + verify in one round-trip.

        ⚠ Use a REAL user sentence as the reference text for this final call.
          If you used self-generated probes during exploration, the final applyAndAnalyze
          should be run on user-provided text to confirm the config holds on real data.
          If no user text is available, state explicitly that the config is PROVISIONAL.

        All threshold parameters are OPTIONAL — omit (or pass null) to keep the current value.
          tauBoundary    — primary recall/precision lever (recommended range 0.35–0.80)
          tauNone        — rejection gate for very weak candidates (rarely needs tuning)
          tauCoarse      — coarse family confidence gate (rarely needs tuning)
          scoreByCoarse  — per-family score overrides, e.g. {"EVENT": 0.85, "PER": 0.60}.
                           An empty map clears all per-family overrides.
                           Pass null to leave the current per-family map unchanged.
          [SVO PREVIEW — optional]
          tauSvoBoundary          — SVO boundary threshold (default 0.50)
          tauSvoAnchoredBoundary  — relaxed NER boundary for SVO-promoted entities (default 0.40)

        Output:
          configApplied  — the full config now active
          configDelta    — fields that changed (old → new)
          entityCount    — total entities detected with the new config
          entities       — full entity list with NER scores + SVO role preview fields
          nestedCount    — number of compound/nested spans (fine ≠ parent fine)
          svoCount       — number of SVO spans detected
    """)
    fun applyAndAnalyze(
        @ToolParam(description = "Reference text to re-analyze after applying the config")
        text: String,
        @ToolParam(description = "New tauBoundary (null = keep current)")
        tauBoundary: Float?,
        @ToolParam(description = "New tauNone (null = keep current)")
        tauNone: Float?,
        @ToolParam(description = "New tauCoarse (null = keep current)")
        tauCoarse: Float?,
        @ToolParam(description = "Per-coarse score overrides map, e.g. {\"EVENT\":0.85}. null = unchanged. Empty map = clear all overrides.")
        scoreByCoarse: Map<String, Float>?,
        @ToolParam(description = "[SVO PREVIEW] New tauSvoBoundary (null = keep current)")
        tauSvoBoundary: Float? = null,
        @ToolParam(description = "[SVO PREVIEW] New tauSvoAnchoredBoundary (null = keep current)")
        tauSvoAnchoredBoundary: Float? = null,
    ): Map<String, Any> {
        val old = nerService.config

        val new = old.copy(
            tauBoundary            = tauBoundary?.coerceIn(0.05f, 0.99f)           ?: old.tauBoundary,
            tauNone                = tauNone?.coerceIn(0.05f, 1.00f)               ?: old.tauNone,
            tauCoarse              = tauCoarse?.coerceIn(0.00f, 0.99f)             ?: old.tauCoarse,
            tauSvoBoundary         = tauSvoBoundary?.coerceIn(0.05f, 0.99f)        ?: old.tauSvoBoundary,
            tauSvoAnchoredBoundary = tauSvoAnchoredBoundary?.coerceIn(0.05f, 0.99f) ?: old.tauSvoAnchoredBoundary,
            scoreByCoarse          = scoreByCoarse ?: old.scoreByCoarse,
        )
        nerService.updateConfig(new)

        val delta = buildMap<String, Any> {
            if (new.tauBoundary            != old.tauBoundary)            put("tauBoundary",            "${old.tauBoundary} → ${new.tauBoundary}")
            if (new.tauNone                != old.tauNone)                put("tauNone",                "${old.tauNone} → ${new.tauNone}")
            if (new.tauCoarse              != old.tauCoarse)              put("tauCoarse",              "${old.tauCoarse} → ${new.tauCoarse}")
            if (new.tauSvoBoundary         != old.tauSvoBoundary)         put("tauSvoBoundary",         "${old.tauSvoBoundary} → ${new.tauSvoBoundary}")
            if (new.tauSvoAnchoredBoundary != old.tauSvoAnchoredBoundary) put("tauSvoAnchoredBoundary", "${old.tauSvoAnchoredBoundary} → ${new.tauSvoAnchoredBoundary}")
            if (new.scoreByCoarse          != old.scoreByCoarse)          put("scoreByCoarse",          "${old.scoreByCoarse} → ${new.scoreByCoarse}")
        }

        val result = nerService.analyseSingle(text)
        val nestedCount = result.entities.count { it.metadata["nested"] == true }

        return mapOf(
            "configApplied" to getConfig(),
            "configDelta"   to delta.ifEmpty { "(no changes)" },
            "entityCount"   to result.entities.size,
            "nestedCount"   to nestedCount,
            "svoCount"      to result.svoSpans.size,
            "entities" to result.entities.map { e ->
                buildMap {
                    put("text",      e.text)
                    put("coarse",    e.metadata["coarse"] ?: "NONE")
                    put("fine",      e.type)
                    put("score",     fmt(e.metadata["score"]))
                    put("pBoundary", fmt(e.metadata["pBoundary"]))
                    put("pFine",     fmt(e.metadata["pFine"]))
                    put("chars",     "[${e.span?.start}:${e.span?.end}]")
                    if (e.metadata["nested"] == true) {
                        put("nested",     true)
                        put("parentText", e.metadata["parentText"])
                        put("parentFine", e.metadata["parentFine"])
                    }
                    (e.metadata["syntacticRole"] as? String)?.let { put("syntacticRole", it) }
                    (e.metadata["gender"]        as? String)?.let { put("gender",        it) }
                    (e.metadata["number"]        as? String)?.let { put("number",        it) }
                    if (e.metadata["svoAnchored"] == true) put("svoAnchored", true)
                }
            },
        )
    }

    // ── 8. Mesurer les performances d'inférence ──────────────────────────────

    @Tool(description = """
        Run an inference benchmark and return detailed timing + hardware info.
        Useful to understand if the model is bottlenecked by ONNX inference, tokenization,
        or post-processing, and to track the impact of config changes on throughput.

        Two modes:
          single  — runs the provided text once (cold + warm) and reports both timings.
                    Good for latency measurement (real-time NER use case).
          batch   — replicates the text N times (default: batchSize from config) and runs
                    them as a single ONNX batch. Reports total time, ms/sentence, entities/s.
                    Good for throughput measurement (bulk indexing use case).

        Output fields:
          runtime          — device (CPU / CoreML), thread counts, ORT optimization level
          coldMs           — first-call latency (includes any lazy JIT / graph compilation)
          warmMs           — second-call latency (steady-state, comparable to Python ORT)
          batchTotalMs     — total time for the batch run (only if mode=batch)
          batchSizeUsed    — actual number of sentences in the batch run
          msPerSentence    — batchTotalMs / batchSizeUsed
          entitiesPerSec   — throughput in entities/second (batch mode)
          interpretation   — text hint comparing measured ms/sentence to expected values

        Typical values on Apple Silicon M-series (CPU, ALL_OPT, no CoreML):
          single warm  :  ~25–60 ms/sentence depending on sentence length
          batch (N=8)  :  ~15–30 ms/sentence (parallelism gain from span batching)
          If you see >100ms/sentence → likely a configuration issue (threads, CoreML disabled).
    """)
    fun probePerformance(
        @ToolParam(description = "Reference text for timing (a typical sentence from the target domain)")
        text: String,
        @ToolParam(description = "Mode: 'single' (latency) or 'batch' (throughput). Default: 'both'")
        mode: String = "both",
        @ToolParam(description = "Number of sentences in batch mode (null = use batchSize from config)")
        batchSize: Int? = null,
    ): Map<String, Any> {
        val runtime = nerService.runtimeInfo()
        val effectiveBatch = batchSize ?: nerService.config.batchSize

        // ── Warm-up silencieux (1 appel non chronométré) ──────────────────────
        nerService.analyse(text)

        val result = buildMap<String, Any> {
            put("runtime", runtime)

            // ── Mode single ────────────────────────────────────────────────────
            if (mode in setOf("single", "both")) {
                val cold0 = System.currentTimeMillis()
                nerService.analyse(text)
                val coldMs = System.currentTimeMillis() - cold0

                val warm0 = System.currentTimeMillis()
                val warmResult = nerService.analyse(text)
                val warmMs = System.currentTimeMillis() - warm0

                put("single", mapOf(
                    "coldMs"      to coldMs,
                    "warmMs"      to warmMs,
                    "entityCount" to warmResult.entities.size,
                    "note"        to "warmMs is the steady-state latency (post JIT, post CPU cache warm)"
                ))
            }

            // ── Mode batch ─────────────────────────────────────────────────────
            if (mode in setOf("batch", "both")) {
                val sentences = List(effectiveBatch) { text }
                var totalEntities = 0
                val t0 = System.currentTimeMillis()
                nerService.analyseStream(sentences) { _, results ->
                    results.forEach { totalEntities += it.entities.size }
                }
                val batchMs = System.currentTimeMillis() - t0
                val msPerSent = if (effectiveBatch > 0) batchMs.toDouble() / effectiveBatch else 0.0
                val entPerSec = if (batchMs > 0) totalEntities * 1000.0 / batchMs else 0.0

                put("batch", mapOf(
                    "batchSize"      to effectiveBatch,
                    "totalMs"        to batchMs,
                    "msPerSentence"  to "%.1f".format(msPerSent),
                    "totalEntities"  to totalEntities,
                    "entitiesPerSec" to "%.0f".format(entPerSec),
                ))
            }

            // ── Interprétation ─────────────────────────────────────────────────
            val batchEntry = (get("batch") as? Map<*, *>)
            val msPerSent  = (batchEntry?.get("msPerSentence") as? String)?.toDoubleOrNull()
                ?: (get("single") as? Map<*, *>)?.let { (it["warmMs"] as? Long)?.toDouble() }
                ?: 0.0
            put("interpretation", when {
                msPerSent <= 0   -> "No timing available"
                msPerSent < 20   -> "✅ Excellent — well within real-time range"
                msPerSent < 50   -> "✅ Good — typical for CPU DeBERTa on Apple Silicon"
                msPerSent < 100  -> "⚠ Acceptable but slower than expected — check thread config"
                else             -> "❌ Slow — possible issue: CoreML disabled, too few threads, or very long sentences"
            })
        }

        return result
    }

    // ── 9. Évaluation dédiée SVO preview ─────────────────────────────────────

    @Tool(description = """
        [SVO PREVIEW — ÉVALUATION DÉDIÉE]
        Lance l'inférence sur un texte et retourne UNIQUEMENT les informations SVO
        pour une évaluation qualitative ciblée par le LLM judge.

        ══════════════════════════════════════════════════════
        PROTOCOLE D'ÉVALUATION OBLIGATOIRE POUR LE LLM JUDGE
        ══════════════════════════════════════════════════════
        Pour chaque span dans "argumentSpans" :
          1. Lire le texte du span et son rôle (subject / object / iobj)
          2. Vérifier que le rôle est cohérent avec la phrase source (field "inputText")
          3. Vérifier l'association entité NER : entityText + entityFine corrects ?
          4. Évaluer la confiance : p_svo_bnd ≥ 0.70 → fiable, < 0.55 → suspect
          5. Si voix PASSIVE → vérifier que le sujet grammatical est bien l'agent passif
        Pour "verbSpans" :
          - Vérifier que le verbe détecté est bien le verbe principal (ou un verbe auxiliaire)
        Conclure par : nb rôles corrects / nb total, cas problématiques identifiés,
        recommandation de réglage seuil si nécessaire.
        ══════════════════════════════════════════════════════

        Rappel : tauSvoBoundary (défaut 0.50) contrôle le seuil de détection SVO.
        tauSvoAnchoredBoundary (défaut 0.40) contrôle la promotion d'entités borderline.
        Ces seuils sont INDÉPENDANTS de la calibration NER.
    """)
    fun evaluateSvoPreview(
        @ToolParam(description = "Le texte à analyser (une phrase ou un court paragraphe)")
        text: String,
    ): Map<String, Any> {
        val result = nerService.analyseSingle(text)

        val argRoles = setOf("svo_subject", "svo_object", "svo_iobj", "pron_subj", "pron_obj")
        val argSpans  = result.svoSpans.filter { it.role in argRoles }
        val verbSpans = result.svoSpans.filter { it.role == "svo_verb" }

        return mapOf(
            "inputText"    to text,
            "svoThresholds" to mapOf(
                "tauSvoBoundary"         to nerService.config.tauSvoBoundary,
                "tauSvoAnchoredBoundary" to nerService.config.tauSvoAnchoredBoundary,
            ),
            "argumentSpans" to argSpans.map { s ->
                buildMap {
                    put("text",       s.text)
                    put("role",       s.role)
                    put("p_svo_bnd",  "%.3f".format(s.svoBoundaryProb))
                    put("p_role",     "%.3f".format(s.roleProb))
                    put("voice",      "${s.voice} (%.2f)".format(s.voiceProb))
                    put("chars",      "[${s.charStart}:${s.charEnd}]")
                    if (s.entity != null) {
                        put("entityText",   s.entity!!.text)
                        put("entityFine",   s.entity!!.type)
                        put("entityCoarse", s.entity!!.metadata["coarse"] ?: "NONE")
                        put("entityPBoundary", fmt(s.entity!!.metadata["pBoundary"]))
                    } else {
                        put("entityText", null)
                        put("entityMatchNote", "Aucune entité NER n'a pu être associée à ce span SVO.")
                    }
                    s.gender?.let { put("gender", it) }
                    s.number?.let { put("number", it) }
                    put("confidence", when {
                        s.svoBoundaryProb >= 0.70f -> "✅ fiable"
                        s.svoBoundaryProb >= 0.55f -> "⚠ incertain"
                        else -> "❌ suspect (p_svo_bnd faible)"
                    })
                }
            },
            "verbSpans" to verbSpans.map { s ->
                mapOf(
                    "text"      to s.text,
                    "p_svo_bnd" to "%.3f".format(s.svoBoundaryProb),
                    "p_role"    to "%.3f".format(s.roleProb),
                    "voice"     to "${s.voice} (%.2f)".format(s.voiceProb),
                    "chars"     to "[${s.charStart}:${s.charEnd}]",
                )
            },
            "nerEntitiesWithRole" to result.entities
                .filter { it.metadata["syntacticRole"] != null }
                .map { e ->
                    mapOf(
                        "text"          to e.text,
                        "fine"          to e.type,
                        "syntacticRole" to e.metadata["syntacticRole"],
                        "gender"        to (e.metadata["gender"] ?: "—"),
                        "number"        to (e.metadata["number"] ?: "—"),
                        "svoAnchored"   to (e.metadata["svoAnchored"] == true),
                    )
                },
            "summary" to mapOf(
                "totalSvoSpans" to result.svoSpans.size,
                "argumentCount" to argSpans.size,
                "verbCount"     to verbSpans.size,
                "withEntityMatch" to argSpans.count { it.entity != null },
                "withoutEntityMatch" to argSpans.count { it.entity == null },
            ),
        )
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun fmt(v: Any?): String = when (v) {
        is Float  -> "%.4f".format(v)
        is Double -> "%.4f".format(v)
        null      -> "—"
        else      -> v.toString()
    }
}

