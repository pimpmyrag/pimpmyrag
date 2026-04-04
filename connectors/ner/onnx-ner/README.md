# connectors/ner/onnx-ner

Pipeline NER à deux niveaux produisant des `EntityCandidate` enrichis
avec métadonnées UD, morphologie et distance au trigger eventlet.

## Pipeline

```
1. OnnxBilouEntityExtractor  (XLM-RoBERTa BILOU)
   → spans coarse : PER / LOC / ORG / TIME / EVENT / OBJECT

2. mergeNerLabelWithUD
   → raffinement des frontières de span via l'arbre UD
   → rogner DET/ADP/PUNCT en tête
   → split rôle/nom : "général De Gaulle" → ["général"] + ["De Gaulle"]

3. OnnxSpanNerExtractor  (DeBERTa-v3-base SpanClassifier)
   → 22 labels fin-grained (HINT_PERSON_NAME, HINT_GROUP_ROLE, HINT_GPE…)
   → masquage structurel COARSE_TO_FINE (garantit cohérence coarse/fine)

4. buildEntityCandidates
   → EntityCandidate avec tous les champs UD + NER
```

## EntityCandidate — champs clés

```kotlin
data class EntityCandidate(
    val text:            String,        // "De Gaulle"
    val lemma:           String,        // "gaulle"
    val nerType:         NerCoarseType, // PER
    val nerHint:         EntityType,    // HINT_PERSON_NAME
    val headDeprel:      String?,       // "nsubj" | "obj" | "obl:agent" | …
    val headUpos:        UPOS?,         // PROPN
    val feats:           UDFeats?,      // voice, gender, number, tense…
    val hopFromTrigger:  Int,           // distance BFS depuis le trigger UD
    val isDirectChildOfTrigger: Boolean // hop == 1
)
```

## Labels fin-grained (22)

| Coarse | Fine-grained |
|---|---|
| PER | `HINT_PERSON_NAME`, `HINT_PERSON_ROLE`, `HINT_NORP`, `HINT_GROUP_ROLE` |
| LOC | `HINT_GPE`, `HINT_FAC_NAME`, `HINT_LOC_GENERIC`, `HINT_INFRA` |
| ORG | `HINT_ORG_NAME` |
| TIME | `HINT_TIME_DATE`, `HINT_TIME_CLOCK`, `HINT_TIME_DURATION` |
| EVENT | `HINT_EVENT_NOMINAL`, `HINT_EVENT_NAMED` |
| OBJECT | `HINT_WEAPON`, `HINT_VEHICLE`, `HINT_SUBSTANCE`, `HINT_FOOD`, `HINT_TOOL`, `HINT_OBJECT_GENERIC`, `HINT_OBJECT_NAME`, `HINT_QUANTITY` |

## Hop distance (usage eventlet)

```kotlin
// Dans la couche eventlet — associer les candidats au trigger :
val args = candidates
    .map { it.withHopFrom(triggerToken.id, sentence.tokens) }
    .filter { it.hopFromTrigger <= 2 }

// Features LR prêtes :
val features = floatArrayOf(
    it.hopFromTrigger.toFloat(),
    if (it.isDirectChildOfTrigger) 1f else 0f,
    it.nerType.ordinal.toFloat(),
    it.nerHint.ordinal.toFloat(),
    if (it.feats?.voice == VoiceValue.PASS) 1f else 0f,
    // headDeprel one-hot …
)
```

## Modèles requis

| Fichier | Description |
|---|---|
| `best_model-v2.onnx` + `.data` | DeBERTa-v3-base SpanClassifier exporté |
| `debertav3-ner/tokenizer_from_hf/` | Tokenizer local |
| XLM-RoBERTa BILOU | Modèle NER coarse (chemin dans `application.yml`) |

## Résultats courants (POC v0 — 6k phrases)

```
Global      : 78%
LOC         : 97%   TIME  : 88%   PER   : 87%
ORG         : 77%   EVENT : 71%   OBJECT: 43%
Latence moy.: 254 ms/texte (batch 90)
```

Cible après retraining 15k phrases équilibrées : **~90%**

## Training

Voir [`training/README.md`](../../../training/README.md).

