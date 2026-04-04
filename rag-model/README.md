# rag-model

Modèles de données partagés par tous les modules. **Aucune dépendance externe.**

## Contenu

| Type | Description |
|---|---|
| `RagDocument` | Document de base : texte, id, metadata |
| `RagUnitType` | Granularité de traitement (`DOCUMENT / PARAGRAPH / SENTENCE / …`) |
| `UDDocument` | Racine de l'analyse Universal Dependencies |
| `UDSentence` | Phrase UD avec offsets globaux dans le document |
| `UDToken` | Token UD complet : `id`, `lemma`, `upos`, `head`, `deprel`, `feats`, offsets char |
| `UDFeats` | Features morphologiques UD v2 (genre, nombre, voix, temps, mode…) |
| `UPOS` | POS universel UD — 17 valeurs (NOUN, PROPN, VERB, ADJ…) |
| `Entity` / `Span` | Entité NER avec span caractère + tokens UD |
| Enums morpho | `GenderValue`, `NumberValue`, `VoiceValue`, `TenseValue`, `MoodValue`… |

## Points clés

- `UDToken.feats.voice` : `VoiceValue.PASS` → argument passif (agent/patient flippés)
- `UDToken.deprel` : relation UD brute (`"nsubj"`, `"obl:agent"`, `"flat:name"`…)
- `UDToken.head` : id du token parent (0 = root de la phrase)

## Dépendances

Aucune. Socle de tous les autres modules.

