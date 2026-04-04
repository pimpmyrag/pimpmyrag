# connectors/ud/ms-ud

Client HTTP pour le parseur Universal Dependencies.
Compatible Stanza (Python) et UDPipe 2.

## Usage

```kotlin
val parser: UDParser = WebUdParser(baseUrl = "http://localhost:9000")
val udDocs: List<UDDocument> = parser.parse(listOf(ragDocument))
```

## Sortie

Chaque `UDDocument` contient :
- `sentences` : `UDSentence` avec `start`/`end` globaux dans le texte
- Par sentence : `tokens` `UDToken` avec `id`, `lemma`, `upos`, `head`, `deprel`, `feats`, offsets char

## Deprels clés pour l'eventlet

| deprel | Rôle sémantique typique |
|---|---|
| `nsubj` | Agent (voix active) |
| `nsubj:pass` | Patient (voix passive) |
| `obl:agent` | Agent (voix passive, "par X") |
| `obj` | Patient / thème |
| `obl` | Circonstant (LOC ou TIME selon NER) |
| `nmod` | Modificateur nominal |
| `flat:name` | Extension du nom propre |

## Service UD (Docker)

```zsh
docker-compose up ud-parser   # Stanza server sur :9000
```

## Dépendances

- `rag-model`
- `rag-engine`

