
# KMongo Ingestion (UUID‑first)

Ce module montre **comment modéliser `documents`, `elements` et `sentences` avec KMongo** en production, en utilisant **UUID** comme `_id` principaux, et **des index uniques** sur `(docId, elementIndex)` et `(docId, sentenceId)`.

## Pourquoi UUID plutôt que des IDs concaténés ?
- **Robustesse & simplicité** : un UUID n’est pas limité par la longueur/format d’une concaténation et n’expose pas d’info métier sensible.
- **Interop & sharding** : UUID (subtype 4) se comporte bien avec les drivers/ORMs, et évite les collisions sur clusters.
- **Business keys séparées** : conservez vos identifiants stables (ex. `docKey`, `(docId, elementIndex)`) **dans des champs indexés**, pas comme `_id` string concaténé.

## Collections & index
- `documents` : `_id=UUID`, index sur `docKey`, `createdAt`.
- `elements`  : `_id=UUID`, **index unique** `(docId, elementIndex)`, index `(docId, type)`, `(docId, metadata.pageNumber)`.
- `sentences` : `_id=UUID`, **index unique** `(docId, sentenceId)`, index `(docId, elementId)`, `(docId, pageNumber)`, `(docId, sectionTitle)`.

## Démarrage
```bash
./gradlew clean build
MONGO_URI="mongodb://localhost:27017" MONGO_DB="ingestion"   java -jar build/libs/kmongo-ingestion-0.1.0.jar
```

## Exemple d’ingestion
Voir `IngestService.kt` : `ingest(elementsFromUnstructured, filename, mime)`
- insère 1 `DocumentRecord`
- insère les `ElementRecord` correspondants (avec `elementIndex` séquentiel)
- segmente naïvement en phrases → `SentenceRecord` (remplace par ICU en prod)

## Notes
- Le DTO `ElementLike` correspond au **JSON retourné par Unstructured** (sous‑ensemble utile) ; vous pouvez le **mapper directement** depuis votre export.
- Vous pouvez propager `sectionTitle` en parcourant les `elements` et en mémorisant le dernier `Title` rencontré.
- Pour une empreinte stable supplémentaire, conservez un `docKey` (ex. `sha256(path|size|mtime)`), mais gardez `_id` en UUID.
