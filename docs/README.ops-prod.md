# PimpMyRAG - Guide ops / production

Ce guide cible l'exploitation du service en environnement de production (ou pre-prod).

## 1) Scope

Composants critiques:

- API applicative: `rag-app`
- Extraction NER/SVO ONNX: `connectors/ner/onnx-ner`
- Services externes: MongoDB, Qdrant, Infinity, UD parser

Objectif: service stable, observable, avec procedure de rollback claire.

## 2) Architecture runtime

Flux simplifie:

1. Requete API -> `rag-app`
2. Pipeline RAG -> connecteurs (NER/embed/rerank/store)
3. Ecriture/lecture sur MongoDB + Qdrant
4. Reponse client

Points de contention typiques:

- latence ONNX (CPU/CoreML)
- saturation IO des stores
- indisponibilite d'un service externe (Infinity/UD)

## 3) Configuration essentielle

Verifier/parametrer:

- `rag-app/src/main/resources/application.yml`
- `docker-compose.yml`
- `ner-demo/src/main/resources/application.properties` (environnement demo)

Variables/valeurs a figer par environnement:

- URLs des services externes
- seuils NER (`tauBoundary`, `tauNone`, `tauCoarse`, `minScore`)
- taille de batch ONNX
- nombre de threads ONNX Runtime

## 4) Demarrage et healthcheck

Demarrage minimal:

```zsh
docker-compose up -d
./gradlew :rag-app:bootRun
```

Checks recommandés:

- endpoint API joignable sur `http://localhost:8080`
- connectivite MongoDB/Qdrant/Infinity/UD
- latence p95 extraction sur un lot fixe de phrases de reference

## 4.1) Deploiement Render de `ner-demo`

Configuration repo:

- spec Render: `render.yaml`
- script bootstrap: `scripts/render/start-ner-demo-render.sh`
- port Spring: `server.port=${PORT:8090}`
- chemins modele/tokenizer via env vars: `NER_MODEL_PATH`, `NER_TOKENIZER_PATH`

Pipeline minimal:

1. Build Render: `./gradlew :ner-demo:bootJar -x test`
2. Start Render: `bash scripts/render/start-ner-demo-render.sh`
3. Push Git sur la branche suivie par Render pour declencher un deploy automatique

Pipeline automatise recommande:

1. `publish-ner-assets.yml` (manuel) pour publier ONNX/tokenizer en release GitHub
2. MAJ des variables Render (`MODEL_URL`, `MODEL_SHA256`, `TOKENIZER_URL`, `TOKENIZER_SHA256`)
3. Push sur `main` -> workflow `CI` -> workflow `deploy-render-ner-demo.yml` (hook Render)

Precondition artefacts:

- Le fichier ONNX et le dossier tokenizer ne sont pas stockes dans Git dans l'etat actuel.
- Provisionner via `MODEL_URL`/`TOKENIZER_URL` (+ checksums SHA-256) ou fournir `NER_MODEL_PATH`/`NER_TOKENIZER_PATH` precharges.
- Le service est configure pour fail-fast au boot si le chemin modele/tokenizer est invalide.

## 5) Runbook incident

### Incident A - Forte hausse de latence

Actions:

1. Verifier la saturation CPU/memoire
2. Baisser la taille de batch si spikes
3. Confirmer la disponibilite des services externes
4. Basculer temporairement sur config seuils plus stricte si backlog

### Incident B - Chute du rappel NER

Actions:

1. Verifier le modele ONNX charge (checksum/version)
2. Verifier la config des seuils runtime
3. Rejouer un jeu de phrases de reference
4. Rollback vers le dernier modele valide

### Incident C - Erreurs connecteurs externes

Actions:

1. Isoler le connecteur en echec (Infinity/UD/Qdrant/Mongo)
2. Appliquer retry/circuit breaker cote app si disponible
3. Degrader le service de facon explicite (code + message)

## 6) Checklist release

- [ ] Build vert `./gradlew build`
- [ ] Test fonctionnel NER `python scripts/ner_candidates_test.py`
- [ ] Export ONNX valide sur checkpoint cible
- [ ] Bench inferencedoc compare avec baseline (latence + throughput)
- [ ] Plan rollback documente (artefact precedent + config precedente)

## 7) Monitoring minimum

A suivre en continu:

- latence p50/p95/p99 par endpoint
- taux d'erreur 4xx/5xx
- nombre moyen d'entites par document (drift signal)
- temps d'inference ONNX par lot
- timeout/dependance externe par service

## 8) Liens utiles

- Hub documentation: `README.md`
- Guide externe: `docs/README.opensource.md`
- Guide bilingue FR/EN: `docs/README.bilingual.md`
- App detail: `rag-app/README.md`

