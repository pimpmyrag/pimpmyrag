# radar-nli-toolkit

Code Kotlin prêt à l'emploi pour :
- **lire** un pack d'axes au format JSON (`axes_pack_v2.json`),
- **construire les directions** (mean(pos) − mean(neg)),
- **calculer le radar** (scores [0,1]) pour une phrase,
- **calculer le NLI** (scores [0,1]) sur un set de probes.

> Remplace les stubs `BgeM3Embedder` et `MDebertaXnliCrossEncoder` par tes implémentations.

## Démarrage rapide

```bash
./gradlew run --quiet
```

Le `main` charge `axes_pack_v2.json` depuis `src/main/resources/` et affiche les scores **radar** et **NLI** pour une phrase de test.

## Intégration à ton projet
- Utilise `AxisBuilder.fromJson(...)` pour construire `List<SemanticAxis>`.
- `Radar(axes).compute(sentence, embedder)` renvoie `Map<String, Float>`.
- `NliSensor(cross).sense(sentence, DEFAULT_PROBES)` renvoie `Map<String, Float>`.

## Remplacer les stubs
- Remplace `io.axes.stubs.BgeM3Embedder` par ta classe réelle (ONNX + tokenizer HF).
- Remplace `io.axes.stubs.MDebertaXnliCrossEncoder` par ton cross‑encoder (mDeBERTa XNLI ONNX).

Si tu relies ONNX/DJL, décommente les dépendances dans `build.gradle.kts`.
