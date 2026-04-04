UdHttpClient (Ktor Netty + coroutines)

Fichier: `src/main/kotlin/rag/connectors/ud/UdHttpClient.kt`

But: un client léger pour consommer le service UD exposé par `server.py` (FastAPI + stanza).

Dépendances (Gradle Kotlin DSL) à ajouter dans le module `build.gradle.kts` ou `build.gradle`:

```
implementation("io.ktor:ktor-client-core:2.4.3")
implementation("io.ktor:ktor-client-netty:2.4.3")
implementation("io.ktor:ktor-client-content-negotiation:2.4.3")
implementation("io.ktor:ktor-serialization-kotlinx-json:2.4.3")
implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.0")
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
```

Exemple d'exécution (depuis le module):

```
./gradlew :connectors:svo:onnx-svo:run --args=''
```

ou exécuter la `main` dans `UdHttpClient` depuis ton IDE.

Notes:
- Le client utilise `Netty` engine (Ktor) mais tu peux remplacer par `CIO` si souhaité.
- Gestion d'erreurs typée via `UdResult` et `UdError`.
- Le JSON response est mappé sur `UDResponse` / `UDToken` (ignore les champs inconnus).
- Ajuste `baseUrl`, timeouts et autres paramètres selon ton déploiement.
