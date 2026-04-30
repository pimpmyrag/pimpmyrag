# ── Build stage ────────────────────────────────────────────────────
FROM eclipse-temurin:21-jdk-jammy AS builder
WORKDIR /build

# Node.js 20 requis par le plugin Vaadin pour compiler le bundle frontend
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates gnupg curl unzip && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Pre-download Gradle wrapper
COPY gradlew gradlew.bat ./
COPY gradle/ gradle/
RUN chmod +x ./gradlew && ./gradlew --version

# Copy build descriptors first for layer caching
COPY settings.gradle.kts build.gradle.kts ./
COPY buildSrc/ buildSrc/

# Copy all sources
COPY . .

# Le plugin Vaadin (com.vaadin 24.7.2) est incompatible avec Gradle 9 car il utilise
# ResolvedConfiguration.getFiles() supprimé. On switche temporairement vers Gradle 8.13
# dans ce stage Docker uniquement (sans toucher gradle-wrapper.properties versionné).
# Le plugin est injecté via sed dans build.gradle.kts (absent du repo pour ne pas casser les builds locaux Gradle 9).
RUN sed -i 's|id("io.spring.dependency-management")|id("io.spring.dependency-management")\n    id("com.vaadin") version "24.7.2"|' ner-demo/build.gradle.kts && \
    sed -i 's|gradle-9.0.0-bin.zip|gradle-8.13-bin.zip|g' gradle/wrapper/gradle-wrapper.properties

# Build du bundle Vaadin en mode production, puis du JAR Spring Boot
RUN ./gradlew :ner-demo:vaadinBuildFrontend :ner-demo:bootJar \
    -x test --no-daemon -Pvaadin.productionMode=true

# ── Runtime stage ──────────────────────────────────────────────────
# Pas de Node.js : le bundle Vaadin est pré-compilé dans le JAR (production mode)
FROM eclipse-temurin:21-jre-jammy

RUN apt-get update && apt-get install -y --no-install-recommends curl unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Modèle ONNX (708 MB) téléchargé au BUILD, pas au démarrage ─────
# Mettre cette couche AVANT le JAR pour bénéficier du cache Docker :
# si le code change mais pas l'URL du modèle, cette couche est réutilisée.
ARG MODEL_URL=https://github.com/pimpmyrag/pimpmyrag/releases/download/v1.1.0-ner-model/best_model_multitask_full.onnx
RUN mkdir -p model && \
    echo "[docker] downloading model from $MODEL_URL" && \
    curl -fL --retry 3 "$MODEL_URL" -o model/best_model_multitask_full.onnx && \
    echo "[docker] model size: $(du -sh model/best_model_multitask_full.onnx)"
ENV NER_MODEL_PATH=/app/model/best_model_multitask_full.onnx

# JAR avec bundle Vaadin pré-compilé
COPY --from=builder /build/ner-demo/build/libs/ ner-demo/build/libs/

# Tokenizer versionné dans le repo (NOT in LFS, ~8 MB)
COPY training/multi-head/tokenizer_export_clean/ training/multi-head/tokenizer_export_clean/

# Start script
COPY scripts/render/ scripts/render/
RUN chmod +x scripts/render/start-ner-demo-render.sh

EXPOSE 10000
CMD ["bash", "scripts/render/start-ner-demo-render.sh"]
