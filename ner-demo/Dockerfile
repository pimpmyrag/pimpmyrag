# ── Build stage ────────────────────────────────────────────────────
FROM eclipse-temurin:21-jdk-jammy AS builder
WORKDIR /build

# Pre-download Gradle wrapper
COPY gradlew gradlew.bat ./
COPY gradle/ gradle/
RUN chmod +x ./gradlew && ./gradlew --version

# Copy build descriptors first for layer caching
COPY settings.gradle.kts build.gradle.kts ./
COPY buildSrc/ buildSrc/

# Copy all sources
COPY . .

RUN ./gradlew :ner-demo:bootJar -x test --no-daemon

# ── Runtime stage ──────────────────────────────────────────────────
FROM eclipse-temurin:21-jre-jammy

# curl/unzip needed by start script; Node.js 20 needed by Vaadin dev-mode (no Gradle plugin on Gradle 9)
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip ca-certificates gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# JAR (built above)
COPY --from=builder /build/ner-demo/build/libs/ ner-demo/build/libs/

# Tokenizer versioned in the repo (NOT in LFS, ~8 MB)
COPY training/multi-head/tokenizer_export_clean/ training/multi-head/tokenizer_export_clean/

# Start script
COPY scripts/render/ scripts/render/
RUN chmod +x scripts/render/start-ner-demo-render.sh

EXPOSE 10000
CMD ["bash", "scripts/render/start-ner-demo-render.sh"]

