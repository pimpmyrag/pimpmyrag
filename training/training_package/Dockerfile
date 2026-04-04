FROM --platform=linux/arm64 python:3.10-slim

# Installe les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie les fichiers de dépendances Python
COPY requirements.txt .

# Installe les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie tous les scripts de training
COPY train_ner.py .
COPY train.py .
COPY dataset.py .
COPY model.py .

# Copie le reste des fichiers de l'application
COPY . .

# Définit le point d'entrée par défaut (peut être écrasé par docker-compose)
ENTRYPOINT ["python", "train_ner.py"]
