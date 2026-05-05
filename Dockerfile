# ============================================================
# BCCS — Image Docker de l'interface citoyenne
# Build multi-stage pour minimiser la surface d'attaque
# ============================================================

# --- Stage 1 : Builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

# Dépendances système minimales pour la compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2 : Runtime ---
FROM python:3.11-slim AS runtime

LABEL maintainer="BCCS Project <contact@collectivite.fr>"
LABEL description="Base de Connaissance Citoyenne Souveraine — Interface citoyenne"
LABEL org.opencontainers.image.source="https://github.com/VOTRE_ORG/bccs-souveraine-2026"

WORKDIR /app

# Dépendances runtime (sans gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    tesseract-ocr-fra \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copie des packages depuis le builder
COPY --from=builder /install /usr/local

# Code source
COPY src/ ./src/
COPY data/ ./data/

# Utilisateur non-root pour la sécurité
RUN useradd --system --uid 1001 --create-home bccs
RUN chown -R bccs:bccs /app
USER bccs

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Europe/Paris

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
