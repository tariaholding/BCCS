# 🏛️ BCCS — Base de Connaissance Citoyenne Souveraine

> **IA citoyenne souveraine · RAG privé · 100 % on-premise · Données qui restent sur le territoire**

[![Licence EUPL-1.2](https://img.shields.io/badge/Licence-EUPL--1.2-blue.svg)](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-green.svg)](https://www.python.org/)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.com/)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-EF4444.svg)](https://qdrant.tech/)

---

## 🎯 Contexte et problématique

Les collectivités territoriales françaises font face à un triple défi :

1. **Engorgement des accueils physiques** — demandes répétitives sur les horaires, les démarches administratives, les règlements d'urbanisme.
2. **Données sensibles** — informations sur les aides sociales, données personnelles des administrés : elles ne peuvent pas transiter vers des serveurs tiers (RGPD, souveraineté numérique).
3. **Disponibilité 24/7** — les citoyens ont besoin de réponses en dehors des heures d'ouverture.

**BCCS** résout ces trois points avec une architecture entièrement locale : aucune donnée ne quitte le serveur de la collectivité.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Réseau PUBLIC                      │
│  ┌──────────────────────────────────────────────┐   │
│  │            bccs-app  (port 8501)             │   │
│  │         Interface citoyenne Streamlit         │   │
│  └──────────────┬──────────────────────────────-┘   │
└─────────────────┼───────────────────────────────────┘
                  │ Réseau INTERNE privé (bccs-internal)
     ┌────────────┴────────────┐
     │                         │
┌────▼─────┐           ┌───────▼──────┐
│  ollama  │           │    qdrant    │
│  LLM     │           │  Vector DB   │
│ :11434   │           │  :6333       │
│(interne) │           │ (interne)    │
└──────────┘           └──────────────┘
```

### Composants

| Composant | Rôle | Technologie |
|-----------|------|-------------|
| **bccs-app** | Interface citoyenne + orchestration RAG | Python / Streamlit / LangChain |
| **ollama** | Inférence LLM locale (aucune API externe) | Ollama + Mistral 7B / Mixtral |
| **qdrant** | Base vectorielle pour la recherche sémantique | Qdrant OSS |

---

## 📁 Structure du projet

```
bccs-souveraine-2026/
├── .github/
│   └── workflows/
│       ├── ci.yml            # Tests et lint automatiques
│       └── docker-build.yml  # Build et publication de l'image
├── data/
│   ├── raw/                  # Documents sources (PDF, CSV, HTML)
│   ├── processed/            # Chunks indexés (gitignorés)
│   ├── qdrant_storage/       # Persistance Qdrant (gitignorée)
│   └── ollama_models/        # Modèles LLM (gitignorés)
├── src/
│   ├── ingestion/            # Pipeline d'indexation des documents
│   │   ├── loader.py         # Chargeurs PDF, CSV, HTML, DOCX
│   │   ├── chunker.py        # Découpage sémantique des textes
│   │   └── embedder.py       # Génération des embeddings
│   ├── retrieval/            # Moteur RAG
│   │   ├── retriever.py      # Recherche vectorielle Qdrant
│   │   └── reranker.py       # Re-ranking des résultats
│   ├── generation/           # Génération de réponses
│   │   ├── chain.py          # Chaîne LangChain RAG
│   │   └── prompts.py        # Prompts système (ton officiel, RGPD-safe)
│   ├── api/                  # API interne FastAPI
│   │   └── main.py
│   └── ui/                   # Interface Streamlit
│       └── app.py
├── docker-compose.yml        # Stack production complète
├── Dockerfile                # Image bccs-app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Démarrage rapide

### Prérequis

- Docker Engine ≥ 24 et Docker Compose ≥ 2.20
- GPU NVIDIA recommandé (CUDA 12+) — fonctionne aussi en CPU
- 16 Go RAM minimum (32 Go recommandés pour Mixtral)

### Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/VOTRE_ORG/bccs-souveraine-2026.git
cd bccs-souveraine-2026

# 2. Copier et adapter la configuration
cp .env.example .env
# Éditer .env : nom de la collectivité, modèle LLM, etc.

# 3. Démarrer la stack complète
docker compose up -d

# 4. Télécharger le modèle LLM (première fois)
docker exec bccs-ollama ollama pull mistral:7b-instruct

# 5. Indexer vos premiers documents
docker exec bccs-app python src/ingestion/loader.py --source data/raw/
```

L'interface est accessible sur **http://localhost:8501**

---

## 📄 Sources de données supportées

| Format | Exemples |
|--------|----------|
| PDF | Règlements d'urbanisme, délibérations, PLU |
| CSV / Excel | Horaires, aides sociales, annuaires |
| HTML | Pages du site institutionnel |
| DOCX | Procédures internes, guides usagers |
| Markdown | Bases de connaissances existantes |

---

## 🔐 Sécurité et souveraineté

- **Zéro exfiltration** : aucun appel à une API externe (OpenAI, Anthropic, Google…)
- **RGPD natif** : les données restent sur l'infrastructure de la collectivité
- **Réseau isolé** : LLM et base vectorielle inaccessibles depuis l'extérieur
- **Audit trail** : chaque question/réponse peut être journalisée localement
- **Licence EUPL-1.2** : compatible avec les exigences des marchés publics français

---

## 🤝 Contribuer

Les contributions sont les bienvenues ! Merci de lire [CONTRIBUTING.md](CONTRIBUTING.md) avant de soumettre une PR.

Ce projet est publié sous licence **EUPL-1.2** — European Union Public Licence.

---

*Projet porté par la communauté des agents du service public numérique français.*
