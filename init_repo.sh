#!/usr/bin/env bash
# ============================================================
# init_repo.sh — Initialise le dépôt GitHub bccs-souveraine-2026
#
# Usage :
#   chmod +x init_repo.sh
#   ./init_repo.sh VOTRE_NOM_GITHUB
#
# Prérequis :
#   - Git configuré (git config --global user.email/name)
#   - GitHub CLI authentifié : gh auth login
# ============================================================

set -euo pipefail

GITHUB_USER="${1:-}"
REPO_NAME="bccs-souveraine-2026"

if [[ -z "$GITHUB_USER" ]]; then
  echo "❌  Usage : ./init_repo.sh VOTRE_NOM_OU_ORG_GITHUB"
  exit 1
fi

echo "🏛️  Initialisation du projet BCCS..."
echo "   Dépôt cible : github.com/${GITHUB_USER}/${REPO_NAME}"
echo ""

# --- 1. Init Git local ---
git init
git checkout -b main

# --- 2. Créer les dossiers vides avec .gitkeep ---
mkdir -p data/{raw,processed,qdrant_storage,ollama_models}
mkdir -p logs
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch logs/.gitkeep

# --- 3. Commit initial ---
git add .
git commit -m "feat: initialisation du projet BCCS 🏛️

Base de Connaissance Citoyenne Souveraine
- Structure complète du projet (data/, src/, .github/workflows/)
- .gitignore Python + Docker
- README.md technique
- docker-compose.yml production (ollama + qdrant + bccs-app)
- Réseau Docker isolé (seule l'UI est exposée au Web)
- Workflow CI GitHub Actions
- Fichier .env.example"

# --- 4. Créer le dépôt GitHub (public ou privé selon choix) ---
echo ""
echo "📡 Création du dépôt GitHub..."

gh repo create "${GITHUB_USER}/${REPO_NAME}" \
  --public \
  --description "🏛️ IA citoyenne souveraine — RAG privé pour collectivités territoriales françaises" \
  --homepage "https://github.com/${GITHUB_USER}/${REPO_NAME}" \
  --push \
  --source .

echo ""
echo "✅ Dépôt créé et code poussé sur main !"
echo "   👉 https://github.com/${GITHUB_USER}/${REPO_NAME}"
echo ""
echo "Prochaines étapes :"
echo "  1. docker compose up -d"
echo "  2. docker exec bccs-ollama ollama pull mistral:7b-instruct"
echo "  3. Déposer vos documents dans data/raw/"
echo "  4. docker exec bccs-app python src/ingestion/loader.py --source data/raw/"
