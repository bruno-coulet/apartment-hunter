# 1. Image de base : stable et légère
FROM python:3.12-slim

# 2. Installation de 'uv' : on récupère l'exécutable depuis l'image officielle
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Répertoire de travail
WORKDIR /app

# 4. Copie des fichiers de configuration uniquement
# On fait cela en premier pour profiter du cache Docker (plus rapide)
COPY pyproject.toml uv.lock ./

# 5. Installation des dépendances
# --frozen : utilise strictement le fichier lock
# --no-dev : n'installe PAS matplotlib/seaborn (inutile en production)
RUN uv sync --frozen --no-cache --no-dev

# 6. Copie du reste du projet (ton code, ton modèle .pkl)
COPY . .

# 7. Port utilisé par l'application
EXPOSE 8000

# 8. Commande pour lancer l'API
# uv run permet d'exécuter dans l'environnement virtuel créé par uv sync
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]