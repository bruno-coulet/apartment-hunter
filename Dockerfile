# Utilisation de Python 3.12 (plus performant)
FROM python:3.12-slim

# On récupère l'exécutable uv depuis l'image officielle
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# On copie uniquement les fichiers de configuration uv (indispensable pour le cache)
COPY pyproject.toml uv.lock ./

# Installation des dépendances avec uv (fini pip et requirements.txt !)
RUN uv sync --frozen --no-cache --no-dev

# Copier tout le projet (api.py, cleaning_utils.py, dossiers models et data_model)
COPY . .

# Configuration de l'environnement
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Lancement sécurisé via uv
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]