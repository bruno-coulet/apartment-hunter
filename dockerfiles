# Dockerfile simple pour l'API FastAPI
FROM python:3.12-slim

# Configuration Python
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY . .

# Port d'exposition
EXPOSE 8000

# Démarrage de l'API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]