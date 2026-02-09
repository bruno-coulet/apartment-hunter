# Documentation Docker - Apartment Hunter

## Configuration Docker

Ce projet inclut une configuration Docker pour déployer l'API FastAPI de prédiction immobilière.

### Prérequis

- Docker Desktop installé et en cours d'exécution
- Projet cloné localement

### Structure des fichiers Docker

```
apartment-hunter/
├── Dockerfile              # Configuration Docker pour l'API
├── docker-compose.yml      # Orchestration multi-services (optionnel)
├── .dockerignore           # Fichiers à exclure du build
└── requirements.txt        # Dépendances Python
```

## Utilisation

### 1. Build de l'image Docker

Construire l'image Docker à partir du Dockerfile :

```bash
docker build -t apartment-hunter .
```

### 2. Lancement du container

Démarrer le container en exposant le port 8000 :

```bash
docker run -p 8000:8000 apartment-hunter
```

### 3. Test de l'API

Une fois le container démarré, l'API est accessible sur :

- Page d'accueil : http://localhost:8000
- Documentation interactive : http://localhost:8000/docs
- API Schema : http://localhost:8000/redoc

### 4. Test d'une prédiction

Exemple de requête POST pour prédire le prix d'une maison :

```bash
curl -X POST http://localhost:8000/predict/maisons \
-H "Content-Type: application/json" \
-d '{
    "property_type": "maisons",
    "sq_mt_built": 120.0,
    "n_rooms": 4,
    "n_bathrooms": 2.0,
    "has_garden": 1,
    "has_pool": 0,
    "neighborhood": 1
}'
```

## Configuration avancée

### Variables d'environnement

Le Dockerfile configure automatiquement :

- `PYTHONUNBUFFERED=1` : Affichage en temps réel des logs
- `WORKDIR=/app` : Répertoire de travail dans le container

### Optimisations incluses

- Image Python slim pour réduire la taille
- Cache pip désactivé pour l'optimisation de l'espace
- Port 8000 exposé pour l'accès externe
- Copie optimisée des dépendances avant le code source

### Gestion des volumes

Pour persister les données ou développer en mode live :

```bash
docker run -p 8000:8000 -v $(pwd):/app apartment-hunter
```

## Déploiement avec Docker Compose

Pour un déploiement complet incluant potentiellement d'autres services :

```bash
# Démarrage de tous les services
docker-compose up --build

# Démarrage en arrière-plan
docker-compose up -d

# Arrêt des services
docker-compose down
```

## Debugging

### Accès aux logs

Voir les logs du container en temps réel :

```bash
docker logs -f <container_id>
```

### Accès au shell du container

Pour débugger à l'intérieur du container :

```bash
docker exec -it <container_id> /bin/bash
```

### Vérification de l'état

Lister les containers en cours d'exécution :

```bash
docker ps
```

## Performance

### Métriques du container

Surveiller les ressources utilisées :

```bash
docker stats <container_id>
```

### Optimisations recommandées

- L'image utilise Python slim pour réduire la taille
- Les dépendances sont installées avant la copie du code pour optimiser le cache Docker
- Le .dockerignore exclut les fichiers inutiles (.venv, notebooks, data)

## Troubleshooting

### Port déjà utilisé

Si le port 8000 est occupé, utiliser un port différent :

```bash
docker run -p 8001:8000 apartment-hunter
```

### Problèmes de build

Vérifier que tous les fichiers requis sont présents :

- requirements.txt
- api.py
- models/ (dossier avec les modèles ML)

### Redémarrage propre

Arrêter tous les containers et redémarrer :

```bash
docker stop $(docker ps -q)
docker system prune -f
docker build -t apartment-hunter .
docker run -p 8000:8000 apartment-hunter
```

## Production

### Considérations pour la production

- Utiliser un serveur WSGI comme Gunicorn pour la production
- Configurer un reverse proxy (Nginx)
- Mettre en place un système de monitoring
- Gérer les secrets et variables d'environnement de manière sécurisée

### Exemple de déploiement production

```dockerfile
# Ajout de Gunicorn pour la production
RUN pip install gunicorn

# Démarrage avec Gunicorn
CMD ["gunicorn", "api:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```