# apartment-hunter

**Outil de prédiction de prix immobilier** basé sur FastAPI, Streamlit et Docker.

---

## 📋 Vue d'ensemble

Ce projet estime les prix immobiliers à Madrid en utilisant des modèles d'apprentissage automatique entraînés sur des données immobilières. Il expose une **API FastAPI** pour les prédictions et une **interface Streamlit** pour l'UX.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│         Streamlit UI (Port 8501)                    │
│    - Formulaire d'entrée utilisateur                │
│    - Affichage résultats                            │
└──────────────┬──────────────────────────────────────┘
               │ HTTP Requests
               ↓
┌─────────────────────────────────────────────────────┐
│         FastAPI Server (Port 8000)                  │
│    - POST /predict - Prédictions                    │
│    - GET / - Santé de l'API                         │
└──────────────┬──────────────────────────────────────┘
               │
               ↓
      ML Model + Preprocessing
```

---

## 📁 Structure du projet

```
apartment-hunter/
├── api.py                  # API FastAPI
├── streamlit_app/
│   ├── app.py             # Interface Streamlit
│   └── style.css          # Styling CSS
├── cleaning_utils.py      # Utilitaires de nettoyage
├── data_cleaned/          # Données nettoyées
├── data_model/            # Train/Test split
├── models/                # Modèles sauvegardés
├── raw_data/              # Données brutes
├── requirements.txt       # Dépendances Python
├── pyproject.toml         # Config uv + projet
├── Dockerfile             # Build Docker
├── docker-compose.yml     # Orchestration (optionnel)
└── README.md             # Documentation
```

### Notebooks (Développement)

- **1_cleaning.ipynb** - Import et nettoyage des données
- **2_analysis.ipynb** - Analyse exploratoire et sélection de variables
- **3_model.ipynb** - Entraînement et validation du modèle

---

## 🚀 Installation

### Avec `uv` (recommandé)

```bash
# Installer uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Créer environnement virtuel
uv venv

# Installer dépendances
uv pip install -r requirements.txt
```

### Avec pip classique

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

---

## 🐳 Docker

### Build l'image

```bash
docker build -t apartment-api .
```

### Lancer l'API seule

```bash
docker run -p 8000:8000 apartment-api
# L'API est disponible à http://localhost:8000
```

### Lancer avec Docker Compose (API + Streamlit)

```bash
docker-compose up
```

Puis accédez à:
- **Streamlit**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs

---

## 🔧 Utilisation

### API FastAPI

**GET /** - Vérifier la santé

```bash
curl http://localhost:8000/
```

**POST /predict** - Prédire un prix

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sq_mt_built": 100,
    "n_rooms": 3,
    "n_bathrooms": 2,
    "neighborhood": 50,
    "has_lift": 1,
    "has_parking": 1,
    "has_pool": 0,
    "has_garden": 0,
    "has_storage_room": 0,
    "is_floor_under": 0
  }'
```

### Interface Streamlit

```bash
streamlit run streamlit_app/app.py
```

Accès: http://localhost:8501

---

## 📦 Gestion des dépendances

Garder `requirements.txt` à jour avec `uv`:

```bash
uv export --format requirements-txt --no-dev -o requirements.txt
```


Pour garder le fichier requirements.txt reflète toujours la réalité (par exemple si les collègues n'utilisent pas encore uv), on peut faut le régénérer avec la commande :

```shell
uv export --format requirements-txt --no-dev -o requirements.txt
```

Arrêter l'ancien conteneur (pour libérer le port 8000) :
```shell
docker stop $(docker ps -q --filter "ancestor=apartment-api")
```


Créer l'image Docker
```shell
docker build -t apartment-api .
```

### Run l'image Docker
```shell
docker run -p 8000:8000 apartment-api
```

---

## 📝 Licence

Projet de groupe - 2026
