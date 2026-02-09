# Guide API Complet

## **Vue d'Ensemble**

L'API Apartment Hunter est construite avec **FastAPI** et offre des endpoints RESTful pour la prédiction de prix immobilier avec des modèles ML optimisés.

**Base URL** : `http://localhost:8000`

## **Démarrage Rapide**

### Installation et Lancement

```bash
# Installation des dépendances
pip install fastapi uvicorn scikit-learn pandas numpy

# Lancement du serveur
uvicorn api:app --host 0.0.0.0 --port 8000

# Vérification
curl http://localhost:8000/
```

### Documentation Interactive

- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

## **Architecture API**

### Structure des Modèles

```
models/
├── model_appartements.pkl    # GradientBoosting optimisé
└── model_maisons.pkl         # RandomForest simple
```

### Métadonnées des Modèles

Chaque modèle inclut :
```python
{
    'model': sklearn_model,           # Modèle entraîné
    'scaler': StandardScaler|None,    # Preprocessing (si requis)
    'selector': SelectKBest|None,     # Feature selection (si appliquée)  
    'features': list,                 # Features attendues
    'metadata': {                     # Informations de performance
        'model_name': str,
        'performance_r2': float,
        'performance_mae': float,
        'property_type': str,
        'grid_search_params': dict|None,
        'feature_selection': str
    }
}
```

## 🏢 **Endpoint Appartements**

### `POST /predict/appartements`

Prédiction de prix pour un appartement.

#### Paramètres Requis

```json
{
    "property_type": "appartements",    // Toujours "appartements"
    "sq_mt_built": 80.0,               // Surface construite (m²)
    "n_rooms": 3,                      // Nombre de pièces
    "n_bathrooms": 1.0,                // Nombre de salles de bain
    "has_lift": 1,                     // Ascenseur (0=Non, 1=Oui)
    "has_parking": 0,                  // Parking (0=Non, 1=Oui)
    "has_central_heating": 1           // Chauffage central (0=Non, 1=Oui)
}
```

#### Exemple de Requête

```bash
curl -X POST "http://localhost:8000/predict/appartements" \
  -H "Content-Type: application/json" \
  -d '{
    "property_type": "appartements",
    "sq_mt_built": 80.0,
    "n_rooms": 3,
    "n_bathrooms": 1.0,
    "has_lift": 1,
    "has_parking": 0,
    "has_central_heating": 1
  }'
```

#### Réponse

```json
{
    "predicted_price": 425650.32,
    "property_type": "appartements",
    "model_used": "GradientBoosting_optimized",
    "confidence_info": {
        "model_r2": 0.7781,
        "model_mae": 147911.0,
        "feature_selection": "SelectKBest_5features",
        "grid_search_applied": true
    },
    "input_features": {
        "sq_mt_built": 80.0,
        "n_rooms": 3,
        "n_bathrooms": 1.0,
        "has_lift": 1,
        "has_parking": 0,
        "has_central_heating": 1
    },
    "features_used": ["sq_mt_built", "n_bathrooms", "n_rooms", "has_lift", "has_parking"]
}
```

#### Features Sélectionnées

L'algorithme SelectKBest sélectionne automatiquement les 5 features les plus prédictives :

1. **sq_mt_built** - Surface construite (Score: 45,321.89)
2. **n_bathrooms** - Nombre de salles de bain (Score: 18,929.79)  
3. **n_rooms** - Nombre de pièces (Score: 5,384.45)
4. **has_lift** - Présence ascenseur (Score: 1,702.67)
5. **has_parking** - Parking disponible (Score: 1,136.13)

> **Note** : `has_central_heating` est fourni mais non utilisé par le modèle (feature éliminée).

---

## **Endpoint Maisons**

### `POST /predict/maisons`

Prédiction de prix pour une maison.

#### Paramètres Requis

```json
{
    "property_type": "maisons",        // Toujours "maisons"
    "sq_mt_built": 120.0,             // Surface construite (m²)
    "n_rooms": 4,                     // Nombre de pièces
    "n_bathrooms": 2.0,               // Nombre de salles de bain
    "has_garden": 1,                  // Jardin (0=Non, 1=Oui)
    "has_pool": 0,                    // Piscine (0=Non, 1=Oui)  
    "neighborhood": 1                 // Quartier (ID numérique)
}
```

#### Exemple de Requête

```bash
curl -X POST "http://localhost:8000/predict/maisons" \
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

#### Réponse

```json
{
    "predicted_price": 687230.45,
    "property_type": "maisons", 
    "model_used": "RandomForest_default",
    "confidence_info": {
        "model_r2": 0.7951,
        "model_mae": 285420.0,
        "feature_selection": "all_features",
        "grid_search_applied": false
    },
    "input_features": {
        "sq_mt_built": 120.0,
        "n_rooms": 4,
        "n_bathrooms": 2.0,
        "has_garden": 1,
        "has_pool": 0,
        "neighborhood": 1
    },
    "features_used": ["sq_mt_built", "n_bathrooms", "n_rooms", "has_garden", "has_pool", "neighborhood"]
}
```

#### Features Utilisées

Toutes les 6 features sont conservées (pas de feature selection) :

1. **sq_mt_built** - Surface construite
2. **n_bathrooms** - Nombre de salles de bain
3. **n_rooms** - Nombre de pièces  
4. **has_garden** - Présence jardin
5. **has_pool** - Présence piscine
6. **neighborhood** - Quartier (encodage numérique)

> **Stratégie** : Aucune feature selection pour éviter l'overfitting sur le petit dataset (2,617 échantillons).

---

## **Endpoint de Santé et Métadonnées**

### `GET /`

Point d'entrée principal avec informations système.

```bash
curl http://localhost:8000/
```

```json
{
    "message": "Apartment Hunter API",
    "status": "active",
    "models_loaded": {
        "appartements": "GradientBoosting_optimized",
        "maisons": "RandomForest_default"
    },
    "version": "1.0",
    "documentation": "/docs"
}
```

### `GET /health`

Vérification de l'état de l'API (si implémenté).

---

## **Gestion des Erreurs**

### Codes de Statut

| Code | Signification | Cause Typique |
|------|---------------|---------------|
| **200** | Succès | Prédiction réussie |
| **400** | Bad Request | Paramètres manquants/invalides |
| **422** | Validation Error | Types de données incorrects |
| **500** | Internal Error | Erreur du modèle ML |

### Exemples d'Erreurs

#### 400 - Paramètre Manquant

```json
{
    "detail": "Field 'sq_mt_built' is required for appartements"
}
```

#### 422 - Type Incorrect

```json
{
    "detail": [
        {
            "loc": ["body", "sq_mt_built"],
            "msg": "value is not a valid float",
            "type": "type_error.float"
        }
    ]
}
```

#### 500 - Erreur Modèle

```json
{
    "detail": "Model prediction failed: feature dimension mismatch"
}
```

---

## 🔒 **Validation des Données**

### Contraintes par Type de Bien

#### Appartements
```python
sq_mt_built: float > 0        # Surface positive
n_rooms: int >= 1             # Au moins 1 pièce
n_bathrooms: float >= 0       # Peut être 0.5 (WC)
has_lift: int in [0, 1]       # Booléen
has_parking: int in [0, 1]    # Booléen  
has_central_heating: int in [0, 1]  # Booléen
```

#### Maisons  
```python
sq_mt_built: float > 0        # Surface positive
n_rooms: int >= 1             # Au moins 1 pièce
n_bathrooms: float >= 1       # Au moins 1 SDB complète
has_garden: int in [0, 1]     # Booléen
has_pool: int in [0, 1]       # Booléen
neighborhood: int >= 0        # ID quartier positif
```

### Preprocessing Automatique

L'API applique automatiquement :

1. **Feature Selection** (appartements uniquement)
2. **Scaling** (si requis par le modèle)  
3. **Validation** des types et contraintes
4. **Imputation** des valeurs par défaut si nécessaire

---

## **Monitoring et Performance**

### Métriques Exposées

Chaque réponse inclut des informations de confiance :

```json
"confidence_info": {
    "model_r2": 0.7781,              // R² sur dataset de test
    "model_mae": 147911.0,           // Erreur absolue moyenne (€)
    "feature_selection": "...",      // Type de feature selection
    "grid_search_applied": true      // Optimisation appliquée
}
```

### Logging

L'API log automatiquement :
- Requêtes reçues avec timestamp
- Erreurs de validation  
- Prédictions effectuées
- Performance des modèles

---

## **Configuration Avancée**

### Variables d'Environnement

```bash
# Configuration du serveur
HOST=0.0.0.0
PORT=8000
RELOAD=true

# Chemins des modèles
MODELS_PATH=./models/
APPARTEMENTS_MODEL=model_appartements.pkl
MAISONS_MODEL=model_maisons.pkl

# Logging
LOG_LEVEL=INFO
```

### Optimisations Performance

```python
# Chargement unique des modèles au startup
@app.on_event("startup")
async def load_models():
    global appartements_model, maisons_model
    appartements_model = load_model("appartements")
    maisons_model = load_model("maisons")

# Cache des prédictions (optionnel)
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_predict(features_tuple):
    return model.predict([features_tuple])
```

---

## 🧪 **Tests API**

### Suite de Tests Complète

```bash
# Test appartement valide
curl -X POST http://localhost:8000/predict/appartements \
  -H "Content-Type: application/json" \
  -d '{"property_type":"appartements","sq_mt_built":80,"n_rooms":3,"n_bathrooms":1,"has_lift":1,"has_parking":0,"has_central_heating":1}'

# Test maison valide  
curl -X POST http://localhost:8000/predict/maisons \
  -H "Content-Type: application/json" \
  -d '{"property_type":"maisons","sq_mt_built":120,"n_rooms":4,"n_bathrooms":2,"has_garden":1,"has_pool":0,"neighborhood":1}'

# Test validation (erreur attendue)
curl -X POST http://localhost:8000/predict/appartements \
  -H "Content-Type: application/json" \
  -d '{"property_type":"appartements","sq_mt_built":-10}'
```

### Benchmarking

```bash
# Test de charge avec Apache Bench
ab -n 1000 -c 10 -T application/json \
   -p test_data.json \
   http://localhost:8000/predict/appartements

# Test de performance avec wrk
wrk -t4 -c100 -d30s \
    -s post_appartement.lua \
    http://localhost:8000/predict/appartements
```

---

## **Changelog API**

### v1.0 (Actuel)
- Endpoints appartements et maisons  
- Modèles optimisés différenciés
- Validation automatique des données
- Métadonnées de confiance
- Documentation Swagger

### Roadmap v1.1
- 🔄 Cache intelligent des prédictions
- 🔄 Endpoints de monitoring `/metrics`
- 🔄 Support batch predictions
- 🔄 Authentification API keys