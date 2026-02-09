# Apartment Hunter - Prédiction de Prix Immobilier

> Système de prédiction de prix pour appartements et maisons utilisant des algorithmes de Machine Learning optimisés selon la taille du dataset.

## **Performances**

| Type de Bien | Algorithme | R² Score | MAE | Stratégie |
|--------------|------------|----------|-----|-----------|
| **Appartements** | GradientBoosting | **77.81%** | 147,911€ | Feature Selection + Grid Search |
| **Maisons** | RandomForest | **79.51%** | 285,420€ | Modèle simple (anti-overfitting) |

## **Quick Start**

```bash
# Installation
git clone <repository>
cd apartment-hunter
python -m pip install -r requirements.txt

```

## **Architecture**

```
apartment-hunter/
├── Data & Analysis
│   ├── 1_cleaning.ipynb       # Nettoyage des données
│   ├── 2_analysis.ipynb       # Analyse exploratoire  
│   └── notebook/3_model.ipynb # Modélisation ML
├── ML Models
│   ├── api.py                 # API FastAPI
│   ├── models/                # Modèles entraînés (.pkl)
│   └── cleaning_utils.py      # Utilitaires de nettoyage
├── Frontend
│   ├── frontent/app.py        # Interface Streamlit
│   └── frontent/style.css     # Styles personnalisés
├── Documentation
│   ├── docs/methodology.md    # Méthodologie scientifique
│   ├── docs/algorithms.md     # Documentation technique
│   ├── docs/api_guide.md      # Guide API complet
│   └── docs/results.md        # Analyse des résultats
└── Deployment
    ├── Dockerfile             # Container API
    ├── Dockerfile.streamlit    # Container Frontend
    └── docker-compose.yml      # Orchestration
```

## **Méthodologie ML**

### Stratégie Adaptative par Dataset

Notre approche innovante adapte la complexité du modèle selon la taille du dataset :

#### **Appartements (19,125 échantillons)**
- - **Feature Selection** (SelectKBest) : 6 → 5 features
- - **Grid Search** : Optimisation hyperparamètres
- - **GradientBoosting** : Algorithme complexe robuste

#### **Maisons (2,617 échantillons)**  
- - **Pas de Feature Selection** : Toutes les features conservées
- - **Pas de Grid Search** : Évite l'overfitting
- - **RandomForest** : Algorithme simple et robuste

### Algorithmes Comparés

| Algorithme | Appartements R² | Maisons R² | Complexité | Usage |
|------------|----------------|------------|------------|-------|
| **RandomForest** | 75.32% | **79.51%** | Moyenne | - Maisons |
| **Ridge** | 72.33% | 52.64% | Faible | - |
| **GradientBoosting** | **77.81%** | 78.37% | Élevée | - Appartements |

## **Features Utilisées**

### Appartements (après sélection)
1. `sq_mt_built` - Surface construite (Score: 45,321)
2. `n_bathrooms` - Nombre de salles de bain (Score: 18,929)
3. `n_rooms` - Nombre de pièces (Score: 5,384)
4. `has_lift` - Présence ascenseur (Score: 1,702)
5. `has_parking` - Parking disponible (Score: 1,136)

### Maisons (toutes conservées)
1. `sq_mt_built` - Surface construite
2. `n_bathrooms` - Nombre de salles de bain  
3. `n_rooms` - Nombre de pièces
4. `has_garden` - Présence jardin
5. `has_pool` - Présence piscine
6. `neighborhood` - Quartier

## **Données**

- **Sources** : Données immobilières nettoyées
- **Appartements** : 19,125 propriétés
- **Maisons** : 2,617 propriétés
- **Split** : 80% train / 20% test
- **Validation** : Cross-validation 5-fold pour Grid Search

## **API Usage**

### Prédiction Appartement
```bash
curl -X POST http://localhost:8000/predict/appartements \
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

### Prédiction Maison
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

## **Interface Web**

Interface Streamlit intuitive accessible sur `http://localhost:8501`

**Fonctionnalités :**
- Sélection type de bien (appartement/maison)
- Formulaire adaptatif selon le type
- Prédiction en temps réel
- Interface responsive et moderne

## **Résultats Détaillés**

### Évolution des Performances

| Étape | Appartements R² | Maisons R² | Amélioration |
|-------|----------------|------------|-------------|
| **Baseline** | 75.32% (RF) | 61.11% (RF) | - |
| **Avec Feature Selection** | 77.17% (GB) | 63.43% (GB) | +2% / +2% |
| **Avec Grid Search** | **77.81% (GB)** | **79.51% (RF)** | +0.6% / +16% |

### Points Clés
- **Appartements** : Feature selection + Grid Search = gain de 2.5%
- **Maisons** : Suppression feature selection = **gain de 16%** -
- **Anti-overfitting** : Stratégie adaptative cruciale pour petits datasets

## **Technologies**

**Backend**
- ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
- ![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)
- ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)

**Frontend**
- ![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
- ![CSS3](https://img.shields.io/badge/CSS3-Custom-blue.svg)

**MLOps**
- ![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)
- ![Pickle](https://img.shields.io/badge/Pickle-Model%20Persistence-green.svg)

## **Travaux Académiques**

Ce projet répond aux exigences suivantes :
- - Comparaison de 3+ algorithmes ML
- - Feature Selection avec justification
- - Grid Search et optimisation hyperparamètres  
- - Validation croisée
- - Analyse de performance détaillée
- - Déploiement en production
- - Interface utilisateur fonctionnelle

## **Documentation Complète**

- [Méthodologie](docs/methodology.md) - Approche scientifique détaillée
- [Algorithmes](docs/algorithms.md) - Explication technique des modèles
- [Guide API](docs/api_guide.md) - Documentation complète de l'API
- [Résultats](docs/results.md) - Analyse approfondie des performances

## 👨‍💻 **Auteur**

**Sulivan Moreau**  
Projet académique - Prédiction de prix immobilier par Machine Learning

---

> **Innovation** : Stratégie adaptative selon la taille du dataset pour optimiser les performances et éviter l'overfitting.
- Quartier (126 options Madrid)
- Équipements (ascenseur, parking, piscine, etc.)

Estimation instantanée avec détails de l'analyse.