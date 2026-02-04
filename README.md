# apartment-hunter

🏠 **Application de prédiction de prix immobilier avec interface Streamlit**

## 🚀 Fonctionnalités

- **Interface web intuitive** : Streamlit avec formulaire de saisie complet
- **Modèle ML performant** : Random Forest avec R² = 94.74%
- **API REST** : FastAPI pour servir les prédictions
- **Dataset Madrid** : 126 quartiers, 134 features après encoding

## 📋 Installation

```bash
# Cloner le projet
git clone https://github.com/bruno-coulet/apartment-hunter.git
cd apartment-hunter

# Installer les dépendances
pip install -r requirements.txt
```

## 🔧 Utilisation

### 1. Lancer l'API
```bash
cd apartment-hunter
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Lancer l'interface Streamlit
```bash
streamlit run streamlit_app/app.py --server.port 8501
```

### 3. Accéder à l'application
- **Interface web** : http://localhost:8501
- **API documentation** : http://localhost:8000/docs

## 🏗️ Architecture

```
apartment-hunter/
├── streamlit_app/          # Interface utilisateur Streamlit
├── api.py                  # API FastAPI 
├── notebook/              # Notebooks ML (training, analysis)
├── models/                # Modèles entraînés (128MB)
├── data_model/            # Datasets train/test (11MB)
└── requirements.txt       # Dépendances Python
```

## 🎯 Performance

- **Modèle** : Random Forest Regressor
- **Score R²** : 94.74% sur le test set
- **Features** : 134 variables (surface, quartier, équipements)
- **Preprocessing** : StandardScaler + OneHotEncoder

## 💡 Utilisation du modèle

L'interface permet de saisir :
- Surface construite (m²)
- Nombre de chambres/salles de bain
- Quartier (126 options Madrid)
- Équipements (ascenseur, parking, piscine, etc.)

Estimation instantanée avec détails de l'analyse.