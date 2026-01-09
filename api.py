from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import os
import pandas as pd
import numpy as np
import traceback

app = FastAPI()

# --- CONFIGURATION DES CHEMINS ---
MODEL_PATH = "models/ridge_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
CONFIG_PATH = "models/model_config.json"

# --- VARIABLES GLOBALES ---
model = None
preprocessor = None
config = None

# --- FONCTION DE CHARGEMENT ---
def load_assets():
    global model, preprocessor, config
    try:
        # 1. Chargement de la configuration JSON
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
            print("✅ Configuration JSON chargée")
        else:
            print("⚠️ Config manquante: models/model_config.json")
            # Fallback sur les 10 colonnes si le fichier manque
            config = {"input_columns": [
                "sq_mt_built", "n_rooms", "n_bathrooms", "neighborhood",
                "has_lift", "has_parking", "has_pool", "has_garden",
                "has_storage_room", "is_floor_under"
            ]}

        # 2. Chargement du Modèle et Préprocesseur
        if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            model = joblib.load(MODEL_PATH)
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print("✅ Modèle et Préprocesseur chargés")
        else:
            print("❌ Erreur : Fichiers .pkl introuvables dans /models")
            
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation : {e}")

# Exécuter le chargement au démarrage
load_assets()

# --- SCHÉMA DE DONNÉES (Pydantic) ---
class PropertyData(BaseModel):
    sq_mt_built: float
    n_rooms: int
    n_bathrooms: int
    neighborhood: int
    has_lift: int = 0
    has_parking: int = 0
    has_pool: int = 0
    has_garden: int = 0
    has_storage_room: int = 0
    is_floor_under: int = 0
    has_parking: int
    has_pool: int
    has_garden: int
    has_storage_room: int
    is_floor_under: int

# --- ROUTES ---

@app.get("/")
def home():
    return {
        "status": "API is running",
        "model_loaded": model is not None,
        "config_loaded": config is not None
    }

@app.post("/predict")
def predict(data: PropertyData):
    try:
        # 1. Préparation des données (on ne garde que nos 10 colonnes)
        input_dict = data.model_dump()
        df_input = pd.DataFrame([input_dict])
        
        # On s'assure que l'ordre des colonnes est identique à l'entraînement
        useful_features = [
            "sq_mt_built", "n_rooms", "n_bathrooms", "neighborhood",
            "has_lift", "has_parking", "has_pool", "has_garden",
            "has_storage_room", "is_floor_under"
        ]
        df_final = df_input[useful_features]

        # 2. Transformation par le preprocessor (Scaling, OneHot, etc.)
        X_processed = preprocessor.transform(df_final)

        # 3. Prédiction brute (Le modèle répond 13.0 car il a appris des logs)
        prediction_log = model.predict(X_processed)[0]

        # 4. TRANSFORMATION INVERSE (Passage du Log à l'Euro)
        # C'est ici qu'on gère le np.exp()
        prediction_euros = np.exp(prediction_log)

        # 5. Retour du résultat
        return {
            "prediction": float(prediction_euros), # Conversion en float standard pour JSON
            "status": "success",
            "unit": "EUR"
        }

    except Exception as e:
        # En cas d'erreur (ex: quartier inconnu), on renvoie le message à Streamlit
        return {"error": str(e)}