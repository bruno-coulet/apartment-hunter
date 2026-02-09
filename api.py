"""API FastAPI pour la prédiction de prix immobilier.

Expose une route de santé et une route de prédiction s'appuyant sur
un modèle scikit-learn et son préprocesseur.
"""

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
    """Charge la configuration, le modèle et le préprocesseur en mémoire."""
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
            
            # Extraire les catégories valides du OneHotEncoder
            print("\n📊 Catégories du preprocessor:")
            for name, transformer, cols in preprocessor.transformers_:
                print(f"  {name}: {cols}")
                if name == "cat" and hasattr(transformer, 'named_steps'):
                    onehot = transformer.named_steps.get('onehotencoder')
                    if onehot and hasattr(onehot, 'categories_'):
                        for i, col in enumerate(cols):
                            print(f"    - {col}: {list(onehot.categories_[i][:10])}...")
        else:
            print("❌ Erreur : Fichiers .pkl introuvables dans /models")
            
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation : {e}")

# Exécuter le chargement au démarrage
load_assets()

# --- SCHÉMA DE DONNÉES (Pydantic) ---
class PropertyData(BaseModel):
    sq_mt_built: int
    n_rooms: int
    n_bathrooms: int
    neighborhood: int
    has_lift: int = 0
    has_parking: int = 0
    has_pool: int = 0
    has_garden: int = 0
    has_storage_room: int = 0
    is_floor_under: int = 0

# --- ROUTES ---

@app.get("/")
def home():
    """Retourne l'état de santé de l'API."""
    return {
        "status": "API is running",
        "model_loaded": model is not None,
        "config_loaded": config is not None
    }

@app.post("/predict")
def predict(data: PropertyData):
    """Génère une prédiction de prix à partir des caractéristiques reçues."""
    try:
        # 1. Préparation des données
        input_dict = data.model_dump()
        df_input = pd.DataFrame([input_dict])
        
        print(f"\n📥 Input reçu: {input_dict}")
        
        # 1b. Conserver le type utilisé à l'entraînement (entier)
        #    Les catégories du OneHotEncoder sont des entiers
        df_input["neighborhood"] = df_input["neighborhood"].astype("int")
        print(f"   neighborhood (dtype int): {df_input['neighborhood'].iloc[0]}")
        
        # 2. Sélectionner les 10 colonnes dans le bon ordre
        useful_features = [
            "sq_mt_built", "n_rooms", "n_bathrooms", "neighborhood",
            "has_lift", "has_parking", "has_pool", "has_garden",
            "has_storage_room", "is_floor_under"
        ]
        df_final = df_input[useful_features]
        
        print(f"📋 DataFrame:\n{df_final}")
        print(f"   Types: {df_final.dtypes.to_dict()}")
        
        # 3. Transformation par le preprocessor
        X_processed = preprocessor.transform(df_final)
        
        print(f"✅ Preprocessing OK - shape: {X_processed.shape}")
        print(f"   Min: {X_processed.min():.6f}, Max: {X_processed.max():.6f}")
        print(f"   Valeurs (premiers 15): {X_processed[0][:15]}")
        
        # 4. Prédiction (en LOG)
        prediction_log = model.predict(X_processed)[0]
        
        print(f"📊 Prédiction LOG: {prediction_log:.6f}")
        
        # 5. Vérifier que la prédiction est valide
        if np.isnan(prediction_log) or np.isinf(prediction_log):
            print(f"❌ Prédiction LOG invalide: {prediction_log}")
            return {"error": f"Prédiction invalide: {prediction_log}"}
        
        # 6. Conversion inverse (LOG1P -> EUROS)
        prediction_euros = np.expm1(prediction_log)
        
        print(f"💰 Prédiction EUROS: {prediction_euros:.2f}")
        
        # Vérifier le résultat final
        if np.isnan(prediction_euros) or np.isinf(prediction_euros):
            print(f"❌ Prix final invalide: {prediction_euros}")
            return {"error": f"Prix final invalide après conversion"}
        
        return {
            "prediction": float(prediction_euros),
            "prediction_log": float(prediction_log),
            "status": "success"
        }

    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    # --- Cartouche ---
    # Fichier : api.py
    # Rôle : API de prédiction (FastAPI)
    # Date : 2026-02-07