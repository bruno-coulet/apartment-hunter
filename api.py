from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import os
import pandas as pd
import numpy as np
import traceback

app = FastAPI()

# Initialisation par défaut
model = None
preprocessor = None
model_config = None
load_error = None

# Définir les chemins vers tes fichiers dans le dossier models/
MODEL_PATH = "models/ridge_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
CONFIG_PATH = "models/model_config.json"

# Chargement au démarrage
try:
    # Charger la configuration
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            model_config = json.load(f)
        print("✅ Configuration modèle chargée")
    else:
        print(f"⚠️ Config manquante: {CONFIG_PATH}")
    
    # Charger le modèle
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Modèle Ridge chargé")
    else:
        load_error = f"Modèle manquant: {MODEL_PATH}"
        print(f"❌ {load_error}")
    
    # Charger le preprocessor
    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("✅ Préprocesseur chargé")
    else:
        load_error = f"Préprocesseur manquant: {PREPROCESSOR_PATH}"
        print(f"❌ {load_error}")
    
    if model and preprocessor and model_config:
        print("✅ Système prêt pour les prédictions")
        
except Exception as e:
    print("❌ Erreur de chargement détaillée:")
    traceback.print_exc()
    load_error = str(e)



# --------- INPUT SCHEMA ----------
class InputData(BaseModel):
    sq_mt_built: float
    n_rooms: int
    n_bathrooms: float
    floor: int
    is_floor_under: int = 0
    rent_price: float
    buy_price_by_area: float
    is_renewal_needed: int = 0
    is_new_development: int = 0
    has_central_heating: int = 0
    has_individual_heating: int = 0
    has_ac: int = 0
    has_fitted_wardrobes: int = 0
    has_lift: int = 0
    is_exterior: int = 0
    has_garden: int = 0
    has_pool: int = 0
    has_terrace: int = 0
    has_balcony: int = 0
    has_storage_room: int = 0
    is_accessible: int = 0
    has_green_zones: int = 0
    has_parking: int = 0
    product: str
    neighborhood: int

@app.get("/")
def read_root():
    return {
        "message": "API d'estimation immobilière Madrid",
        "status": "ready" if model and preprocessor else "error",
        "error": load_error
    }

@app.get("/config")
def get_config():
    """Retourne la configuration du modèle (colonnes attendues, etc)"""
    if not model_config:
        return {"error": "Configuration non disponible"}
    
    return {
        "input_columns": model_config["input_columns"],
        "n_input_features": len(model_config["input_columns"]),
        "n_preprocessed_features": len(model_config["preprocessed_columns"]),
        "model_type": model_config["model_type"]
    }

@app.get("/columns")
def get_columns():
    """Retourne les listes complètes des colonnes"""
    if not model_config:
        return {"error": "Configuration non disponible"}
    
    return model_config

def preprocess(payload: InputData) -> pd.DataFrame:
    df = pd.DataFrame([payload.model_dump()])
    
    # Conversion forcée en numérique pour toutes les colonnes numériques
    numeric_cols = [
        "sq_mt_built", "n_rooms", "n_bathrooms", "floor", 
        "rent_price", "buy_price_by_area"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Nettoyage des colonnes binaires (0/1)
    binary_cols = [
        "is_floor_under", "is_renewal_needed", "is_new_development",
        "has_central_heating", "has_individual_heating", "has_ac",
        "has_fitted_wardrobes", "has_lift", "is_exterior", "has_garden",
        "has_pool", "has_terrace", "has_balcony", "has_storage_room",
        "is_accessible", "has_green_zones", "has_parking"
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1).astype(int)

    return df

@app.post("/predict")
def predict(data: InputData):
    if model is None or preprocessor is None:
        return {"error": "Modèle ou préprocesseur non chargé"}

    try:
        # 1. Créer DataFrame depuis les données d'entrée
        df = preprocess(data)
        
        # 2. Vérifier que toutes les colonnes requises sont présentes
        required_cols = model_config["input_columns"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            return {"error": f"Colonnes manquantes: {missing_cols}"}
        
        # 3. S'assurer de l'ordre des colonnes
        df = df[required_cols]
        
        # 4. Transformation avec le preprocessor
        X_transformed = preprocessor.transform(df)
        
        # 5. Prédiction
        prediction = model.predict(X_transformed)[0]
        
        return {
            "prediction": float(prediction),
            "status": "success"
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Erreur prédiction: {str(e)}"}