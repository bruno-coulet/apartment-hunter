from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np



import pickle
import os
from pathlib import Path

app = FastAPI()

# --------- CHARGEMENT DU MODÈLE ET DU PREPROCESSEUR ----------
MODEL_DIR = Path("models")

# Charger le modèle entraîné
try:
    with open(MODEL_DIR / "best_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open(MODEL_DIR / "preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    
    with open(MODEL_DIR / "model_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    print(f"✅ Modèle chargé: {metadata['model_name']}")
    print(f"✅ Performance: {metadata['test_score']:.4f} R²")
    print(f"✅ Features: {len(metadata['features'])} variables")
    
except FileNotFoundError as e:
    print(f"❌ Erreur: Fichier modèle non trouvé: {e}")
    model = None
    preprocessor = None
    metadata = None

# --------- INPUT SCHEMA (match Streamlit payload) ----------
class InputData(BaseModel):
    sq_mt_built: float
    n_rooms: int
    n_bathrooms: float
    neighborhood: int
    product: str  # Type de bien (appartement, maison, etc.)

    has_lift: int = 0
    has_parking: int = 0
    has_pool: int = 0
    has_garden: int = 0
    has_storage_room: int = 0
    is_floor_under: int = 0


@app.get("/")
def read_root():
    return {
        "message": "API d'estimation immobilière avec StandardScaler",
        "model": metadata["model_name"] if metadata else "Non chargé",
        "performance": f"{metadata['test_score']:.4f} R²" if metadata else "N/A",
        "features": len(metadata['features']) if metadata else 0
    }


def preprocess_input(payload: InputData) -> pd.DataFrame:
    """Convertit les données d'entrée en DataFrame avec les bonnes colonnes et le bon ordre"""
    
    # Créer un DataFrame avec l'ordre EXACT du notebook 
    # (d'après vos métadonnées : numeric_features + categorical_features)
    data_dict = {
        # Ordre des features numériques (comme dans le notebook)
        'sq_mt_built': [payload.sq_mt_built],
        'n_rooms': [payload.n_rooms],
        'n_bathrooms': [payload.n_bathrooms],
        'has_lift': [payload.has_lift],
        'has_parking': [payload.has_parking],
        'has_pool': [payload.has_pool],
        'has_garden': [payload.has_garden],
        'has_storage_room': [payload.has_storage_room],
        'is_floor_under': [payload.is_floor_under],
        # Feature catégorielle en dernier
        'neighborhood': [payload.neighborhood]
    }
    
    df = pd.DataFrame(data_dict)
    
    # Réorganiser selon l'ordre exact des métadonnées du modèle
    if metadata and 'features' in metadata:
        # Utiliser l'ordre exact sauvegardé lors de l'entraînement
        df = df[metadata['features']]
        print(f"✅ Colonnes réorganisées selon métadonnées: {list(df.columns)}")
    
    # Forcer neighborhood en catégorie (comme dans le notebook)
    df["neighborhood"] = df["neighborhood"].astype("category")
    
    print(f"✅ DataFrame final: colonnes = {list(df.columns)}")
    print(f"✅ Types: {dict(df.dtypes)}")
    
    return df


@app.post("/predict")
def predict(data: InputData):
    """Prédiction de prix avec le modèle entraîné"""
    
    if model is None or preprocessor is None:
        return {"error": "Modèle non chargé. Vérifiez les fichiers dans /models/"}
    
    try:
        print(f"📥 Requête reçue: {data}")
        
        # 1. Preprocessing des données d'entrée
        df_input = preprocess_input(data)
        print(f"✅ DataFrame créé: {df_input}")
        print(f"✅ Shape: {df_input.shape}")
        print(f"✅ Colonnes: {list(df_input.columns)}")
        
        # 2. Appliquer le même preprocesseur que dans le notebook
        X_scaled = preprocessor.transform(df_input)
        print(f"✅ Transformation appliquée: {X_scaled.shape}")
        
        # 3. Prédiction selon le modèle utilisé
        if metadata["model_name"] == "Linear Regression":
            # Linear Regression utilise les données scalées (comme dans le notebook)
            log_price_pred = model.predict(X_scaled)[0]
            print(f"✅ Prédiction LR (données scalées): {log_price_pred}")
        else:
            # Random Forest utilise les données BRUTES avec l'ordre exact du training
            # df_input a déjà l'ordre correct grâce à preprocess_input()
            log_price_pred = model.predict(df_input)[0]
            print(f"✅ Prédiction RF (données brutes): {log_price_pred}")
            print(f"✅ Features utilisées: {list(df_input.columns)}")
        
        # 4. Conversion log -> prix réel
        price_pred = np.exp(log_price_pred)
        print(f"✅ Prix final: {price_pred}")
        
        result = {
            "prediction": int(price_pred),
            "log_prediction": float(log_price_pred),
            "model_used": metadata["model_name"],
            "preprocessing_applied": True,
            "features_count": X_scaled.shape[1],
            "input_data": data.dict(),
            "r2_score": metadata["test_score"]
        }
        
        print(f"📤 Réponse envoyée: {result}")
        return result

    except Exception as e:
        error_msg = f"Erreur lors de la prédiction: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {"error": error_msg}


@app.get("/model-info")
def model_info():
    """Informations détaillées sur le modèle"""
    if metadata is None:
        return {"error": "Modèle non chargé"}
    
    return {
        "model_name": metadata["model_name"],
        "performance_r2": metadata["test_score"],
        "total_features": len(metadata["features"]),
        "numeric_features": metadata["numeric_features"],
        "categorical_features": metadata["categorical_features"],
        "features_list": metadata["features"]
    }