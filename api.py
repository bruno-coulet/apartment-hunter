from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np



import pickle
import os
from pathlib import Path

app = FastAPI()

# --------- CHARGEMENT DES MODÈLES ----------
# --------- CHARGEMENT DES MODÈLES ----------
MODEL_DIR = Path("models")

# Pour l'instant, utiliser le modèle existant pour les deux types
try:
    with open(MODEL_DIR / "best_model.pkl", "rb") as f:
        base_model = pickle.load(f)
    
    with open(MODEL_DIR / "preprocessor.pkl", "rb") as f:
        base_preprocessor = pickle.load(f)
    
    with open(MODEL_DIR / "model_metadata.pkl", "rb") as f:
        base_metadata = pickle.load(f)
    
    # Utiliser le même modèle pour les deux types pour l'instant
    models = {
        'appartements': {
            'model': base_model,
            'preprocessor': base_preprocessor,  
            'metadata': base_metadata
        },
        'maisons': {
            'model': base_model,
            'preprocessor': base_preprocessor,
            'metadata': base_metadata
        }
    }
    
    print(f"✅ Modèles chargés pour appartements et maisons")
    print(f"✅ Performance: {base_metadata.get('test_score', 0):.4f} R²")
    
except FileNotFoundError as e:
    print(f"❌ Erreur: Fichier modèle non trouvé: {e}")
    models = {'appartements': None, 'maisons': None}

# --------- INPUT SCHEMAS SPÉCIFIQUES PAR TYPE ----------
class AppartementInput(BaseModel):
    """Schema pour les appartements"""
    property_type: str = "appartements"  # Type de bien
    sq_mt_built: float
    n_rooms: int
    n_bathrooms: float
    has_lift: int = 0
    has_parking: int = 0
    has_central_heating: int = 0

class MaisonInput(BaseModel):
    """Schema pour les maisons"""  
    property_type: str = "maisons"  # Type de bien
    sq_mt_built: float
    n_rooms: int
    n_bathrooms: float
    has_garden: int = 0
    has_pool: int = 0
    neighborhood: str = "Unknown"  # Ajouté pour les maisons si nécessaire


@app.get("/")
def read_root():
    available_models = [k for k, v in models.items() if v is not None]
    return {
        "message": "API d'estimation immobilière - Appartements et Maisons",
        "available_models": available_models,
        "appartements_loaded": models.get("appartements") is not None,
        "maisons_loaded": models.get("maisons") is not None
    }


def preprocess_input(payload, property_type: str) -> pd.DataFrame:
    """Convertit les données d'entrée en DataFrame selon le type de bien"""
    
    if property_type not in models or models[property_type] is None:
        raise ValueError(f"Modèle {property_type} non disponible")
    
    metadata = models[property_type]['metadata']
    
    # Convertir le payload en dictionnaire
    if hasattr(payload, 'dict'):
        data_dict = payload.dict()
    else:
        data_dict = payload
    
    # Supprimer property_type des données (pas utilisé dans le modèle)
    data_dict.pop('property_type', None)
    
    # Créer DataFrame avec une seule ligne
    df = pd.DataFrame([data_dict])
    
    # Réorganiser selon l'ordre exact des métadonnées du modèle
    if 'features' in metadata:
        # S'assurer que toutes les colonnes requises sont présentes
        missing_cols = set(metadata['features']) - set(df.columns)
        for col in missing_cols:
            df[col] = 0  # Valeur par défaut
        
        # Réorganiser dans l'ordre exact du training
        df = df[metadata['features']]
        print(f"✅ Colonnes réorganisées pour {property_type}: {list(df.columns)}")
    
    print(f"✅ DataFrame final pour {property_type}: colonnes = {list(df.columns)}")
    print(f"✅ Types: {dict(df.dtypes)}")
    
    return df


@app.post("/predict/appartements")
def predict_appartement(data: AppartementInput):
    """Prédiction de prix pour un appartement"""
    return make_prediction(data, "appartements")


@app.post("/predict/maisons") 
def predict_maison(data: MaisonInput):
    """Prédiction de prix pour une maison"""
    return make_prediction(data, "maisons")


def make_prediction(data, property_type: str):
    """Fonction générique de prédiction"""
    
    if property_type not in models or models[property_type] is None:
        return {"error": f"Modèle {property_type} non chargé"}
    
    model_data = models[property_type]
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    metadata = model_data['metadata']
    
    try:
        print(f"📥 Requête {property_type} reçue: {data}")
        
        # 1. Preprocessing des données d'entrée
        df_input = preprocess_input(data, property_type)
        print(f"✅ DataFrame créé: {df_input}")
        
        # 2. Appliquer le preprocesseur si nécessaire
        if preprocessor is not None:
            X_processed = preprocessor.transform(df_input)
            print(f"✅ Transformation appliquée: {X_processed.shape}")
        else:
            X_processed = df_input.values
            
        # 3. Prédiction
        prediction = model.predict(X_processed)[0] if hasattr(X_processed, 'shape') and len(X_processed.shape) > 1 else model.predict(df_input)[0]
        print(f"✅ Prédiction {property_type}: {prediction}")
        
        # 4. Conversion si nécessaire (prix réel vs log-prix)
        # Assumons que le modèle retourne déjà le prix réel
        price_pred = float(prediction)
        
        result = {
            "prediction": int(price_pred),
            "property_type": property_type,
            "model_used": metadata.get("model_name", "Unknown"),
            "preprocessing_applied": preprocessor is not None,
            "features_count": len(metadata.get('features', [])),
            "input_data": data.dict(),
            "r2_score": metadata.get("test_score", 0.0)
        }
        
        print(f"📤 Réponse {property_type} envoyée: {result}")
        return result

    except Exception as e:
        error_msg = f"Erreur lors de la prédiction {property_type}: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {"error": error_msg}


@app.get("/model-info")
def model_info():
    """Informations détaillées sur tous les modèles"""
    result = {}
    
    for prop_type, model_data in models.items():
        if model_data is not None:
            metadata = model_data['metadata']
            result[prop_type] = {
                "model_name": metadata.get("model_name", "Unknown"),
                "performance_r2": metadata.get("test_score", 0.0),
                "total_features": len(metadata.get("features", [])),
                "features_list": metadata.get("features", [])
            }
        else:
            result[prop_type] = {"error": "Modèle non chargé"}
    
    return result


@app.get("/model-info/{property_type}")
def model_info_specific(property_type: str):
    """Informations détaillées sur un modèle spécifique"""
    if property_type not in models:
        return {"error": f"Type de bien '{property_type}' non reconnu. Types disponibles: {list(models.keys())}"}
    
    model_data = models[property_type]
    if model_data is None:
        return {"error": f"Modèle {property_type} non chargé"}
    
    metadata = model_data['metadata']
    return {
        "property_type": property_type,
        "model_name": metadata.get("model_name", "Unknown"),
        "performance_r2": metadata.get("test_score", 0.0),
        "total_features": len(metadata.get("features", [])),
        "features_list": metadata.get("features", []),
        "model_loaded": True
    }