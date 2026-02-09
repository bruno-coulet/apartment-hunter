from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

app = FastAPI(
    title="Apartment Hunter API",
    description="API de prédiction de prix immobilier pour appartements et maisons",
    version="1.0.0"
)

# --------- CHARGEMENT DES MODÈLES ----------
MODEL_DIR = Path("models")

# Chargement des modèles spécialisés
try:
    # Modèle appartements RandomForest
    with open(MODEL_DIR / "model_appartements.pkl", "rb") as f:
        apt_data = pickle.load(f)
    
    # Modèle maisons RandomForest
    with open(MODEL_DIR / "model_maisons.pkl", "rb") as f:
        mai_data = pickle.load(f)
    
    models = {
        'appartements': {
            'model': apt_data['model'],
            'scaler': apt_data.get('scaler'),
            'features': apt_data['features'],
            'metadata': apt_data.get('metadata', {
                'model_name': 'RandomForest',
                'performance_r2': 0.7474,
                'property_type': 'appartements'
            })
        },
        'maisons': {
            'model': mai_data['model'],
            'scaler': mai_data.get('scaler'),
            'features': mai_data['features'],
            'metadata': mai_data.get('metadata', {
                'model_name': 'RandomForest',
                'performance_r2': 0.7965,
                'property_type': 'maisons'
            })
        }
    }
    
    print(f"Modèles chargés:")
    print(f"  Appartements: {models['appartements']['metadata']['model_name']} (R² = {models['appartements']['metadata']['performance_r2']:.4f})")
    print(f"  Maisons: {models['maisons']['metadata']['model_name']} (R² = {models['maisons']['metadata']['performance_r2']:.4f})")
    
except FileNotFoundError as e:
    print(f"Erreur: Fichier modèle non trouvé: {e}")
    models = {'appartements': None, 'maisons': None}

# --------- INPUT SCHEMAS SPÉCIFIQUES PAR TYPE ----------
class AppartementInput(BaseModel):
    """Schema pour les appartements"""
    property_type: str = "appartements"
    sq_mt_built: float
    n_rooms: int
    n_bathrooms: float
    has_lift: int = 0
    has_parking: int = 0
    has_central_heating: int = 0

class MaisonInput(BaseModel):
    """Schema pour les maisons"""
    property_type: str = "maisons"
    sq_mt_built: float
    n_rooms: int
    n_bathrooms: float
    has_garden: int = 0
    has_pool: int = 0
    neighborhood: int = 0
    # Colonnes dupliquées du dataset original
    n_bathrooms_1: Optional[float] = None  # sera automatiquement dupliqué
    has_pool_1: Optional[int] = None  # sera automatiquement dupliqué




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
        print(f"Colonnes réorganisées pour {property_type}: {list(df.columns)}")
    
    print(f"DataFrame final pour {property_type}: colonnes = {list(df.columns)}")
    print(f"Types: {dict(df.dtypes)}")
    
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
    scaler = model_data.get('scaler')
    features = model_data['features']
    metadata = model_data['metadata']
    
    try:
        print(f"📥 Requête {property_type} reçue: {data}")
        
        # 1. Conversion en dictionnaire et extraction des features
        input_dict = data.dict()
        input_dict.pop('property_type', None)  # Retirer le type
        
        # 2. Créer DataFrame avec les bonnes features
        df_input = pd.DataFrame([input_dict])
        
        # Pour les maisons, adapter aux features exactes du modèle entraîné
        if property_type == "maisons":
            # Ajouter les colonnes manquantes avec les valeurs correspondantes
            if 'n_bathrooms.1' not in df_input.columns:
                df_input['n_bathrooms.1'] = df_input['n_bathrooms']
            if 'has_pool.1' not in df_input.columns:
                df_input['has_pool.1'] = df_input['has_pool']
        
        df_input = df_input[features]  # Réorganiser selon les features du modèle
        
        print(f"Features extraites: {list(df_input.columns)}")
        print(f"Valeurs: {df_input.iloc[0].to_dict()}")
        
        # 3. Preprocessing si nécessaire
        if scaler is not None:
            X_processed = scaler.transform(df_input)
            print(f"Scaling appliqué")
        else:
            X_processed = df_input.values
            print(f"Pas de scaling (RandomForest)")
            
        # 4. Prédiction
        prediction = model.predict(X_processed)[0]
        print(f"Prédiction {property_type}: {prediction}")
        
        # 5. Résultat
        result = {
            "prediction": int(prediction),
            "property_type": property_type,
            "model_used": metadata.get("model_name", "Unknown"),
            "features_used": features,
            "r2_score": metadata.get("test_score", 0.0),
            "input_data": input_dict
        }
        
        print(f"📤 Réponse {property_type} envoyée: prix = {int(prediction)}€")
        return result

    except Exception as e:
        error_msg = f"Erreur lors de la prédiction {property_type}: {str(e)}"
        print(f"{error_msg}")
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


# --------- POINT D'ENTRÉE PRINCIPAL ----------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Démarrage du serveur API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)