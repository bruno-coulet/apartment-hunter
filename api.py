from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle

app = FastAPI()

# Simulation d'un pipeline de preprocessing
class PropertyPreprocessor:
    def __init__(self):
        # Simuler des statistiques du dataset d'entraînement
        self.scaler = StandardScaler()
        # Valeurs moyennes fictives pour la normalisation
        self.scaler.mean_ = np.array([85.0, 3.2, 1.8])  # surface, rooms, bathrooms
        self.scaler.scale_ = np.array([35.0, 1.1, 0.6])
        
    def preprocess(self, surface: float, rooms: int, bathrooms: int) -> np.array:
        """
        Pipeline de preprocessing :
        1. Création du DataFrame
        2. Feature engineering
        3. Normalisation
        """
        # 1. Créer DataFrame comme pendant l'entraînement
        data = pd.DataFrame({
            'surface': [surface],
            'rooms': [rooms], 
            'bathrooms': [bathrooms]
        })
        
        # 2. Feature engineering (exemple : ratios utiles)
        data['surface_per_room'] = data['surface'] / data['rooms']
        data['surface_per_bath'] = data['surface'] / data['bathrooms']
        
        # 3. Validation des données
        data['surface'] = np.clip(data['surface'], 10, 500)  # entre 10 et 500 m²
        data['rooms'] = np.clip(data['rooms'], 1, 20)
        
        # 4. Features finales pour le modèle
        features = data[['surface', 'rooms', 'bathrooms', 'surface_per_room', 'surface_per_bath']].values
        
        # 5. Normalisation (simulation)
        # Pour l'instant juste une transformation simple
        normalized_features = features / np.array([100, 5, 3, 30, 50])  # normalisation manuelle
        
        return normalized_features

# Initialiser le preprocessor
preprocessor = PropertyPreprocessor()

# Schéma d'entrée
class InputData(BaseModel):
    surface: float
    rooms: int
    bathrooms: int = 1

@app.get("/")
def read_root():
    return {"message": "API d'estimation immobilière avec pipeline de preprocessing"}

@app.post("/predict")
def predict(data: InputData):
    try:
        # 1. Preprocessing des données d'entrée
        processed_features = preprocessor.preprocess(data.surface, data.rooms, data.bathrooms)
        
        # 2. Simulation de prédiction (remplacer par votre vrai modèle)
        # Prix basé sur les features processées + logique business
        base_price = data.surface * 3500  # prix de base par m²
        room_bonus = (data.rooms - 1) * 15000  # bonus par chambre supplémentaire
        bath_bonus = (data.bathrooms - 1) * 8000  # bonus par salle de bain
        
        # Facteur de qualité basé sur les ratios (du preprocessing)
        surface_per_room = data.surface / data.rooms
        if surface_per_room > 25:  # spacieux
            quality_factor = 1.15
        elif surface_per_room < 15:  # compact
            quality_factor = 0.9
        else:
            quality_factor = 1.0
            
        final_price = (base_price + room_bonus + bath_bonus) * quality_factor
        
        # Ajout de variabilité realistic
        import random
        variation = random.uniform(-0.05, 0.1)  # -5% à +10%
        final_price = final_price * (1 + variation)
        
        return {
            "prediction": int(final_price),
            "preprocessing_applied": True,
            "features_used": ["surface", "rooms", "bathrooms", "surface_per_room", "surface_per_bath"],
            "quality_factor": quality_factor
        }
        
    except Exception as e:
        return {"error": f"Erreur lors du preprocessing: {str(e)}"}
