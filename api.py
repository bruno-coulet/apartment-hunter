from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI()

# --------- INPUT SCHEMA (match Streamlit payload) ----------
class InputData(BaseModel):
    sq_mt_built: float
    n_rooms: int
    n_bathrooms: float
    neighborhood: int

    has_lift: int = 0
    has_parking: int = 0
    has_pool: int = 0
    has_garden: int = 0
    has_storage_room: int = 0
    is_floor_under: int = 0


@app.get("/")
def read_root():
    return {"message": "API d'estimation immobilière (features df_model)"}


def preprocess(payload: InputData) -> pd.DataFrame:
    """Préprocessing simple : types + bornes réalistes + dataframe 1 ligne."""
    df = pd.DataFrame([payload.model_dump()])

    # Types
    df["sq_mt_built"] = pd.to_numeric(df["sq_mt_built"], errors="coerce")
    df["n_rooms"] = pd.to_numeric(df["n_rooms"], errors="coerce")
    df["n_bathrooms"] = pd.to_numeric(df["n_bathrooms"], errors="coerce")
    df["neighborhood"] = pd.to_numeric(df["neighborhood"], errors="coerce")

    bin_cols = ["has_lift","has_parking","has_pool","has_garden","has_storage_room","is_floor_under"]
    for c in bin_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(0, 1).astype(int)

    # Bornes simples (évite valeurs incohérentes)
    df["sq_mt_built"] = df["sq_mt_built"].clip(10, 1000)
    df["n_rooms"] = df["n_rooms"].clip(0, 24)
    df["n_bathrooms"] = df["n_bathrooms"].clip(1, 16)

    # Neighborhood (dans ton dataset : 1..135)
    df["neighborhood"] = df["neighborhood"].clip(1, 135).astype(int)

    return df


@app.post("/predict")
def predict(data: InputData):
    try:
        X = preprocess(data)

        # ---------- PREDICTION FAKE (demo) ----------
        # Base €/m² (Madrid) -> juste pour une démo cohérente
        base_eur_m2 = 3500

        # Effet quartier (simple) : on simule un coefficient qui varie selon l'ID
        # (dans un vrai modèle, neighborhood serait encodé)
        neigh_factor = 0.85 + (X.loc[0, "neighborhood"] / 135) * 0.35  # ~0.85 -> ~1.20

        # Bonus équipements (en €)
        equip_bonus = (
            X.loc[0, "has_lift"] * 12000
            + X.loc[0, "has_parking"] * 18000
            + X.loc[0, "has_pool"] * 25000
            + X.loc[0, "has_garden"] * 20000
            + X.loc[0, "has_storage_room"] * 8000
        )

        # Pénalité sous-sol
        under_penalty = -15000 if X.loc[0, "is_floor_under"] == 1 else 0

        # Bonus pièces / SDB (petit bonus, car déjà lié à la surface)
        rooms_bonus = max(X.loc[0, "n_rooms"] - 1, 0) * 6000
        baths_bonus = max(X.loc[0, "n_bathrooms"] - 1, 0) * 9000

        price = (
            X.loc[0, "sq_mt_built"] * base_eur_m2 * neigh_factor
            + equip_bonus
            + under_penalty
            + rooms_bonus
            + baths_bonus
        )

        return {
            "prediction": int(price),
            "preprocessing_applied": True,
            "features_used": list(X.columns),
            "quality_factor": float(neigh_factor),
        }

    except Exception as e:
        return {"error": f"Erreur lors du preprocessing/predict: {str(e)}"}
