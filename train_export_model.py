"""Entraîne et exporte le modèle Ridge pour la prédiction du prix immobilier.

Ce script recharge les jeux d'entraînement/test, reconstruit le préprocesseur,
entraîne un modèle Ridge sur la cible en log, puis sauvegarde tous les artefacts
(utiles pour l'API et l'UI Streamlit).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_MODEL_DIR = ROOT / "data_model"
MODELS_DIR = ROOT / "models"

USEFUL_FEATURES = [
    "sq_mt_built",
    "n_rooms",
    "n_bathrooms",
    "neighborhood",
    "has_lift",
    "has_parking",
    "has_pool",
    "has_garden",
    "has_storage_room",
    "is_floor_under",
]

NUMERIC_FEATURES = ["sq_mt_built", "n_rooms", "n_bathrooms"]
CATEGORICAL_FEATURES = ["neighborhood"]
BINARY_FEATURES = [
    "has_lift",
    "has_parking",
    "has_pool",
    "has_garden",
    "has_storage_room",
    "is_floor_under",
]


def charger_donnees() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Charge les jeux d'entraînement et de test depuis data_model."""
    x_train = pd.read_feather(DATA_MODEL_DIR / "X_train.feather")
    y_train = pd.read_feather(DATA_MODEL_DIR / "y_train.feather").squeeze()
    x_test = pd.read_feather(DATA_MODEL_DIR / "X_test.feather")
    y_test = pd.read_feather(DATA_MODEL_DIR / "y_test.feather").squeeze()
    return x_train, y_train, x_test, y_test


def preparer_features(x: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne les colonnes utiles et homogénéise les types."""
    x = x[USEFUL_FEATURES].copy()
    # On force le quartier en texte pour un OneHotEncoder stable
    x["neighborhood"] = x["neighborhood"].astype("string")
    return x


def construire_preprocesseur() -> ColumnTransformer:
    """Construit le préprocesseur (numérique, catégoriel, binaire)."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY_FEATURES),
        ]
    )
    return preprocessor


def entrainer_modele(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[Ridge, ColumnTransformer]:
    """Entraîne un modèle Ridge avec prétraitement."""
    preprocessor = construire_preprocesseur()
    x_train_processed = preprocessor.fit_transform(x_train)
    model = Ridge()
    model.fit(x_train_processed, y_train)
    return model, preprocessor


def evaluer_modele(
    model: Ridge,
    preprocessor: ColumnTransformer,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Évalue le modèle sur le test set et retourne les métriques."""
    x_test_processed = preprocessor.transform(x_test)
    y_pred = model.predict(x_test_processed)
    return {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "y_pred_mean": float(np.mean(y_pred)),
    }


def sauvegarder_artefacts(
    model: Ridge,
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Sauvegarde modèle, préprocesseur et fichiers de configuration."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODELS_DIR / "ridge_model.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")

    config = {
        "input_columns": USEFUL_FEATURES,
        "model_type": "Ridge",
        "target": y_train.name,
        "use_log": True,
    }
    with open(MODELS_DIR / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    streamlit_config = {
        "input_columns": USEFUL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "binary_features": BINARY_FEATURES,
        "ranges": {
            col: {
                "min": float(x_train[col].min()),
                "max": float(x_train[col].max()),
                "mean": float(x_train[col].mean()),
            }
            for col in NUMERIC_FEATURES
        },
        "categorical_values": {
            col: x_train[col].dropna().unique().tolist() for col in CATEGORICAL_FEATURES
        },
    }
    with open(MODELS_DIR / "streamlit_config.json", "w", encoding="utf-8") as f:
        json.dump(streamlit_config, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Point d'entrée principal : entraînement, évaluation, export."""
    x_train, y_train, x_test, y_test = charger_donnees()

    x_train = preparer_features(x_train)
    x_test = preparer_features(x_test)

    model, preprocessor = entrainer_modele(x_train, y_train)
    metrics = evaluer_modele(model, preprocessor, x_test, y_test)
    sauvegarder_artefacts(model, preprocessor, x_train, y_train)

    print("✅ Modèle et préprocesseur exportés dans models/")
    print(
        "📊 Métriques test - "
        f"R²: {metrics['r2']:.4f} | "
        f"MAE: {metrics['mae']:.4f} | "
        f"RMSE: {metrics['rmse']:.4f} | "
        f"Moyenne prédite (log): {metrics['y_pred_mean']:.4f}"
    )


if __name__ == "__main__":
    main()

# --- Cartouche ---
# Fichier : train_export_model.py
# Rôle : réentraîner et exporter le modèle Ridge + préprocesseur
# Date : 2026-02-07
