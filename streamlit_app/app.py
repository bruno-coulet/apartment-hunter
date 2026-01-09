import streamlit as st
import requests, os, json
import pickle
import numpy as np
import pandas as pd


# À l'intérieur de Docker, on utilise le nom du service 'api'
url = "http://api:8000/predict"

# ========= CHARGER LA CONFIGURATION STREAMLIT =========
@st.cache_resource
def load_streamlit_config():
    """Charge la configuration depuis le notebook"""
    config_paths = ["models/streamlit_config.json", "../models/streamlit_config.json"]
    
    for path in config_paths:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    
    # Configuration par défaut si fichier non trouvé
    return {
        "input_columns": ["sq_mt_built", "n_rooms", "n_bathrooms", "neighborhood", 
                         "has_lift", "has_parking", "has_pool", "has_garden", 
                         "has_storage_room", "is_floor_under"],
        "numerical_features": ["sq_mt_built", "n_rooms", "n_bathrooms"],
        "categorical_features": ["neighborhood", "product"],
        "ranges": {},
        "categorical_values": {}
    }

config = load_streamlit_config()
input_columns = config.get("input_columns", [])
numerical_features = config.get("numerical_features", [])
categorical_features = config.get("categorical_features", [])
ranges = config.get("ranges", {})
categorical_values = config.get("categorical_values", {})

# --- Load CSS gère les deux environnements (Docker et local) ---
css_locations = ["style.css", "streamlit_app/style.css"]
css_content = None

for loc in css_locations:
    if os.path.exists(loc):
        with open(loc) as f:
            css_content = f.read()
        break

if css_content:
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
else:
    st.info("Style par défaut appliqué (style.css non trouvé).")



# ---------- HEADER ----------
st.markdown("""
    <div style='display: flex; align-items: center; justify-content: space-between;'>
        <h2 style='color:#1e88e5;'>ImmoPredict</h2>
        <div style='font-size:18px;'>
            <a style='margin-right:20px;'>Estimer</a>
            <a style='margin-right:20px;'>Acheter</a>
            <a style='margin-right:20px;'>Vendre</a>
            <a style='margin-right:20px;'>Contact</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class='hero'>
    <h1 style='text-align:center; color:white; font-size:45px;'>Estimation Immobilière Instantanée</h1>
    <p style='text-align:center; color:white; font-size:20px;'>Obtenez la valeur de votre bien en quelques secondes</p>
</div>
""", unsafe_allow_html=True)

# SEARCH BAR (cosmétique)
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([4, 1])

with col1:
    adresse = st.text_input(
        "Adresse",
        placeholder="Ex : Calle de Alcalá, Madrid",
        key="search",
        label_visibility="collapsed"
    )

with col2:
    st.button("Rechercher")

# ---------- HOW IT WORKS ----------
st.markdown("<br><h2>Comment ça marche ?</h2>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
        <div class='card'>
            <h3>1. Renseignez votre bien</h3>
            <p>Surface, chambres, quartier, équipements…</p>
        </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
        <div class='card'>
            <h3>2. Analyse du marché</h3>
            <p>Comparaison avec des biens similaires.</p>
        </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
        <div class='card'>
            <h3>3. Obtenez votre estimation</h3>
            <p>Un prix instantané basé sur votre modèle.</p>
        </div>
    """, unsafe_allow_html=True)

# ---------- PROPERTY INPUTS ----------
st.markdown("<br><h2>Caractéristiques du bien</h2>", unsafe_allow_html=True)

input_values = {}

# ===== Récupérer les vraies colonnes attendues =====
if not input_columns:
    input_columns = [
        "sq_mt_built", "n_rooms", "n_bathrooms", "floor", "is_floor_under",
        "rent_price", "buy_price_by_area", "is_renewal_needed", "is_new_development",
        "has_central_heating", "has_individual_heating", "has_ac", "has_fitted_wardrobes",
        "has_lift", "is_exterior", "has_garden", "has_pool", "has_terrace", "has_balcony",
        "has_storage_room", "is_accessible", "has_green_zones", "has_parking", "product", "neighborhood"
    ]

# ===== Section 1: Infos structurelles =====
st.markdown("<h3>📐 Propriétés structurelles</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    feature_range = ranges.get("sq_mt_built", {})
    input_values["sq_mt_built"] = st.number_input(
        "Surface construite (m²)",
        min_value=float(feature_range.get("min", 10)),
        max_value=float(feature_range.get("max", 1000)),
        value=float(feature_range.get("mean", 80)),
        step=1.0
    )

with col2:
    feature_range = ranges.get("n_rooms", {})
    input_values["n_rooms"] = st.number_input(
        "Chambres",
        min_value=float(feature_range.get("min", 0)),
        max_value=float(feature_range.get("max", 24)),
        value=float(feature_range.get("mean", 3)),
        step=1.0
    )

with col3:
    feature_range = ranges.get("n_bathrooms", {})
    input_values["n_bathrooms"] = st.number_input(
        "Salles de bain",
        min_value=float(feature_range.get("min", 1)),
        max_value=float(feature_range.get("max", 16)),
        value=float(feature_range.get("mean", 2)),
        step=1.0
    )

col1, col2, col3 = st.columns(3)

with col1:
    feature_range = ranges.get("floor", {})
    input_values["floor"] = st.number_input(
        "Étage",
        min_value=float(feature_range.get("min", -1)),
        max_value=float(feature_range.get("max", 50)),
        value=float(feature_range.get("mean", 3)),
        step=1.0
    )

with col2:
    feature_range = ranges.get("rent_price", {})
    input_values["rent_price"] = st.number_input(
        "Prix de location (€)",
        min_value=float(feature_range.get("min", 0)),
        max_value=float(feature_range.get("max", 10000)),
        value=float(feature_range.get("mean", 500)),
        step=10.0
    )

with col3:
    feature_range = ranges.get("buy_price_by_area", {})
    input_values["buy_price_by_area"] = st.number_input(
        "Prix/m² (€)",
        min_value=float(feature_range.get("min", 0)),
        max_value=float(feature_range.get("max", 20000)),
        value=float(feature_range.get("mean", 5000)),
        step=100.0
    )

# ===== Section 2: Localisation =====
st.markdown("<h3>📍 Localisation</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    input_values["is_exterior"] = st.checkbox("Extérieur", value=False)

with col2:
    input_values["is_floor_under"] = st.checkbox("Sous-sol", value=False)

with col3:
    options = categorical_values.get("neighborhood", list(range(1, 136)))
    input_values["neighborhood"] = st.selectbox("Quartier", options=options)

# ===== Section 3: Équipements chauffage/climatisation =====
st.markdown("<h3>🔥 Chauffage & Climatisation</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    input_values["has_central_heating"] = st.checkbox("Chauffage central", value=False)

with col2:
    input_values["has_individual_heating"] = st.checkbox("Chauffage individuel", value=False)

with col3:
    input_values["has_ac"] = st.checkbox("Climatisation", value=False)

# ===== Section 4: Équipements généraux =====
st.markdown("<h3>✨ Équipements</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    input_values["has_fitted_wardrobes"] = st.checkbox("Dressings intégrés", value=False)
    input_values["has_lift"] = st.checkbox("Ascenseur", value=True)
    input_values["has_garden"] = st.checkbox("Jardin", value=False)

with col2:
    input_values["has_pool"] = st.checkbox("Piscine", value=False)
    input_values["has_terrace"] = st.checkbox("Terrasse", value=False)
    input_values["has_balcony"] = st.checkbox("Balcon", value=False)

with col3:
    input_values["has_storage_room"] = st.checkbox("Cave/Débarras", value=False)
    input_values["is_accessible"] = st.checkbox("Accessible handicapés", value=False)
    input_values["has_green_zones"] = st.checkbox("Zones vertes", value=False)

# ===== Section 5: Parking & État =====
st.markdown("<h3>🚗 Parking & État</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    input_values["has_parking"] = st.checkbox("Parking", value=False)

with col2:
    input_values["is_renewal_needed"] = st.checkbox("Nécessite rénovation", value=False)

with col3:
    input_values["is_new_development"] = st.checkbox("Nouveau programme", value=False)

# ===== Section 6: Type de propriété =====
st.markdown("<h3>🏠 Type de propriété</h3>", unsafe_allow_html=True)
options = categorical_values.get("product", ["piso", "casa", "chalet adosado", "chalet pareado", "duplex", "estudio", "finca", "casa o chalet"])
input_values["product"] = st.selectbox("Type", options=options)

# ---------- CALL API ----------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Estimer le bien"):
    # Construire le payload avec toutes les colonnes attendues par l'API
    expected_columns = [
        "sq_mt_built", "n_rooms", "n_bathrooms", "floor", "is_floor_under",
        "rent_price", "buy_price_by_area", "is_renewal_needed", "is_new_development",
        "has_central_heating", "has_individual_heating", "has_ac", "has_fitted_wardrobes",
        "has_lift", "is_exterior", "has_garden", "has_pool", "has_terrace", "has_balcony",
        "has_storage_room", "is_accessible", "has_green_zones", "has_parking", "product", "neighborhood"
    ]
    
    payload = {}
    missing = []
    
    for col in expected_columns:
        if col in input_values:
            payload[col] = input_values[col]
        else:
            missing.append(col)
    
    if missing:
        st.warning(f"⚠️ Colonnes manquantes: {missing}")

    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()
            # Vérification de sécurité pour éviter le 'NoneType'
            if result is not None and isinstance(result, dict):
                if 'prediction' in result:
                    st.success(f"💰 **Valeur estimée : {result['prediction']:,.0f} €**")
                else:
                    st.error(f"Clé 'prediction' absente de la réponse. Reçu : {result}")
            else:
                st.error("L'API a renvoyé une réponse vide ou invalide.")

            # Affichage principal du prix
            st.success(f"💰 **Valeur estimée : {result['prediction']:,.0f} €**")

            # Détails (si ton API renvoie ces champs)
            with st.expander("🔍 Détails de l'analyse"):
                st.write("**Payload envoyé au modèle :**")
                st.json(payload)

                st.write("**Infos API :**")
                st.write(f"✅ Preprocessing: {result.get('preprocessing_applied', 'N/A')}")
                st.write(f"📊 Facteur qualité: {result.get('quality_factor', 'N/A')}")

                st.write("**Features utilisées :**")
                for feature in result.get("features_used", []):
                    st.write(f"• {feature}")

        else:
            st.error(f"Erreur API: {response.status_code}")
            try:
                st.code(response.text)
            except Exception:
                pass

    except Exception as e:
        st.error(f"Erreur lors de l'appel au modèle: {str(e)}")

# ---------- FOOTER ----------
st.markdown("""
<br><br>
<div class='footer'>ImmoPredict © 2025 — Estimation immobilière basée sur l'IA</div>
""", unsafe_allow_html=True)