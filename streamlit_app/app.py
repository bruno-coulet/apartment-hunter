import streamlit as st
import requests
import os
import json
import numpy as np

# --- CONFIGURATION ---
API_URL = "http://api:8000/predict"

# --- HELPERS ---
def format_euros(value: float) -> str:
    try:
        # Format with US locale then swap separators to FR style
        s = f"{float(value):,.2f}"  # e.g., 389,788.00
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")  # -> 389.788,00
        return f"{s} €"
    except Exception:
        # Fallback: round to 2 decimals without locale changes
        return f"{round(float(value), 2)} €"

@st.cache_resource
def load_config():
    """Charge la config depuis le notebook"""
    config_paths = ["models/streamlit_config.json", "../models/streamlit_config.json"]
    for path in config_paths:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    # Fallback
    return {
        "input_columns": ["sq_mt_built", "n_rooms", "n_bathrooms", "neighborhood",
                         "has_lift", "has_parking", "has_pool", "has_garden",
                         "has_storage_room", "is_floor_under"],
        "numeric_features": ["sq_mt_built", "n_rooms", "n_bathrooms"],
        "categorical_features": ["neighborhood"],
        "binary_features": ["has_lift", "has_parking", "has_pool", "has_garden", "has_storage_room", "is_floor_under"]
    }

config = load_config()

st.set_page_config(
    page_title="Madrid Apartment Hunter",
    page_icon="🏙️",
    layout="centered"
)

# --- CHARGEMENT DU CSS ---
def load_css():
    css_paths = ["style.css", "streamlit_app/style.css"]
    for path in css_paths:
        if os.path.exists(path):
            with open(path) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            return True
    return False

load_css()

# --- INTERFACE UTILISATEUR ---
st.title("🏙️ Madrid Apartment Hunter")
st.markdown("""
    Estimation du prix d'achat d'un appartement à Madrid basée sur le **Machine Learning**.
""")

st.divider()

# Récupérer les valeurs de quartier depuis la config
neighborhoods = config.get("categorical_values", {}).get("neighborhood", list(range(1, 136)))

with st.form("prediction_form"):
    st.subheader("📋 Caractéristiques du bien")
    
    col1, col2 = st.columns(2)
    
    numeric_features = config.get("numeric_features", [])
    ranges = config.get("ranges", {})
    
    # Inputs numériques
    with col1:
        feature_range = ranges.get("sq_mt_built", {})
        sq_mt_built = st.number_input(
            "Surface (m²)",
            min_value=int(feature_range.get("min", 10)),
            max_value=int(feature_range.get("max", 1000)),
            value=int(feature_range.get("mean", 75)),
            step=1,
            format="%d"
        )
        
        feature_range = ranges.get("n_rooms", {})
        n_rooms = st.number_input(
            "Chambres",
            min_value=int(feature_range.get("min", 0)),
            max_value=int(feature_range.get("max", 15)),
            value=int(feature_range.get("mean", 2)),
            step=1
        )
        
        feature_range = ranges.get("n_bathrooms", {})
        n_bathrooms = st.number_input(
            "Salles de bain",
            min_value=int(feature_range.get("min", 1)),
            max_value=int(feature_range.get("max", 10)),
            value=int(feature_range.get("mean", 1)),
            step=1,
            format="%d"
        )
    
    with col2:
        neighborhood = st.selectbox("Quartier", options=neighborhoods)
        is_floor_under = st.checkbox("Sous-sol", value=False)
        st.write("**Équipements :**")
    
    # Checkboxes pour les équipements
    col1, col2, col3 = st.columns(3)
    with col1:
        has_lift = st.checkbox("Ascenseur", value=True)
        has_parking = st.checkbox("Parking")
    with col2:
        has_pool = st.checkbox("Piscine")
        has_garden = st.checkbox("Jardin")
    with col3:
        has_storage_room = st.checkbox("Cave/Débarras")
    
    submit_button = st.form_submit_button("🔮 Estimer le prix")

# --- LOGIQUE DE PRÉDICTION ---
if submit_button:
    payload = {
        "sq_mt_built": int(sq_mt_built),
        "n_rooms": int(n_rooms),
        "n_bathrooms": int(n_bathrooms),
        "neighborhood": int(neighborhood),
        "has_lift": int(has_lift),
        "has_parking": int(has_parking),
        "has_pool": int(has_pool),
        "has_garden": int(has_garden),
        "has_storage_room": int(has_storage_room),
        "is_floor_under": int(is_floor_under)
    }

    with st.spinner("⏳ Calcul en cours..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                if "prediction" in result:
                    prix_euros = result["prediction"]
                    st.balloons()
                    st.success("✅ Estimation terminée!")
                    st.metric(label="💰 Prix estimé", value=format_euros(prix_euros))
                    # Info additionnelle si disponible
                    if "prediction_log" in result:
                        st.caption(f"(log-prix: {result['prediction_log']:.4f})")
                    
                elif "error" in result:
                    st.error(f"❌ Erreur API: {result['error']}")

                else:
                    st.warning("⚠️ Format de réponse API inattendu.")
            
            else:
                st.error(f"❌ L'API a répondu avec un code erreur : {response.status_code}")
                st.write("Vérifiez que le service 'api' est bien démarré.")

        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de contacter l'API. Vérifiez que le conteneur Docker 'api' fonctionne sur le port 8000.")
        except Exception as e:
            st.error(f"❌ Une erreur imprévue est survenue : {e}")

st.divider()
st.caption("Projet étudiant - Data Science & Cloud Deployment")