import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page principale
st.set_page_config(
    page_title="ImmoPredict ML Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Chargement du CSS externe
with open('frontent/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



# ========== FONCTIONS UTILITAIRES ==========
@st.cache_data
def load_dataset():
    """Charge le dataset avec gestion d'erreur"""
    try:
        df = pd.read_feather("data_model/houses.feather")
        df['prix_reel'] = np.expm1(df['log_buy_price'])
        return df
    except FileNotFoundError:
        st.error("Dataset non trouvé")
        return None

@st.cache_data
def load_model_info():
    """Charge les informations du modèle"""
    try:
        import pickle
        with open('models/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return metadata
    except:
        return None

def predict_price(input_data, property_type):
    """Fonction pour faire une prédiction selon le type de bien"""
    try:
        endpoint = f"http://localhost:8000/predict/{property_type}"
        response = requests.post(endpoint, json=input_data)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

@st.cache_data
def get_model_info():
    """Récupère les informations sur tous les modèles"""
    try:
        response = requests.get("http://localhost:8000/model-info")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# ========== NAVIGATION ==========
# Ajout de la navigation dans la sidebar
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Choisissez une page:",
    ["Estimateur", "Dashboard Dataset", "Explorer Modèle", "Performance"]
)

# ========== PAGE PRINCIPALE ==========
if page == "Estimateur":
    # Header principal
    st.markdown("""
    <div class='main-header'>
        <h1>ImmoPredict ML Platform</h1>
        <p>Intelligence Artificielle pour l'estimation immobilière</p>
    </div>
    """, unsafe_allow_html=True)

    
    # Informations du modèle
    model_info = get_model_info()
    if model_info:
        st.markdown("### Modèles Disponibles")
        col1, col2 = st.columns(2)
        
        with col1:
            if "appartements" in model_info:
                apt_info = model_info["appartements"]
                st.metric("Appartements", 
                         apt_info.get('model_name', 'N/A'), 
                         f"R²: {apt_info.get('performance_r2', 0)*100:.1f}%")
        
        with col2:
            if "maisons" in model_info:
                mai_info = model_info["maisons"]
                st.metric("Maisons", 
                         mai_info.get('model_name', 'N/A'), 
                         f"R²: {mai_info.get('performance_r2', 0)*100:.1f}%")

    st.markdown("---")

    # SÉLECTION DU TYPE DE BIEN
    st.markdown("## Choisissez le type de bien")
    property_type = st.radio(
        "Type de bien à estimer:",
        ["appartements", "maisons"],
        horizontal=True,
        help="Chaque type utilise un modèle spécialisé avec des variables adaptées"
    )

    st.markdown("---")

    # Interface de prédiction adaptée selon le type
    st.markdown(f"## Estimer votre {'appartement' if property_type == 'appartements' else 'maison'}")

    col1, col2 = st.columns([2, 1])

    
    with col1:
        st.markdown("### Caractéristiques du bien")
        
        # Variables communes
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            sq_mt_built = st.number_input("Surface (m²)", min_value=20, max_value=1000, value=80 if property_type == "appartements" else 150, step=5)
        with col_b:
            n_rooms = st.number_input("Chambres", min_value=1, max_value=15, value=3 if property_type == "appartements" else 4)
        with col_c:
            n_bathrooms = st.number_input("Salles de bain", min_value=1, max_value=10, value=2)
    
        
        st.markdown("### Équipements")
        
        if property_type == "appartements":
            # Features spécifiques aux appartements
            col_eq1, col_eq2 = st.columns(2)
            
            with col_eq1:
                has_lift = st.checkbox("Ascenseur", value=True, help="Présence d'un ascenseur dans l'immeuble")
                has_parking = st.checkbox("Parking", help="Place de parking incluse")
            with col_eq2:
                has_central_heating = st.checkbox("Chauffage central", value=True, help="Système de chauffage central")
            
            # Variables non utilisées pour les appartements
            has_garden = 0
            has_pool = 0
            neighborhood = 0
            
        else:  # maisons
            # Features spécifiques aux maisons
            col_eq1, col_eq2 = st.columns(2)
            
            with col_eq1:
                has_garden = st.checkbox("Jardin", value=True, help="Présence d'un jardin privé")
                has_pool = st.checkbox("Piscine", help="Piscine privée")
            with col_eq2:
                neighborhood = st.selectbox("Quartier (code)", 
                                          list(range(1, 200)), 
                                          index=50,
                                          help="Code numérique du quartier (1-200)")
            
            # Variables non utilisées pour les maisons
            has_lift = 0
            has_parking = 0
            has_central_heating = 0

    
    with col2:
        st.markdown("### Informations ML")
        
        # Informations spécifiques au modèle appartements
        if model_info and "appartements" in model_info:
            current_model = model_info["appartements"]
            if "error" not in current_model:
                confidence = current_model.get('performance_r2', 0) * 100
                features_count = current_model.get('total_features', 6)  # 6 features sans buy_price
                
                st.markdown(f"""
                <div class='model-info'>
                    <h4>Modèle Appartements</h4>
                    <div class='confidence-bar'>
                        <div class='confidence-fill' style='width: {confidence}%'></div>
                    </div>
                    <p><strong>{confidence:.1f}%</strong> de précision (R²)</p>
                    <hr>
                    <p><strong>Features utilisées :</strong></p>
                    <p>• Surface construite</p>
                    <p>• Chambres & SdB</p>
                    <p>• Ascenseur & Parking</p>
                    <p>• Chauffage central</p>
                    <p>• Prix d'achat</p>
                    <p><strong>Total: {features_count} variables</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Modèle appartements non disponible")
        
        # Bouton de prédiction
        button_text = "ESTIMER L'APPARTEMENT" if property_type == "appartements" else "ESTIMER LA MAISON"
        if st.button(button_text, type="primary", use_container_width=True):
            
            # Préparation des données selon le type
            if property_type == "appartements":
                payload = {
                    "property_type": "appartements",
                    "sq_mt_built": float(sq_mt_built),
                    "n_rooms": int(n_rooms),
                    "n_bathrooms": float(n_bathrooms),
                    "has_lift": int(has_lift),
                    "has_parking": int(has_parking),
                    "has_central_heating": int(has_central_heating)
                }
            else:  # maisons
                payload = {
                    "property_type": "maisons",
                    "sq_mt_built": float(sq_mt_built),
                    "n_rooms": int(n_rooms),
                    "n_bathrooms": float(n_bathrooms),
                    "has_garden": int(has_garden),
                    "has_pool": int(has_pool),
                    "neighborhood": int(neighborhood)
                }
            
            # Prédiction selon le type
            result = predict_price(payload, property_type)
            
            if result and 'prediction' in result:
                real_price = result['prediction']
                price_per_m2 = real_price / sq_mt_built
                
                # Affichage du résultat
                st.markdown(f"""
                <div class='prediction-result'>
                    <h2>Estimation: {real_price:,.0f} €</h2>
                    <p>Prix par m²: {price_per_m2:,.0f} €/m²</p>
                    <p>Type: {property_type.title()}</p>
                    <p>Modèle: {result.get('model_used', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Détails de l'analyse
                with st.expander("Détails de l'analyse ML"):
                    st.write("**Données envoyées au modèle:**")
                    st.json(payload)
                    
                    st.write("**Pipeline de traitement:**")
                    st.write(f"1. Validation des données pour {property_type}")
                    st.write("2. Preprocessing adapté au type de bien")
                    st.write(f"3. Prédiction {result.get('model_used', 'N/A')}")
                    st.write("4. Estimation finale")
                    
                    if 'features_used' in result:
                        st.write(f"**Features utilisées:** {', '.join(result['features_used'])}")
                    if 'r2_score' in result:
                        st.write(f"**Performance du modèle:** {result['r2_score']*100:.1f}% R²")
            else:
                st.error(f"Erreur lors de l'estimation. Vérifiez que l'API est démarrée et que le modèle {property_type} est disponible.")

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div class='footer'>
    <p>ImmoPredict ML Platform - Estimation immobilière par Intelligence Artificielle</p>
    <p>Modèles: RandomForest optimisé pour appartements et maisons</p>
</div>
""", unsafe_allow_html=True)