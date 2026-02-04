import streamlit as st
import requests
import pandas as pd

# Fonction pour charger les quartiers disponibles
@st.cache_data
def load_available_neighborhoods():
    """Charge les quartiers disponibles depuis le dataset"""
    try:
        # Charger le dataset pour récupérer les quartiers réels
        df = pd.read_feather("data_model/houses.feather")
        neighborhoods = sorted(df['neighborhood'].unique())
        return neighborhoods
    except FileNotFoundError:
        # Si le fichier n'est pas trouvé, utiliser une gamme par défaut
        return list(range(1, 136))

# Load CSS
try:
    with open("streamlit_app/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    # Si le fichier CSS n'est pas trouvé, continuer sans style
    pass

# ---------- HEADER ----------
st.markdown("""
    <div style='display: flex; align-items: center; justify-content: space-between;'>
        <h2 style='color:#1e88e5;'>ImmoPredict</h2>
        <div style='font-size:18px;'>
            <a style='margin-right:20px;'>Estimer</a>
    </div>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class='hero'>
    <h1 style='text-align:center; color:white; font-size:45px;'>Estimation Immobilière Instantanée</h1>
    <p style='text-align:center; color:white; font-size:20px;'>Obtenez la valeur de votre bien en quelques secondes</p>
</div>
""", unsafe_allow_html=True)


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

# (A) Variables structurelles
colA, colB, colC = st.columns(3)
with colA:
    sq_mt_built = st.number_input("Surface construite (m²)", min_value=10, max_value=1000, value=80, step=1)
with colB:
    n_rooms = st.number_input("Nombre de chambres", min_value=0, max_value=24, value=3, step=1)
with colC:
    n_bathrooms = st.number_input("Salles de bain", min_value=1, max_value=16, value=2, step=1)

# (B) Quartier
st.markdown("<br>", unsafe_allow_html=True)

# Charger les quartiers réellement disponibles dans le dataset
NEIGHBORHOOD_VALUES = load_available_neighborhoods()

# Créer un mapping pour l'affichage (optionnel : pour rendre plus user-friendly)
neighborhood_options = [f"Quartier {n}" for n in NEIGHBORHOOD_VALUES]

# Menu déroulant avec les vrais quartiers
selected_neighborhood_display = st.selectbox(
    "Quartier (neighborhood)", 
    options=neighborhood_options, 
    index=4 if len(NEIGHBORHOOD_VALUES) > 4 else 0,  # Quartier par défaut
    help=f"Choisissez parmi les {len(NEIGHBORHOOD_VALUES)} quartiers disponibles dans notre base de données"
)

# Extraire le numéro du quartier sélectionné
neighborhood = NEIGHBORHOOD_VALUES[neighborhood_options.index(selected_neighborhood_display)]

# Affichage d'info pour l'utilisateur
st.caption(f"📍 {len(NEIGHBORHOOD_VALUES)} quartiers disponibles dans notre base (de {min(NEIGHBORHOOD_VALUES)} à {max(NEIGHBORHOOD_VALUES)})")

# (C) Type de bien
st.markdown("<br>", unsafe_allow_html=True)

# Options disponibles pour le type de bien (basées sur les données françaises)
PRODUCT_OPTIONS = [
    "appartement",
    "penthouse / appartement au dernier étage", 
    "maison ou chalet",
    "duplex",
    "maison mitoyenne",
    "maison jumelée",
    "studio",
    "domaine / propriété rurale",
    "maison"
]

product = st.selectbox(
    "Type de bien (product)",
    options=PRODUCT_OPTIONS,
    index=0,  # "appartement" par défaut
    help="Sélectionnez le type de bien immobilier"
)

# (D) Equipements (binaires)
st.markdown("<br><h3>Équipements</h3>", unsafe_allow_html=True)

e1, e2, e3 = st.columns(3)
with e1:
    has_lift = st.checkbox("Ascenseur (has_lift)", value=True)
    has_parking = st.checkbox("Parking (has_parking)", value=False)
with e2:
    has_pool = st.checkbox("Piscine (has_pool)", value=False)
    has_garden = st.checkbox("Jardin (has_garden)", value=False)
with e3:
    has_storage_room = st.checkbox("Cave / débarras (has_storage_room)", value=False)
    is_floor_under = st.checkbox("Sous-sol (is_floor_under)", value=False)

# ---------- CALL API ----------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Estimer le bien"):
    payload = {
        # noms = ceux de ton df_model
        "sq_mt_built": float(sq_mt_built),
        "n_rooms": int(n_rooms),
        "n_bathrooms": float(n_bathrooms),
        "neighborhood": int(neighborhood),
        "product": str(product),  # Type de bien
        "has_lift": int(has_lift),
        "has_parking": int(has_parking),
        "has_pool": int(has_pool),
        "has_garden": int(has_garden),
        "has_storage_room": int(has_storage_room),
        "is_floor_under": int(is_floor_under),
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            
            # 🔍 DEBUG : Affichons toute la réponse
            st.write("**🔍 DEBUG - Réponse complète de l'API :**")
            st.json(result)
            
            # Vérifions si 'prediction' existe
            if 'prediction' in result:
                # Affichage principal du prix
                st.success(f"💰 **Valeur estimée : {result['prediction']:,.0f} €**")

                # Détails (si ton API renvoie ces champs)
                with st.expander("🔍 Détails de l'analyse"):
                    st.write("**Payload envoyé au modèle :**")
                    st.json(payload)

                    st.write("**Infos API :**")
                    st.write(f"✅ Preprocessing: {result.get('preprocessing_applied', 'N/A')}")
                    st.write(f"📊 Modèle utilisé: {result.get('model_used', 'N/A')}")
                    st.write(f"📈 Performance R²: {result.get('r2_score', 'N/A')}")

                    st.write("**Features count :**")
                    st.write(f"• Nombre de features: {result.get('features_count', 'N/A')}")
            else:
                # Si pas de 'prediction', c'est probablement une erreur
                st.error("❌ L'API a renvoyé une erreur :")
                if 'error' in result:
                    st.code(result['error'])
                else:
                    st.write("Réponse inattendue de l'API")
                    st.json(result)

        else:
            st.error(f"Erreur API: {response.status_code}")
            st.write("**📱 Détails de la réponse :**")
            try:
                error_json = response.json()
                st.json(error_json)
            except:
                st.code(response.text)

    except Exception as e:
        st.error(f"❌ Erreur lors de l'appel au modèle: {str(e)}")
        st.write("**🔍 Détails de l'erreur :**")
        st.code(str(e))

# ---------- FOOTER ----------
st.markdown("""
<br><br>
<div class='footer'>ImmoPredict © 2025 — Estimation immobilière basée sur l'IA</div>
""", unsafe_allow_html=True)