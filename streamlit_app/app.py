import streamlit as st
import requests

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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

# Mets ici les valeurs possibles de neighborhood.
# Si tu ne connais pas la liste exacte, tu peux laisser un champ numérique.
NEIGHBORHOOD_VALUES = list(range(1, 136))  # 1..135 d'après ton dataset

neighborhood = st.selectbox("Quartier (neighborhood)", options=NEIGHBORHOOD_VALUES, index=58)  # 59 par défaut approx

# (C) Equipements (binaires)
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