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

# SEARCH BAR
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([4,1])

with col1:
    adresse = st.text_input("Adresse", placeholder="Ex : 10 rue de Marseille, 13001", key="search", label_visibility="collapsed")

with col2:
    if st.button("Rechercher"):
        st.session_state.estimate_click = True


# ---------- HOW IT WORKS ----------
st.markdown("<br><h2>Comment ça marche ?</h2>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
        <div class='card'>
            <h3>1. Renseignez votre bien</h3>
            <p>Surface, chambres, étage…</p>
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
            <p>Un prix fiable, instantanément.</p>
        </div>
    """, unsafe_allow_html=True)


# ---------- PROPERTY INPUTS ----------
st.markdown("<br><h2>Caractéristiques du bien</h2>", unsafe_allow_html=True)

colA, colB, colC = st.columns(3)

with colA:
    surface = st.number_input("Surface (m²)", 10, 500)

with colB:
    rooms = st.number_input("Nombre de pièces", 1, 20)

with colC:
    bathrooms = st.number_input("Salles de bain", 1, 10)


# ---------- CALL API ----------
if st.button("Estimer le bien"):
    payload = {
        "surface": surface,
        "rooms": rooms,
        "bathrooms": bathrooms
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Affichage principal du prix
            st.success(f"💰 **Valeur estimée : {result['prediction']:,.0f} €**")
            
            # Affichage des détails du preprocessing (collapsible)
            with st.expander("🔍 Détails de l'analyse"):
                st.write("**Pipeline de preprocessing appliqué :**")
                st.write(f"✅ Preprocessing: {result.get('preprocessing_applied', 'Non')}")
                st.write(f"📊 Facteur qualité: {result.get('quality_factor', 'N/A')}")
                
                st.write("**Features utilisées :**")
                features = result.get('features_used', [])
                for feature in features:
                    st.write(f"• {feature}")
                
                # Calcul du ratio surface/pièces pour info
                surface_per_room = surface / rooms if rooms > 0 else 0
                st.write(f"📏 Surface par pièce: {surface_per_room:.1f} m²")
                
        else:
            st.error(f"Erreur API: {response.status_code}")
            
    except Exception as e:
        st.error(f"Erreur lors de l'appel au modèle: {str(e)}")

# ---------- FOOTER ----------
st.markdown("""
<br><br>
<div class='footer'>ImmoPredict © 2025 — Estimation immobilière basée sur l'IA</div>
""", unsafe_allow_html=True)
