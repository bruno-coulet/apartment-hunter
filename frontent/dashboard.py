import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="Dataset Dashboard - ImmoPredict",
    page_icon="📊",
    layout="wide"
)

# Chargement du dataset
@st.cache_data
def load_dataset():
    """Charge le dataset avec gestion d'erreur"""
    try:
        df = pd.read_feather("data_model/houses.feather")
        # Ajouter le prix réel
        df['prix_reel'] = np.expm1(df['log_buy_price'])
        return df
    except FileNotFoundError:
        st.error("❌ Dataset non trouvé. Assurez-vous que le fichier houses.feather existe.")
        return None

# Chargement des données
df = load_dataset()

if df is not None:
    # ========== HEADER ==========
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color:#1e88e5;'>📊 Dashboard Dataset ImmoPredict</h1>
            <p style='font-size: 18px; color: #666;'>Analyse complète du dataset immobilier Madrid</p>
        </div>
    """, unsafe_allow_html=True)

    # ========== MÉTRIQUES GÉNÉRALES ==========
    st.markdown("## 🔢 Métriques Générales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🏠 Total Biens", 
            value=f"{len(df):,}",
            help="Nombre total d'annonces immobilières"
        )
    
    with col2:
        st.metric(
            label="💰 Prix Médian", 
            value=f"{df['prix_reel'].median():,.0f} €",
            help="Prix médian des biens immobiliers"
        )
    
    with col3:
        st.metric(
            label="📐 Surface Médiane", 
            value=f"{df['sq_mt_built'].median():.0f} m²",
            help="Surface construite médiane"
        )
    
    with col4:
        st.metric(
            label="🏘️ Quartiers", 
            value=f"{df['neighborhood'].nunique()}",
            help="Nombre de quartiers différents"
        )

    # ========== TYPES DE DONNÉES ==========
    st.markdown("## 📋 Structure du Dataset")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🔍 Types de Données")
        
        # Création du tableau des types
        type_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_vals = df[col].nunique()
            missing = df[col].isnull().sum()
            
            if col == 'product':
                data_type = "Catégoriel (Type de bien)"
            elif col == 'neighborhood':
                data_type = "Catégoriel (Quartier)"
            elif col in ['has_lift', 'has_parking', 'has_pool', 'has_garden', 'has_storage_room', 'is_floor_under']:
                data_type = "Binaire (0/1)"
            elif col in ['sq_mt_built', 'n_bathrooms', 'log_buy_price', 'prix_reel']:
                data_type = "Numérique (Continu)"
            elif col == 'n_rooms':
                data_type = "Numérique (Discret)"
            else:
                data_type = "Autre"
            
            type_info.append({
                'Colonne': col,
                'Type': data_type,
                'Valeurs Uniques': unique_vals,
                'Manquantes': missing
            })
        
        type_df = pd.DataFrame(type_info)
        st.dataframe(type_df, use_container_width=True, height=400)
    
    with col2:
        st.markdown("### 🏠 Types de Biens")
        
        # Graphique des types de biens
        product_counts = df['product'].value_counts()
        
        fig_product = px.pie(
            values=product_counts.values,
            names=product_counts.index,
            title="Répartition par Type de Bien",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_product.update_layout(
            showlegend=True,
            height=400,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5
            )
        )
        st.plotly_chart(fig_product, use_container_width=True)

    # ========== STATISTIQUES DÉTAILLÉES ==========
    st.markdown("## 📊 Statistiques Détaillées")
    
    # Bien médian
    st.markdown("### 🎯 Profil du Bien Médian")
    
    median_price = df['prix_reel'].median()
    median_surface = df['sq_mt_built'].median()
    median_rooms = df['n_rooms'].median()
    median_bathrooms = df['n_bathrooms'].median()
    most_common_product = df['product'].mode()[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        **💰 Prix :** {median_price:,.0f} €  
        **📐 Surface :** {median_surface:.0f} m²  
        **💶 Prix/m² :** {median_price/median_surface:,.0f} €/m²
        """)
    
    with col2:
        st.markdown(f"""
        **🛏️ Chambres :** {median_rooms:.0f}  
        **🚿 SdB :** {median_bathrooms:.0f}  
        **🏠 Type :** {most_common_product}
        """)
    
    with col3:
        # Équipements les plus fréquents
        equipements = ['has_lift', 'has_parking', 'has_pool', 'has_garden', 'has_storage_room']
        eq_names = ['Ascenseur', 'Parking', 'Piscine', 'Jardin', 'Cave']
        
        st.markdown("**🔧 Équipements fréquents :**")
        for eq, name in zip(equipements, eq_names):
            pct = (df[eq].sum() / len(df)) * 100
            st.markdown(f"• {name}: {pct:.0f}%")

    # ========== DISTRIBUTIONS ==========
    st.markdown("## 📈 Distributions des Variables")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des prix
        fig_price = px.histogram(
            df, 
            x='prix_reel', 
            nbins=50,
            title="Distribution des Prix (€)",
            labels={'prix_reel': 'Prix (€)', 'count': 'Nombre de biens'}
        )
        fig_price.update_layout(showlegend=False)
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        # Distribution des surfaces
        fig_surface = px.histogram(
            df, 
            x='sq_mt_built', 
            nbins=50,
            title="Distribution des Surfaces (m²)",
            labels={'sq_mt_built': 'Surface (m²)', 'count': 'Nombre de biens'}
        )
        fig_surface.update_layout(showlegend=False)
        st.plotly_chart(fig_surface, use_container_width=True)

    # ========== RELATIONS ENTRE VARIABLES ==========
    st.markdown("## 🔗 Relations entre Variables")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prix vs Surface par type de bien
        fig_scatter = px.scatter(
            df.sample(1000),  # Échantillon pour performance
            x='sq_mt_built', 
            y='prix_reel',
            color='product',
            title="Prix vs Surface par Type de Bien",
            labels={'sq_mt_built': 'Surface (m²)', 'prix_reel': 'Prix (€)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Box plot prix par quartier (top 10)
        top_neighborhoods = df['neighborhood'].value_counts().head(10).index
        df_top_neigh = df[df['neighborhood'].isin(top_neighborhoods)]
        
        fig_box = px.box(
            df_top_neigh,
            x='neighborhood',
            y='prix_reel',
            title="Prix par Quartier (Top 10)",
            labels={'neighborhood': 'Quartier', 'prix_reel': 'Prix (€)'}
        )
        fig_box.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_box, use_container_width=True)

    # ========== CORRÉLATIONS ==========
    st.markdown("## 🎯 Matrice de Corrélations")
    
    # Sélection des variables numériques
    numeric_cols = ['prix_reel', 'sq_mt_built', 'n_rooms', 'n_bathrooms'] + \
                   ['has_lift', 'has_parking', 'has_pool', 'has_garden', 'has_storage_room', 'is_floor_under']
    
    corr_matrix = df[numeric_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        title="Corrélations entre Variables",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

    # ========== ÉCHANTILLON DU DATASET ==========
    st.markdown("## 🔍 Aperçu du Dataset")
    
    st.markdown("### 📋 Premières Lignes")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistiques descriptives
    st.markdown("### 📊 Statistiques Descriptives")
    
    # Colonnes numériques pour les stats
    stats_cols = ['prix_reel', 'sq_mt_built', 'n_rooms', 'n_bathrooms']
    stats_df = df[stats_cols].describe()
    
    # Formatage pour meilleure lisibilité
    stats_formatted = stats_df.copy()
    for col in ['prix_reel', 'sq_mt_built']:
        stats_formatted[col] = stats_formatted[col].apply(lambda x: f"{x:,.0f}")
    
    st.dataframe(stats_formatted, use_container_width=True)

else:
    st.error("Impossible de charger le dataset. Vérifiez que le fichier existe.")

# ========== FOOTER ==========
st.markdown("""
<br><br>
<div style='text-align: center; padding: 20px; border-top: 1px solid #eee;'>
    <p style='color: #666;'>📊 Dashboard Dataset ImmoPredict © 2025 — Analyse immobilière basée sur l'IA</p>
</div>
""", unsafe_allow_html=True)