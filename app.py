import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importation depuis notre fichier métier
from meteo_utils import get_clean_weather_data, train_temperature_regressor, train_temperature_classifier

# --- CONFIGURATION ---
st.set_page_config(page_title="Dashboard Météo Paris", page_icon="🌤️", layout="wide")
st.title("Dashboard Météo Infoclimat & IA")

# --- CHARGEMENT DES DONNÉES ET MODÈLES ---
@st.cache_data
def load_data():
    return get_clean_weather_data()

@st.cache_resource
def load_models(df):
    return train_temperature_regressor(df), train_temperature_classifier(df)

df = load_data()
reg_model, clf_model = load_models(df)

# --- INTERFACE (ONGLETS) ---
tab1, tab2, tab3 = st.tabs([" Données Brutes", " Visualisations", " Prédictions IA"])

with tab1:
    st.header("Aperçu du jeu de données")
    st.dataframe(df)

with tab2:
    st.header("Exploration Visuelle")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Température vs Humidité")
        st.line_chart(df[['Température 2m (°C)', 'Humidité (%)']])
        
    with col2:
        st.subheader("Rose des Vents")
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
        theta = np.deg2rad(df['Vent direction (°)'])
        radii = df['Vent moyen (km/h)']
        scatter = ax.scatter(theta, radii, c=radii, cmap='viridis', alpha=0.7)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        st.pyplot(fig)

with tab3:
    st.header("Testez les modèles de Machine Learning")
    st.write("Le modèle prend désormais en compte l'heure de la journée pour plus de précision !")
    
    c1, c2, c3, c4 = st.columns(4)
    hum = c1.slider(" Humidité (%)", 0, 100, 60)
    pres = c2.slider(" Pression (hPa)", 980, 1050, 1015)
    vent = c3.slider(" Vent (km/h)", 0, 100, 15)
    heure = c4.slider(" Heure", 0, 23, 14)
    
    # Prédiction
    input_df = pd.DataFrame({'Humidité (%)': [hum], 'Pression (hPa)': [pres], 'Vent moyen (km/h)': [vent], 'Heure': [heure]})
    
    pred_temp = reg_model.predict(input_df)[0]
    pred_class = clf_model.predict(input_df)[0]
    
    st.success(f"###  Température prédite : **{pred_temp:.2f} °C**")
    st.info(f"###  Tendance générale : **{pred_class}**")