import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def get_clean_weather_data():
    """Récupère et nettoie les données météo depuis l'API Infoclimat."""
    url = "http://www.infoclimat.fr/public-api/gfs/json?_ll=48.85341,2.3488&_auth=CRMEE1IsV3VQfVNkBnBWfwJqBDEAdgIlVysCYVg9Ui8FbgVkAGBTNVI8UC1XeFFnWXRSMVtgVWUFbgJ6CngEZQljBGhSOVcwUD9TNgYpVn0CLARlACACJVc2AmVYPFIvBWQFYQBlUy9SNVAsV2RRYVl1Ui1bZVVqBWECZwpgBGUJbgRpUjhXPFAgUy4GMFZrAmAEbQBuAjlXNAJiWDxSYAU0BTQAa1M0UiNQO1diUWJZa1I1W2xVbgVkAnoKeAQeCRkEfVJxV3dQalN3BitWNwJvBDA%3D&_c=3f2c8309cb21866b814bfce5f4727bbe"
    
    response = requests.get(url)
    data = response.json()
    
    # 1. Suppression des clés inutiles
    cles_a_supprimer = ["request_state", "request_key", "message", "model_run", "source"]
    for cle in cles_a_supprimer:
        data.pop(cle, None)
        
    # 2. Conversion en DataFrame
    dfjson = pd.DataFrame.from_dict(data, orient='index')
    dfjson.index = pd.to_datetime(dfjson.index)
    
    # 3. Aplatissement des dictionnaires
    df_temp = pd.json_normalize(dfjson['temperature']).set_index(dfjson.index)
    df_humidite = pd.json_normalize(dfjson['humidite']).set_index(dfjson.index)
    df_pression = pd.json_normalize(dfjson['pression']).set_index(dfjson.index)
    df_vent = pd.json_normalize(dfjson['vent_moyen']).set_index(dfjson.index)
    df_vent_dir = pd.json_normalize(dfjson['vent_direction']).set_index(dfjson.index)
    
    # 4. Création du DataFrame final
    df_clean = pd.DataFrame({
        'Température 2m (°C)': df_temp['2m'] - 273.15,
        'Température Sol (°C)': df_temp['sol'] - 273.15,
        'Humidité (%)': df_humidite['2m'],
        'Pression (hPa)': df_pression['niveau_de_la_mer'] / 100,
        'Vent moyen (km/h)': df_vent['10m'],
        'Vent direction (°)': df_vent_dir['10m'],
        'Pluie (mm)': dfjson['pluie'].astype(float),
        'Heure': dfjson.index.hour 
    })
    
    return df_clean

def train_temperature_regressor(df):
    """Entraîne un Random Forest pour la Régression."""
    X = df[['Humidité (%)', 'Pression (hPa)', 'Vent moyen (km/h)', 'Heure']]
    y = df['Température 2m (°C)']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def get_temperature_class(temp):
    """Définit la catégorie météo."""
    if temp < 0: return "Froid "
    elif temp <= 15: return "Tempéré "
    else: return "Chaud "

def train_temperature_classifier(df):
    """Entraîne un Random Forest pour la Classification."""
    df_class = df.copy()
    df_class['temp_class'] = df_class['Température 2m (°C)'].apply(get_temperature_class)
    
    X = df_class[['Humidité (%)', 'Pression (hPa)', 'Vent moyen (km/h)', 'Heure']]
    y = df_class['temp_class']
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf