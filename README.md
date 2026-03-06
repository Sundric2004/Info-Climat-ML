#  Infoclimat Paris : Analyse Prédictive & Dashboard Météo

Ce projet a pour objectif de collecter, traiter et modéliser les données météorologiques de Paris sur une semaine via le webservice de l'association **Infoclimat**. Il combine des techniques de **Data Engineering** pour la récolte, de **Data Visualization** pour l'exploration, et de **Machine Learning** pour la prédiction de tendances climatiques.

##  Aperçu du Projet

Le pipeline utilise les prévisions détaillées à 7 jours fournies en format JSON. La solution est bâtie sur une architecture modulaire permettant une maintenance aisée et une réutilisation du code entre le carnet de recherche (Notebook) et l'outil de production (Streamlit).

###  Structure du Dépôt

* **`meteo_utils.py`** : Module central regroupant la logique métier (récolte API, nettoyage des métadonnées, normalisation Pandas et entraînement des modèles).
* **`app.py`** : Interface utilisateur interactive développée avec **Streamlit**.
* **`notebook.ipynb`** : Analyse exploratoire détaillée, visualisation et évaluation rigoureuse des modèles de Machine Learning.

---

##  Méthodologie Data

### 1. Récolte et Nettoyage (Data Engineering)
* **Ingestion :** Récupération des prévisions météo de Paris via l'API JSON d'Infoclimat.
* **Nettoyage :** Suppression des clés système (`request_state`, `request_key`, `message`, `model_run`, `source`) pour ne conserver que les relevés temporels.
* **Normalisation :** Utilisation de `pd.json_normalize` pour transformer les dictionnaires imbriqués (température, vent, pression, etc.) en colonnes structurées.
* **Feature Engineering :**
    * Conversion des températures de Kelvin vers Celsius.
    * Conversion des index de chaînes de caractères en objets `datetime`.
    * Extraction de la variable **Heure** pour capturer le cycle diurne et améliorer les prédictions.

### 2. Visualisation des Données
Le projet inclut plusieurs représentations graphiques (Matplotlib) pour explorer les corrélations :
* Évolution comparée des températures à 2m et au sol.
* Cycle de l'humidité relative.
* **Rose des Vents :** Visualisation polaire de la direction et de la force du vent.
* Histogramme des précipitations (Pluie en mm).

---

##  Machine Learning

Nous avons implémenté et comparé plusieurs approches pour répondre aux problématiques de régression et de classification justifiées par les données.

### Performance des Modèles
| Tâche | Modèle | Résultat Clé |
| :--- | :--- | :--- |
| **Régression** | Régression Linéaire | $R^2 \approx 0.67$ |
| **Régression** | Random Forest Regressor | $RMSE \approx 2.39^\circ C$ |
| **Classification** | Random Forest Classifier | Accuracy $\approx 77\%$ |

*Note : La Régression Linéaire s'avère ici légèrement plus robuste que le Random Forest en raison du volume restreint de données sur une semaine, évitant ainsi le surapprentissage (overfitting) inhérent aux modèles ensemblistes sur de petits échantillons.*

---

## 🚀 Installation et Utilisation

### Prérequis
* Python 3.8+
* Un gestionnaire de paquets (`pip` ou `conda`)

### Installation
1.  **Cloner le dépôt :**
    
2.  **Installer les dépendances :**
    

### Lancement
* **Pour lancer le Dashboard Interactif :** ```bash
  streamlit run app.py
