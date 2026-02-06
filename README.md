# 🔥 Prédiction des Incendies de Forêt en Algérie

Application de Machine Learning pour la prédiction des feux de forêt en Algérie, utilisant des données météorologiques pour détecter les risques d'incendie et estimer leur gravité.

## 📋 Description

Ce projet utilise l'Intelligence Artificielle pour :
1. **Détecter** s'il y a un risque de feu (Classification binaire)
2. **Estimer** la gravité potentielle via l'indice FWI (Fire Weather Index)

Le modèle est entraîné sur le dataset **Algerian Forest Fires** contenant des données des régions de Bejaia et Sidi-Bel Abbes.

## 🛠️ Technologies Utilisées

- **Python 3**
- **Streamlit** - Interface web interactive
- **Scikit-learn** - Modèles de Machine Learning
  - KNeighborsClassifier (Classification)
  - Ridge Regression avec PolynomialFeatures (Régression)
- **Pandas** - Manipulation des données
- **Matplotlib & Seaborn** - Visualisation

## 📁 Structure du Projet

```
├── app.py                          # Application Streamlit
├── mon_script_ml.py                # Script d'entraînement des modèles
├── Algerian_forest_fires_dataset.csv  # Dataset
├── mon_modele_classification.pkl   # Modèle de classification sauvegardé
├── mon_modele_regression.pkl       # Modèle de régression sauvegardé
├── mon_scaler.pkl                  # StandardScaler sauvegardé
├── mon_poly_features.pkl           # PolynomialFeatures sauvegardé
└── README.md
```

## 🚀 Installation

1. Cloner le repository :
```bash
git clone https://github.com/oussamabedjaoui/PredictionDesIncendies.git
cd PredictionDesIncendies
```

2. Créer un environnement virtuel :
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# ou
source .venv/bin/activate  # Linux/Mac
```

3. Installer les dépendances :
```bash
pip install streamlit pandas scikit-learn matplotlib seaborn joblib
```

## 💻 Utilisation

### Entraîner les modèles (optionnel)
```bash
python mon_script_ml.py
```

### Lancer l'application web
```bash
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`

## 📊 Fonctionnalités

- **Interface intuitive** : Ajustez les paramètres météo via des sliders
- **Paramètres d'entrée** :
  - Température (°C)
  - Humidité Relative (%)
  - Vitesse du vent (km/h)
  - Précipitations (cm)
- **Résultats** :
  - Alerte de risque de feu (Oui/Non)
  - Indice FWI estimé avec niveau de danger

## 📈 Dataset

Le dataset contient des observations météorologiques des régions algériennes :
- **Bejaia** (Nord-Est)
- **Sidi-Bel Abbes** (Nord-Ouest)

Variables utilisées : Temperature, RH (Humidité), Ws (Vent), Rain (Pluie)

## 👤 Auteur

**Oussama Bedjaoui**
- GitHub: [@oussamabedjaoui](https://github.com/oussamabedjaoui)

## 📄 Licence

Ce projet est open source.
