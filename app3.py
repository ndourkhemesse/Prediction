import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Configuration de la page
st.set_page_config(page_title="Prédiction de la PA", layout="centered")

# Titre designé
st.markdown("""
    <div style='background-color: #2196F3; padding: 15px; border-radius: 8px; text-align: center;'>
        <h1 style='color: white; font-size: 30px; margin: 0;'>
            🩺 Prédiction de la pression artérielle 
        </h1>
    </div>
    """, unsafe_allow_html=True)

# Charger le modèle et l’ordre des colonnes
model, feature_order = joblib.load("xgb_model1.pkl")

# Dictionnaires utiles
periode_dict = {'Nuit': 0, 'Matin': 1, 'Après-midi': 2, 'Soir': 3}
anticoag_options = ['Aucun', 'HS', 'HC', 'Lovenox']

# Mapping des classes numériques vers texte
prediction_labels = {
    0: "Hypertension",
    1: "Hypotension",
    2: "Normal"
}

# Style CSS global pour les containers et le bouton
st.markdown("""
    <style>
    .card {
        background-color: #F8F9FA;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .section-title {
        font-size: 22px;
        color: #2196F3;
        margin-bottom: 10px;
    }
    .stButton > button {
        background-color: #2196F3;
        color: white;
        border: none;
        padding: 0.6em 2em;
        border-radius: 8px;
        font-size: 18px;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0b7dda;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Section Inputs médicaux
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🩺 Paramètres de la séance</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    Debit_sang_pompe = st.number_input("💉 Débit sang pompe (ml/min)", value=300, step=1, format="%d")
    UF_H = st.number_input("💧 UF_H (ml)", value=20, step=1, format="%d")
    Debit_eau_dialysat = st.number_input("💧 Débit eau dialysat (ml/min)", value=500, step=1, format="%d")
    PA = st.number_input("📈 PA (mmHg)", value=-80, step=1, format="%d")

with col2:
    PV = st.number_input("📉 PV (mmHg)", value=40, step=1, format="%d")
    PTM = st.number_input("📊 PTM (mmHg)", value=200, step=1, format="%d")
    Poul = st.number_input("❤️ Pouls (bpm)", value=70, step=1, format="%d")

st.markdown("</div>", unsafe_allow_html=True)

# Section Choix anticoagulant et période
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>⚙️ Options supplémentaires</div>", unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    anticoag_nom = st.selectbox("🧪 Anticoagulant", options=anticoag_options)
with col4:
    periode_nom = st.selectbox("🕒 Période", options=list(periode_dict.keys()))

st.markdown("</div>", unsafe_allow_html=True)

# Variables binaires Anticoagulant
Anticoagulant_HS = Anticoagulant_HC = Anticoagulant_Lovenox = 0
if anticoag_nom == 'HS':
    Anticoagulant_HS = 1
elif anticoag_nom == 'HC':
    Anticoagulant_HC = 1
elif anticoag_nom == 'Lovenox':
    Anticoagulant_Lovenox = 1

# Encodage période
periode_enc = periode_dict[periode_nom]

# Préparation du dictionnaire avec noms corrects
input_dict = {
    'Debit_sang_pompe': Debit_sang_pompe,
    'UF_H': UF_H,
    'Debit_eau_dialysat': Debit_eau_dialysat,
    'PA': PA,
    'PV': PV,
    'PTM': PTM,
    'Poul': Poul,
    'Anticoagulant_Héparine Standard': Anticoagulant_HS,
    'Anticoagulant_Héparine calcique': Anticoagulant_HC,
    'Anticoagulant_lovenox 0.4': Anticoagulant_Lovenox,
    'Periode_enc': periode_enc
}

# Construction du DataFrame avec ordre correct
features_df = pd.DataFrame([input_dict])

if list(features_df.columns) != feature_order:
    st.error("⚠️ Attention : l’ordre des colonnes ne correspond pas à celui attendu par le modèle !")
else:
    features_df = features_df[feature_order]
    st.write("📦 Données envoyées au modèle :", features_df)
 #   st.write("📊 Types des colonnes :", features_df.dtypes)

# Prédiction
if st.button("Prédire"):
    try:
        # Prédire la classe
        prediction = model.predict(features_df)[0]
        proba = model.predict_proba(features_df)[0]

        st.write("🎯 Classe prédite (predict):", prediction)

        # Affichage des probabilités avec labels texte
        proba_df = pd.DataFrame([proba], columns=[prediction_labels.get(c, c) for c in model.classes_])
        st.write("📊 Probabilités par classe :")
        st.write(proba_df)

        # Prédiction finale en texte
        prediction_text = prediction_labels.get(prediction, "Inconnu")
        st.success(f"✅ Prédiction finale : {prediction_text}")

    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
