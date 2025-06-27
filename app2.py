
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Prédiction de la PA", layout="centered")
st.title("🩺 Prédiction de la pression artérielle avec XGBoost")

# Charger le modèle et l’ordre des colonnes
model, feature_order = joblib.load("xgb_model1.pkl")  # fichier doit contenir (model, X_train.columns.tolist())

# Dictionnaires utiles
periode_dict = {'Nuit': 0, 'Matin': 1, 'Après-midi': 2, 'Soir': 3}
anticoag_options = ['Aucun', 'HS', 'HC', 'Lovenox']

# Interface utilisateur
Debit_sang_pompe = st.number_input("Débit sang pompe (ml/min)", value=300.0)
UF_H = st.number_input("UF_H (ml)", value=1.5)
Debit_eau_dialysat = st.number_input("Débit eau dialysat (ml/min)", value=500.0)
PA = st.number_input("PA (mmHg)", value=120.0)
PV = st.number_input("PV (mmHg)", value=40.0)
PTM = st.number_input("PTM (mmHg)", value=200.0)
Poul = st.number_input("Pouls (bpm)", value=70.0)

# Anticoagulant
anticoag_nom = st.selectbox("Anticoagulant", options=anticoag_options)
Anticoagulant_HS = Anticoagulant_HC = Anticoagulant_Lovenox = 0
if anticoag_nom == 'HS':
    Anticoagulant_HS = 1
elif anticoag_nom == 'HC':
    Anticoagulant_HC = 1
elif anticoag_nom == 'Lovenox':
    Anticoagulant_Lovenox = 1

# Période
periode_nom = st.selectbox("Période", options=list(periode_dict.keys()))
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
features_df = features_df[feature_order]

# Prédiction
if st.button("Prédire"):
    try:
        prediction = model.predict(features_df)[0]
        proba = model.predict_proba(features_df)[0]

        prediction_labels = {
            model.classes_[0]: "Hypertension",
            model.classes_[1]: "Hypotension",
            model.classes_[2]: "Normal"
        }
        prediction_text = prediction_labels.get(prediction, "Inconnu")

        st.success(f"✅ Prédiction : {prediction_text}")
        st.write("📊 Probabilités :", proba.tolist())

    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
j'ai excuter ça pour avoir le fichier py
