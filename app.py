import streamlit as st
import joblib
import numpy as np

st.title("🩺 Blood pressure prediction with XGBoost")

# Charger le modèle
model = joblib.load('xgb_model (3).pkl')

# Dictionnaires et options
periode_dict = {'Nuit': 0, 'Matin': 1, 'Après-midi': 2, 'Soir': 3}
anticoag_options = ['Aucun', 'HS', 'HC', 'Lovenox']

# Saisie des données utilisateur
Debit_sang_pompe = st.number_input("Débit sang pompe (ml/min)", value=300.0)
UF_H = st.number_input("UF_H (ml)", value=1.5)
Debit_eau_dialysat = st.number_input("Débit eau dialysat (ml/min)", value=500.0)
PA = st.number_input("PA (mmHg)", value=120.0)
PV = st.number_input("PV (mmHg)", value=40.0)
PTM = st.number_input("PTM (mmHg)", value=200.0)
Poul = st.number_input("Pouls (bpm)", value=70.0)

# Anticoagulants
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

# Bouton de prédiction
if st.button("Prédire"):
    try:
        # Préparer les données pour la prédiction
        features = np.array([
            Debit_sang_pompe,
            UF_H,
            Debit_eau_dialysat,
            PA,
            PV,
            PTM,
            Poul,
            Anticoagulant_HS,
            Anticoagulant_HC,
            Anticoagulant_Lovenox,
            periode_enc
        ]).reshape(1, -1)

        # Effectuer la prédiction
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0].tolist()

        # Mapper la prédiction en étiquette texte
        prediction_labels = {0: "Hypertension", 1: "Normal", 2: "Hypotension"}
        prediction_text = prediction_labels.get(prediction, "Inconnu")

        st.success(f"✅ Prédiction : {prediction_text}")
        st.write("📊 Probabilités :", proba)

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
