
import streamlit as st
import requests

st.title("🩺 Prédiction de la pression artérielle avec XGBoost")

periode_dict = {'Nuit': 0, 'Matin': 1, 'Après-midi': 2, 'Soir': 3}
anticoag_options = ['Aucun', 'HS', 'HC', 'Lovenox']

Debit_sang_pompe = st.number_input("Débit sang pompe (ml/min)", value=300.0)
UF_H = st.number_input("UF_H (ml)", value=1.5)
Debit_eau_dialysat = st.number_input("Débit eau dialysat (ml/min)", value=500.0)
PA = st.number_input("PA (mmHg)", value=120.0)
PV = st.number_input("PV (mmHg)", value=40.0)
PTM = st.number_input("PTM (mmHg)", value=200.0)
Poul = st.number_input("Pouls (bpm)", value=70.0)

anticoag_nom = st.selectbox("Anticoagulant", options=anticoag_options)
Anticoagulant_HS = Anticoagulant_HC = Anticoagulant_Lovenox = 0

if anticoag_nom == 'HS':
    Anticoagulant_HS = 1
elif anticoag_nom == 'HC':
    Anticoagulant_HC = 1
elif anticoag_nom == 'Lovenox':
    Anticoagulant_Lovenox = 1

periode_nom = st.selectbox("Période", options=list(periode_dict.keys()))
periode_enc = periode_dict[periode_nom]

if st.button("Prédire"):
    payload = {
        "Debit_sang_pompe": Debit_sang_pompe,
        "UF_H": UF_H,
        "Debit_eau_dialysat": Debit_eau_dialysat,
        "PA": PA,
        "PV": PV,
        "PTM": PTM,
        "Poul": Poul,
        "Anticoagulant_HS": Anticoagulant_HS,
        "Anticoagulant_HC": Anticoagulant_HC,
        "Anticoagulant_Lovenox": Anticoagulant_Lovenox,
        "Periode_enc": periode_enc
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Prédiction : {result['prediction_label']}")
            st.write("📊 Probabilités :", result["proba"])
        else:
            st.error(f"Erreur API : code {response.status_code}")

    except Exception as e:
        st.error(f"Erreur de connexion : {e}")
