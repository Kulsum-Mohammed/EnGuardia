import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and encoders
scaler = joblib.load('saved_models/nids_scaler.joblib')
model = joblib.load('saved_models/nids_xgb_model.joblib')
label_encoder = joblib.load('saved_models/nids_label_encoder.joblib')
input_encoders = joblib.load('saved_models/nids_input_encoders.joblib')
selected_features = joblib.load('saved_models/nids_selected_features.joblib')

# Load state encoder and options
state_encoder = input_encoders['state']
state_options = list(state_encoder.classes_)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #262730;
        }
        .main {
            background-color: #262730;
            color: #ffffff;
        }
        h1, h2, h3 {
            color: #f63366;
        }
        .stTextInput > div > input {
            background-color: #1e1f25;
            color: #ffffff;
        }
        .stSelectbox > div {
            background-color: #1e1f25;
            color: #ffffff;
        }
        .stButton>button {
            background-color: #f63366;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Page title
st.title("🔐 EnGuardia - Network Attack Predictor")
st.markdown("Input the session-level network features below to predict the cyberattack type:")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        dur = st.number_input("Duration", value=0.0)
        state = st.selectbox("State (categorical)", options=state_options)
        dpkts = st.number_input("Destination Packets", value=0.0)
        sbytes = st.number_input("Source Bytes", value=0.0)
        dbytes = st.number_input("Destination Bytes", value=0.0)
        rate = st.number_input("Packet Rate", value=0.0)
        sttl = st.number_input("Source TTL", value=0.0)
        dttl = st.number_input("Destination TTL", value=0.0)

    with col2:
        sload = st.number_input("Source Load", value=0.0)
        dload = st.number_input("Destination Load", value=0.0)
        dinpkt = st.number_input("Destination Interpacket Time", value=0.0)
        smean = st.number_input("Source Mean Packet Size", value=0.0)
        dmean = st.number_input("Destination Mean Packet Size", value=0.0)
        ct_state_ttl = st.number_input("Connection State/TTL", value=0.0)
        ct_srv_dst = st.number_input("Connections to Same Service", value=0.0)
        ct_flw_http_mthd = st.number_input("HTTP Method Count", value=0.0)

    submitted = st.form_submit_button("🔍 Predict Attack Type")

    if submitted:
        try:
            # Create feature vector
            input_data = {
                'dur': dur,
                'state': state_encoder.transform([state])[0],
                'dpkts': dpkts,
                'sbytes': sbytes,
                'dbytes': dbytes,
                'rate': rate,
                'sttl': sttl,
                'dttl': dttl,
                'sload': sload,
                'dload': dload,
                'dinpkt': dinpkt,
                'smean': smean,
                'dmean': dmean,
                'ct_state_ttl': ct_state_ttl,
                'ct_srv_dst': ct_srv_dst,
                'ct_flw_http_mthd': ct_flw_http_mthd
            }

            X = pd.DataFrame([input_data])[selected_features]
            X_scaled = scaler.transform(X)

            # Predict
            proba = model.predict_proba(X_scaled)[0]
            pred_idx = np.argmax(proba)
            predicted_class = label_encoder.inverse_transform([pred_idx])[0]
            confidence = proba[pred_idx]

            st.success(f"🛡️ Predicted Attack Type: **{predicted_class}**")
            st.info(f"Confidence: **{confidence:.2%}**")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
