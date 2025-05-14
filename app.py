import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="EnGuardia - NIDS", layout="centered")

# Custom Styling with your color palette
st.markdown("""
    <style>
        body, .main {
            background-color: #ffffff;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #262730;
            border-bottom: 2px solid #fffd80;
            padding-bottom: 0.2em;
        }
        .stButton>button {
            background-color: #f63366;
            color: white;
            border-radius: 12px;
            padding: 0.5em 1em;
            font-weight: bold;
            border: none;
        }
        .stTextInput>div>div>input, .stNumberInput input {
            border: 1px solid #fffd80;
            padding: 0.4em;
            border-radius: 6px;
            background-color: #ffffff;
            color: #262730;
        }
        .stForm {
            background-color: #fffd8011;
            padding: 2em;
            border-radius: 10px;
            border: 2px dashed #f63366;
        }
        .stDataFrame th {
            background-color: #f63366;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Load model components
model = joblib.load('saved_models/nids_xgb_model.joblib')
scaler = joblib.load('saved_models/nids_scaler.joblib')
label_encoder_y = joblib.load('saved_models/nids_label_encoder.joblib')
label_encoders = joblib.load('saved_models/nids_input_encoders.joblib')
selected_features = joblib.load('saved_models/nids_selected_features.joblib')

# Feature display names
feature_names = {
    'dur': 'Duration',
    'state': 'State',
    'dpkts': 'Destination Packets',
    'sbytes': 'Source Bytes',
    'dbytes': 'Destination Bytes',
    'rate': 'Packet Rate',
    'sttl': 'Source TTL',
    'dttl': 'Destination TTL',
    'sload': 'Source Load',
    'dload': 'Destination Load',
    'dinpkt': 'Destination Interpacket Time',
    'smean': 'Source Mean Packet Size',
    'dmean': 'Destination Mean Packet Size',
    'ct_state_ttl': 'Connection State/TTL',
    'ct_srv_dst': 'Connections to Same Service',
    'ct_flw_http_mthd': 'HTTP Method Count'
}

# Attack descriptions
attack_details = {
    "Analysis": "### 🕵️ Analysis\n- Attackers scan and analyze system behavior.",
    "Backdoor": "### 🔓 Backdoor\n- Hidden access to systems for remote control.",
    "DoS": "### 🚫 DoS\n- Service flooding to exhaust resources.",
    "Exploits": "### 🧨 Exploits\n- Taking advantage of software flaws.",
    "Fuzzers": "### 💥 Fuzzers\n- Crash systems with malformed input.",
    "Generic": "### 📦 Generic\n- Catch-all for encryption attacks like brute force.",
    "Normal": "### ✅ Normal\n- Legitimate network session.",
    "Reconnaissance": "### 🔭 Reconnaissance\n- Information gathering and scanning.",
    "Shellcode": "### 🐚 Shellcode\n- Executable code injection.",
    "Worms": "### 🐛 Worms\n- Self-replicating malware attacks."
}

# App title
st.title("🔐 EnGuardia - Network Attack Predictor")
st.markdown("Input the session-level network features below to predict the cyberattack type:")

# Encoding map display
if 'state' in label_encoders:
    st.markdown("### 🧬 State Encoding Map")
    state_encoder = label_encoders['state']
    df_state = pd.DataFrame({
        "State": state_encoder.classes_,
        "Encoded Value": state_encoder.transform(state_encoder.classes_)
    })
    st.dataframe(df_state)

# Input Form
with st.form("attack_form"):
    st.subheader("📥 Input Network Session Features")
    user_input = {}
    for feature in selected_features:
        label = feature_names.get(feature, feature)
        if feature in label_encoders:
            user_input[feature] = st.text_input(f"{label} (categorical)", value="")
        else:
            user_input[feature] = st.number_input(f"{label} (numeric)", value=0.0)
    submitted = st.form_submit_button("🎯 Predict Attack Type")

# Prediction Logic
if submitted:
    try:
        input_df = pd.DataFrame([user_input])

        for col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

        for col in input_df.columns:
            if col not in label_encoders:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        if input_df.isnull().any().any():
            st.error("⚠️ One or more fields have invalid or missing values.")
        else:
            scaled = scaler.transform(input_df)
            pred = model.predict(scaled)[0]
            pred_label = label_encoder_y.inverse_transform([pred])[0]

            st.success(f"🛡️ Predicted Attack Type: **{pred_label}**")
            st.markdown(attack_details.get(pred_label, "ℹ️ No description available."))

            # Visualization
            st.markdown("---")
            st.subheader("📊 Feature Input Visualization")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                x=[feature_names.get(f, f) for f in user_input.keys()],
                y=list(user_input.values()),
                palette=sns.color_palette(["#f63366"])
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylabel("Value")
            ax.set_title("Input Feature Distribution", color="#262730")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"🚨 Error during prediction: {e}")


