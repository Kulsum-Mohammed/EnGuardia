import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Custom Styling
st.markdown("""
    <style>
        body, .main { background-color: #F8F9FA; }
        h1, h2, h3, h4, h5, h6 { color: #0A66C2; }
        .stButton>button {
            background-color: #0066FF;
            color: white;
            border-radius: 10px;
            padding: 0.5em 1em;
            font-weight: bold;
        }
        .stTextInput>div>div>input, .stNumberInput input {
            border: 1px solid #198754;
            padding: 0.5em;
            border-radius: 5px;
        }
        .stForm {
            background-color: #FFFFFF;
            padding: 2em;
            border-radius: 10px;
            border: 1px solid #DEE2E6;
        }
    </style>
""", unsafe_allow_html=True)

# Load components
model = joblib.load('saved_models/nids_xgb_model.joblib')
scaler = joblib.load('saved_models/nids_scaler.joblib')
label_encoder_y = joblib.load('saved_models/nids_label_encoder.joblib')
label_encoders = joblib.load('saved_models/nids_input_encoders.joblib')
selected_features = joblib.load('saved_models/nids_selected_features.joblib')

# Feature full names mapping
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
    'ct_srv_dst': 'Connections to Same Service' 
}

# Attack descriptions
attack_details = {
    "Analysis": """
### 🕵️ Analysis Attack
- **What it is**: Probing systems to discover vulnerabilities.
- **Why it occurs**: To gather intel before launching deeper attacks.
- **What causes it**: Inadequate firewall rules or IDS.
- **How to deal with it**: Deploy anomaly-based IDS and traffic monitoring.
- **Risks & Potential Damage**: System reconnaissance, intelligence leak.
""",
    "Backdoor": """
### 🔓 Backdoor Attack
- **What it is**: Hidden access points to bypass security controls.
- **Why it occurs**: Often left by malware or insiders.
- **What causes it**: Poorly secured applications.
- **How to deal with it**: Endpoint monitoring and access audits.
- **Risks & Potential Damage**: Unauthorized control, data breaches.
""",
    "DoS": """
### 🚫 Denial of Service (DoS)
- **What it is**: Flooding services to make them unavailable.
- **Why it occurs**: Disruption or extortion.
- **What causes it**: Exploiting network weaknesses.
- **How to deal with it**: Traffic filtering, rate-limiting.
- **Risks & Potential Damage**: Service outage, financial loss.
""",
    "Exploits": """
### 🧨 Exploits
- **What it is**: Use of software bugs to gain control.
- **Why it occurs**: To escalate privileges or steal data.
- **What causes it**: Unpatched vulnerabilities.
- **How to deal with it**: Regular patching, code audits.
- **Risks & Potential Damage**: System compromise.
""",
    "Fuzzers": """
### 💥 Fuzzing Attack
- **What it is**: Sending malformed data to crash systems.
- **Why it occurs**: Bug discovery or sabotage.
- **What causes it**: Lack of input validation.
- **How to deal with it**: Sanitize inputs and use fuzz testing.
- **Risks & Potential Damage**: System crashes, exposed flaws.
""",
    "Generic": """
### 📦 Generic Attack
- **What it is**: General, non-categorized attack types.
- **Why it occurs**: Automated scans or default exploits.
- **What causes it**: Weak baseline defenses.
- **How to deal with it**: Use firewalls, IP filtering.
- **Risks & Potential Damage**: Unauthorized access or info leak.
""",
    "Normal": """
### ✅ Normal Traffic
- No attack detected. Session looks safe.
""",
    "Reconnaissance": """
### 🔭 Reconnaissance
- **What it is**: Info-gathering phase of an attack.
- **Why it occurs**: To plan targeted exploitation.
- **What causes it**: Insecure services or exposed ports.
- **How to deal with it**: Use honeypots and strict access logs.
- **Risks & Potential Damage**: Future targeted attacks.
""",
    "Worms": """
### 🐛 Worm Attack
- **What it is**: Self-replicating malware that spreads.
- **Why it occurs**: To infect and propagate automatically.
- **What causes it**: Network service vulnerabilities.
- **How to deal with it**: Apply patches, network segmentation.
- **Risks & Potential Damage**: Widespread infection, data loss.
"""
}

# App title
st.title("🔐 EnGuardia - NIDS Attack Predictor")
st.markdown("Enter session features to detect the attack type.")

# Form Input
with st.form("attack_form"):
    user_input = {}
    for feature in selected_features:
        label = feature_names.get(feature, feature)
        if feature in label_encoders:
            user_input[feature] = st.text_input(f"{label} (categorical)")
        else:
            user_input[feature] = st.number_input(f"{label} (numeric)", value=0.0)
    submitted = st.form_submit_button("Predict Attack Type")

# Prediction and Results
if submitted:
    try:
        input_df = pd.DataFrame([user_input])

        for col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

        for col in input_df.columns:
            if col not in label_encoders:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        if input_df.isnull().any().any():
            st.error("⚠️ Invalid or missing input values.")
        else:
            scaled_input = scaler.transform(input_df)
            pred = model.predict(scaled_input)[0]
            label = label_encoder_y.inverse_transform([pred])[0]

            st.success(f"🛡️ Predicted Attack Type: **{label}**")
            st.markdown(attack_details.get(label, "No description available."))

            # Visualization
            st.markdown("---")
            st.subheader("📊 Input Feature Visualization")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                x=[feature_names.get(k, k) for k in user_input.keys()],
                y=list(user_input.values()),
                palette='coolwarm'
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title("Session Input Features")
            ax.set_ylabel("Value")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")

