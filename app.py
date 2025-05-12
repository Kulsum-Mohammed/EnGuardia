import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set up custom UI theme
st.markdown("""
    <style>
        body {
            background-color: white;
        }
        .main {
            background-color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #008FFF;
        }
        .stButton>button {
            background-color: #6200EF;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-weight: bold;
        }
        .stTextInput>div>div>input, .stNumberInput input {
            border: 1px solid #A4C013;
            padding: 0.5em;
        }
        .stForm {
            border: 1px solid #E7A930;
            border-radius: 10px;
            padding: 1em;
        }
        .stMarkdown {
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)

# Load ML components
model = joblib.load('saved_models/nids_xgb_model.joblib')
scaler = joblib.load('saved_models/nids_scaler.joblib')
selector = joblib.load('saved_models/nids_feature_selector.joblib')
label_encoder_y = joblib.load('saved_models/nids_label_encoder.joblib')
label_encoders = joblib.load('saved_models/nids_input_encoders.joblib')
selected_feature_names = joblib.load('saved_models/nids_selected_features.joblib')

# Attack descriptions
descriptions = {
    "Analysis": """
1. Analysis Attack
🔍 What it is: This involves probing a system to identify vulnerabilities or gather information.
❓ Why it occurs: To prepare for more severe intrusions.
⚙️ What causes it: Lack of proper firewall rules or IDS.
🛡️ How to deal with it: Use up-to-date firewalls and anomaly-based IDS.
☠️ Risks & Potential Damage: Data leakage and reconnaissance for major attacks.
""",
    "Backdoor": """
2. Backdoor Attack
🔍 What it is: A method to bypass authentication to access a system secretly.
❓ Why it occurs: Often installed by malware or insider threat.
⚙️ What causes it: Insecure applications or poor patch management.
🛡️ How to deal with it: Use endpoint protection and monitor traffic.
☠️ Risks & Potential Damage: Total system compromise.
""",
    "DoS": """
3. DoS (Denial of Service) Attack
🔍 What it is: Overwhelming a system with requests to make it unavailable.
❓ Why it occurs: To disrupt services.
⚙️ What causes it: Exploiting network or protocol weaknesses.
🛡️ How to deal with it: Rate limiting, firewalls, and traffic analysis.
☠️ Risks & Potential Damage: Downtime, revenue loss.
""",
    "Exploits": """
4. Exploits
🔍 What it is: Taking advantage of software vulnerabilities.
❓ Why it occurs: To gain control or access data.
⚙️ What causes it: Unpatched systems.
🛡️ How to deal with it: Apply updates, use vulnerability scanners.
☠️ Risks & Potential Damage: Full system takeover.
""",
    "Fuzzers": """
5. Fuzzers
🔍 What it is: Inputting random data to crash or expose bugs.
❓ Why it occurs: Testing or malicious intent.
⚙️ What causes it: No input validation.
🛡️ How to deal with it: Sanitize inputs and monitor for unusual behavior.
☠️ Risks & Potential Damage: Crashes, exposure of sensitive data.
""",
    "Generic": """
6. Generic
🔍 What it is: Common patterns or attacks not tied to a specific method.
❓ Why it occurs: Often automated scans or bots.
⚙️ What causes it: Lack of traffic filtering.
🛡️ How to deal with it: Intrusion Prevention Systems (IPS).
☠️ Risks & Potential Damage: System access or enumeration.
""",
    "Normal": "This is normal traffic. No malicious activity detected.",
    "Reconnaissance": """
7. Reconnaissance
🔍 What it is: Mapping out the network or gathering intel.
❓ Why it occurs: To plan a more severe attack.
⚙️ What causes it: Lack of detection rules.
🛡️ How to deal with it: Use honeypots and alerting.
☠️ Risks & Potential Damage: Full-blown attacks if unchecked.
""",
    "Worms": """
8. Worms
🔍 What it is: Self-replicating malware spreading across systems.
❓ Why it occurs: To infect and propagate automatically.
⚙️ What causes it: Vulnerable network services.
🛡️ How to deal with it: Patch regularly, monitor anomalies.
☠️ Risks & Potential Damage: Widespread disruption.
"""
}

# Title and instructions
st.title("🔐 EnGuardia - NIDS Attack Predictor")
st.markdown("Input the network session features to predict the type of cyberattack.")

# Input form
with st.form("attack_form"):
    user_input = {}
    for feature in selected_feature_names:
        if feature in label_encoders:
            user_input[feature] = st.text_input(f"{feature} (categorical)")
        else:
            user_input[feature] = st.number_input(f"{feature} (numeric)", value=0.0)

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    try:
        input_df = pd.DataFrame([user_input], columns=selected_feature_names)

        for col in label_encoders:
            if col in input_df.columns:
                input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

        for col in input_df.columns:
            if col not in label_encoders:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        if input_df.isnull().any().any():
            st.error("Invalid or missing input values.")
        else:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            label = label_encoder_y.inverse_transform([prediction])[0]

            st.success(f"🛡️ Predicted Attack Type: {label}")
            st.markdown(descriptions.get(label, "No description available."))

            st.markdown("---")
            st.subheader("📊 Feature Visualization")

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=list(user_input.keys()), y=list(user_input.values()), palette=['#008FFF']*len(user_input))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_title("Input Feature Values", color="#6200EF")
            ax.tick_params(colors='#333333')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
