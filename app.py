import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="EnGuardia - Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color scheme
COLORS = ["#008FFF", "#A4C013", "#6200EF", "#E7A930"]
BACKGROUND_COLOR = "white"

# Load model and components
@st.cache_resource
def load_components():
    try:
        model = joblib.load('saved_models/nids_xgb_model.joblib')
        scaler = joblib.load('saved_models/nids_scaler.joblib')
        selector = joblib.load('saved_models/nids_feature_selector.joblib')
        label_encoder_y = joblib.load('saved_models/nids_label_encoder.joblib')
        label_encoders = joblib.load('saved_models/nids_input_encoders.joblib')
        selected_feature_names = joblib.load('saved_models/nids_selected_features.joblib')
        return model, scaler, selector, label_encoder_y, label_encoders, selected_feature_names
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        st.stop()

model, scaler, selector, label_encoder_y, label_encoders, selected_feature_names = load_components()

# Attack information
attack_info = {
    "Analysis": {
        "🔍 What it is":
"The Analysis attack category refers to malicious activities that involve the inspection or manipulation of system or network behaviors with the intent to exploit, understand, or destabilize the environment.
This includes the use of tools or techniques designed to gather detailed data about system processes, network behavior, or vulnerabilities."
"Examples may include":
"1.Packet sniffers"
"2.Traffic analyzers"
"3.System call tracing"
"4.Protocol manipulation"
"5.Malicious scanning scripts",
"❓ Why it occurs":
"1. Attackers perform analysis to":
"2. Study the system for weaknesses"
"3. Develop tailored attack strategies"
"4. Extract intelligence about defenses"
"5. Trigger specific faults in network behavior"
"6. It’s often a preparatory or probing step that precedes more aggressive attacks.",
"⚙️ What causes it":
"1. Insider threats or compromised accounts"
"2. Installation of spyware or analysis tools"
"3. Exploitation of legitimate monitoring systems (e.g., misused Wireshark)"
"4. Remote attackers testing how the system reacts under certain conditions"
"5. Tools used to reverse engineer protocol or traffic behavior",
"🛡️ How to deal with it":
"1. Monitor and log system analysis tools and unusual access patterns"
"2. Restrict access to debugging and monitoring interfaces"
"3. Implement host-based intrusion detection (HIDS)"
"4. Use endpoint security to detect unauthorized behavior or tool usage"
"5. Secure sensitive data-in-motion with encryption to prevent packet sniffing",
"☠️ Risks & Potential Damage":
"1. Leakage of system internals (services, architecture, vulnerabilities)"
"2. Exploitable intelligence for targeted exploits"
"3. Weakening of defense mechanisms through learned behavior"
"4. Unnoticed footholds in the system"
"5. Precursor to privilege escalation, exploitation, or full-scale breaches"
    },
    "Backdoor": {
        "description": "Backdoor attacks create hidden entry points to bypass normal authentication.",
        "why": "Attackers maintain persistent access to compromised systems.",
        "causes": "Malware infections, vulnerable software, or weak credentials.",
        "prevention": "Regular system scans, strict access controls, and monitoring for unusual outbound connections.",
        "risks": "Complete system compromise, data theft, and lateral movement in the network.",
    },
    # ... [keep other attack types as in your original code]
    "Normal": {
        "description": "Normal network traffic with no malicious indicators.",
    }
}

# State mapping for encoding
state_mapping = {
    "ACC": 0, "CLO": 1, "CON": 2, "ECO": 3, "ECR": 4, 
    "FIN": 5, "INT": 6, "MAS": 7, "PAR": 8, "REQ": 9, 
    "RST": 10, "TST": 11, "TXD": 12, "URH": 13
}

def make_prediction(input_data):
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=selected_feature_names)
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        return label_encoder_y.inverse_transform([prediction])[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    # Custom CSS
    st.markdown(f"""
        <style>
            .main {{ background-color: {BACKGROUND_COLOR}; }}
            .stButton>button {{
                background-color: {COLORS[0]};
                color: white;
                font-weight: bold;
                border: none;
            }}
            .stButton>button:hover {{
                background-color: {COLORS[2]};
                color: white;
            }}
            .attack-card {{
                border-left: 5px solid {COLORS[0]};
                padding: 10px;
                margin: 10px 0;
                background-color: #f8f9fa;
            }}
        </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ EnGuardia - Network Intrusion Detection System")
    st.markdown("""
        Predict network attacks based on traffic characteristics. Enter the network flow details below 
        and our AI model will analyze the patterns to detect potential threats.
    """)

    # Create input form
    with st.form("prediction_form"):
        st.subheader("Network Traffic Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dur = st.number_input("Duration (dur)", min_value=0.0, value=0.12, step=0.01)
            dpkts = st.number_input("Destination Packets (dpkts)", min_value=0, value=4)
            sbytes = st.number_input("Source Bytes (sbytes)", min_value=0, value=258)
            
        with col2:
            dbytes = st.number_input("Destination Bytes (dbytes)", min_value=0, value=172)
            rate = st.number_input("Packet Rate (rate)", min_value=0.0, value=74.08, step=0.01)
            sttl = st.number_input("Source TTL (sttl)", min_value=0, value=252)
            
        with col3:
            dttl = st.number_input("Destination TTL (dttl)", min_value=0, value=254)
            state = st.selectbox("Connection State (state)", options=list(state_mapping.keys()), index=5)
            submit_button = st.form_submit_button("Predict Attack Type")

    if submit_button:
        input_data = {
            "dur": dur,
            "state": state_mapping[state],
            "dpkts": dpkts,
            "sbytes": sbytes,
            "dbytes": dbytes,
            "rate": rate,
            "sttl": sttl,
            "dttl": dttl,
            # Include other features as needed
        }
        
        attack_type = make_prediction(input_data)
        
        if attack_type:
            st.success(f"Predicted Attack Type: **{attack_type}**")
            display_attack_info(attack_type, input_data)

def display_attack_info(attack_type, input_data):
    st.markdown("---")
    st.header(f"Attack Analysis: {attack_type}")
    
    if attack_type != "Normal":
        with st.container():
            st.markdown(f"""
                <div class="attack-card">
                    <h3>🔍 What it is:</h3>
                    <p>{attack_info[attack_type]['description']}</p>
                    <h3>❓ Why it occurs:</h3>
                    <p>{attack_info[attack_type]['why']}</p>
                    <h3>⚙️ What causes it:</h3>
                    <p>{attack_info[attack_type]['causes']}</p>
                    <h3>🛡️ How to deal with it:</h3>
                    <p>{attack_info[attack_type]['prevention']}</p>
                    <h3>☠️ Risks & Potential Damage:</h3>
                    <p>{attack_info[attack_type]['risks']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Visualization section
    st.subheader("Traffic Analysis")
    visualize_attack(attack_type, input_data)

def visualize_attack(attack_type, input_data):
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic metrics visualization
        fig = px.bar(
            x=["Duration", "Packets", "Bytes"],
            y=[input_data["dur"], input_data["dpkts"], input_data["sbytes"]],
            title="Key Traffic Metrics",
            color_discrete_sequence=COLORS
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Radar chart for multiple features
        categories = ['Rate', 'Duration', 'Packets', 'Bytes', 'TTL']
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[
                input_data.get("rate", 0)/10, 
                input_data.get("dur", 0)*10, 
                input_data.get("dpkts", 0),
                input_data.get("sbytes", 0)/100, 
                input_data.get("sttl", 0)
            ],
            theta=categories,
            fill='toself',
            name='Current Traffic'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 150])),
            showlegend=True,
            title="Traffic Metrics Radar"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
