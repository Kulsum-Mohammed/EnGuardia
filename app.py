import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Custom CSS for the color palette
def set_custom_style():
    st.markdown(f"""
    <style>
        :root {{
            --primary: #8956F1;
            --secondary: #CB80FF;
            --dark: #241B35;
            --darker: #020102;
        }}
        
        .stApp {{
            background-color: var(--darker);
            color: white;
        }}
        
        .st-b7 {{
            color: white !important;
        }}
        
        .st-c0 {{
            background-color: var(--dark);
        }}
        
        .css-1aumxhk {{
            background-color: var(--dark);
            background-image: none;
            color: white;
        }}
        
        .st-bb {{
            background-color: transparent;
        }}
        
        .st-at {{
            background-color: var(--primary);
        }}
        
        .st-ae {{
            background-color: var(--dark);
        }}
        
        .st-cg {{
            color: white;
        }}
        
        .st-ci {{
            background-color: var(--dark);
            border-color: var(--primary);
        }}
        
        .stAlert {{
            background-color: var(--dark);
            border-left: 0.5rem solid var(--secondary) !important;
        }}
        
        .st-bh {{
            border: 1px solid var(--primary);
        }}
        
        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: var(--dark);
            color: white;
            border-color: var(--primary);
        }}
        
        .stNumberInput div[data-baseweb="input"] > div {{
            background-color: var(--dark);
            color: white;
            border-color: var(--primary);
        }}
        
        .stSlider div[data-baseweb="slider"] > div > div > div {{
            background-color: var(--secondary);
        }}
        
        .stButton button {{
            background-color: var(--primary);
            color: white;
            border: none;
            transition: all 0.3s ease;
        }}
        
        .stButton button:hover {{
            background-color: var(--secondary);
            color: var(--darker);
        }}
        
        .stTab {{
            background-color: var(--dark);
        }}
        
        .stTab .stTabActive {{
            border-bottom: 3px solid var(--secondary);
        }}
        
        .stDataFrame {{
            background-color: var(--dark);
        }}
        
        footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

# Initialize custom style
set_custom_style()

# Page Config
st.set_page_config(
    page_title="EnGuardia NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model
@st.cache_resource
def load_model():
    try:
        return joblib.load('enguardia_model.pkl')
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Prediction Function
def predict_attack(input_data, pkg):
    try:
        processed = pd.DataFrame(0, index=[0], columns=pkg['feature_names'])
        
        # Fill numerical features
        for col in pkg['num_cols']:
            processed[col] = input_data.get(col, 0)
        
        # Fill categorical features
        for col in pkg['cat_cols']:
            val = input_data.get(col)
            encoded_col = f"{col}_{val}"
            if encoded_col in processed.columns:
                processed[encoded_col] = 1
        
        # Scale numerical features
        processed[pkg['num_cols']] = pkg['scaler'].transform(processed[pkg['num_cols']])
        
        # Predict
        pred = pkg['model'].predict(processed)[0]
        proba = pkg['model'].predict_proba(processed)[0]
        
        return {
            'attack_type': pkg['attack_mapping'][pred],
            'confidence': float(np.max(proba)),
            'probabilities': {pkg['attack_mapping'][i]: float(p) for i, p in enumerate(proba)},
            'status': 'success'
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# Initialize Session State
if 'attack_history' not in st.session_state:
    st.session_state.attack_history = pd.DataFrame(columns=[
        'timestamp', 'attack_type', 'confidence', 'src_bytes', 'dst_bytes'
    ])

# Custom Components
def metric_card(title, value, delta=None):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #241B35 0%, #020102 100%);
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 5px solid #8956F1;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    ">
        <h3 style="color: #CB80FF; margin-top: 0;">{title}</h3>
        <h1 style="color: white; margin-bottom: 0.5rem;">{value}</h1>
        {f'<p style="color: #CB80FF; margin-bottom: 0;">{delta}</p>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

def threat_alert(threat_type, source, protocol, bytes):
    color_map = {
        'dos': '#FF4D4D',
        'probe': '#FFA34D',
        'r2l': '#FF4DFF',
        'u2r': '#FF4D4D'
    }
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #241B35 0%, #020102 100%);
        border-radius: 8px;
        padding: 1rem;
        border-left: 5px solid {color_map.get(threat_type.lower(), '#8956F1')};
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    ">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span style="
                background-color: {color_map.get(threat_type.lower(), '#8956F1')};
                color: white;
                padding: 0.3rem 0.6rem;
                border-radius: 20px;
                font-weight: bold;
                font-size: 0.8rem;
            ">🚨 {threat_type.upper()}</span>
            <span style="color: #CB80FF;">{source} → {protocol} ({bytes} bytes)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main App
def main():
    st.title("EnGuardia Network Intrusion Detection")
    st.markdown("""
    <style>
        .title {
            color: #CB80FF;
        }
        .subtitle {
            color: #8956F1;
        }
    </style>
    """, unsafe_allow_html=True)
    
    pkg = load_model()
    
    if pkg is None:
        st.warning("Please train the model first using train_model.py")
        return
    
    tab1, tab2, tab3 = st.tabs(["📡 Real-Time Monitor", "📊 Threat Analysis", "🔍 Manual Inspection"])
    
    with tab1:
        st.header("Network Traffic Dashboard", anchor=False)
        st.markdown("---")
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Total Packets", "1,247", "↑ 12%")
        with col2:
            metric_card("Threats Detected", "23", "↓ 3%")
        with col3:
            metric_card("Detection Accuracy", "99.2%", "↑ 0.4%")
        with col4:
            metric_card("Response Time", "42ms", "↓ 8ms")
        
        # Live Traffic Section
        st.subheader("Live Traffic Stream", anchor=False)
        
        # Simulate live data
        live_data = pd.DataFrame({
            'timestamp': [datetime.now().strftime("%H:%M:%S") for _ in range(50)],
            'source': [f"192.168.1.{np.random.randint(1,50)}" for _ in range(50)],
            'destination': [f"10.0.0.{np.random.randint(1,20)}" for _ in range(50)],
            'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], 50),
            'bytes': np.random.randint(100, 5000, 50),
            'threat': np.random.choice(['normal', 'dos', 'probe', 'r2l'], 50, p=[0.85, 0.05, 0.07, 0.03])
        })
        
        # Display alerts
        alerts = live_data[live_data['threat'] != 'normal']
        if not alerts.empty:
            st.subheader("Active Alerts", anchor=False)
            for _, row in alerts.iterrows():
                threat_alert(row['threat'], row['source'], row['protocol'], row['bytes'])
        
        # Show live chart
        st.subheader("Traffic Visualization", anchor=False)
        fig = px.scatter(
            live_data,
            x='timestamp',
            y='bytes',
            color='threat',
            color_discrete_map={
                'normal': '#8956F1',
                'dos': '#FF4D4D',
                'probe': '#FFA34D',
                'r2l': '#FF4DFF'
            },
            size='bytes',
            hover_data=['source', 'destination', 'protocol'],
            template='plotly_dark'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Time",
            yaxis_title="Bytes Transferred",
            legend_title="Threat Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Threat Intelligence Center", anchor=False)
        st.markdown("---")
        
        if not st.session_state.attack_history.empty:
            # Attack Distribution
            st.subheader("Attack Distribution", anchor=False)
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.pie(
                    st.session_state.attack_history,
                    names='attack_type',
                    color_discrete_sequence=['#8956F1', '#FF4D4D', '#FFA34D', '#FF4DFF'],
                    hole=0.4,
                    template='plotly_dark'
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Recent Attacks", anchor=False)
                st.dataframe(
                    st.session_state.attack_history.sort_values('timestamp', ascending=False).head(5),
                    column_config={
                        "timestamp": "Time",
                        "attack_type": "Threat Type",
                        "confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            format="%.1f%%",
                            min_value=0,
                            max_value=1
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
            
            # Timeline Analysis
            st.subheader("Attack Timeline", anchor=False)
            timeline = st.session_state.attack_history.copy()
            timeline['count'] = 1
            timeline = timeline.set_index('timestamp').resample('5T').sum().reset_index()
            
            fig = px.area(
                timeline,
                x='timestamp',
                y='count',
                color_discrete_sequence=['#CB80FF'],
                template='plotly_dark'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Time",
                yaxis_title="Attack Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No threats detected yet")
    
    with tab3:
        st.header("Packet Inspector", anchor=False)
        st.markdown("---")
        
        with st.form("packet_form"):
            st.subheader("Network Packet Details", anchor=False)
            col1, col2 = st.columns(2)
            
            with col1:
                protocol = st.selectbox("Protocol", pkg['encoder'].categories_[0], key='protocol')
                service = st.selectbox("Service", pkg['encoder'].categories_[1], key='service')
                flag = st.selectbox("Flag", pkg['encoder'].categories_[2], key='flag')
                src_bytes = st.number_input("Source Bytes", min_value=0, value=100, key='src_bytes')
                dst_bytes = st.number_input("Destination Bytes", min_value=0, value=50, key='dst_bytes')
            
            with col2:
                logged_in = st.selectbox("Logged In", [0, 1], key='logged_in')
                count = st.number_input("Count", min_value=0, value=10, key='count')
                srv_count = st.number_input("Service Count", min_value=0, value=5, key='srv_count')
                same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0, 0.8, key='same_srv_rate')
            
            if st.form_submit_button("Analyze Packet", type='primary'):
                input_data = {
                    'protocol_type': protocol,
                    'service': service,
                    'flag': flag,
                    'src_bytes': src_bytes,
                    'dst_bytes': dst_bytes,
                    'logged_in': logged_in,
                    'count': count,
                    'srv_count': srv_count,
                    'same_srv_rate': same_srv_rate,
                    'diff_srv_rate': 1.0 - same_srv_rate
                }
                
                with st.spinner("Analyzing network packet..."):
                    result = predict_attack(input_data, pkg)
                
                if result['status'] == 'success':
                    st.success(f"Prediction: **{result['attack_type'].upper()}** (Confidence: {result['confidence']*100:.1f}%)")
                    
                    # Show probabilities
                    st.subheader("Threat Probability Distribution", anchor=False)
                    prob_df = pd.DataFrame({
                        'Attack': list(result['probabilities'].keys()),
                        'Probability': [p*100 for p in result['probabilities'].values()]
                    })
                    
                    fig = go.Figure(go.Bar(
                        x=prob_df['Probability'],
                        y=prob_df['Attack'],
                        orientation='h',
                        marker_color=['#8956F1', '#FF4D4D', '#FFA34D', '#FF4DFF', '#CB80FF'],
                        text=prob_df['Probability'].round(1).astype(str) + '%',
                        textposition='auto'
                    ))
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_title="Probability (%)",
                        yaxis_title="Attack Type",
                        template='plotly_dark',
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Log the detection
                    new_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'attack_type': result['attack_type'],
                        'confidence': result['confidence'],
                        'src_bytes': src_bytes,
                        'dst_bytes': dst_bytes
                    }
                    st.session_state.attack_history = pd.concat([
                        st.session_state.attack_history,
                        pd.DataFrame([new_entry])
                    ], ignore_index=True)
                else:
                    st.error(f"Analysis failed: {result['message']}")

if __name__ == "__main__":
    main()