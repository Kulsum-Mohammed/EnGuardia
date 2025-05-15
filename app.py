import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

# Load models and encoders
scaler = joblib.load('saved_models/nids_scaler.joblib')
model = joblib.load('saved_models/nids_xgb_model.joblib')
label_encoder = joblib.load('saved_models/nids_label_encoder.joblib')
input_encoders = joblib.load('saved_models/nids_input_encoders.joblib')
selected_features = joblib.load('saved_models/nids_selected_features.joblib')

# Load state encoder and options
state_encoder = input_encoders['state']
state_options = list(state_encoder.classes_)

# Detailed attack descriptions
attack_info = {
    "Normal": """
    **Normal** traffic represents baseline, non-malicious activity in a network or system.  

    **🔍 Why it happens:**  
    - **Legitimate User Actions:** Web browsing, email, API calls, database queries.  
    - **System Processes:** Scheduled tasks (cron jobs), backups, cloud sync (AWS S3, Azure Blob).  
    - **Automated Traffic:** CDN requests (Cloudflare), bot traffic (Googlebot), IoT heartbeats.  

    **⚠️ Risks:**  
    - **False Negatives:** Attackers may mimic normal behavior (e.g., DNS tunneling hidden in HTTP traffic).  
    - **Insider Threats:** Authorized users performing malicious actions under the guise of normal activity.  

    **🛡️ Mitigation:**  
    - **Behavioral Baselining:** Use ML models (Darktrace, Splunk UBA) to detect deviations.  
    - **Log Correlation:** SIEM (Splunk, ELK) to cross-check logs for inconsistencies.  

    **🔒 Prevention:**  
    - **Zero Trust Architecture (ZTA):** Enforce strict identity verification (Okta, Azure AD).  
    - **Anomaly Detection:** Deploy unsupervised learning models to flag subtle shifts in traffic.  
    """,

    "Backdoor": """
    **Backdoor** attacks create hidden access points, allowing attackers persistent control.  

    **🔍 Why it happens:**  
    - **Exploited Vulnerabilities:** CVE-2024-XXXX (unpatched RCE in web apps).  
    - **Malware Payloads:** Trojans (Emotet), RATs (Cobalt Strike), or webshells (China Chopper).  
    - **Misconfigurations:** Default credentials (admin:admin), exposed Docker APIs, or unprotected Redis instances.  

    **⚠️ Risks:**  
    - **Persistence:** Attackers maintain access even after reboots (Windows Registry, cronjobs).  
    - **Lateral Movement:** Use backdoors for privilege escalation (Pass-the-Hash, Kerberoasting).  
    - **Data Exfiltration:** Silent data theft (e.g., FTP transfers masked as normal traffic).  

    **🛡️ Mitigation:**  
    - **EDR Solutions:** CrowdStrike, SentinelOne to detect suspicious process injections.  
    - **Honeypots:** Deploy fake backdoors to lure and identify attackers.  

    **🔒 Prevention:**  
    - **Patch Management:** Automate updates (WSUS, Ansible).  
    - **Network Segmentation:** Isolate critical servers (PCI-DSS compliance).  
    - **Memory Scanning:** Use tools like YARA to detect in-memory backdoors.  
    """,

    "Fuzzers": """
    **Fuzzers** automate input testing to crash software or uncover 0-day vulnerabilities.  

    **🔍 Why it happens:**  
    - **Bug Bounty Hunting:** Ethical hackers using AFL, Peach Fuzzer.  
    - **Malicious Actors:** Exploit kits (Metasploit’s `auxiliary/fuzzers`).  
    - **Targets:** APIs (Swagger), IoT firmware, SCADA systems.  

    **⚠️ Risks:**  
    - **DoS Conditions:** Buffer overflow crashes (e.g., `ping of death`).  
    - **Arbitrary Code Execution:** Format string exploits (CWE-134).  
    - **Logic Flaws:** SQLi, XXE, or SSRF via malformed XML/JSON.  

    **🛡️ Mitigation:**  
    - **Input Sanitization:** Regex whitelisting, parameterized queries.  
    - **Rate Limiting:** Block IPs sending excessive malformed requests.  

    **🔒 Prevention:**  
    - **Fuzz Testing in DevSecOps:** OWASP ZAP, Burp Suite in CI/CD pipelines.  
    - **WAF Rules:** Cloudflare’s OWASP CRS to filter malicious payloads.  
    - **Memory Protections:** ASLR, DEP, and Stack Canaries.  
    """,

    "Reconnaissance": """
    **Reconnaissance** gathers intel for future attacks (MITRE ATT&CK: TA0043).  

    **🔍 Why it happens:**  
    - **Scanning Tools:** Nmap (`-sV` for service versions), Shodan (`org:"Company"`).  
    - **OSINT:** LinkedIn scraping, GitHub dorking (`password.txt`).  
    - **Example:** DNS zone transfers exposing internal hosts.  

    **⚠️ Risks:**  
    - **Attack Surface Mapping:** Discover weak points (e.g., outdated Apache 2.4.49).  
    - **Social Engineering:** Tailgating based on employee roles (from LinkedIn).  

    **🛡️ Mitigation:**  
    - **Deception Tech:** IllusionBlack, Thinkst Canaries.  
    - **Log Aggregation:** Detect repeated scans (ELK + Suricata rules).  

    **🔒 Prevention:**  
    - **Network Hardening:** Disable ICMP replies, use port knocking.  
    - **Threat Intelligence Feeds:** Block known scanner IPs (AbuseIPDB).  
    """,

    "Exploits": """
    **Exploits** weaponize vulnerabilities (CVEs) for unauthorized access.  

    **🔍 Why it happens:**  
    - **Unpatched Software:** EternalBlue (MS17-010), ProxyShell (CVE-2021-34473).  
    - **Zero-Days:** Pegasus spyware (NSO Group), FORCEDENTRY (Apple).  

    **⚠️ Risks:**  
    - **Privilege Escalation:** Dirty Pipe (CVE-2022-0847 → root access).  
    - **Ransomware:** WannaCry, LockBit exploiting RDP flaws.  

    **🛡️ Mitigation:**  
    - **Vulnerability Scanning:** Nessus, OpenVAS for CVE detection.  
    - **Sandboxing:** Isolate suspicious processes (FireJail, Qubes OS).  

    **🔒 Prevention:**  
    - **Patch Tuesdays:** Automate Windows/Linux updates.  
    - **Exploit Mitigations:** EMET, Windows Defender Exploit Guard.  
    """,

    "Analysis": """
    **Analysis** attacks reverse-engineer protocols or extract metadata.  

    **🔍 Why it happens:**  
    - **Traffic Inspection:** Wireshark decrypting weak TLS 1.0.  
    - **Metadata Leaks:** PDFs exposing internal IPs, GPS coordinates.  

    **⚠️ Risks:**  
    - **Deanonymization:** Tor exit node sniffing.  
    - **Side-Channel Attacks:** CPU cache timing (Meltdown/Spectre).  

    **🛡️ Mitigation:**  
    - **Encryption:** AES-256, Signal Protocol for messaging.  
    - **Data Masking:** Redact PII in logs (GDPR compliance).  

    **🔒 Prevention:**  
    - **Obfuscation:** VPNs, Tor, or domain fronting.  
    - **Strict Logging Policies:** Avoid storing sensitive data in plaintext.  
    """,

    "DoS": """
    **Denial of Service (DoS)** overwhelms systems to disrupt services.  

    **🔍 Why it happens:**  
    - **Volumetric Attacks:** UDP floods (Memcached amplification).  
    - **Application-Layer:** Slowloris (partial HTTP requests).  

    **⚠️ Risks:**  
    - **Financial Loss:** $20K/min downtime for e-commerce.  
    - **Smokescreen:** Distraction for data exfiltration.  

    **🛡️ Mitigation:**  
    - **Cloudflare Magic Transit:** Absorb DDoS traffic.  
    - **Anycast Routing:** Distribute load (AWS Shield).  

    **🔒 Prevention:**  
    - **Rate Limiting:** Nginx `limit_req_zone`.  
    - **BGP Flowspec:** ISP-level blackholing.  
    """,

    "Worms": """
    **Worms** self-replicate across networks (e.g., WannaCry, NotPetya).  

    **🔍 Why it happens:**  
    - **Protocol Exploits:** SMBv1 (EternalBlue), RDP (BlueKeep).  
    - **Phishing Payloads:** Malicious macros (Emotet → TrickBot).  

    **⚠️ Risks:**  
    - **Botnet Recruitment:** Mirai IoT worms for DDoS.  
    - **Wiper Malware:** Destructive payloads (Industroyer).  

    **🛡️ Mitigation:**  
    - **Air-Gapping:** Isolate critical ICS/SCADA systems.  
    - **Network Traffic Analysis:** Darktrace AI for lateral movement.  

    **🔒 Prevention:**  
    - **Patch Legacy Systems:** Windows XP embedded devices.  
    - **Microsegmentation:** VMware NSX for zero-trust networks.  
    """,

    "Generic": """
    **Generic** attacks are pattern-based detections (e.g., Snort rules).  

    **🔍 Why it happens:**  
    - **Signature Matches:** Known malware hashes (VirusTotal).  
    - **Heuristics:** Behavioral AI (Cylance, SentinelOne).  

    **⚠️ Risks:**  
    - **Polymorphic Malware:** Evades static signatures.  
    - **False Positives:** Legitimate tools flagged (PsExec).  

    **🛡️ Mitigation:**  
    - **Sandbox Analysis:** Joe Sandbox, Cuckoo.  
    - **Threat Hunting:** MITRE ATT&CK framework mapping.  

    **🔒 Prevention:**  
    - **UEBA:** User Entity Behavior Analytics (Exabeam).  
    - **EDR + XDR:** Endpoint + extended detection (Palo Alto Cortex).  
    """
}

# Updated visual style
st.markdown("""
    <style>
        body, .main {
            background-color: #1c1e26;
            color: #e0e0e0;
        }
        h1, h2, h3 {
            color: #fbbf24;
        }
        input, select, textarea {
            background-color: #2a2d3e !important;
            color: #ffffff !important;
            border: 1px solid #4b5563 !important;
        }
        .stButton>button {
            background-color: #10b981;
            color: white;
            font-weight: bold;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
        }
        .stNumberInput label, .stSelectbox label {
            color: white;  /* Change to black for light theme */
            font-weight: bold;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🔐 EnGuardia – Turning Packets into Patterns. Patterns into Protection.")

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

            proba = model.predict_proba(X_scaled)[0]
            pred_idx = np.argmax(proba)
            predicted_class = label_encoder.inverse_transform([pred_idx])[0]
            confidence = proba[pred_idx]

            st.success(f"🛡️ Predicted Attack Type: **{predicted_class}**")
            st.info(f"Confidence: **{confidence:.2%}**")

            if st.button("📖 About the Attack"):
                st.markdown(attack_info.get(predicted_class, "No information available."))

            st.subheader("📊 Visual Dashboard for Input Features")

            # Barplot for feature values
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            sns.barplot(x=list(X.columns), y=list(X.iloc[0]), palette="crest", ax=ax1)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
            ax1.set_ylabel("Value")
            ax1.set_title("Feature Input Values")
            st.pyplot(fig1)

            # Pie chart: Source vs Destination bytes
            fig2, ax2 = plt.subplots()
            ax2.pie([sbytes, dbytes], labels=['Source Bytes', 'Destination Bytes'], autopct='%1.1f%%', colors=['#3b82f6','#f59e0b'])
            ax2.set_title("Traffic Direction Volume")
            st.pyplot(fig2)

            # Histogram of packet sizes
            fig3, ax3 = plt.subplots()
            ax3.hist([smean, dmean], bins=10, color=['#10b981','#ef4444'], label=['Source Mean', 'Destination Mean'])
            ax3.legend()
            ax3.set_title("Distribution of Mean Packet Sizes")
            st.pyplot(fig3)

            # Radar-style plot (polar) for flow-related features
            radar_labels = ['rate', 'sload', 'dload', 'dinpkt']
            radar_values = [input_data[feat] for feat in radar_labels]
            radar_angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
            radar_values += radar_values[:1]
            radar_angles += radar_angles[:1]

            fig4, ax4 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax4.plot(radar_angles, radar_values, 'o-', linewidth=2, label='Flow Metrics')
            ax4.fill(radar_angles, radar_values, alpha=0.25)
            ax4.set_xticks(radar_angles[:-1])
            ax4.set_xticklabels(radar_labels)
            ax4.set_title("Flow Metrics Radar View")
            st.pyplot(fig4)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
