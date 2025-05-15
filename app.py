import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load models and encoders
scaler = joblib.load('saved_models/nids_scaler.joblib')
model = joblib.load('saved_models/nids_xgb_model.joblib')
label_encoder = joblib.load('saved_models/nids_label_encoder.joblib')
input_encoders = joblib.load('saved_models/nids_input_encoders.joblib')
selected_features = joblib.load('saved_models/nids_selected_features.joblib')

# Load state encoder and options
state_encoder = input_encoders['state']
state_options = list(state_encoder.classes_)
attack_info = {
    "Normal": """
    ### 🟢 Normal

    **Normal** traffic represents expected, benign activity within a network or system. It's the baseline upon which anomalies are identified.

    **🔍 Why it happens:**  
    Normal behavior can be initiated by both users and systems:
    - **Legitimate User Activity:** Accessing web pages, logging into apps, checking emails, downloading files, remote working via VPN.  
    - **Automated System Behavior:** Scheduled jobs (e.g., `cron`, Windows Task Scheduler), antivirus updates, software patching, or cloud service communication (like syncing with AWS S3).  
    - **Network Management Tasks:** Pings, SNMP polls, heartbeat messages from IoT devices or Kubernetes health checks.

    **⚠️ Risks:**  
    - **False Negatives:** Malicious actors often mimic normal behavior (e.g., data exfiltration disguised as legitimate HTTPS traffic).  
    - **Insider Threats:** Trusted users abusing access privileges, making it harder to distinguish from legitimate actions.  
    - **Living off the Land (LotL):** Attackers use built-in tools like PowerShell or `certutil` to blend in.

    **🛡️ Mitigation Strategies:**  
    - **Behavioral Analytics:** Use UEBA (User and Entity Behavior Analytics) and anomaly detection (e.g., Splunk UBA, Vectra AI) to flag deviations from baselines.  
    - **Log Correlation:** Combine logs across services (via SIEM platforms like ELK Stack or Splunk) to identify contextual anomalies.  

    **🔒 Prevention Techniques:**  
    - **Zero Trust Security Models:** Validate identity and access regardless of network location.  
    - **Strict Role-Based Access Control (RBAC):** Only allow access necessary for each role.  
    - **Continuous Monitoring:** Real-time visibility into every access attempt and data flow.
    """,

    "Backdoor": """
    ### 🚪 Backdoor

    **Backdoor** attacks establish covert channels that allow attackers to bypass normal authentication, granting remote control over compromised systems.

    **🔍 Why it happens:**  
    - **Software Vulnerabilities:** Remote Code Execution (RCE) flaws in outdated applications (e.g., CVE-2024-XXXX).  
    - **Malware Payloads:** Trojans and Remote Access Trojans (RATs) like Cobalt Strike, PlugX, or NjRAT create backdoor entry points.  
    - **Weak Configurations:** Systems with default credentials, open ports, or misconfigured containers (like Docker daemons) are frequent targets.

    **⚠️ Risks:**  
    - **Persistence Mechanisms:** Attackers use Windows Registry keys, Linux cronjobs, or startup scripts to ensure backdoor survival after reboots.  
    - **Command and Control (C2):** Attackers can execute commands, extract data, or deploy additional payloads remotely.  
    - **Lateral Movement:** Once inside, attackers can pivot to other systems using techniques like Pass-the-Hash or Kerberoasting.

    **🛡️ Mitigation Strategies:**  
    - **EDR/XDR Tools:** Use CrowdStrike, SentinelOne, or Microsoft Defender to detect and respond to abnormal process behaviors.  
    - **Network Deception:** Implement honeypots or honeytokens to trick and trace intruders.  
    - **Integrity Monitoring:** File integrity tools like Tripwire can detect unauthorized changes.

    **🔒 Prevention Techniques:**  
    - **Frequent Patching:** Automate vulnerability patching with tools like Ansible, WSUS, or Patch My PC.  
    - **Access Controls:** Enforce MFA, limit administrative privileges, and monitor privileged access logs.  
    - **Memory Forensics:** Tools like Volatility or YARA rules can uncover in-memory implants.
    """,

    "Fuzzers": """
    ### 🧪 Fuzzers

    **Fuzzers** are tools or techniques used to test software robustness by sending unexpected or malformed input to uncover vulnerabilities.

    **🔍 Why it happens:**  
    - **Security Researchers:** Ethical hackers use fuzzers like AFL, Peach, or Sulley for bug bounty hunting.  
    - **Cybercriminals:** Adversaries use fuzzing modules in frameworks like Metasploit to discover crash-prone endpoints.  
    - **Common Targets:** REST APIs, IoT firmware, web apps, protocol implementations, and industrial systems.

    **⚠️ Risks:**  
    - **Denial of Service (DoS):** Input flooding can crash services, often unintentionally.  
    - **Arbitrary Code Execution:** Poorly handled inputs may allow buffer overflows or format string exploits.  
    - **Information Disclosure:** Fuzzing can uncover endpoints revealing sensitive error messages or stack traces.

    **🛡️ Mitigation Strategies:**  
    - **Strict Input Validation:** Use whitelisting, regex, and JSON/XML schema validation.  
    - **Rate Limiting & Throttling:** Prevent abuse by limiting the frequency of requests.  
    - **Crash Monitoring:** Tools like Sentry or AppDynamics can alert on unexpected failures.

    **🔒 Prevention Techniques:**  
    - **Secure Development Lifecycle (SDLC):** Incorporate fuzz testing early using tools like OWASP ZAP, Boofuzz, or Burp Suite.  
    - **Runtime Protections:** Implement memory safety techniques such as ASLR (Address Space Layout Randomization), DEP (Data Execution Prevention), and Stack Canaries.  
    - **API Gateways:** Apply schema validation, access control, and throttling policies at the gateway level.
    """,
    "Reconnaissance": """
    ### 🔍 Reconnaissance

    **Reconnaissance** is the initial stage of an attack lifecycle, where adversaries collect information about their targets to identify potential attack vectors.

    **🔍 Why it happens:**  
    - **Scanning Tools:** Tools like Nmap, Masscan, and Shodan help attackers enumerate services and open ports.  
    - **Passive OSINT Collection:** Scraping employee data from LinkedIn, reading technical documentation, scanning GitHub for sensitive files (e.g., `.env`, `passwords.txt`).  
    - **DNS and Network Discovery:** Attackers perform zone transfers, subdomain enumeration (using tools like `dnsenum` or `Amass`), or traceroute mapping.

    **⚠️ Risks:**  
    - **Attack Surface Mapping:** Reveals exposed infrastructure such as outdated software versions or forgotten services (e.g., old admin portals).  
    - **Social Engineering:** Intel gathered from employees or organizational structure can fuel phishing and pretexting attacks.  
    - **Credential Stuffing Prep:** Recon helps identify valid usernames for brute force campaigns.

    **🛡️ Mitigation Strategies:**  
    - **Deception Technology:** Plant fake services or credentials (e.g., Canarytokens, Thinkst Canaries) to detect early probing.  
    - **Threat Detection:** Correlate logs to spot scanning patterns using Suricata/Zeek or ELK Stack dashboards.  
    - **Darknet Monitoring:** Monitor dark web and forums for discussions or data dumps about your organization.

    **🔒 Prevention Techniques:**  
    - **Service Hardening:** Disable unused ports/services and enforce least privilege.  
    - **Obfuscation & Port Knocking:** Use techniques like port knocking or single-packet authorization to conceal services.  
    - **Threat Intel Feeds:** Integrate feeds like AbuseIPDB, AlienVault OTX to blacklist known scanners.
    """,

    "Exploits": """
    ### 💣 Exploits

    **Exploits** are malicious programs or commands that take advantage of software or hardware vulnerabilities to gain unauthorized access or escalate privileges.

    **🔍 Why it happens:**  
    - **Unpatched Systems:** Legacy applications and OSes often contain well-known vulnerabilities like EternalBlue (SMBv1) or Heartbleed (OpenSSL).  
    - **0-Day Vulnerabilities:** Attackers may use undisclosed vulnerabilities to bypass defenses before patches are available.  
    - **Exploit Kits:** Tools like Metasploit and RIG contain prebuilt exploit scripts for common CVEs.

    **⚠️ Risks:**  
    - **Full System Compromise:** Exploits can lead to root/admin access.  
    - **Ransomware Delivery:** Used to drop and execute payloads like LockBit or REvil.  
    - **Espionage or Sabotage:** Nation-state actors use exploits for stealthy operations (e.g., Stuxnet targeting PLCs).

    **🛡️ Mitigation Strategies:**  
    - **Vulnerability Management:** Regular scanning using tools like Qualys, OpenVAS, or Nessus to identify known CVEs.  
    - **Application Sandboxing:** Run potentially vulnerable apps in isolated environments (e.g., Docker, Firejail).  
    - **Security Patching:** Establish SLA-based patch timelines for critical vulnerabilities.

    **🔒 Prevention Techniques:**  
    - **Attack Surface Reduction:** Disable unused features (e.g., ActiveX in browsers, RDP on public interfaces).  
    - **Exploit Mitigation Features:** Enable protections like Control Flow Guard (CFG), Data Execution Prevention (DEP), and Enhanced Mitigation Experience Toolkit (EMET).  
    - **Firmware & BIOS Updates:** Keep low-level firmware updated to prevent hardware exploit chains.
    """,

    "Analysis": """
    ### 🧬 Analysis

    **Analysis attacks** involve dissecting software, network traffic, or protocols to understand behavior and identify weaknesses—either for defense or exploitation.

    **🔍 Why it happens:**  
    - **Packet Sniffing:** Tools like Wireshark or tcpdump can extract credentials from unencrypted sessions or analyze protocol misconfigurations.  
    - **Reverse Engineering:** Decompiling binaries with IDA Pro, Ghidra, or using dynamic analysis to observe software behavior.  
    - **Metadata Inspection:** Attackers extract document metadata to reveal internal filenames, IPs, or usernames.

    **⚠️ Risks:**  
    - **Sensitive Data Exposure:** TLS 1.0/1.1 or weak ciphers may allow attackers to decrypt communications.  
    - **Side-Channel Attacks:** Techniques like Spectre, Meltdown, and RAMBleed exploit CPU/memory behavior to extract secrets.  
    - **Leakage via Logs:** Improper logging practices can expose tokens, passwords, or internal infrastructure.

    **🛡️ Mitigation Strategies:**  
    - **Strong Encryption Standards:** Enforce TLS 1.3, PFS (Perfect Forward Secrecy), and disable legacy ciphers.  
    - **Metadata Scrubbing:** Remove EXIF data from images/documents before sharing.  
    - **Endpoint Hardening:** Prevent debuggers or memory dumping tools from running on production systems.

    **🔒 Prevention Techniques:**  
    - **Obfuscation & Anti-Debugging:** Use software hardening and code obfuscation to prevent reverse engineering.  
    - **Security Policies for Logs:** Mask or hash PII and sensitive fields before storage (for GDPR/CCPA compliance).  
    - **Secure Protocols:** Prefer SSH over Telnet, SFTP over FTP, and ensure strict headers (HSTS, X-Content-Type).
    """,

    "DoS": """
    ### 🌩️ Denial of Service (DoS)

    **DoS** attacks aim to overwhelm network, application, or system resources to render them inaccessible to legitimate users.

    **🔍 Why it happens:**  
    - **Volumetric Attacks:** Flood the network with massive traffic—often using amplification techniques (e.g., DNS, NTP, or Memcached reflection).  
    - **Application Layer (Layer 7) Attacks:** Send slow or malformed requests (like Slowloris) that exhaust application threads.  
    - **Botnets & DDoS-as-a-Service:** Platforms like Booter services or Mirai botnet variants offer easy DoS capabilities.

    **⚠️ Risks:**  
    - **Downtime & Revenue Loss:** Even short disruptions can cost thousands to millions depending on the business model.  
    - **Operational Disruption:** Healthcare systems, banking, and utilities are prime targets due to the high impact.  
    - **Cover for Intrusion:** DoS can mask or distract from lateral movement or data exfiltration.

    **🛡️ Mitigation Strategies:**  
    - **Traffic Filtering:** Use upstream filtering via CDNs (e.g., Cloudflare, Akamai) or blackhole routing via ISPs.  
    - **Auto-Scaling & Rate Limits:** Deploy elastic services that absorb or deflect spikes in traffic.  
    - **DDoS Detection:** Solutions like AWS Shield, Azure DDoS Protection detect and respond to abnormal patterns.

    **🔒 Prevention Techniques:**  
    - **Rate-Limiting Middleware:** Tools like NGINX, HAProxy can apply IP-based request limits.  
    - **Anycast Networking:** Distribute services across geolocations to prevent single point overload.  
    - **BGP Flowspec / Null Routing:** Network-layer defense by diverting malicious traffic.
    """,

    "Worms": """
    ### 🦠 Worms

    **Worms** are self-replicating malware that spread autonomously across networks, exploiting vulnerabilities or weak credentials without human interaction.

    **🔍 Why it happens:**  
    - **Software Exploits:** Vulnerabilities in outdated protocols (e.g., SMBv1 exploited by EternalBlue) allow for rapid worm propagation.  
    - **Email & Phishing:** Embedded scripts or macros in documents (e.g., Emotet, Dridex) initiate worm-like spread.  
    - **Brute Forcing:** SSH/RDP brute force attacks used to spread across poorly secured servers.

    **⚠️ Risks:**  
    - **Botnet Formation:** Worms like Mirai enlist infected IoT devices for coordinated DDoS.  
    - **Destructive Payloads:** Wipers like NotPetya and Industroyer destroy data and render systems inoperable.  
    - **Rapid Outbreaks:** Worms spread faster than manual attacks, often impacting thousands of hosts within minutes.

    **🛡️ Mitigation Strategies:**  
    - **Lateral Movement Detection:** Use NDR tools like Darktrace or Corelight to detect unusual internal communication.  
    - **Quarantine Policies:** Isolate infected segments using NAC (Network Access Control) systems.  
    - **Patch Automation:** Continuously deploy security updates to close vulnerabilities before they’re exploited.

    **🔒 Prevention Techniques:**  
    - **Microsegmentation:** Break networks into security zones using tools like VMware NSX or Cisco TrustSec.  
    - **Strong Credentials:** Enforce long passwords and disable password-based authentication on critical systems.  
    - **Air-Gapping:** Physically isolate critical systems (ICS/SCADA) from the internet.
    """,

    "Generic": """
    ### 🧾 Generic

    **Generic** attacks refer to signature or heuristic-based detections of behavior that matches common malware traits or known bad patterns.

    **🔍 Why it happens:**  
    - **Signature Matching:** IDS/IPS tools (like Snort, Suricata) flag traffic based on predefined signatures (e.g., known malware hash).  
    - **Heuristic Engines:** Behavioral analysis detects anomalous patterns like repeated failed logins, memory injection, or registry manipulation.  
    - **Static Indicators:** Simple string matches or header anomalies often used in generic rules.

    **⚠️ Risks:**  
    - **Polymorphic Malware:** Modern threats mutate signatures to evade detection.  
    - **False Positives:** Legitimate tools (e.g., PsExec, Powershell scripts) may be flagged due to overlap in behavior.  
    - **Signature Obsolescence:** Older IDS rules may not cover evolving threats.

    **🛡️ Mitigation Strategies:**  
    - **Sandboxing:** Analyze suspicious files in isolated environments (e.g., Joe Sandbox, Any.Run).  
    - **Behavioral Profiling:** Use AI/ML-based endpoint tools like SentinelOne, Cybereason, or CrowdStrike to identify unknown threats.  
    - **Threat Intel Enrichment:** Integrate IOC feeds to contextualize alerts (VirusTotal, MISP).

    **🔒 Prevention Techniques:**  
    - **Defense-in-Depth:** Layered security including firewalls, EDR, NDR, and SIEM.  
    - **Adaptive Detection:** Leverage XDR (Extended Detection & Response) platforms for cross-domain analysis.  
    - **UEBA:** Detect abnormal behavior across users, endpoints, and network for early threat identification.
    """


}


# 🎨 Custom cyber-themed styling
st.markdown("""
    <style>
        body, .main {
            background-color: #0f172a;
            color: #e2e8f0;
        }
        h1, h2, h3, .st-emotion-cache-10trblm, .st-emotion-cache-1v0mbdj {
            color: #38bdf8;
        }
        .stButton>button {
            background-color: #38bdf8;
            color: black;
            font-weight: bold;
            border-radius: 8px;
            padding: 8px 16px;
        }
        .stNumberInput label, .stSelectbox label {
            color: #f8fafc;
            font-weight: bold;
        }
        .stTextInput>div>div>input, .stSelectbox>div>div>select {
            background-color: #1e293b;
            color: #f8fafc;
        }
    </style>
""", unsafe_allow_html=True)

# 🚀 Title
st.title("🔐 EnGuardia – Turning Packets into Patterns. Patterns into Protection.")
st.markdown("Input the session-level network features below to predict the cyberattack type:")

# 🧾 Input form
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

        # 🔎 Show attack info immediately
        st.markdown("## 🧠 Attack Insights")
        st.markdown(attack_info.get(predicted_class, "No description available."), unsafe_allow_html=True)
        # 📊 Visualizations
        st.subheader("📊 Visual Dashboard for Input Features")

        # Barplot
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        sns.barplot(x=list(X.columns), y=list(X.iloc[0]), palette="crest", ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_ylabel("Value")
        ax1.set_title("Feature Input Values")
        st.pyplot(fig1)

        # Pie chart for byte distribution
        fig2, ax2 = plt.subplots()
        ax2.pie([sbytes, dbytes], labels=['Source Bytes', 'Destination Bytes'],
                autopct='%1.1f%%', colors=['#38bdf8','#f59e0b'])
        ax2.set_title("Traffic Direction Volume")
        st.pyplot(fig2)

        # Histogram for mean packet sizes
        fig3, ax3 = plt.subplots()
        ax3.hist([[smean], [dmean]], bins=5, color=['#10b981','#ef4444'], label=['Source Mean', 'Destination Mean'])
        ax3.legend()
        ax3.set_title("Distribution of Mean Packet Sizes")
        ax3.set_xlabel("Packet Size")
        ax3.set_ylabel("Frequency")
        st.pyplot(fig3)

        # Radar plot for selected flow metrics
        radar_labels = ['rate', 'sload', 'dload', 'dinpkt']
        radar_values = [input_data[feat] for feat in radar_labels]
        radar_angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
        radar_values += radar_values[:1]
        radar_angles += radar_angles[:1]

        fig4, ax4 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax4.plot(radar_angles, radar_values, 'o-', linewidth=2, label='Flow Metrics', color='#6366f1')
        ax4.fill(radar_angles, radar_values, alpha=0.25, color='#6366f1')
        ax4.set_xticks(radar_angles[:-1])
        ax4.set_xticklabels(radar_labels)
        ax4.set_title("Flow Metrics Radar View")
        st.pyplot(fig4)



    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

