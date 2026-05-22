import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnLens · Prediction Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark luxury analytics aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root & Base ── */
:root {
    --bg-base:       #0a0d14;
    --bg-card:       #111520;
    --bg-card-hover: #161b2e;
    --border:        #1e2540;
    --border-light:  #2a3050;
    --teal:          #00d4b8;
    --teal-dim:      #00a896;
    --amber:         #ffb347;
    --rose:          #ff4d6d;
    --text-primary:  #e8ecf4;
    --text-muted:    #8892a4;
    --text-dim:      #4a5568;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-base);
    color: var(--text-primary);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer {
    visibility: hidden;
}
.stDeployButton { display: none; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1018 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Hero header ── */
.hero-wrap {
    padding: 2.5rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.4rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.1;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}
.hero-title span { color: var(--teal); }
.hero-sub {
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
    font-weight: 300;
}

/* ── Section label ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin: 1.8rem 0 0.8rem;
    border-left: 2px solid var(--teal);
    padding-left: 0.6rem;
}

/* ── Cards ── */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.info-card:hover { border-color: var(--border-light); }

/* ── Result card ── */
.result-card {
    border-radius: 16px;
    padding: 2rem 2.2rem;
    border: 1px solid;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-card.churn {
    background: linear-gradient(135deg, #1a0a0e 0%, #1f0d13 100%);
    border-color: #4a1020;
}
.result-card.stay {
    background: linear-gradient(135deg, #031a18 0%, #041f1c 100%);
    border-color: #0a3530;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
}
.result-card.churn::before { background: var(--rose); }
.result-card.stay::before  { background: var(--teal); }

.result-verdict {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.result-card.churn .result-verdict { color: var(--rose); }
.result-card.stay  .result-verdict { color: var(--teal); }

.result-desc {
    font-size: 0.875rem;
    color: var(--text-muted);
    font-weight: 300;
}

/* ── Risk pill ── */
.risk-pill {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.25rem 0.75rem;
    border-radius: 100px;
    margin-top: 0.8rem;
    font-weight: 500;
}
.risk-high   { background: rgba(255,77,109,0.15); color: var(--rose);  border: 1px solid rgba(255,77,109,0.3); }
.risk-medium { background: rgba(255,179,71,0.12); color: var(--amber); border: 1px solid rgba(255,179,71,0.3); }
.risk-low    { background: rgba(0,212,184,0.1);   color: var(--teal);  border: 1px solid rgba(0,212,184,0.25); }

/* ── Metric tiles ── */
.metric-tile {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--teal);
}
.metric-lbl {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* ── Action box ── */
.action-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
}
.action-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
    color: var(--text-primary);
}
.action-item {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
    color: var(--text-muted);
}
.action-item:last-child { border-bottom: none; }
.action-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}

/* ── Streamlit widget overrides ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider { 
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--teal), var(--teal-dim)) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    margin-top: 1rem !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

div[data-testid="stHorizontalBlock"] > div { gap: 1rem; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL (graceful fallback for demo)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model   = joblib.load('models/churn_model.pkl')
        columns = joblib.load('models/model_columns.pkl')
        return model, columns
    except Exception:
        return None, None

model, model_columns = load_model()


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-label">Customer Intelligence Platform</div>
    <div class="hero-title">Churn<span>Lens</span></div>
    <div class="hero-sub">Real-time churn risk scoring · ML-powered retention signals</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR — all inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;
                color:#e8ecf4;padding:1rem 0 0.5rem;border-bottom:1px solid #1e2540;
                margin-bottom:1rem;">
        ⚙️ Customer Profile
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
    gender      = st.selectbox("Gender", ["Male", "Female"])
    senior      = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner     = st.selectbox("Has Partner", ["Yes", "No"])
    dependents  = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure      = st.slider("Tenure (months)", 0, 72, 12)

    st.markdown('<div class="section-label">Services</div>', unsafe_allow_html=True)
    phone_service    = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines   = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security  = st.selectbox("Online Security",  ["Yes", "No", "No internet service"])
    online_backup    = st.selectbox("Online Backup",    ["Yes", "No", "No internet service"])
    device_protection= st.selectbox("Device Protection",["Yes", "No", "No internet service"])
    tech_support     = st.selectbox("Tech Support",     ["Yes", "No", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV",     ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.markdown('<div class="section-label">Billing</div>', unsafe_allow_html=True)
    contract         = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless        = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method   = st.selectbox("Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges  = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0, step=0.5)
    total_charges    = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges), step=1.0)

    predict_btn = st.button("🔮 Analyze Churn Risk")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def build_gauge(probability):
    color = "#ff4d6d" if probability > 0.7 else "#ffb347" if probability > 0.4 else "#00d4b8"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color, "family": "Syne"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a5568",
                     "tickfont": {"color": "#4a5568", "size": 10}},
            "bar":  {"color": color, "thickness": 0.22},
            "bgcolor": "#161b2e",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40],  "color": "#0d1018"},
                {"range": [40, 70],  "color": "#0d1018"},
                {"range": [70, 100], "color": "#0d1018"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": probability * 100
            }
        },
        title={"text": "CHURN PROBABILITY", "font": {"size": 11, "color": "#4a5568", "family": "DM Mono"}},
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#111520",
        font_color="#e8ecf4",
    )
    return fig


def build_radar(input_dict):
    """Simple risk-factor radar based on known churn signals."""
    factors = {
        "Month-to-Month": 1 if input_dict["Contract"] == "Month-to-month" else 0,
        "Fiber Optic":    1 if input_dict["InternetService"] == "Fiber optic" else 0,
        "No Security":    1 if input_dict["OnlineSecurity"] == "No" else 0,
        "E-Check Pay":    1 if input_dict["PaymentMethod"] == "Electronic check" else 0,
        "Short Tenure":   max(0, (24 - input_dict["tenure"]) / 24),
        "High Charges":   min(1, (input_dict["MonthlyCharges"] - 20) / 80),
        "No Support":     1 if input_dict["TechSupport"] == "No" else 0,
    }
    cats   = list(factors.keys())
    vals   = list(factors.values())
    vals  += [vals[0]]
    cats  += [cats[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill='toself',
        fillcolor='rgba(255,77,109,0.12)',
        line=dict(color='#ff4d6d', width=2),
        marker=dict(size=5, color='#ff4d6d'),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0d1018",
            radialaxis=dict(visible=True, range=[0,1], showticklabels=False,
                            gridcolor="#1e2540", linecolor="#1e2540"),
            angularaxis=dict(gridcolor="#1e2540", linecolor="#1e2540",
                             tickfont=dict(size=10, color="#8892a4")),
        ),
        paper_bgcolor="#111520",
        plot_bgcolor="#111520",
        showlegend=False,
        margin=dict(t=30, b=20, l=30, r=30),
        height=280,
    )
    return fig


def build_tenure_bar(tenure_val, monthly_val):
    """Contextual bar: customer's tenure vs typical churn cohorts."""
    cohorts = ["0–6 mo", "7–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"]
    churn_rates = [0.62, 0.48, 0.32, 0.19, 0.11]
    colors = ["#ff4d6d", "#ff7a54", "#ffb347", "#7ecba1", "#00d4b8"]

    bucket_idx = 0
    if tenure_val <= 6:    bucket_idx = 0
    elif tenure_val <= 12: bucket_idx = 1
    elif tenure_val <= 24: bucket_idx = 2
    elif tenure_val <= 48: bucket_idx = 3
    else:                  bucket_idx = 4

    bar_colors = [c if i != bucket_idx else "#ffffff" for i, c in enumerate(colors)]

    fig = go.Figure(go.Bar(
        x=cohorts,
        y=[r * 100 for r in churn_rates],
        marker_color=bar_colors,
        marker_line_width=0,
        text=[f"{r*100:.0f}%" for r in churn_rates],
        textposition="outside",
        textfont=dict(size=10, color="#8892a4"),
    ))
    fig.update_layout(
    height=220,
    margin=dict(t=20, b=10, l=10, r=10),
    paper_bgcolor="#111520",
    plot_bgcolor="#111520",
    xaxis=dict(
        gridcolor="#1e2540",
        linecolor="#1e2540",
        tickfont=dict(size=10, color="#8892a4")
    ),
    yaxis=dict(
        gridcolor="#1e2540",
        linecolor="#1e2540",
        tickfont=dict(size=10, color="#4a5568"),
        title=dict(
            text="Avg Churn %",
            font=dict(size=9, color="#4a5568")
        )
    ),
    showlegend=False,
)
    
    fig.add_annotation(x=cohorts[bucket_idx], y=churn_rates[bucket_idx]*100 + 7,
                       text="▲ You", showarrow=False,
                       font=dict(size=10, color="#ffffff"))
    return fig


# ─────────────────────────────────────────────
# DEFAULT DASHBOARD STATE
# ─────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-tile">
            <div class="metric-val">{tenure}</div>
            <div class="metric-lbl">Tenure (mo)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-tile">
            <div class="metric-val">${monthly_charges:.0f}</div>
            <div class="metric-lbl">Monthly</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        contract_short = {"Month-to-month": "MoM", "One year": "1 yr", "Two year": "2 yr"}[contract]
        st.markdown(f"""<div class="metric-tile">
            <div class="metric-val">{contract_short}</div>
            <div class="metric-lbl">Contract</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────
if predict_btn:
    input_dict = {
        'gender': gender, 'SeniorCitizen': senior,
        'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone_service,
        'MultipleLines': multiple_lines, 'InternetService': internet_service,
        'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
        'DeviceProtection': device_protection, 'TechSupport': tech_support,
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': paperless,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }

    if model is not None:
        input_df      = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)
        for col in model_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_columns]
        prediction    = model.predict(input_encoded)[0]
        probability   = model.predict_proba(input_encoded)[0][1]
    else:
        # Demo fallback — simulate a probability
        risk_score = (
            (0.3 if contract == "Month-to-month" else 0) +
            (0.2 if internet_service == "Fiber optic" else 0) +
            (0.15 if payment_method == "Electronic check" else 0) +
            (0.1 if online_security == "No" else 0) +
            max(0, (30 - tenure) / 100) +
            (monthly_charges / 500)
        )
        probability = min(0.97, max(0.03, risk_score))
        prediction  = 1 if probability >= 0.5 else 0

    if probability > 0.7:
        risk = "High Risk"
        risk_class = "risk-high"
    elif probability > 0.4:
        risk = "Medium Risk"
        risk_class = "risk-medium"
    else:
        risk = "Low Risk"
        risk_class = "risk-low"

    # ── Left column: verdict + gauge + tenure chart ──
    with col_left:
        is_churn = prediction == 1
        card_cls = "churn" if is_churn else "stay"
        icon     = "⚠️" if is_churn else "✅"
        verdict  = "Likely to Churn" if is_churn else "Likely to Stay"
        desc     = "This customer shows elevated churn signals. Intervention recommended." \
                   if is_churn else "Low departure signals detected. Focus on loyalty & upsell."

        st.markdown(f"""
        <div class="result-card {card_cls}">
            <div style="font-size:1.6rem;margin-bottom:0.4rem">{icon}</div>
            <div class="result-verdict">{verdict}</div>
            <div class="result-desc">{desc}</div>
            <div class="risk-pill {risk_class}">{risk}</div>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(build_gauge(probability), use_container_width=True)

        st.markdown('<div class="section-label">Tenure vs Churn Cohorts</div>', unsafe_allow_html=True)
        st.plotly_chart(build_tenure_bar(tenure, monthly_charges), use_container_width=True)

    # ── Right column: radar + actions ──
    with col_right:
        st.markdown('<div class="section-label">Risk Factor Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(build_radar(input_dict), use_container_width=True)

        st.markdown('<div class="section-label">Recommended Actions</div>', unsafe_allow_html=True)

        if risk == "High Risk":
            actions = [
                ("🎯", "Personal outreach within 48 hours", "#ff4d6d"),
                ("💰", "Offer 20–30% discount on next 3 months", "#ff4d6d"),
                ("📋", "Propose contract upgrade incentive", "#ff4d6d"),
                ("🛡️", "Assign dedicated account manager", "#ff7a54"),
            ]
        elif risk == "Medium Risk":
            actions = [
                ("📬", "Targeted email campaign with service highlights", "#ffb347"),
                ("⭐", "Enrol in loyalty rewards programme", "#ffb347"),
                ("📞", "Proactive service quality check-in", "#ffb347"),
            ]
        else:
            actions = [
                ("🎁", "Reward with loyalty tier upgrade", "#00d4b8"),
                ("📈", "Present upsell for premium services", "#00d4b8"),
                ("📰", "Regular value-add newsletter", "#00d4b8"),
            ]

        action_html = '<div class="action-box"><div class="action-title">Action Plan</div>'
        for emoji, text, dot_col in actions:
            action_html += f"""
            <div class="action-item">
                <div class="action-dot" style="background:{dot_col}"></div>
                <span>{emoji} {text}</span>
            </div>"""
        action_html += '</div>'
        st.markdown(action_html, unsafe_allow_html=True)

        # ── Key signal summary ──
        st.markdown('<div class="section-label">Key Signals Detected</div>', unsafe_allow_html=True)
        signals = []
        if contract == "Month-to-month":      signals.append(("↑ No long-term commitment", "rose"))
        if internet_service == "Fiber optic": signals.append(("↑ High-cost internet tier",  "rose"))
        if online_security == "No":           signals.append(("↑ No security add-on",       "amber"))
        if payment_method == "Electronic check": signals.append(("↑ E-check payment",       "amber"))
        if tenure < 12:                       signals.append(("↑ New customer (< 1 yr)",    "amber"))
        if tech_support == "No":              signals.append(("↑ No tech support",          "amber"))
        if partner == "Yes":                  signals.append(("↓ Has partner (retention+)", "teal"))
        if contract == "Two year":            signals.append(("↓ Long-term contract",       "teal"))

        color_map = {"rose": "#ff4d6d", "amber": "#ffb347", "teal": "#00d4b8"}
        for label, col_key in signals[:6]:
            col = color_map[col_key]
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.5rem;padding:0.35rem 0;
                        border-bottom:1px solid #1e2540;font-size:0.82rem;color:#8892a4;">
                <div style="width:8px;height:8px;border-radius:50%;background:{col};flex-shrink:0"></div>
                {label}
            </div>""", unsafe_allow_html=True)

else:
    # Placeholder state when no prediction yet
    with col_left:
        st.markdown("""
        <div class="info-card" style="text-align:center;padding:3rem 2rem;">
            <div style="font-size:3rem;margin-bottom:1rem">🔮</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                        color:#e8ecf4;margin-bottom:0.5rem;">Ready to Analyze</div>
            <div style="font-size:0.85rem;color:#4a5568;">
                Fill in the customer profile in the sidebar,<br>
                then click <strong style="color:#00d4b8">Analyze Churn Risk</strong> to generate predictions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Quick-reference churn stats card
        st.markdown('<div class="section-label">Industry Benchmarks</div>', unsafe_allow_html=True)

        benchmarks = [
            ("Avg telecom churn rate",       "~21% / yr"),
            ("Month-to-month contract churn","~42%"),
            ("Fiber optic churn rate",        "~30%"),
            ("Customers retained after yr 2", "~85%"),
        ]
        for label, val in benchmarks:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:0.7rem 0;border-bottom:1px solid #1e2540;font-size:0.85rem;">
                <span style="color:#8892a4">{label}</span>
                <span style="font-family:'DM Mono',monospace;color:#00d4b8;font-size:0.8rem">{val}</span>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1.5rem;border-top:1px solid #1e2540;
            display:flex;justify-content:space-between;align-items:center;
            font-family:'DM Mono',monospace;font-size:0.65rem;color:#2a3050;">
    <span>CHURNLENS · ML PREDICTION ENGINE</span>
    <span>MODEL: RANDOM FOREST · TELCO DATASET</span>
</div>
""", unsafe_allow_html=True)