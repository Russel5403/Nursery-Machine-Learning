import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model & encoders
base_dir = os.path.dirname(os.path.abspath(__file__))
model    = joblib.load(os.path.join(base_dir, 'nursery_rf_model.pkl'))
encoders = joblib.load(os.path.join(base_dir, 'nursery_encoders.pkl'))

# Prescriptive table
PRESCRIPTIVE = {
    "not_recom":  {
        "priority": "NOT RECOMMENDED",
        "color": "#c0392b",
        "bg": "#fdecea",
        "icon": "❌",
        "action": "Reject application. Send official rejection letter with explanation.",
        "justification": "Application does not meet minimum criteria for nursery admission."
    },
    "recommend":  {
        "priority": "FULLY RECOMMENDED",
        "color": "#1a5276",
        "bg": "#eaf0fb",
        "icon": "⭐",
        "action": "Immediate acceptance. Fast-track and assign dedicated case officer.",
        "justification": "Exceptionally rare qualifying case. Warrants highest-level attention."
    },
    "very_recom": {
        "priority": "HIGHLY RECOMMENDED",
        "color": "#b7770d",
        "bg": "#fef9e7",
        "icon": "🌟",
        "action": "Strong acceptance. Expedited processing. Notify family within 48 hrs.",
        "justification": "High-priority application with strong indicators across all criteria."
    },
    "priority":   {
        "priority": "STANDARD PRIORITY",
        "color": "#1e8449",
        "bg": "#eafaf1",
        "icon": "✅",
        "action": "Accept. Place in standard admission queue. Provide timeline to family.",
        "justification": "Meets standard criteria. Normal processing appropriate."
    },
    "spec_prior": {
        "priority": "SPECIAL PRIORITY",
        "color": "#1a7a4a",
        "bg": "#e8f8f5",
        "icon": "📋",
        "action": "Accept with special review. Admissions committee evaluation for tailored support.",
        "justification": "Special circumstances identified. May require additional resources or accommodations."
    },
}

# Feature options
OPTIONS = {
    "parents":  ["Usual", "Pretentious", "Great Pretentious"],
    "has_nurs": ["Proper", "Less Proper", "Improper", "Critical", "Very Critical"],
    "form":     ["Complete", "Completed", "Incomplete", "Foster"],
    "children": ["1", "2", "3", "More"],
    "housing":  ["Convenient", "Less Convenient", "Critical"],
    "finance":  ["Convenient", "Inconvenient"],
    "social":   ["Non Problematic", "Slightly Problematic", "Problematic"],
    "health":   ["Recommended", "Priority", "Not Recommended"],
}

# Page config
st.set_page_config(page_title="Nursery Admission Advisor", page_icon="🏫", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

/* Page background */
.stApp {
    background: linear-gradient(135deg, #eaf2fb 0%, #fdfefe 50%, #eafaf1 100%);
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #1a3a5c 0%, #2471a3 60%, #1a7a4a 100%);
    border-radius: 16px;
    padding: 36px 40px 28px 40px;
    margin-bottom: 28px;
    color: white;
    box-shadow: 0 8px 32px rgba(36,113,163,0.18);
}
.header-banner h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    margin: 0 0 8px 0;
    letter-spacing: 0.5px;
}
.header-banner p {
    font-size: 1.05rem;
    opacity: 0.88;
    margin: 0;
    font-weight: 300;
}

/* Section cards */
.section-card {
    background: white;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 2px 16px rgba(36,113,163,0.08);
    border-left: 4px solid #2471a3;
}
.section-card.green {
    border-left-color: #1a7a4a;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: #1a3a5c;
    font-weight: 600;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Result box */
.result-banner {
    border-radius: 14px;
    padding: 24px 28px;
    margin: 20px 0 12px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
}
.result-class {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 4px;
}
.result-priority {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.75;
}

/* Info rows */
.info-row {
    background: #f8fbff;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    border: 1px solid #d6eaf8;
}
.info-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #2471a3;
    margin-bottom: 4px;
}
.info-value {
    font-size: 1rem;
    color: #1a3a5c;
}

/* Streamlit button override */
div.stButton > button {
    background: linear-gradient(135deg, #1a3a5c, #2471a3);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Source Sans 3', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 14px 0;
    width: 100%;
    transition: all 0.2s ease;
    box-shadow: 0 4px 14px rgba(36,113,163,0.25);
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #2471a3, #1a7a4a);
    box-shadow: 0 6px 20px rgba(36,113,163,0.35);
    transform: translateY(-1px);
}

/* Expander styling */
.streamlit-expanderHeader {
    background: white !important;
    border-radius: 12px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: #1a3a5c !important;
    border: 2px solid #d6eaf8 !important;
    padding: 14px 20px !important;
}
.streamlit-expanderContent {
    background: white !important;
    border: 2px solid #d6eaf8 !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 20px !important;
}

/* Selectbox label */
label[data-testid="stWidgetLabel"] p {
    font-weight: 600 !important;
    color: #1a3a5c !important;
    font-size: 0.9rem !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #7f8c8d;
    font-size: 0.82rem;
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid #d5d8dc;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="header-banner">
    <h1>🏫 Nursery School Admission Advisor</h1>
    <p>A machine learning–powered decision support tool for nursery school admissions officers.<br>
    Fill in the applicant's family profile below to receive an AI-generated admission recommendation.</p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="section-card">
    <div class="section-title">📌 How to Use</div>
    <p style="margin:6px 0 0 0; color:#555; font-size:0.95rem;">
        Expand the <strong>Applicant Profile Form</strong> below, select the appropriate values for each field,
        then click <strong>Predict Admission</strong> to receive a recommendation.
    </p>
</div>
""", unsafe_allow_html=True)


with st.expander("📝  Applicant Profile Form — Click to expand", expanded=False):
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**👨‍👩‍👧 Family Information**")
        parents  = st.selectbox("Parents Status",     OPTIONS["parents"],  key="parents")
        form     = st.selectbox("Family Form",         OPTIONS["form"],     key="form")
        children = st.selectbox("Number of Children", OPTIONS["children"], key="children")
        finance  = st.selectbox("Financial Standing", OPTIONS["finance"],  key="finance")

    with col2:
        st.markdown("**🏠 Living & Social Conditions**")
        housing  = st.selectbox("Housing Conditions", OPTIONS["housing"],  key="housing")
        social   = st.selectbox("Social Conditions",  OPTIONS["social"],   key="social")
        has_nurs = st.selectbox("Nursery Quality",    OPTIONS["has_nurs"], key="has_nurs")
        health   = st.selectbox("Health Conditions",  OPTIONS["health"],   key="health")

    st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


predict = st.button("🔍  Predict Admission", use_container_width=True)

if predict:
    input_data = {
        "parents":  parents,
        "has_nurs": has_nurs,
        "form":     form,
        "children": children,
        "housing":  housing,
        "finance":  finance,
        "social":   social,
        "health":   health,
    }

    encoded = []
    for col, val in input_data.items():
        le = encoders[col]
        encoded.append(le.transform([val])[0])

    X_input = np.array(encoded).reshape(1, -1)
    prediction_encoded = model.predict(X_input)[0]
    predicted_class = encoders['class'].inverse_transform([prediction_encoded])[0]

    info = PRESCRIPTIVE[predicted_class]


    st.markdown(f"""
    <div class="result-banner" style="background:{info['bg']}; border-left: 5px solid {info['color']};">
        <div class="result-priority" style="color:{info['color']};">{info['icon']} Admission Decision</div>
        <div class="result-class" style="color:{info['color']};">{info['priority']}</div>
        <div style="font-size:0.95rem; color:#555; margin-top:4px;">Predicted class: <strong>{predicted_class}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    # Details
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="info-row">
            <div class="info-label">📋 Recommended Action</div>
            <div class="info-value">{info['action']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="info-row">
            <div class="info-label">💡 Justification</div>
            <div class="info-value">{info['justification']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Input summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-card green">
        <div class="section-title">🗂 Applicant Profile Summary</div>
    </div>
    """, unsafe_allow_html=True)

    s_col1, s_col2 = st.columns(2)
    items = list(input_data.items())
    labels = {
        "parents": "Parents Status", "has_nurs": "Nursery Quality",
        "form": "Family Form", "children": "No. of Children",
        "housing": "Housing", "finance": "Finance",
        "social": "Social Conditions", "health": "Health"
    }
    for i, (k, v) in enumerate(items):
        with (s_col1 if i % 2 == 0 else s_col2):
            st.markdown(f"""
            <div style="background:#f4f6f7; border-radius:8px; padding:10px 14px; margin:5px 0;">
                <span style="font-size:0.75rem; font-weight:700; color:#2471a3; text-transform:uppercase; letter-spacing:1px;">{labels[k]}</span><br>
                <span style="font-size:0.97rem; color:#1a3a5c; font-weight:600;">{v}</span>
            </div>
            """, unsafe_allow_html=True)


st.markdown("""
<div class="footer">
    🏫 Nursery Admission Advisor &nbsp;|&nbsp; Powered by Random Forest &nbsp;|&nbsp; CMU Machine Learning Project
</div>
""", unsafe_allow_html=True)