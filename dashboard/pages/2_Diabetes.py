"""
2_Diabetes.py — Diabetes Screening Dashboard Page

Provides individual prediction with risk factor breakdown,
recommendations, and dataset exploration.
"""

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from database.services import save_prediction
from modules.diabetes.explainability import (
    summary_plot,
    waterfall_plot
)
# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
sys.path.insert(0, PROJECT_ROOT)

from modules.diabetes.prediction import (
    predict, load_model, load_dataset, get_dataset_statistics,
    FEATURE_NAMES, NORMAL_RANGES, _classify_glucose, _classify_bmi, _classify_bp,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MedInsight AI — Diabetes Screening",
    page_icon="💉",
    layout="wide",
)

from dashboard.ui.theme import load_css
load_css()

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>

.result-positive 
.result-negative 
.result-moderate 

.risk-factor-card {
    background: rgba(30,27,75,0.4);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
}
.rf-high     { border-left: 4px solid #ef4444; }
.rf-moderate { border-left: 4px solid #f59e0b; }
.rf-low      { border-left: 4px solid #22c55e; }

.rec-card {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 12px;
    padding: 0.85rem 1.25rem;
    margin-bottom: 0.5rem;
    color: #a7f3d0;
    font-size: 0.9rem;
}

.metric-box {
    background: linear-gradient(145deg, #1e1b4b, #312e81);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 14px;
    padding: 1.25rem;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #34d399, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    color: #94a3b8;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.2rem;
}
.metric-status {
    font-size: 0.8rem;
    margin-top: 0.3rem;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    display: inline-block;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="page-header">
    <h1>💉 Diabetes Screening</h1>
    <p>Type 2 Diabetes risk assessment with personalised recommendations</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_explore = st.tabs([
    "🔬 Risk Assessment",
    "📊 Dataset Explorer",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Prediction
# ═══════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("#### Enter your clinical parameters")
    st.caption("Values from routine blood work / physical examination")
    patient_name = st.text_input(
    "Patient Name",
    placeholder="Enter patient name"
)
    col_l, col_r = st.columns(2)

    with col_l:
        pregnancies = st.number_input(
            "Number of Pregnancies", 0, 20, 0, key="d_preg",
            help="Number of times pregnant (0 for males or nulliparous)"
        )
        glucose = st.number_input(
            "Fasting Plasma Glucose (mg/dL)", 0, 300, 100, key="d_glucose",
            help="Normal: 70–99 | Pre-diabetes: 100–125 | Diabetes: ≥ 126"
        )
        bp = st.number_input(
            "Diastolic Blood Pressure (mm Hg)", 0, 150, 72, key="d_bp",
            help="Normal: 60–80"
        )
        skin = st.number_input(
            "Skin Fold Thickness (mm)", 0, 100, 20, key="d_skin",
            help="Triceps skin fold measurement"
        )

    with col_r:
        insulin = st.number_input(
            "2-Hour Serum Insulin (μU/mL)", 0, 900, 80, key="d_insulin",
            help="Normal: 16–166"
        )
        bmi = st.number_input(
            "BMI (kg/m²)", 0.0, 70.0, 25.0, 0.1, key="d_bmi",
            help="Normal: 18.5–24.9 | Overweight: 25–29.9 | Obese: ≥ 30"
        )
        dpf = st.number_input(
            "Diabetes Pedigree Function", 0.0, 3.0, 0.35, 0.01, key="d_dpf",
            help="Genetic predisposition score (higher → more family history)"
        )
        age = st.number_input(
            "Age (years)", 1, 120, 33, key="d_age"
        )

    # ── Quick metrics ─────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    glu_cat = _classify_glucose(glucose)
    bmi_cat = _classify_bmi(bmi)
    bp_cat = _classify_bp(bp)

    def _status_color(cat):
        if cat in ('Normal', 'Desirable'):
            return 'background:rgba(34,197,94,0.2); color:#4ade80;'
        elif cat in ('Pre-diabetes', 'Overweight', 'Elevated'):
            return 'background:rgba(234,179,8,0.2); color:#facc15;'
        else:
            return 'background:rgba(239,68,68,0.2); color:#f87171;'

    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{glucose}</div>
            <div class="metric-label">Glucose (mg/dL)</div>
            <div class="metric-status" style="{_status_color(glu_cat)}">{glu_cat}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{bmi}</div>
            <div class="metric-label">BMI (kg/m²)</div>
            <div class="metric-status" style="{_status_color(bmi_cat)}">{bmi_cat}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{bp}</div>
            <div class="metric-label">Blood Pressure (mm Hg)</div>
            <div class="metric-status" style="{_status_color(bp_cat)}">{bp_cat}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    if st.button("🔍  Assess Diabetes Risk", type="primary",
                  use_container_width=True, key="btn_diabetes"):
        user_input = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': bp,
            'SkinThickness': skin,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age,
        }

        with st.spinner("Analysing..."):
            try:
                result = predict(user_input)
            except FileNotFoundError as e:
                st.error(f"Model not found: {e}")
                st.stop()

        # ── Result card ───────────────────────────────────────────────
        risk = result['risk_level']
        css_class = (
            'result-positive' if risk == 'High'
            else 'result-moderate' if risk == 'Moderate'
            else 'result-negative'
        )
        verdict = ("Diabetes Risk Elevated"
                    if result['prediction'] == 1
                    else "Low Diabetes Risk")

        st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="result-title">{verdict}</div>
            <div style="color:#e2e8f0; font-size:0.95rem;">
                Probability: <b>{result['probability']:.1%}</b> &nbsp;|&nbsp;
                Risk Level: <b>{risk}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Gauge chart ───────────────────────────────────────────────
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['probability'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': 'Diabetes Probability'},
            number={'suffix': '', 'valueformat': '.1%'},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': '#ef4444' if result['prediction'] == 1 else '#22c55e'},
                'steps': [
                    {'range': [0, 0.3], 'color': 'rgba(34,197,94,0.2)'},
                    {'range': [0.3, 0.7], 'color': 'rgba(234,179,8,0.2)'},
                    {'range': [0.7, 1], 'color': 'rgba(239,68,68,0.2)'},
                ],
                'threshold': {
                    'line': {'color': '#ef4444', 'width': 4},
                    'thickness': 0.75,
                    'value': result['probability'],
                },
            },
        ))
        gauge.update_layout(height=300, margin=dict(t=50, b=0))
        st.plotly_chart(gauge, use_container_width=True)

        # ── Risk factors ──────────────────────────────────────────────
        risk_factors = result.get('risk_factors', [])
        if risk_factors:
            st.markdown("##### ⚠️ Identified Risk Factors")
            for rf in risk_factors:
                sev = rf.get('severity', 'low')
                css = f"rf-{sev}"
                icon = '🔴' if sev == 'high' else '🟡' if sev == 'moderate' else '🟢'
                st.markdown(f"""
                <div class="risk-factor-card {css}">
                    {icon} <b>{rf['factor']}</b>: {rf['value']} {rf['unit']}
                    &nbsp;—&nbsp; <em>{rf['category']}</em>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No major risk factors identified.")

        # ── Recommendations ───────────────────────────────────────────
        recs = result.get('recommendations', [])
        if recs:
            st.markdown("##### 💡 Personalised Recommendations")
            for i, rec in enumerate(recs, 1):
                st.markdown(f"""
                <div class="rec-card">
                    <b>{i}.</b> {rec}
                </div>
                """, unsafe_allow_html=True)

        # ── Full interpretation ───────────────────────────────────────
        with st.expander("📝 Full Interpretation", expanded=False):
            st.text(result['interpretation'])

        # ── Explainable AI ─────────────────────────────────────────────
        st.divider()
        st.subheader("🔍 Explainable AI")

        shap_tab1, shap_tab2 = st.tabs([
            "Feature Importance",
            "Waterfall Plot"
        ])

        with shap_tab1:
            st.pyplot(summary_plot(result['input_df']))

        with shap_tab2:
            st.pyplot(waterfall_plot(result['input_df']))

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Dataset Explorer
# ═══════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("#### 📊 Diabetes Dataset Explorer")

    try:
        df = load_dataset()
    except FileNotFoundError:
        st.warning("Dataset not found. Place `diabetes.csv` in "
                    "`modules/diabetes/data/`.")
        st.stop()

    st.caption(f"Pima Indians Diabetes Dataset: {df.shape[0]} records × {df.shape[1]} columns")

    # ── Outcome distribution ──────────────────────────────────────────
    e1, e2 = st.columns(2)

    with e1:
        dist = df['Outcome'].value_counts().reset_index()
        dist.columns = ['Status', 'Count']
        dist['Status'] = dist['Status'].map({1: 'Diabetic', 0: 'Non-Diabetic'})
        fig_pie = px.pie(dist, names='Status', values='Count',
                         title='Outcome Distribution',
                         color_discrete_sequence=['#f87171', '#34d399'])
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with e2:
        fig_glucose = px.histogram(
            df, x='Glucose', color=df['Outcome'].map({1: 'Diabetic', 0: 'Non-Diabetic'}),
            marginal='box', nbins=30,
            title='Glucose Distribution by Outcome',
            color_discrete_sequence=['#34d399', '#f87171'],
        )
        fig_glucose.update_layout(height=350)
        st.plotly_chart(fig_glucose, use_container_width=True)

    # ── Feature selector ──────────────────────────────────────────────
    selected = st.selectbox("Explore feature distribution",
                            FEATURE_NAMES, key="d_explore_feat")
    fig_feat = px.histogram(
        df, x=selected,
        color=df['Outcome'].map({1: 'Diabetic', 0: 'Non-Diabetic'}),
        marginal='box', nbins=30,
        title=f'{selected} Distribution by Outcome',
        color_discrete_sequence=['#34d399', '#f87171'],
    )
    fig_feat.update_layout(height=400)
    st.plotly_chart(fig_feat, use_container_width=True)

    # ── Correlation with outcome ──────────────────────────────────────
    with st.expander("📈 Feature Correlation with Diabetes", expanded=False):
        corr = df[FEATURE_NAMES].corrwith(df['Outcome']).sort_values(ascending=True)
        fig_corr = px.bar(
            x=corr.values, y=corr.index, orientation='h',
            title='Correlation of Features with Diabetes Outcome',
            labels={'x': 'Correlation', 'y': 'Feature'},
            color=corr.values,
            color_continuous_scale='RdYlGn_r',
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────
    with st.expander("🔥 Full Correlation Heatmap", expanded=False):
        full_corr = df[FEATURE_NAMES + ['Outcome']].corr().round(2)
        fig_heat = px.imshow(full_corr, text_auto=True, aspect='auto',
                             color_continuous_scale='RdBu_r',
                             title='Correlation Heatmap')
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Raw data ──────────────────────────────────────────────────────
    with st.expander("📄 Raw Data", expanded=False):
        st.dataframe(df, use_container_width=True, height=400)

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-small">
    <p>⚠️ <b>Disclaimer:</b> This tool is for screening purposes only.
    Consult a healthcare professional for HbA1c testing, diagnosis, and treatment.</p>
</div>
""", unsafe_allow_html=True)
