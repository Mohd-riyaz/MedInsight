"""
1_Anemia.py — Anemia Screening Dashboard Page

Provides two prediction modes (blood-test & symptom-based), parameter
analysis, and dataset exploration.
"""

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from modules.anemia.explainability import (
    summary_plot,
    waterfall_plot
)
# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
ANEMIA_MODULE = os.path.join(PROJECT_ROOT, "modules", "anemia")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, ANEMIA_MODULE)

from modules.anemia.predictor import (
    predict_blood, predict_symptoms, predict_combined,
    load_blood_model, load_symptom_model, load_dataset,
    BLOOD_FEATURES, SYMPTOM_FEATURES,
)
from modules.anemia.config.feature_config import FEATURES, NORMAL_RANGES
from modules.anemia.visualization import create_prediction_gauge, create_feature_comparison_radar

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MedInsight AI — Anemia Screening",
    page_icon="🩸",
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

.param-card {
    background: rgba(30,27,75,0.4);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}
.param-normal  { border-left: 4px solid #22c55e; }
.param-abnormal { border-left: 4px solid #ef4444; }

.info-chip {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.8rem;
    color: #a5b4fc;
    margin: 0.15rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="page-header">
    <h1>🩸 Anemia Screening</h1>
    <p>Blood-test analysis &amp; symptom-based risk assessment</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_blood, tab_symptom, tab_explore = st.tabs([
    "🔬 Blood Test Analysis",
    "📋 Symptom Questionnaire",
    "📊 Dataset Explorer",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Blood Test
# ═══════════════════════════════════════════════════════════════════════════
with tab_blood:
    st.markdown("#### Enter your blood test parameters")
    st.caption("Values from a Complete Blood Count (CBC) report")

    col_left, col_right = st.columns(2)

    with col_left:
        gender = st.selectbox("Gender", ["Male", "Female"], key="blood_gender")
        age = st.number_input("Age (years)", 1, 100, 35, key="blood_age")
        hb = st.number_input(
            "Haemoglobin — Hb (g/dL)", 5.0, 20.0, 12.0, 0.1, key="blood_hb",
            help="Normal: F 12.0–15.5 | M 13.5–17.5"
        )
        rbc = st.number_input(
            "Red Blood Cells — RBC (million/μL)", 1.0, 8.0, 4.5, 0.1,
            key="blood_rbc",
            help="Normal: F 4.0–5.2 | M 4.5–5.9"
        )

    with col_right:
        pcv = st.number_input(
            "Packed Cell Volume — PCV (%)", 10.0, 60.0, 35.0, 0.5,
            key="blood_pcv",
            help="Normal: F 37–47 | M 40–52"
        )
        mcv = st.number_input(
            "Mean Corpuscular Volume — MCV (fL)", 20.0, 120.0, 85.0, 0.5,
            key="blood_mcv", help="Normal: 80–100"
        )
        mch = st.number_input(
            "Mean Corpuscular Hb — MCH (pg)", 10.0, 40.0, 28.0, 0.5,
            key="blood_mch", help="Normal: 27–33"
        )
        mchc = st.number_input(
            "MCHC (g/dL)", 25.0, 40.0, 33.0, 0.5,
            key="blood_mchc", help="Normal: 32–36"
        )

    if st.button("🔍  Analyse Blood Test", type="primary", use_container_width=True,
                  key="btn_blood"):
        user_input = {
            'Gender': gender,
            'Age': age,
            'Hb': hb,
            'RBC': rbc,
            'PCV': pcv,
            'MCV': mcv,
            'MCH': mch,
            'MCHC': mchc,
        }

        with st.spinner("Analysing..."):
            try:
                result = predict_blood(user_input)
            except FileNotFoundError as e:
                st.error(f"Model not found: {e}")
                st.stop()
        input_df = result["input_df"]
        # ── Result card ────────────────────────────────────────────────
        risk = result['risk_level']
        css_class = (
            'result-positive' if risk == 'High'
            else 'result-moderate' if risk == 'Moderate'
            else 'result-negative'
        )
        verdict = "Anemia Likely" if result['prediction'] == 1 else "Anemia Unlikely"

        st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="result-title">{verdict}</div>
            <div style="color:#e2e8f0; font-size:0.95rem;">
                Probability: <b>{result['probability']:.1%}</b> &nbsp;|&nbsp;
                Risk Level: <b>{risk}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Gauge + Radar ─────────────────────────────────────────────
        g1, g2 = st.columns(2)
        with g1:
            gauge_fig = create_prediction_gauge(
                result['probability'], result['prediction']
            )
            gauge_fig.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(gauge_fig, use_container_width=True)

        with g2:
            radar_input = dict(user_input)
            radar_input['Gender_Encoded'] = 1 if gender == 'Female' else 0
            radar_fig = create_feature_comparison_radar(radar_input, NORMAL_RANGES)
            radar_fig.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(radar_fig, use_container_width=True)

        # ── Parameter breakdown ───────────────────────────────────────
        st.markdown("##### 📋 Parameter Analysis")
        for p in result.get('parameter_analysis', []):
            css = 'param-normal' if p['status'] == 'Normal' else 'param-abnormal'
            icon = '✅' if p['status'] == 'Normal' else '⚠️'
            st.markdown(f"""
            <div class="param-card {css}">
                {icon} <b>{p['parameter']}</b>: {p['value']} {p['unit']}
                &nbsp;—&nbsp; {p['status']}
                <span style="color:#94a3b8; font-size:0.85rem;">
                    (Normal range: {p['normal_range']} {p['unit']})
                </span>
            </div>
            """, unsafe_allow_html=True)

        # ── Interpretation ────────────────────────────────────────────
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
            st.pyplot(summary_plot(input_df))

        with shap_tab2:
            st.pyplot(waterfall_plot(input_df))

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Symptom Questionnaire
# ═══════════════════════════════════════════════════════════════════════════
with tab_symptom:
    st.markdown("#### Symptom & Risk Factor Questionnaire")
    st.caption("Select all symptoms you are currently experiencing")

    s_col1, s_col2 = st.columns(2)

    with s_col1:
        sym_gender = st.selectbox("Gender", ["Male", "Female"], key="sym_gender")
        sym_age = st.number_input("Age (years)", 1, 100, 35, key="sym_age")
        st.markdown("**Core Symptoms**")
        fatigue = st.checkbox("Persistent fatigue / tiredness", key="sym_fatigue")
        breath = st.checkbox("Shortness of breath", key="sym_breath")
        dizzy = st.checkbox("Dizziness / lightheadedness", key="sym_dizzy")
        pale = st.checkbox("Pale skin / pallor", key="sym_pale")
        heart = st.checkbox("Heart racing / palpitations", key="sym_heart")
        headache = st.checkbox("Frequent headaches", key="sym_headache")
        cold = st.checkbox("Cold hands and feet", key="sym_cold")

    with s_col2:
        st.markdown("**Additional Symptoms**")
        nails = st.checkbox("Brittle nails", key="sym_nails")
        conc = st.checkbox("Poor concentration", key="sym_conc")
        st.markdown("**Risk Factors**")
        periods = st.checkbox("Heavy menstrual periods", key="sym_periods",
                              disabled=(sym_gender == "Male"))
        blood_loss = st.checkbox("Recent significant blood loss", key="sym_blood")
        veg = st.checkbox("Vegetarian / low-iron diet", key="sym_veg")
        gi = st.checkbox("GI disorders (celiac, Crohn's, etc.)", key="sym_gi")
        chronic = st.checkbox("Chronic disease (kidney, liver, etc.)", key="sym_chronic")

    if st.button("🔍  Assess Symptom Risk", type="primary",
                  use_container_width=True, key="btn_symptom"):
        sym_input = {
            'Gender': 1 if sym_gender == 'Female' else 0,
            'Age': sym_age,
            'Fatigue': int(fatigue),
            'Shortness_of_Breath': int(breath),
            'Dizziness': int(dizzy),
            'Pale_Skin': int(pale),
            'Heart_Racing': int(heart),
            'Headaches': int(headache),
            'Cold_Hands_Feet': int(cold),
            'Brittle_Nails': int(nails),
            'Poor_Concentration': int(conc),
            'Heavy_Periods': int(periods) if sym_gender == 'Female' else 0,
            'Recent_Blood_Loss': int(blood_loss),
            'Vegetarian_Diet': int(veg),
            'GI_Disorders': int(gi),
            'Chronic_Disease': int(chronic),
        }

        with st.spinner("Analysing symptoms..."):
            try:
                result = predict_symptoms(sym_input)
            except FileNotFoundError as e:
                st.error(f"Model not found: {e}")
                st.stop()

        risk = result['risk_level']
        css_class = (
            'result-positive' if risk == 'High'
            else 'result-moderate' if risk == 'Moderate'
            else 'result-negative'
        )
        verdict = "Anemia Risk Detected" if result['prediction'] == 1 else "Low Anemia Risk"

        st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="result-title">{verdict}</div>
            <div style="color:#e2e8f0; font-size:0.95rem;">
                Probability: <b>{result['probability']:.1%}</b> &nbsp;|&nbsp;
                Risk Level: <b>{risk}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge
        gauge_fig = create_prediction_gauge(
            result['probability'], result['prediction'],
            title="Symptom-Based Anemia Probability"
        )
        gauge_fig.update_layout(height=300, margin=dict(t=40, b=0))
        st.plotly_chart(gauge_fig, use_container_width=True)

        # Symptom summary chips
        summary = result.get('symptom_summary', {})
        reported = summary.get('reported_symptoms', [])
        if reported:
            st.markdown("**Reported symptoms:**")
            chips_html = " ".join(
                f'<span class="info-chip">{s}</span>' for s in reported
            )
            st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.info("No symptoms reported.")

        with st.expander("📝 Full Interpretation", expanded=False):
            st.text(result['interpretation'])       

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Dataset Explorer
# ═══════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("#### 📊 Anemia Dataset Explorer")

    try:
        df = load_dataset()
    except FileNotFoundError:
        st.warning("Dataset file not found. Please place `Anemia Dataset.xlsx` "
                    "in `modules/anemia/data/`.")
        st.stop()

    st.caption(f"Dataset: {df.shape[0]} records × {df.shape[1]} columns")

    # ── Class distribution ────────────────────────────────────────────
    e1, e2 = st.columns(2)

    with e1:
        if 'Decision_Class' in df.columns:
            dist = df['Decision_Class'].value_counts().reset_index()
            dist.columns = ['Status', 'Count']
            dist['Status'] = dist['Status'].map({1: 'Anemic', 0: 'Non-Anemic'})
            fig_pie = px.pie(dist, names='Status', values='Count',
                             title='Anemia Status Distribution',
                             color_discrete_sequence=['#f87171', '#60a5fa'])
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

    with e2:
        if 'Gender' in df.columns and 'Decision_Class' in df.columns:
            df_viz = df.copy()
            if df_viz['Gender'].dtype == object:
                df_viz['Gender'] = df_viz['Gender'].replace({'f': 'Female', 'm': 'Male'})
            df_viz['Diagnosis'] = df_viz['Decision_Class'].map({1: 'Anemic', 0: 'Non-Anemic'})
            cross = pd.crosstab(df_viz['Gender'], df_viz['Diagnosis'])
            fig_bar = px.bar(cross, barmode='group',
                             title='Anemia by Gender',
                             color_discrete_sequence=['#f87171', '#60a5fa'])
            fig_bar.update_layout(height=350)
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Parameter distributions ───────────────────────────────────────
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ('Decision_Class',)]
    if numeric_cols:
        selected_param = st.selectbox("Select parameter to explore", numeric_cols,
                                       key="explore_param")

        if 'Decision_Class' in df.columns:
            df_plot = df.copy()
            df_plot['Diagnosis'] = df_plot['Decision_Class'].map(
                {1: 'Anemic', 0: 'Non-Anemic'}
            )
            fig_hist = px.histogram(
                df_plot, x=selected_param, color='Diagnosis',
                marginal='box', nbins=30,
                title=f'Distribution of {selected_param} by Anemia Status',
                color_discrete_sequence=['#f87171', '#60a5fa'],
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────
    with st.expander("🔥 Correlation Heatmap", expanded=False):
        corr_df = df.select_dtypes(include=[np.number])
        if 'Gender' in df.columns and df['Gender'].dtype == object:
            corr_df = corr_df  # already numeric-only
        corr = corr_df.corr().round(2)
        fig_heat = px.imshow(corr, text_auto=True, aspect='auto',
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
    Consult a healthcare professional for diagnosis and treatment.</p>
</div>
""", unsafe_allow_html=True)
