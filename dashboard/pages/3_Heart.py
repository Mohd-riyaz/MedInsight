"""
3_Heart.py — Heart Disease Screening Dashboard Page

Provides individual prediction with risk factor analysis,
clinical recommendations, and dataset exploration.
"""

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from modules.heart.explainability import (
    summary_plot,
    waterfall_plot
)
# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
sys.path.insert(0, PROJECT_ROOT)

from modules.heart.predictor import (
    predict, load_model, load_dataset, preprocess_dataset,
    get_dataset_statistics, MODEL_FEATURES,
    CP_MAPPING, RESTECG_MAPPING, SLOPE_MAPPING, THAL_MAPPING,
    _classify_bp, _classify_cholesterol,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MedInsight AI — Heart Disease Screening",
    page_icon="❤️",
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
    background: rgba(139,92,246,0.08);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 12px;
    padding: 0.85rem 1.25rem;
    margin-bottom: 0.5rem;
    color: #e9d5ff;
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
    background: linear-gradient(90deg, #c084fc, #f472b6);
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
    <h1>❤️ Heart Disease Screening</h1>
    <p>Cardiovascular risk assessment from clinical &amp; exercise test data</p>
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
    st.caption("Values from physical examination, blood work, and exercise stress test")

    col_l, col_m, col_r = st.columns(3)

    with col_l:
        st.markdown("**Demographics**")
        h_age = st.number_input("Age (years)", 18, 100, 50, key="h_age")
        h_sex = st.selectbox("Sex", ["Male", "Female"], key="h_sex")

        st.markdown("**Chest Pain**")
        h_cp = st.selectbox(
            "Chest Pain Type", list(CP_MAPPING.keys()), key="h_cp",
            help="Typical angina, atypical angina, non-anginal, or asymptomatic"
        )

    with col_m:
        st.markdown("**Vitals & Blood Work**")
        h_trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)", 80, 220, 120, key="h_bp",
            help="Normal: < 120 | Elevated: 120–129 | High: ≥ 130"
        )
        h_chol = st.number_input(
            "Serum Cholesterol (mg/dL)", 100, 600, 200, key="h_chol",
            help="Desirable: < 200 | Borderline: 200–239 | High: ≥ 240"
        )
        h_fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL?",
            ["No", "Yes"], key="h_fbs"
        )
        h_restecg = st.selectbox(
            "Resting ECG", list(RESTECG_MAPPING.keys()), key="h_restecg"
        )

    with col_r:
        st.markdown("**Exercise Test Results**")
        h_thalch = st.number_input(
            "Max Heart Rate Achieved (bpm)", 60, 220, 150, key="h_thalch",
            help=f"Expected max ≈ {220 - h_age} bpm"
        )
        h_exang = st.selectbox(
            "Exercise-Induced Angina?",
            ["No", "Yes"], key="h_exang"
        )
        h_oldpeak = st.number_input(
            "ST Depression (Oldpeak)", 0.0, 8.0, 0.0, 0.1, key="h_oldpeak",
            help="Normal: < 1.0 | Abnormal: ≥ 1.0"
        )
        h_slope = st.selectbox(
            "ST Slope", list(SLOPE_MAPPING.keys()), key="h_slope"
        )
        h_ca = st.number_input(
            "Major Vessels (Fluoroscopy)", 0, 4, 0, key="h_ca",
            help="Number of major vessels coloured by fluoroscopy"
        )
        h_thal = st.selectbox(
            "Thalassemia", list(THAL_MAPPING.keys()), key="h_thal"
        )
        

    # ── Quick metrics ─────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    def _status_color(cat):
        if cat in ('Normal', 'Desirable'):
            return 'background:rgba(34,197,94,0.2); color:#4ade80;'
        elif cat in ('Elevated', 'Borderline High'):
            return 'background:rgba(234,179,8,0.2); color:#facc15;'
        else:
            return 'background:rgba(239,68,68,0.2); color:#f87171;'

    bp_cat = _classify_bp(h_trestbps)
    chol_cat = _classify_cholesterol(h_chol)
    hr_pct = round(h_thalch / max(220 - h_age, 1) * 100)

    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{h_trestbps}</div>
            <div class="metric-label">Resting BP (mm Hg)</div>
            <div class="metric-status" style="{_status_color(bp_cat)}">{bp_cat}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{h_chol}</div>
            <div class="metric-label">Cholesterol (mg/dL)</div>
            <div class="metric-status" style="{_status_color(chol_cat)}">{chol_cat}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        hr_cat = 'Normal' if hr_pct >= 75 else 'Below Expected'
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{h_thalch}</div>
            <div class="metric-label">Max Heart Rate (bpm)</div>
            <div class="metric-status" style="{_status_color(hr_cat)}">{hr_pct}% of max</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        op_cat = 'Normal' if h_oldpeak < 1.0 else 'Abnormal'
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{h_oldpeak}</div>
            <div class="metric-label">ST Depression (mm)</div>
            <div class="metric-status" style="{_status_color(op_cat)}">{op_cat}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    if st.button("🔍  Assess Heart Disease Risk", type="primary",
                  use_container_width=True, key="btn_heart"):
        user_input = {
            'age': h_age,
            'sex': h_sex,
            'cp': h_cp,
            'trestbps': h_trestbps,
            'chol': h_chol,
            'fbs': h_fbs == 'Yes',
            'restecg': h_restecg,
            'thalch': h_thalch,
            'exang': h_exang == 'Yes',
            'oldpeak': h_oldpeak,
            'slope': h_slope,
            'ca': h_ca,
            'thal': h_thal,
        }

        with st.spinner("Analysing..."):
            try:
                result = predict(user_input)
                input_df = result["input_df"]
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
        verdict = ("Heart Disease Risk Elevated"
                    if result['prediction'] == 1
                    else "Low Heart Disease Risk")

        st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="result-title">{verdict}</div>
            <div style="color:#e2e8f0; font-size:0.95rem;">
                Probability: <b>{result['probability']:.1%}</b> &nbsp;|&nbsp;
                Risk Level: <b>{risk}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # ── Gauge + Radar ─────────────────────────────────────────────
        g1, g2 = st.columns(2)

        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['probability'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': 'Heart Disease Probability'},
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

        with g2:
            # Risk factor severity bar chart
            risk_factors = result.get('risk_factors', [])
            if risk_factors:
                rf_names = [rf['factor'] for rf in risk_factors]
                rf_severities = [
                    3 if rf['severity'] == 'high' else 2 if rf['severity'] == 'moderate' else 1
                    for rf in risk_factors
                ]
                rf_colors = [
                    '#ef4444' if rf['severity'] == 'high'
                    else '#f59e0b' if rf['severity'] == 'moderate'
                    else '#22c55e'
                    for rf in risk_factors
                ]
                fig_rf = go.Figure(go.Bar(
                    y=rf_names, x=rf_severities, orientation='h',
                    marker_color=rf_colors,
                    text=[rf['severity'].title() for rf in risk_factors],
                    textposition='auto',
                ))
                fig_rf.update_layout(
                    title='Risk Factor Severity',
                    xaxis_title='Severity',
                    xaxis=dict(tickvals=[1, 2, 3], ticktext=['Low', 'Moderate', 'High']),
                    height=300, margin=dict(t=50, b=0, l=0),
                )
                st.plotly_chart(fig_rf, use_container_width=True)
            else:
                st.success("No significant risk factors identified.")

        # ── Risk factors detail ───────────────────────────────────────
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

        # ── Recommendations ───────────────────────────────────────────
        recs = result.get('recommendations', [])
        if recs:
            st.markdown("##### 💡 Clinical Recommendations")
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
    st.markdown("#### 📊 Heart Disease Dataset Explorer")

    try:
        df_raw = load_dataset()
    except FileNotFoundError:
        st.warning("Dataset not found. Place `heart_disease_uci.csv` in "
                    "`modules/heart/data/`.")
        st.stop()

    st.caption(f"UCI Heart Disease Dataset: {df_raw.shape[0]} records × {df_raw.shape[1]} columns")

    # Pre-process for viz
    try:
        df = preprocess_dataset(df_raw)
    except Exception:
        df = df_raw.copy()

    # ── Class distribution ────────────────────────────────────────────
    e1, e2 = st.columns(2)

    with e1:
        if 'target' in df.columns:
            dist = df['target'].value_counts().reset_index()
            dist.columns = ['Status', 'Count']
            dist['Status'] = dist['Status'].map({1: 'Heart Disease', 0: 'No Disease'})
            fig_pie = px.pie(dist, names='Status', values='Count',
                             title='Heart Disease Distribution',
                             color_discrete_sequence=['#f472b6', '#60a5fa'])
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

    with e2:
        if 'target' in df.columns and 'sex' in df.columns:
            df_viz = df.copy()
            df_viz['Sex'] = df_viz['sex'].map({1: 'Male', 0: 'Female'})
            df_viz['Status'] = df_viz['target'].map({1: 'Disease', 0: 'No Disease'})
            cross = pd.crosstab(df_viz['Sex'], df_viz['Status'])
            fig_bar = px.bar(cross, barmode='group',
                             title='Heart Disease by Sex',
                             color_discrete_sequence=['#60a5fa', '#f472b6'])
            fig_bar.update_layout(height=350)
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Age distribution ──────────────────────────────────────────────
    if 'age' in df.columns and 'target' in df.columns:
        df_age_viz = df.copy()
        df_age_viz['Status'] = df_age_viz['target'].map({1: 'Disease', 0: 'No Disease'})
        fig_age = px.histogram(
            df_age_viz, x='age', color='Status',
            marginal='box', nbins=25,
            title='Age Distribution by Heart Disease Status',
            color_discrete_sequence=['#60a5fa', '#f472b6'],
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)

    # ── Feature selector ──────────────────────────────────────────────
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ('target',)]
    if numeric_cols:
        selected = st.selectbox("Explore feature", numeric_cols, key="h_explore")
        if 'target' in df.columns:
            df_feat = df.copy()
            df_feat['Status'] = df_feat['target'].map({1: 'Disease', 0: 'No Disease'})
            fig_feat = px.histogram(
                df_feat, x=selected, color='Status',
                marginal='box', nbins=30,
                title=f'{selected} Distribution by Heart Disease Status',
                color_discrete_sequence=['#60a5fa', '#f472b6'],
            )
            fig_feat.update_layout(height=400)
            st.plotly_chart(fig_feat, use_container_width=True)

    # ── Correlation ───────────────────────────────────────────────────
    with st.expander("📈 Feature Correlation with Heart Disease", expanded=False):
        if 'target' in df.columns:
            feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                         if c != 'target']
            corr = df[feat_cols].corrwith(df['target']).sort_values(ascending=True)
            fig_corr = px.bar(
                x=corr.values, y=corr.index, orientation='h',
                title='Correlation of Features with Heart Disease',
                labels={'x': 'Correlation', 'y': 'Feature'},
                color=corr.values,
                color_continuous_scale='RdYlGn_r',
            )
            fig_corr.update_layout(height=450)
            st.plotly_chart(fig_corr, use_container_width=True)

    # ── Heatmap ───────────────────────────────────────────────────────
    with st.expander("🔥 Full Correlation Heatmap", expanded=False):
        corr_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        full_corr = df[corr_cols].corr().round(2)
        fig_heat = px.imshow(full_corr, text_auto=True, aspect='auto',
                             color_continuous_scale='RdBu_r',
                             title='Correlation Heatmap')
        fig_heat.update_layout(height=550)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Raw data ──────────────────────────────────────────────────────
    with st.expander("📄 Raw Data", expanded=False):
        st.dataframe(df_raw, use_container_width=True, height=400)

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-small">
    <p>⚠️ <b>Disclaimer:</b> This tool is for screening purposes only.
    Consult a cardiologist for proper ECG, stress testing, and imaging.</p>
</div>
""", unsafe_allow_html=True)
