import streamlit as st
import pandas as pd
import sys
import os

# Ensure project root is in path for imports
DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analytics.statistics import (
    load_all_datasets,
    get_kpi_summary,
    get_dataset_stats,
    get_ai_insights,
    MODEL_ACCURACIES
)
from analytics.charts import (
    create_pie_chart,
    create_accuracy_bar_chart
)
from dashboard.ui.theme import load_css

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)
load_css()

st.markdown("""
<div class="page-header">
    <h1>📊 Analytics & AI Insights</h1>
    <p>Explore the underlying clinical datasets, machine learning model performances, and automated data insights powering MedInsight AI.</p>
</div>
""", unsafe_allow_html=True)

# --- Load Data ---
with st.spinner("Loading clinical datasets..."):
    datasets = load_all_datasets()
    counts = get_kpi_summary(datasets)
    total_records = counts.pop("Total")

# --- 1. Top Level KPIs ---
st.markdown("<div class='section-title' style='text-align: left; font-size: 1.5rem;'>Global Overview</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Datasets", len(datasets))
c2.metric("Total Combined Records", f"{total_records:,}")
c3.metric("Supported Diseases", "3")

best_model = max(MODEL_ACCURACIES.items(), key=lambda x: x[1])
c4.metric("Highest Model Accuracy", f"{best_model[1]:.2f}%", best_model[0])

st.markdown("<br>", unsafe_allow_html=True)

# --- 2. Interactive Charts ---
left, right = st.columns(2)
with left:
    st.plotly_chart(create_pie_chart(counts), use_container_width=True)

with right:
    st.plotly_chart(create_accuracy_bar_chart(MODEL_ACCURACIES), use_container_width=True)


# --- 3. Dataset Characteristics Table ---
st.markdown("<div class='section-title' style='text-align: left; font-size: 1.5rem; margin-top: 2rem;'>Dataset Characteristics</div>", unsafe_allow_html=True)

stats_list = get_dataset_stats(datasets)
df_stats = pd.DataFrame(stats_list)

# Style the dataframe to blend nicely with the dark theme
st.dataframe(
    df_stats, 
    use_container_width=True, 
    hide_index=True,
    height=160
)

# --- 4. AI Insights ---
st.markdown("<div class='section-title' style='text-align: left; font-size: 1.5rem; margin-top: 2rem;'>🤖 Automated AI Insights</div>", unsafe_allow_html=True)

insights = get_ai_insights(datasets, MODEL_ACCURACIES)
for insight in insights:
    st.info(insight)