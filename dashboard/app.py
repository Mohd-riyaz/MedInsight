"""
MedInsight AI — Main Dashboard Application

Streamlit multipage app entry point. This file renders the Home / landing
page. Disease-specific pages live under ``pages/``.
"""

import streamlit as st
import os
import sys

# ---------------------------------------------------------------------------
# Path configuration — allow importing from modules/
# ---------------------------------------------------------------------------
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MedInsight — Smart Health Screening",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.ui.theme import load_css
load_css()

# ---------------------------------------------------------------------------
# Custom CSS for Premium Landing Page
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Global Resets & Typography */
h1, h2, h3, h4, p, div {
    font-family: 'Inter', 'Roboto', sans-serif;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse-glow {
    0%, 100% { filter: drop-shadow(0 0 15px rgba(56, 189, 248, 0.4)); }
    50% { filter: drop-shadow(0 0 30px rgba(56, 189, 248, 0.8)); }
}

/* 1. Hero Section */
.hero-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #082f49 100%);
    border-radius: 24px;
    padding: 4rem 3rem;
    margin-bottom: 3rem;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    animation: fadeIn 0.8s ease-out;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 80%; height: 200%;
    background: radial-gradient(circle, rgba(56,189,248,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-content {
    flex: 1;
    z-index: 1;
}
.hero-title {
    font-size: 3.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e0f2fe, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 1.4rem;
    color: #bae6fd;
    font-weight: 500;
    margin-bottom: 1.5rem;
}
.hero-desc {
    font-size: 1.05rem;
    color: #94a3b8;
    line-height: 1.7;
    margin-bottom: 2rem;
    max-width: 90%;
}
.hero-btn {
    display: inline-block;
    background: linear-gradient(90deg, #0ea5e9, #3b82f6);
    color: #ffffff;
    font-size: 1.1rem;
    font-weight: 600;
    padding: 0.8rem 2rem;
    border-radius: 50px;
    text-decoration: none;
    box-shadow: 0 10px 20px rgba(14, 165, 233, 0.3);
    transition: all 0.3s ease;
    border: none;
    cursor: pointer;
}
.hero-btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 25px rgba(14, 165, 233, 0.4);
    color: #ffffff;
}
.hero-visual {
    flex: 1;
    text-align: center;
    z-index: 1;
}
.hero-icon-large {
    font-size: 12rem;
    animation: pulse-glow 4s infinite ease-in-out;
}

/* 2. About Card */
.glass-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 2.5rem;
    margin-bottom: 3rem;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
    animation: fadeIn 1s ease-out 0.2s both;
}
.section-title {
    font-size: 2rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 1.5rem;
    text-align: center;
}
.about-text {
    font-size: 1.1rem;
    color: #cbd5e1;
    line-height: 1.8;
    text-align: justify;
    column-count: 2;
    column-gap: 3rem;
}
@media (max-width: 768px) {
    .about-text { column-count: 1; }
    .hero-container { flex-direction: column; padding: 2rem 1.5rem; }
    .hero-visual { display: none; }
}

/* 3. Core Features Grid */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    margin-bottom: 3rem;
}
.feature-item {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    position: relative;
    overflow: hidden;
    animation: fadeIn 1s ease-out 0.4s both;
}
.feature-item:hover {
    transform: translateY(-10px);
    border-color: rgba(56, 189, 248, 0.6);
    box-shadow: 0 20px 40px rgba(56, 189, 248, 0.15);
}
.feature-item::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    opacity: 0;
    transition: opacity 0.4s ease;
}
.feature-item:hover::before { opacity: 1; }
.feature-item-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    display: block;
}
.feature-item-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.75rem;
}
.feature-item-desc {
    font-size: 0.95rem;
    color: #94a3b8;
    line-height: 1.5;
}

/* 4. How It Works - Workflow Timeline */
.workflow-container {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    margin-bottom: 4rem;
    gap: 1rem;
    animation: fadeIn 1s ease-out 0.6s both;
}
.workflow-step {
    flex: 1 1 150px;
    text-align: center;
    position: relative;
    padding: 1rem;
}
.workflow-icon-box {
    width: 64px;
    height: 64px;
    margin: 0 auto 1rem auto;
    background: rgba(14, 165, 233, 0.1);
    border: 2px solid rgba(14, 165, 233, 0.5);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    color: #38bdf8;
    z-index: 2;
    position: relative;
    transition: all 0.3s ease;
}
.workflow-step:hover .workflow-icon-box {
    background: rgba(14, 165, 233, 0.3);
    transform: scale(1.1);
    box-shadow: 0 0 20px rgba(14, 165, 233, 0.4);
}
.workflow-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #cbd5e1;
}

/* 5. Why Choose MedInsight */
.reasons-container {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0) 0%, rgba(30, 41, 59, 0.4) 100%);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 3rem;
    animation: fadeIn 1s ease-out 0.8s both;
}
.reason-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}
.reason-col {
    flex: 1 1 300px;
    display: flex;
    align-items: flex-start;
    background: rgba(15, 23, 42, 0.6);
    padding: 1.2rem;
    border-radius: 12px;
    border-left: 4px solid #38bdf8;
    transition: transform 0.3s ease, background 0.3s ease, box-shadow 0.3s ease;
}
.reason-col:hover {
    transform: translateY(-5px);
    background: rgba(15, 23, 42, 0.9);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
}
.reason-check {
    color: #38bdf8;
    font-size: 1.2rem;
    margin-right: 1rem;
    margin-top: 2px;
}
.reason-text {
    color: #e2e8f0;
    font-weight: 500;
    font-size: 1.05rem;
}
.reason-subtext {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.2rem;
}
.transparency-note {
    background: rgba(56, 189, 248, 0.05);
    border: 1px dashed rgba(56, 189, 248, 0.3);
    padding: 1.5rem;
    border-radius: 12px;
    color: #bae6fd;
    font-style: italic;
    text-align: center;
}

/* 6. Medical Disclaimer */
.disclaimer-card {
    background: rgba(220, 38, 38, 0.05);
    border-left: 5px solid #ef4444;
    border-radius: 8px;
    padding: 1.5rem;
    margin-top: 3rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    animation: fadeIn 1s ease-out 1s both;
}
.disclaimer-icon {
    font-size: 2.5rem;
}
.disclaimer-text {
    color: #fca5a5;
    font-size: 0.95rem;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar Polish
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("###  MedInsight")
    st.markdown("---")
    st.markdown("""
    <div style="color:#94a3b8; font-size:0.85rem; line-height:1.7;">
    <b>Navigate</b> using the pages above to access
    disease-specific prediction tools.<br><br>
    Each module provides:<br>
    • Risk prediction<br>
    • Parameter analysis<br>
    • Personalised recommendations<br>
    • Explainable AI insights
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("v2.0 · Professional SaaS Edition")


# ---------------------------------------------------------------------------
# 1. Hero Banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-content">
        <div class="hero-title">MedInsight AI</div>
        <div class="hero-subtitle">AI-Powered Clinical Decision Support System</div>
        <div class="hero-desc">
            Empowering early disease risk assessment through Machine Learning and Explainable AI. 
            MedInsight AI provides intelligent screening, transparent predictions, clinical insights, 
            and personalized recommendations for improved healthcare awareness.
        </div>
        <a href="#about-medinsight" class="hero-btn">Start Screening</a>
    </div>
    <div class="hero-visual">
        <div class="hero-icon-large">🧬</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Anchor for the Start Screening button scroll (visual only since streamlit is SPA, it scrolls down)
st.markdown("<div id='about-medinsight'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. About MedInsight AI
# ---------------------------------------------------------------------------
st.markdown("""
<div class="glass-card">
    <div class="section-title">About MedInsight AI</div>
    <div class="about-text">
        <p>MedInsight AI is an advanced clinical decision support platform designed to assist in early disease screening. By harnessing the power of cutting-edge Machine Learning and SHAP (SHapley Additive exPlanations) Explainable AI, we transform complex health data into actionable, easy-to-understand insights.</p>
        <p>Currently supporting robust risk models for <strong>Anemia, Diabetes, and Heart Disease</strong>, the platform analyzes patient parameters against clinically validated datasets. We go beyond simple predictions by offering deep clinical interpretation and personalized recommendations. Emphasizing transparency, accessibility, and healthcare awareness, MedInsight AI ensures you not only see the predicted risk, but understand <em>why</em> it was made.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3. Core Features
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title' style='margin-top: 2rem;'>Core Features</div>", unsafe_allow_html=True)

st.markdown("""
<div class="feature-grid">
    <div class="feature-item">
        <span class="feature-item-icon">🩸</span>
        <div class="feature-item-title">Anemia Prediction</div>
        <div class="feature-item-desc">Advanced analysis of blood parameters and symptoms to accurately assess anemia risk and severity.</div>
    </div>
    <div class="feature-item">
        <span class="feature-item-icon">🍬</span>
        <div class="feature-item-title">Diabetes Prediction</div>
        <div class="feature-item-desc">Evaluates glucose, insulin, and lifestyle factors to predict Type 2 Diabetes risk profiles.</div>
    </div>
    <div class="feature-item">
        <span class="feature-item-icon">❤️</span>
        <div class="feature-item-title">Heart Disease Prediction</div>
        <div class="feature-item-desc">Comprehensive cardiovascular assessment using resting BP, cholesterol, and ECG findings.</div>
    </div>
    <div class="feature-item">
        <span class="feature-item-icon">🔍</span>
        <div class="feature-item-title">Explainable AI (SHAP)</div>
        <div class="feature-item-desc">Transparent model predictions showing exactly how each clinical parameter influences the outcome.</div>
    </div>
    <div class="feature-item">
        <span class="feature-item-icon">📊</span>
        <div class="feature-item-title">Analytics Dashboard</div>
        <div class="feature-item-desc">Explore extensive clinical datasets and health trends through interactive, visually rich charts.</div>
    </div>
    <div class="feature-item">
        <span class="feature-item-icon">💡</span>
        <div class="feature-item-title">Personalized Recommendations</div>
        <div class="feature-item-desc">Receive tailored, evidence-based lifestyle and dietary advice based on your unique risk profile.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 4. How It Works
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title' style='margin-top: 4rem;'>How It Works</div>", unsafe_allow_html=True)

st.markdown("""
<div class="workflow-container">
    <div class="workflow-step">
        <div class="workflow-icon-box">📋</div>
        <div class="workflow-title">Patient Info</div>
    </div>
    <div class="workflow-step">
        <div style="font-size: 2rem; color: #38bdf8; margin-top: 1rem;">➔</div>
    </div>
    <div class="workflow-step">
        <div class="workflow-icon-box">🛡️</div>
        <div class="workflow-title">Data Validation</div>
    </div>
    <div class="workflow-step">
        <div style="font-size: 2rem; color: #38bdf8; margin-top: 1rem;">➔</div>
    </div>
    <div class="workflow-step">
        <div class="workflow-icon-box">⚙️</div>
        <div class="workflow-title">ML Prediction</div>
    </div>
    <div class="workflow-step">
        <div style="font-size: 2rem; color: #38bdf8; margin-top: 1rem;">➔</div>
    </div>
    <div class="workflow-step">
        <div class="workflow-icon-box">📈</div>
        <div class="workflow-title">Risk Analysis</div>
    </div>
    <div class="workflow-step">
        <div style="font-size: 2rem; color: #38bdf8; margin-top: 1rem;">➔</div>
    </div>
    <div class="workflow-step">
        <div class="workflow-icon-box">🧠</div>
        <div class="workflow-title">SHAP AI</div>
    </div>
    <div class="workflow-step">
        <div style="font-size: 2rem; color: #38bdf8; margin-top: 1rem;">➔</div>
    </div>
    <div class="workflow-step">
        <div class="workflow-icon-box">🩺</div>
        <div class="workflow-title">Interpretation</div>
    </div>
    <div class="workflow-step">
        <div style="font-size: 2rem; color: #38bdf8; margin-top: 1rem;">➔</div>
    </div>
    <div class="workflow-step">
        <div class="workflow-icon-box">🌟</div>
        <div class="workflow-title">Recommendations</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 5. Why Choose MedInsight AI?
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title' style='margin-top: 2rem;'>Why Choose MedInsight AI?</div>", unsafe_allow_html=True)

st.markdown("""
<div class="reasons-container">
    <div class="reason-row">
        <div class="reason-col">
            <span class="reason-check">✔</span>
            <div>
                <div class="reason-text">AI-Powered Predictions</div>
                <div class="reason-subtext">Highly accurate screening models trained on vast clinical datasets.</div>
            </div>
        </div>
        <div class="reason-col">
            <span class="reason-check">✔</span>
            <div>
                <div class="reason-text">Explainable AI with SHAP</div>
                <div class="reason-subtext">No black boxes. Understand precisely how every prediction is made.</div>
            </div>
        </div>
        <div class="reason-col">
            <span class="reason-check">✔</span>
            <div>
                <div class="reason-text">Easy-to-Use Interface</div>
                <div class="reason-subtext">Modern, intuitive, and seamlessly integrated workflows for clinicians and patients.</div>
            </div>
        </div>
    </div>
    <div class="reason-row">
        <div class="reason-col">
            <span class="reason-check">✔</span>
            <div>
                <div class="reason-text">Personalized Clinical Insights</div>
                <div class="reason-subtext">Detailed breakdowns tailored to the individual's specific health profile.</div>
            </div>
        </div>
        <div class="reason-col">
            <span class="reason-check">✔</span>
            <div>
                <div class="reason-text">Fast Screening</div>
                <div class="reason-subtext">Real-time processing for immediate risk assessments and triage.</div>
            </div>
        </div>
        <div class="reason-col">
            <span class="reason-check">✔</span>
            <div>
                <div class="reason-text">Data-Driven Support</div>
                <div class="reason-subtext">Continuous improvement ensuring the highest standards of healthcare support.</div>
            </div>
        </div>
    </div>
    <div class="transparency-note">
        <strong>Why Transparency Matters:</strong> In healthcare, trust is paramount. By utilizing Explainable AI, MedInsight AI ensures that medical professionals and users can independently verify the logic behind every risk assessment, fostering confidence and enabling better-informed medical decisions.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 6. Medical Disclaimer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-card">
    <div class="disclaimer-icon">⚠️</div>
    <div class="disclaimer-text">
        <strong>MEDICAL DISCLAIMER:</strong> MedInsight AI is designed for educational and screening purposes only. Predictions generated by this application are not medical diagnoses and should not replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional regarding any health concerns.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 7. FAQ Section
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title' style='margin-top: 4rem;'>Frequently Asked Questions</div>", unsafe_allow_html=True)

st.markdown("""
<style>
/* Style Streamlit expanders to blend with the dark glass theme */
[data-testid="stExpander"] {
    background: rgba(30, 41, 59, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
[data-testid="stExpander"] summary {
    font-weight: 600;
    color: #e2e8f0;
    padding: 0.5rem;
}
[data-testid="stExpander"] summary:hover {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

faq1, faq2 = st.columns(2)
with faq1:
    with st.expander("How accurate are the AI predictions?"):
        st.write("Our models are trained on highly validated clinical datasets. Current accuracy rates are 99% for Anemia, 88% for Diabetes, and 84% for Heart Disease. However, these are screening tools and should not replace professional diagnosis.")
    
    with st.expander("What is Explainable AI (SHAP)?"):
        st.write("SHAP (SHapley Additive exPlanations) is a technique that breaks down exactly how the AI arrived at its prediction. It shows which of your health parameters increased or decreased your risk score, ensuring complete transparency.")

with faq2:
    with st.expander("Is my health data stored or shared?"):
        st.write("No. MedInsight AI operates completely locally during your session. We do not store, track, or share any of the personal health parameters you input into the screening tools.")
    
    with st.expander("Can I use this instead of seeing a doctor?"):
        st.write("Absolutely not. MedInsight AI provides early risk assessment and educational insights. You must always consult a licensed healthcare professional for medical advice, diagnosis, and treatment.")


# ---------------------------------------------------------------------------
# 8. Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; margin-top: 5rem; padding: 2rem 0; border-top: 1px solid rgba(255,255,255,0.05); animation: fadeIn 1s ease-out 1.2s both;">
    <div style="font-size: 1.5rem; font-weight: 800; background: linear-gradient(90deg, #e0f2fe, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">
        MedInsight AI
    </div>
    <div style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;">
        Empowering early detection through Artificial Intelligence.
    </div>
    <div style="color: #64748b; font-size: 0.8rem;">
        &copy; 2026 MedInsight AI. All rights reserved.<br>
        Version 2.0 Professional Edition
    </div>
</div>
""", unsafe_allow_html=True)
