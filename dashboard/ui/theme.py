import streamlit as st

def load_css():
    st.markdown("""
    <style>
    /* =======================================================================
       MedInsight AI Premium Theme — Global CSS
       ======================================================================= */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar polish ─────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* ── Hero / Page Headers ─────────────────────────────────────────────── */
    .page-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0d9488 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 20px 40px -15px rgba(13, 148, 136, 0.4);
        position: relative;
        overflow: hidden;
    }
    .page-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        background: linear-gradient(to right, #ffffff, #ccfbf1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .page-header p {
        font-size: 1.1rem;
        color: #99f6e4;
        margin: 0;
        font-weight: 400;
    }

    /* ── Result Cards ───────────────────────────────────────────────────── */
    .result-card {
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(12px);
    }
    .result-positive {
        background: linear-gradient(135deg, rgba(225, 29, 72, 0.15), rgba(225, 29, 72, 0.05));
        border: 1px solid rgba(225, 29, 72, 0.4);
        box-shadow: 0 10px 30px -10px rgba(225, 29, 72, 0.2);
    }
    .result-negative {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.4);
        box-shadow: 0 10px 30px -10px rgba(16, 185, 129, 0.2);
    }
    .result-moderate {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.05));
        border: 1px solid rgba(245, 158, 11, 0.4);
        box-shadow: 0 10px 30px -10px rgba(245, 158, 11, 0.2);
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .result-positive .result-title { color: #f43f5e; }
    .result-negative .result-title { color: #10b981; }
    .result-moderate .result-title { color: #fbbf24; }

    /* ── Parameter Analysis / Glass Cards ───────────────────────────────── */
    .param-card, .info-box, .chart-container, .insights-panel {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .param-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border-color: rgba(14, 165, 233, 0.4);
    }
    .param-card h4, .info-box h4, .insights-panel h4 {
        color: #38bdf8;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.15rem;
    }
    .param-card p, .info-box p, .insights-panel li {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* ── Metric Highlights ──────────────────────────────────────────────── */
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* ── Status Pills ───────────────────────────────────────────────────── */
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-high   { background: rgba(244, 63, 94, 0.2); color: #fda4af; border: 1px solid rgba(244,63,94,0.3); }
    .status-normal { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }
    .status-low    { background: rgba(14, 165, 233, 0.2); color: #7dd3fc; border: 1px solid rgba(14,165,233,0.3); }

    /* ── Disclaimer ─────────────────────────────────────────────────────── */
    .disclaimer {
        background: linear-gradient(to right, rgba(225, 29, 72, 0.1), rgba(225, 29, 72, 0.02));
        border-left: 4px solid #e11d48;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-top: 2.5rem;
    }
    .disclaimer p {
        color: #fda4af;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)
