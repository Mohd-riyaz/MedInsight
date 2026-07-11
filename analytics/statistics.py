"""
analytics/statistics.py — Data loading, KPI computation, and AI insights
for the MedInsight AI Healthcare Analytics Dashboard.

All dataset values are computed dynamically from the source files.
"""

import os
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANEMIA_DATA_PATH = os.path.join(
    _PROJECT_ROOT, "modules", "anemia", "data", "Anemia Dataset.xlsx"
)
DIABETES_DATA_PATH = os.path.join(
    _PROJECT_ROOT, "modules", "diabetes", "data", "diabetes.csv"
)
HEART_DATA_PATH = os.path.join(
    _PROJECT_ROOT, "modules", "heart", "data", "heart_disease_uci.csv"
)

# ---------------------------------------------------------------------------
# Default model accuracies (%)
# Anemia: from training notebook (Random Forest best model)
# Diabetes / Heart: user-provided
# ---------------------------------------------------------------------------
MODEL_ACCURACIES = {
    "Anemia": 99.00,
    "Diabetes": 88.31,
    "Heart Disease": 84.24,
}

# Target variable names per dataset
_TARGET_VARIABLES = {
    "Anemia": "Decision_Class",
    "Diabetes": "Outcome",
    "Heart Disease": "num",
}


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_anemia_data() -> pd.DataFrame:
    """Load the Anemia Dataset from Excel."""
    return pd.read_excel(ANEMIA_DATA_PATH)


@st.cache_data(show_spinner=False)
def load_diabetes_data() -> pd.DataFrame:
    """Load the Diabetes dataset from CSV."""
    return pd.read_csv(DIABETES_DATA_PATH)


@st.cache_data(show_spinner=False)
def load_heart_data() -> pd.DataFrame:
    """Load the Heart Disease UCI dataset from CSV."""
    return pd.read_csv(HEART_DATA_PATH)


def load_all_datasets() -> dict[str, pd.DataFrame]:
    """Return all three datasets keyed by display name."""
    return {
        "Anemia": load_anemia_data(),
        "Diabetes": load_diabetes_data(),
        "Heart Disease": load_heart_data(),
    }


# ---------------------------------------------------------------------------
# KPI helpers
# ---------------------------------------------------------------------------
def get_kpi_summary(datasets: dict[str, pd.DataFrame]) -> dict:
    """Return record counts for each dataset and the total."""
    counts = {name: len(df) for name, df in datasets.items()}
    counts["Total"] = sum(counts.values())
    return counts


# ---------------------------------------------------------------------------
# Dataset summary table
# ---------------------------------------------------------------------------
_SOURCE_FILES = {
    "Anemia": "modules/anemia/data/Anemia Dataset.xlsx",
    "Diabetes": "modules/diabetes/data/diabetes.csv",
    "Heart Disease": "modules/heart/data/heart_disease_uci.csv",
}


def get_dataset_stats(datasets: dict[str, pd.DataFrame]) -> list[dict]:
    """Build a list of per-dataset summary dicts for the summary table."""
    rows = []
    for name, df in datasets.items():
        # Try to detect the target variable automatically
        target = _TARGET_VARIABLES.get(name, "—")
        # Verify it exists in the dataframe; fall back gracefully
        if target != "—" and target not in df.columns:
            # Case-insensitive lookup
            for col in df.columns:
                if col.lower() == target.lower():
                    target = col
                    break
            else:
                target = df.columns[-1]  # last column as fallback

        missing = int(df.isnull().sum().sum())

        rows.append({
            "Dataset": name,
            "Records": len(df),
            "Features": len(df.columns),
            "Target Variable": target,
            "Missing Values": missing,
            "Source File": _SOURCE_FILES.get(name, "—"),
        })
    return rows


# ---------------------------------------------------------------------------
# AI Insights (dynamically generated)
# ---------------------------------------------------------------------------
def get_ai_insights(
    datasets: dict[str, pd.DataFrame],
    accuracies: dict[str, float] | None = None,
) -> list[str]:
    """Generate a list of human-readable insight strings."""
    if accuracies is None:
        accuracies = MODEL_ACCURACIES

    counts = {name: len(df) for name, df in datasets.items()}
    total = sum(counts.values())
    largest = max(counts, key=counts.get)
    smallest = min(counts, key=counts.get)
    best_model = max(accuracies, key=accuracies.get)

    insights = [
        f"📂 **Largest dataset:** {largest} with **{counts[largest]:,}** records",
        f"📁 **Smallest dataset:** {smallest} with **{counts[smallest]:,}** records",
        f"🏆 **Highest-performing model:** {best_model} at **{accuracies[best_model]:.2f}%** accuracy",
        f"🩺 **Diseases supported:** {len(datasets)}",
        f"📊 **Total combined records:** {total:,}",
    ]
    return insights
