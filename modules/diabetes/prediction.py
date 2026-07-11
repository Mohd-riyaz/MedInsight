"""
prediction.py - Diabetes prediction engine

Loads the pre-trained diabetes model (trained on Pima Indians Diabetes
dataset) and exposes a unified prediction API for the MedInsight AI
dashboard.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODULE_DIR, "models", "diabetes_model.pkl")
DATA_PATH = os.path.join(MODULE_DIR, "data", "diabetes.csv")

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
]

TARGET = 'Outcome'

# Clinical reference ranges (source: ADA / WHO guidelines)
NORMAL_RANGES = {
    'Glucose': {
        'label': 'Fasting Plasma Glucose',
        'unit': 'mg/dL',
        'normal': (70, 99),
        'prediabetes': (100, 125),
        'diabetes': (126, None),  # ≥ 126
    },
    'BloodPressure': {
        'label': 'Diastolic Blood Pressure',
        'unit': 'mm Hg',
        'normal': (60, 80),
        'elevated': (80, 89),
        'high': (90, None),
    },
    'BMI': {
        'label': 'Body Mass Index',
        'unit': 'kg/m²',
        'underweight': (0, 18.5),
        'normal': (18.5, 24.9),
        'overweight': (25.0, 29.9),
        'obese': (30.0, None),
    },
    'Insulin': {
        'label': '2-Hour Serum Insulin',
        'unit': 'μU/mL',
        'normal': (16, 166),
    },
    'SkinThickness': {
        'label': 'Triceps Skin Fold Thickness',
        'unit': 'mm',
        'normal': (10, 50),
    },
    'DiabetesPedigreeFunction': {
        'label': 'Diabetes Pedigree Function',
        'unit': '',
        'low_risk': (0.0, 0.5),
        'moderate_risk': (0.5, 1.0),
        'high_risk': (1.0, None),
    },
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str = MODEL_PATH):
    """
    Load the pre-trained diabetes prediction model.

    Returns
    -------
    model : sklearn estimator / pipeline
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Diabetes model not found at {model_path}. "
            "Please run the training notebook first."
        )
    return joblib.load(model_path)


def load_dataset(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the diabetes CSV dataset.

    Returns
    -------
    df : pd.DataFrame
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}.")
    return pd.read_csv(data_path)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _handle_zeros(input_dict: dict) -> dict:
    """
    Replace biologically implausible zero values with sensible defaults
    (median from the original Pima dataset).
    Zero is valid only for ``Pregnancies`` and ``Outcome``.
    """
    # Approximate medians from the Pima dataset
    medians = {
        'Glucose': 117.0,
        'BloodPressure': 72.0,
        'SkinThickness': 29.0,
        'Insulin': 125.0,
        'BMI': 32.0,
    }
    cleaned = dict(input_dict)
    for feat, median_val in medians.items():
        if feat in cleaned and (cleaned[feat] is None or cleaned[feat] == 0):
            cleaned[feat] = median_val
    return cleaned


def preprocess_input(user_input: dict) -> pd.DataFrame:
    """
    Convert a dictionary of patient data into a DataFrame suitable for
    the diabetes model.

    Parameters
    ----------
    user_input : dict
        Keys should match ``FEATURE_NAMES``.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame in the expected column order.
    """
    cleaned = _handle_zeros(user_input)
    processed = {}
    for feat in FEATURE_NAMES:
        processed[feat] = float(cleaned.get(feat, 0))

    return pd.DataFrame([processed])[FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(user_input: dict, model=None) -> dict:
    """
    Predict diabetes risk from patient clinical parameters.

    Parameters
    ----------
    user_input : dict
        Patient data with keys from ``FEATURE_NAMES``.
    model : sklearn estimator, optional
        Pre-loaded model.  If *None* the model is loaded from disk.

    Returns
    -------
    dict
        ``prediction``   : int (0 = No Diabetes, 1 = Diabetes)
        ``probability``  : float (0-1)
        ``risk_level``   : str ('Low' / 'Moderate' / 'High')
        ``risk_factors`` : list[dict]
        ``interpretation`` : str
        ``recommendations`` : list[str]
    """
    if model is None:
        model = load_model()

    input_df = preprocess_input(user_input)

    prediction = int(model.predict(input_df)[0])

    # Handle models without predict_proba (e.g. SVM without probability=True)
    if hasattr(model, 'predict_proba'):
        proba = float(model.predict_proba(input_df)[0][1])
    elif hasattr(model, 'decision_function'):
        # Convert decision function to pseudo-probability via sigmoid
        decision = float(model.decision_function(input_df)[0])
        proba = 1 / (1 + np.exp(-decision))
    else:
        proba = float(prediction)

    # --- Clinical rule adjustments ----------------------------------------
    glucose = float(user_input.get('Glucose', 0))
    bmi = float(user_input.get('BMI', 0))

    # Strong glucose signal
    if glucose >= 140:
        proba = max(proba, 0.70)
        prediction = 1
    elif glucose >= 126:
        proba = max(proba, 0.55)
        prediction = 1

    # Extremely high BMI compounds risk
    if bmi >= 35 and glucose >= 110:
        proba = max(proba, 0.60)
        prediction = 1

    risk_level = _stratify_risk(proba)
    risk_factors = _analyse_risk_factors(user_input)
    interpretation = _build_interpretation(
        prediction, proba, risk_level, risk_factors, user_input
    )
    recommendations = _generate_recommendations(
        prediction, risk_level, risk_factors, user_input
    )

    return {
        'prediction': prediction,
        'probability': round(proba, 4),
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'interpretation': interpretation,
        'recommendations': recommendations,
        'input_df': input_df,
    }


# ---------------------------------------------------------------------------
# Risk factor analysis
# ---------------------------------------------------------------------------

def _stratify_risk(probability: float) -> str:
    if probability < 0.30:
        return 'Low'
    elif probability < 0.70:
        return 'Moderate'
    else:
        return 'High'


def _classify_glucose(value: float) -> str:
    if value <= 0:
        return 'Unknown'
    elif value < 100:
        return 'Normal'
    elif value < 126:
        return 'Pre-diabetes'
    else:
        return 'Diabetic range'


def _classify_bmi(value: float) -> str:
    if value <= 0:
        return 'Unknown'
    elif value < 18.5:
        return 'Underweight'
    elif value < 25:
        return 'Normal'
    elif value < 30:
        return 'Overweight'
    else:
        return 'Obese'


def _classify_bp(value: float) -> str:
    if value <= 0:
        return 'Unknown'
    elif value < 80:
        return 'Normal'
    elif value < 90:
        return 'Elevated'
    else:
        return 'High'


def _classify_dpf(value: float) -> str:
    if value < 0.5:
        return 'Low genetic risk'
    elif value < 1.0:
        return 'Moderate genetic risk'
    else:
        return 'High genetic risk'


def _analyse_risk_factors(user_input: dict) -> list:
    """Identify and categorise individual risk factors."""
    factors = []
    glucose = float(user_input.get('Glucose', 0))
    bmi = float(user_input.get('BMI', 0))
    bp = float(user_input.get('BloodPressure', 0))
    age = float(user_input.get('Age', 0))
    dpf = float(user_input.get('DiabetesPedigreeFunction', 0))
    insulin = float(user_input.get('Insulin', 0))
    pregnancies = int(user_input.get('Pregnancies', 0))

    # Glucose
    glu_cat = _classify_glucose(glucose)
    if glu_cat in ('Pre-diabetes', 'Diabetic range'):
        factors.append({
            'factor': 'Elevated Glucose',
            'value': glucose,
            'unit': 'mg/dL',
            'category': glu_cat,
            'severity': 'high' if glu_cat == 'Diabetic range' else 'moderate',
        })

    # BMI
    bmi_cat = _classify_bmi(bmi)
    if bmi_cat in ('Overweight', 'Obese'):
        factors.append({
            'factor': 'Elevated BMI',
            'value': bmi,
            'unit': 'kg/m²',
            'category': bmi_cat,
            'severity': 'high' if bmi_cat == 'Obese' else 'moderate',
        })

    # Blood pressure
    bp_cat = _classify_bp(bp)
    if bp_cat in ('Elevated', 'High'):
        factors.append({
            'factor': 'Elevated Blood Pressure',
            'value': bp,
            'unit': 'mm Hg',
            'category': bp_cat,
            'severity': 'high' if bp_cat == 'High' else 'moderate',
        })

    # Age
    if age >= 45:
        factors.append({
            'factor': 'Age ≥ 45',
            'value': int(age),
            'unit': 'years',
            'category': 'Increased age-related risk',
            'severity': 'moderate',
        })

    # Genetic / family history (via DPF)
    dpf_cat = _classify_dpf(dpf)
    if 'Moderate' in dpf_cat or 'High' in dpf_cat:
        factors.append({
            'factor': 'Family History / Genetics',
            'value': round(dpf, 3),
            'unit': '',
            'category': dpf_cat,
            'severity': 'high' if 'High' in dpf_cat else 'moderate',
        })

    # Insulin resistance proxy
    if insulin > 166:
        factors.append({
            'factor': 'Elevated Insulin',
            'value': insulin,
            'unit': 'μU/mL',
            'category': 'Possible insulin resistance',
            'severity': 'moderate',
        })

    # Gestational history
    if pregnancies >= 5:
        factors.append({
            'factor': 'Multiple Pregnancies',
            'value': pregnancies,
            'unit': '',
            'category': 'Gestational risk factor',
            'severity': 'low',
        })

    return factors


# ---------------------------------------------------------------------------
# Interpretation and recommendations
# ---------------------------------------------------------------------------

def _build_interpretation(prediction, probability, risk_level,
                          risk_factors, user_input):
    """Compose human-readable clinical interpretation."""
    lines = [
        f"Risk Level: {risk_level} ({probability:.1%} probability of diabetes)."
    ]

    if prediction == 1:
        lines.append(
            "The clinical parameters indicate an elevated risk of Type 2 Diabetes."
        )
    else:
        lines.append(
            "The clinical parameters do not strongly indicate diabetes at this time."
        )

    # Glucose summary
    glucose = float(user_input.get('Glucose', 0))
    glu_cat = _classify_glucose(glucose)
    lines.append(f"\nGlucose status: {glucose} mg/dL ({glu_cat}).")

    # BMI summary
    bmi = float(user_input.get('BMI', 0))
    bmi_cat = _classify_bmi(bmi)
    lines.append(f"BMI status: {bmi} kg/m² ({bmi_cat}).")

    if risk_factors:
        lines.append(f"\nIdentified risk factors ({len(risk_factors)}):")
        for rf in risk_factors:
            lines.append(
                f"  • {rf['factor']}: {rf['value']} {rf['unit']} — {rf['category']}"
            )

    lines.append(
        "\n⚠ This is a screening tool only. Please consult a healthcare "
        "professional for diagnosis, HbA1c testing, and treatment."
    )
    return "\n".join(lines)


def _generate_recommendations(prediction, risk_level, risk_factors,
                               user_input):
    """Generate personalised lifestyle / clinical recommendations."""
    recs = []

    if prediction == 1 or risk_level in ('Moderate', 'High'):
        recs.append(
            "Schedule an HbA1c test and fasting glucose test with your doctor."
        )

    # BMI-related
    bmi = float(user_input.get('BMI', 0))
    if bmi >= 25:
        recs.append(
            "Aim for a healthy body weight through balanced nutrition "
            "and at least 150 minutes of moderate exercise per week."
        )

    # Glucose-related
    glucose = float(user_input.get('Glucose', 0))
    if glucose >= 100:
        recs.append(
            "Limit refined carbohydrates and sugary beverages. "
            "Focus on whole grains, vegetables, and lean proteins."
        )

    # Blood pressure
    bp = float(user_input.get('BloodPressure', 0))
    if bp >= 80:
        recs.append(
            "Monitor blood pressure regularly. Reduce sodium intake and "
            "manage stress to help maintain healthy blood pressure."
        )

    # Age
    age = float(user_input.get('Age', 0))
    if age >= 45:
        recs.append(
            "Adults aged 45+ should have regular diabetes screenings "
            "at least every 3 years, or more frequently if risk factors exist."
        )

    # Genetic
    dpf = float(user_input.get('DiabetesPedigreeFunction', 0))
    if dpf >= 0.5:
        recs.append(
            "With a family history of diabetes, proactive monitoring and "
            "lifestyle modifications are especially important."
        )

    # Universal
    recs.append(
        "Stay hydrated, get adequate sleep (7-9 hours), and avoid "
        "tobacco and excessive alcohol consumption."
    )

    return recs


# ---------------------------------------------------------------------------
# Batch prediction (for analytics / dashboard)
# ---------------------------------------------------------------------------

def predict_batch(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Run predictions on an entire DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns matching ``FEATURE_NAMES``.
    model : sklearn estimator, optional

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
        ``Prediction``, ``Probability``, ``Risk_Level``.
    """
    if model is None:
        model = load_model()

    result_df = df.copy()
    features = df[FEATURE_NAMES].copy()

    # Handle zeros in batch
    medians = {
        'Glucose': 117.0, 'BloodPressure': 72.0,
        'SkinThickness': 29.0, 'Insulin': 125.0, 'BMI': 32.0,
    }
    for col, med in medians.items():
        if col in features.columns:
            features[col] = features[col].replace(0, med)

    result_df['Prediction'] = model.predict(features)

    if hasattr(model, 'predict_proba'):
        result_df['Probability'] = model.predict_proba(features)[:, 1]
    else:
        result_df['Probability'] = result_df['Prediction'].astype(float)

    result_df['Risk_Level'] = result_df['Probability'].apply(_stratify_risk)

    return result_df


# ---------------------------------------------------------------------------
# Dataset statistics (for dashboard EDA)
# ---------------------------------------------------------------------------

def get_dataset_statistics(df: pd.DataFrame = None) -> dict:
    """
    Compute summary statistics from the diabetes dataset.

    Returns
    -------
    dict with keys: shape, class_distribution, feature_stats, correlation
    """
    if df is None:
        df = load_dataset()

    stats = {
        'shape': df.shape,
        'class_distribution': df[TARGET].value_counts().to_dict(),
        'feature_stats': df[FEATURE_NAMES].describe().to_dict(),
        'correlation_with_target': df[FEATURE_NAMES]
            .corrwith(df[TARGET])
            .sort_values(ascending=False)
            .to_dict(),
    }
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  MedInsight AI — Diabetes Predictor (CLI Demo)")
    print("=" * 60)

    sample_patient = {
        'Pregnancies': 5,
        'Glucose': 166,
        'BloodPressure': 72,
        'SkinThickness': 19,
        'Insulin': 175,
        'BMI': 25.8,
        'DiabetesPedigreeFunction': 0.587,
        'Age': 51,
    }

    try:
        result = predict(sample_patient)
        print(f"\nPrediction: {'Diabetic' if result['prediction'] == 1 else 'Non-Diabetic'}")
        print(f"Probability: {result['probability']:.1%}")
        print(f"\n{result['interpretation']}")
        print("\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")

    # Low-risk patient
    print("\n" + "-" * 60)
    sample_healthy = {
        'Pregnancies': 1,
        'Glucose': 85,
        'BloodPressure': 66,
        'SkinThickness': 29,
        'Insulin': 102.5,
        'BMI': 26.6,
        'DiabetesPedigreeFunction': 0.351,
        'Age': 31,
    }

    try:
        result = predict(sample_healthy)
        print(f"\nPrediction: {'Diabetic' if result['prediction'] == 1 else 'Non-Diabetic'}")
        print(f"Probability: {result['probability']:.1%}")
        print(f"\n{result['interpretation']}")
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
