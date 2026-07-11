"""
predictor.py - Anemia prediction engine

Loads pre-trained models (blood-test based and symptom-based) and provides
unified prediction interface for the MedInsight AI dashboard.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODULE_DIR)
sys.path.insert(0, os.path.dirname(MODULE_DIR))

from config.feature_config import (
    FEATURES, NORMAL_RANGES, TARGET,
    encode_gender, prepare_features_for_model, get_feature_names_for_model
)

# ---------------------------------------------------------------------------
# Paths to pre-trained artefacts
# ---------------------------------------------------------------------------
BLOOD_MODEL_PATH = os.path.join(MODULE_DIR, "models", "anemia_prediction_model.pkl")
SYMPTOM_MODEL_PATH = os.path.join(MODULE_DIR, "models", "symptom_anemia_prediction_model.pkl")
DATA_PATH = os.path.join(MODULE_DIR, "data", "Anemia Dataset.xlsx")

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------
BLOOD_FEATURES = ['Age', 'Gender_Encoded', 'Hb', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC']

SYMPTOM_FEATURES = [
    'Gender', 'Age',
    'Fatigue', 'Shortness_of_Breath', 'Dizziness', 'Pale_Skin',
    'Heart_Racing', 'Headaches', 'Cold_Hands_Feet', 'Brittle_Nails',
    'Poor_Concentration', 'Heavy_Periods', 'Recent_Blood_Loss',
    'Vegetarian_Diet', 'GI_Disorders', 'Chronic_Disease'
]

# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_blood_model(model_path: str = BLOOD_MODEL_PATH):
    """
    Load the blood-test based anemia prediction model.

    Returns
    -------
    model : sklearn estimator / pipeline
        The pre-trained model ready for ``.predict()`` / ``.predict_proba()``.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Blood-test model not found at {model_path}. "
            "Please run the training notebook first."
        )
    return joblib.load(model_path)


def load_symptom_model(model_path: str = SYMPTOM_MODEL_PATH):
    """
    Load the symptom-based anemia prediction model.

    Returns
    -------
    model : sklearn estimator / pipeline
        The pre-trained model ready for ``.predict()`` / ``.predict_proba()``.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Symptom model not found at {model_path}. "
            "Please run symptom_prediction.py first."
        )
    return joblib.load(model_path)


def load_dataset(data_path: str = DATA_PATH):
    """
    Load the anemia dataset (Excel format).

    Returns
    -------
    df : pd.DataFrame
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}.")
    return pd.read_excel(data_path)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_blood_input(user_input: dict) -> pd.DataFrame:
    """
    Convert a dictionary of patient blood-test values into a DataFrame
    suitable for the blood-test model.

    Parameters
    ----------
    user_input : dict
        Keys should include ``Gender`` (or ``Gender_Encoded``), ``Age``,
        ``Hb``, ``RBC``, ``PCV``, ``MCV``, ``MCH``, ``MCHC``.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns in the order the model expects.
    """
    processed = {}

    # Gender encoding
    if 'Gender_Encoded' in user_input:
        processed['Gender_Encoded'] = user_input['Gender_Encoded']
    elif 'Gender' in user_input:
        processed['Gender_Encoded'] = encode_gender(user_input['Gender'])
    else:
        processed['Gender_Encoded'] = 0  # default: Male

    # Numerical features
    for feat in ['Age', 'Hb', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC']:
        if feat in user_input:
            processed[feat] = float(user_input[feat])
        else:
            processed[feat] = float(FEATURES[feat]['default'])

    df = pd.DataFrame([processed])

    # Ensure correct column order
    ordered_cols = [c for c in BLOOD_FEATURES if c in df.columns]
    return df[ordered_cols]


def preprocess_symptom_input(user_input: dict) -> pd.DataFrame:
    """
    Convert a dictionary of patient symptoms / risk factors into a
    DataFrame suitable for the symptom-based model.

    Parameters
    ----------
    user_input : dict
        Keys from ``SYMPTOM_FEATURES``.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame.
    """
    processed = {}
    for feat in SYMPTOM_FEATURES:
        processed[feat] = user_input.get(feat, 0)

    return pd.DataFrame([processed])


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------

def predict_blood(user_input: dict, model=None):
    """
    Predict anemia from blood-test parameters.

    Parameters
    ----------
    user_input : dict
        Patient data containing hematological parameters.
    model : sklearn estimator, optional
        Pre-loaded model.  If *None* the model is loaded from disk.

    Returns
    -------
    dict
        ``prediction``  : int (0 = Non-Anemic, 1 = Anemic)
        ``probability`` : float (0-1)
        ``risk_level``  : str ('Low' / 'Moderate' / 'High')
        ``interpretation`` : str
        ``parameter_analysis`` : list[dict]
    """
    if model is None:
        model = load_blood_model()

    input_df = preprocess_blood_input(user_input)

    prediction = int(model.predict(input_df)[0])
    proba = float(model.predict_proba(input_df)[0][1])

    # Apply clinical rules ------------------------------------------------
    # If Hb is clearly below normal range → boost probability
    gender_key = 'f' if user_input.get('Gender_Encoded', user_input.get('Gender', 0)) in [1, 'f', 'Female'] else 'm'
    hb_value = float(user_input.get('Hb', FEATURES['Hb']['default']))
    hb_low = NORMAL_RANGES['Hb'][gender_key][0]

    if hb_value < hb_low:
        proba = max(proba, 0.65)
        prediction = 1

    # Risk stratification
    risk_level = _stratify_risk(proba)

    # Parameter analysis
    param_analysis = _analyse_blood_parameters(user_input, gender_key)

    interpretation = _build_blood_interpretation(
        prediction, proba, risk_level, param_analysis
    )

    return {
        'prediction': prediction,
        'probability': round(proba, 4),
        'risk_level': risk_level,
        'interpretation': interpretation,
        'parameter_analysis': param_analysis,
        'input_df': input_df,
    }


def predict_symptoms(user_input: dict, model=None):
    """
    Predict anemia from symptom / risk-factor questionnaire.

    Parameters
    ----------
    user_input : dict
        Keys from ``SYMPTOM_FEATURES`` (binary 0/1 except Age and Gender).
    model : sklearn estimator, optional
        Pre-loaded model.

    Returns
    -------
    dict
        Same structure as :func:`predict_blood`.
    """
    if model is None:
        model = load_symptom_model()

    input_df = preprocess_symptom_input(user_input)

    # --- no-symptom short-circuit ----------------------------------------
    symptom_keys = [k for k in SYMPTOM_FEATURES if k not in ('Gender', 'Age')]
    all_zero = all(user_input.get(k, 0) == 0 for k in symptom_keys)

    if all_zero:
        return {
            'prediction': 0,
            'probability': 0.02,
            'risk_level': 'Low',
            'interpretation': (
                "No symptoms or risk factors reported. "
                "Based on the information provided, anemia risk is very low."
            ),
            'symptom_summary': _summarise_symptoms(user_input),
        }

    prediction = int(model.predict(input_df)[0])
    proba = float(model.predict_proba(input_df)[0][1])

    # Clinical knowledge adjustments
    key_symptoms = [
        'Pale_Skin', 'Fatigue', 'Shortness_of_Breath',
        'Heavy_Periods', 'Recent_Blood_Loss',
    ]
    key_count = sum(user_input.get(k, 0) for k in key_symptoms)
    total_count = sum(user_input.get(k, 0) for k in symptom_keys)

    if prediction == 0 and key_count >= 3:
        proba = max(proba, 0.40)
        if proba > 0.5:
            prediction = 1

    if prediction == 1 and total_count <= 2:
        proba = min(proba, 0.70)

    risk_level = _stratify_risk(proba)
    symptom_summary = _summarise_symptoms(user_input)
    interpretation = _build_symptom_interpretation(
        prediction, proba, risk_level, symptom_summary
    )

    return {
        'prediction': prediction,
        'probability': round(proba, 4),
        'risk_level': risk_level,
        'interpretation': interpretation,
        'symptom_summary': symptom_summary,
    }


# ---------------------------------------------------------------------------
# Combined / ensemble prediction
# ---------------------------------------------------------------------------

def predict_combined(blood_input: dict, symptom_input: dict,
                     blood_model=None, symptom_model=None,
                     blood_weight: float = 0.65,
                     symptom_weight: float = 0.35):
    """
    Combine blood-test and symptom-based predictions.

    The weighted average of probabilities is used as the final score.

    Parameters
    ----------
    blood_input, symptom_input : dict
    blood_model, symptom_model : sklearn estimator, optional
    blood_weight, symptom_weight : float
        Must sum to 1.0.

    Returns
    -------
    dict
        Merged prediction result with individual results included.
    """
    blood_result = predict_blood(blood_input, model=blood_model)
    symptom_result = predict_symptoms(symptom_input, model=symptom_model)

    combined_prob = (
        blood_weight * blood_result['probability']
        + symptom_weight * symptom_result['probability']
    )
    combined_pred = 1 if combined_prob >= 0.5 else 0
    risk_level = _stratify_risk(combined_prob)

    interpretation = (
        f"Combined analysis (blood tests {blood_weight:.0%}, "
        f"symptoms {symptom_weight:.0%}) yields a "
        f"{risk_level.lower()}-risk assessment.\n\n"
        f"• Blood-test probability: {blood_result['probability']:.1%}\n"
        f"• Symptom probability: {symptom_result['probability']:.1%}\n"
        f"• Combined probability: {combined_prob:.1%}\n\n"
    )
    if combined_pred == 1:
        interpretation += (
            "The combined assessment suggests possible anemia. "
            "A Complete Blood Count (CBC) is recommended for confirmation."
        )
    else:
        interpretation += (
            "The combined assessment does not strongly indicate anemia, "
            "but consult a healthcare professional if symptoms persist."
        )

    return {
        'prediction': combined_pred,
        'probability': round(combined_prob, 4),
        'risk_level': risk_level,
        'interpretation': interpretation,
        'blood_result': blood_result,
        'symptom_result': symptom_result,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _stratify_risk(probability: float) -> str:
    """Return risk category string from probability."""
    if probability < 0.30:
        return 'Low'
    elif probability < 0.70:
        return 'Moderate'
    else:
        return 'High'


def _analyse_blood_parameters(user_input: dict, gender_key: str) -> list:
    """
    Compare each hematological parameter to its gender-specific normal range.

    Returns a list of dicts with keys:
        parameter, value, normal_range, status, unit
    """
    results = []
    for param, ranges in NORMAL_RANGES.items():
        value = user_input.get(param)
        if value is None:
            continue
        value = float(value)

        if 'all' in ranges:
            low, high = ranges['all']
        else:
            low, high = ranges[gender_key]

        if value < low:
            status = 'Below Normal'
        elif value > high:
            status = 'Above Normal'
        else:
            status = 'Normal'

        unit = FEATURES.get(param, {}).get('unit', '')
        results.append({
            'parameter': param,
            'value': value,
            'normal_range': f"{low} – {high}",
            'status': status,
            'unit': unit,
        })

    return results


def _build_blood_interpretation(prediction, probability, risk_level, param_analysis):
    """Build a human-readable interpretation for blood-test prediction."""
    abnormal = [p for p in param_analysis if p['status'] != 'Normal']

    lines = [
        f"Risk Level: {risk_level} ({probability:.1%} probability of anemia).\n"
    ]

    if prediction == 1:
        lines.append(
            "The blood-test parameters suggest a likelihood of anemia."
        )
    else:
        lines.append(
            "The blood-test parameters do not strongly suggest anemia."
        )

    if abnormal:
        lines.append("\nAbnormal parameters detected:")
        for p in abnormal:
            lines.append(
                f"  • {p['parameter']}: {p['value']} {p['unit']} "
                f"({p['status']}; normal: {p['normal_range']} {p['unit']})"
            )

    lines.append(
        "\n⚠ This is a screening tool only. Please consult a healthcare "
        "professional for diagnosis and treatment."
    )

    return "\n".join(lines)


def _summarise_symptoms(user_input: dict) -> dict:
    """Return summary of reported symptoms."""
    symptom_labels = {
        'Fatigue': 'Fatigue',
        'Shortness_of_Breath': 'Shortness of Breath',
        'Dizziness': 'Dizziness',
        'Pale_Skin': 'Pale Skin',
        'Heart_Racing': 'Heart Racing / Palpitations',
        'Headaches': 'Headaches',
        'Cold_Hands_Feet': 'Cold Hands & Feet',
        'Brittle_Nails': 'Brittle Nails',
        'Poor_Concentration': 'Poor Concentration',
        'Heavy_Periods': 'Heavy Menstrual Periods',
        'Recent_Blood_Loss': 'Recent Blood Loss',
        'Vegetarian_Diet': 'Vegetarian / Low-Iron Diet',
        'GI_Disorders': 'Gastrointestinal Disorders',
        'Chronic_Disease': 'Chronic Disease',
    }
    present = [
        label for key, label in symptom_labels.items()
        if user_input.get(key, 0) == 1
    ]
    return {
        'reported_symptoms': present,
        'total_count': len(present),
    }


def _build_symptom_interpretation(prediction, probability, risk_level,
                                  symptom_summary):
    """Build a human-readable interpretation for symptom-based prediction."""
    count = symptom_summary['total_count']
    lines = [
        f"Risk Level: {risk_level} ({probability:.1%} probability of anemia).",
        f"Symptoms / risk factors reported: {count}.",
    ]

    if symptom_summary['reported_symptoms']:
        lines.append("\nReported symptoms:")
        for s in symptom_summary['reported_symptoms']:
            lines.append(f"  • {s}")

    if prediction == 1:
        lines.append(
            "\nBased on the reported symptoms, there is a notable risk of "
            "anemia. A Complete Blood Count (CBC) test is recommended."
        )
    else:
        lines.append(
            "\nThe reported symptoms alone do not strongly indicate anemia, "
            "but if symptoms persist please seek medical advice."
        )

    lines.append(
        "\n⚠ This is a screening tool only. Please consult a healthcare "
        "professional for diagnosis and treatment."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  MedInsight AI — Anemia Predictor (CLI Demo)")
    print("=" * 60)

    # Demo: blood-test prediction
    sample_blood = {
        'Gender': 'Female',
        'Age': 45,
        'Hb': 9.5,
        'RBC': 3.8,
        'PCV': 30.0,
        'MCV': 75.0,
        'MCH': 24.0,
        'MCHC': 31.0,
    }

    print("\n--- Blood-Test Prediction ---")
    try:
        result = predict_blood(sample_blood)
        print(result['interpretation'])
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")

    # Demo: symptom-based prediction
    sample_symptoms = {
        'Gender': 1,
        'Age': 45,
        'Fatigue': 1,
        'Shortness_of_Breath': 1,
        'Dizziness': 1,
        'Pale_Skin': 1,
        'Heart_Racing': 0,
        'Headaches': 1,
        'Cold_Hands_Feet': 1,
        'Brittle_Nails': 0,
        'Poor_Concentration': 1,
        'Heavy_Periods': 1,
        'Recent_Blood_Loss': 0,
        'Vegetarian_Diet': 0,
        'GI_Disorders': 0,
        'Chronic_Disease': 0,
    }

    print("\n--- Symptom-Based Prediction ---")
    try:
        result = predict_symptoms(sample_symptoms)
        print(result['interpretation'])
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
