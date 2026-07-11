"""
predictor.py - Heart disease prediction engine

Loads the pre-trained heart disease model (trained on UCI Heart Disease
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
MODEL_PATH = os.path.join(MODULE_DIR, "models", "heart_model.pkl")
DATA_PATH = os.path.join(MODULE_DIR, "data", "heart_disease_uci.csv")

# ---------------------------------------------------------------------------
# Feature definitions (UCI Heart Disease dataset)
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    'id', 'age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num',
]

# Categorical value mappings
CP_MAPPING = {
    'typical angina': 0,
    'atypical angina': 1,
    'non-anginal': 2,
    'asymptomatic': 3,
}

RESTECG_MAPPING = {
    'normal': 0,
    'st-t abnormality': 1,
    'lv hypertrophy': 2,
}

SLOPE_MAPPING = {
    'upsloping': 0,
    'flat': 1,
    'downsloping': 2,
}

THAL_MAPPING = {
    'normal': 0,
    'fixed defect': 1,
    'reversable defect': 2,
    'reversible defect': 2,  # common alternate spelling
}

# Features used by the trained model (after encoding)
MODEL_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
]

# Clinical reference ranges
NORMAL_RANGES = {
    'trestbps': {
        'label': 'Resting Blood Pressure',
        'unit': 'mm Hg',
        'normal': (90, 120),
        'elevated': (120, 129),
        'high_stage1': (130, 139),
        'high_stage2': (140, None),
    },
    'chol': {
        'label': 'Serum Cholesterol',
        'unit': 'mg/dL',
        'desirable': (0, 200),
        'borderline': (200, 239),
        'high': (240, None),
    },
    'thalch': {
        'label': 'Maximum Heart Rate Achieved',
        'unit': 'bpm',
        'note': 'Expected max ≈ 220 - age',
    },
    'oldpeak': {
        'label': 'ST Depression',
        'unit': 'mm',
        'normal': (0, 1.0),
        'abnormal': (1.0, None),
    },
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str = MODEL_PATH):
    """
    Load the pre-trained heart disease prediction model.

    Returns
    -------
    model : sklearn estimator / pipeline
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Heart model not found at {model_path}. "
            "Please run the training notebook first."
        )
    return joblib.load(model_path)


def load_dataset(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the UCI Heart Disease CSV dataset.

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

def _encode_categorical(value, mapping: dict, default: int = 0):
    """Safely encode a categorical value using the provided mapping."""
    if isinstance(value, (int, float)):
        if np.isnan(value):
            return default
        return int(value)
    if isinstance(value, str):
        return mapping.get(value.lower().strip(), default)
    return default


def _encode_boolean(value, true_values=('true', 'yes', '1', 1, True)):
    """Encode boolean / TRUE-FALSE string values to 0/1."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        return 1 if value.strip().lower() in ('true', 'yes', '1') else 0
    return 0


def _encode_sex(value):
    """Encode sex: Male → 1, Female → 0."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        return 1 if value.strip().lower() in ('male', 'm', '1') else 0
    return 0


def preprocess_input(user_input: dict) -> pd.DataFrame:
    """
    Convert a dictionary of patient data into a model-ready DataFrame.

    Handles both raw string values (from forms) and pre-encoded numeric
    values.

    Parameters
    ----------
    user_input : dict
        Patient data.  Keys should match UCI feature names.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns in ``MODEL_FEATURES`` order.
    """
    processed = {}

    # --- Numerical features ---
    processed['age'] = float(user_input.get('age', 50))
    processed['trestbps'] = float(user_input.get('trestbps', 120))
    processed['chol'] = float(user_input.get('chol', 200))
    processed['thalch'] = float(user_input.get('thalch', 150))
    processed['oldpeak'] = float(user_input.get('oldpeak', 0.0))
    processed['ca'] = float(user_input.get('ca', 0))

    # --- Boolean / binary features ---
    processed['sex'] = _encode_sex(user_input.get('sex', 0))
    processed['fbs'] = _encode_boolean(user_input.get('fbs', 0))
    processed['exang'] = _encode_boolean(user_input.get('exang', 0))

    # --- Categorical features ---
    processed['cp'] = _encode_categorical(
        user_input.get('cp', 0), CP_MAPPING, default=0
    )
    processed['restecg'] = _encode_categorical(
        user_input.get('restecg', 0), RESTECG_MAPPING, default=0
    )
    processed['slope'] = _encode_categorical(
        user_input.get('slope', 0), SLOPE_MAPPING, default=0
    )
    processed['thal'] = _encode_categorical(
        user_input.get('thal', 0), THAL_MAPPING, default=0
    )

    df = pd.DataFrame([processed])
    return df[MODEL_FEATURES]


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the full UCI Heart Disease dataset for training or batch
    prediction.  Encodes all categorical/boolean columns, handles missing
    values, and binarises the multi-class target ``num`` into 0/1.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with numeric columns + binary target.
    """
    data = df.copy()

    # Encode sex
    data['sex'] = data['sex'].apply(_encode_sex)

    # Encode categoricals
    data['cp'] = data['cp'].apply(lambda x: _encode_categorical(x, CP_MAPPING))
    data['restecg'] = data['restecg'].apply(
        lambda x: _encode_categorical(x, RESTECG_MAPPING)
    )
    data['slope'] = data['slope'].apply(
        lambda x: _encode_categorical(x, SLOPE_MAPPING)
    )
    data['thal'] = data['thal'].apply(
        lambda x: _encode_categorical(x, THAL_MAPPING)
    )

    # Encode booleans
    data['fbs'] = data['fbs'].apply(_encode_boolean)
    data['exang'] = data['exang'].apply(_encode_boolean)

    # Binarise target: 0 → no disease, 1-4 → disease
    data['target'] = (data['num'] > 0).astype(int)

    # Handle missing values — fill with median
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    # Drop non-feature columns
    drop_cols = [c for c in ['id', 'dataset', 'num'] if c in data.columns]
    data = data.drop(columns=drop_cols, errors='ignore')

    return data


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(user_input: dict, model=None) -> dict:
    """
    Predict heart disease risk from patient clinical parameters.

    Parameters
    ----------
    user_input : dict
        Patient data with keys matching UCI Heart Disease features.
    model : sklearn estimator, optional
        Pre-loaded model.

    Returns
    -------
    dict
        ``prediction``    : int (0 = No Heart Disease, 1 = Heart Disease)
        ``probability``   : float (0-1)
        ``risk_level``    : str ('Low' / 'Moderate' / 'High')
        ``risk_factors``  : list[dict]
        ``interpretation``: str
        ``recommendations``: list[str]
    """
    if model is None:
        model = load_model()

    input_df = preprocess_input(user_input)

    prediction = int(model.predict(input_df)[0])

    # Probability estimation
    if hasattr(model, 'predict_proba'):
        proba = float(model.predict_proba(input_df)[0][1])
    elif hasattr(model, 'decision_function'):
        decision = float(model.decision_function(input_df)[0])
        proba = 1 / (1 + np.exp(-decision))
    else:
        proba = float(prediction)

    # --- Clinical rule adjustments ----------------------------------------
    age = float(user_input.get('age', 50))
    chol = float(user_input.get('chol', 200))
    trestbps = float(user_input.get('trestbps', 120))
    oldpeak = float(user_input.get('oldpeak', 0))
    cp = user_input.get('cp', 0)

    # Asymptomatic chest pain + ST depression is a red flag
    cp_val = _encode_categorical(cp, CP_MAPPING) if isinstance(cp, str) else int(cp)
    if cp_val == 3 and oldpeak >= 2.0:
        proba = max(proba, 0.65)
        prediction = 1

    # Very high cholesterol + high BP
    if chol >= 300 and trestbps >= 140:
        proba = max(proba, 0.55)
        prediction = 1

    # Elderly with multiple risk factors
    risk_count = sum([
        chol >= 240,
        trestbps >= 140,
        oldpeak >= 2.0,
        age >= 60,
        _encode_boolean(user_input.get('exang', 0)) == 1,
    ])
    if risk_count >= 3:
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
        'input_df': input_df
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


def _classify_bp(value: float) -> str:
    if value < 120:
        return 'Normal'
    elif value < 130:
        return 'Elevated'
    elif value < 140:
        return 'High (Stage 1)'
    else:
        return 'High (Stage 2)'


def _classify_cholesterol(value: float) -> str:
    if value < 200:
        return 'Desirable'
    elif value < 240:
        return 'Borderline High'
    else:
        return 'High'


def _analyse_risk_factors(user_input: dict) -> list:
    """Identify and categorise individual cardiac risk factors."""
    factors = []

    age = float(user_input.get('age', 0))
    sex = _encode_sex(user_input.get('sex', 0))
    chol = float(user_input.get('chol', 0))
    trestbps = float(user_input.get('trestbps', 0))
    fbs = _encode_boolean(user_input.get('fbs', 0))
    oldpeak = float(user_input.get('oldpeak', 0))
    thalch = float(user_input.get('thalch', 0))
    exang = _encode_boolean(user_input.get('exang', 0))
    cp = user_input.get('cp', 0)
    ca = float(user_input.get('ca', 0))

    # Age
    risk_age = 55 if sex == 0 else 45
    if age >= risk_age:
        factors.append({
            'factor': f'Age ≥ {risk_age}',
            'value': int(age),
            'unit': 'years',
            'category': 'Age-related cardiovascular risk',
            'severity': 'moderate',
        })

    # Cholesterol
    chol_cat = _classify_cholesterol(chol)
    if chol_cat in ('Borderline High', 'High'):
        factors.append({
            'factor': 'Elevated Cholesterol',
            'value': chol,
            'unit': 'mg/dL',
            'category': chol_cat,
            'severity': 'high' if chol_cat == 'High' else 'moderate',
        })

    # Blood pressure
    bp_cat = _classify_bp(trestbps)
    if bp_cat != 'Normal':
        factors.append({
            'factor': 'Elevated Resting Blood Pressure',
            'value': trestbps,
            'unit': 'mm Hg',
            'category': bp_cat,
            'severity': 'high' if 'Stage 2' in bp_cat else 'moderate',
        })

    # Fasting blood sugar
    if fbs == 1:
        factors.append({
            'factor': 'Fasting Blood Sugar > 120 mg/dL',
            'value': '> 120',
            'unit': 'mg/dL',
            'category': 'Diabetic / pre-diabetic indicator',
            'severity': 'moderate',
        })

    # Exercise-induced angina
    if exang == 1:
        factors.append({
            'factor': 'Exercise-Induced Angina',
            'value': 'Present',
            'unit': '',
            'category': 'Significant ischemic indicator',
            'severity': 'high',
        })

    # ST depression
    if oldpeak >= 1.0:
        factors.append({
            'factor': 'ST Depression (Oldpeak)',
            'value': oldpeak,
            'unit': 'mm',
            'category': 'Abnormal' if oldpeak >= 2.0 else 'Borderline',
            'severity': 'high' if oldpeak >= 2.0 else 'moderate',
        })

    # Chest pain type
    cp_val = _encode_categorical(cp, CP_MAPPING) if isinstance(cp, str) else int(cp)
    if cp_val == 3:  # asymptomatic
        factors.append({
            'factor': 'Asymptomatic Chest Pain Type',
            'value': 'Asymptomatic',
            'unit': '',
            'category': 'Often associated with silent ischemia',
            'severity': 'high',
        })

    # Major vessels coloured by fluoroscopy
    if ca >= 1:
        factors.append({
            'factor': 'Major Vessels Coloured by Fluoroscopy',
            'value': int(ca),
            'unit': 'vessels',
            'category': f'{int(ca)} vessel(s) affected',
            'severity': 'high' if ca >= 2 else 'moderate',
        })

    # Low max heart rate (below expected)
    expected_max_hr = 220 - age
    if thalch > 0 and thalch < 0.75 * expected_max_hr:
        factors.append({
            'factor': 'Reduced Max Heart Rate',
            'value': int(thalch),
            'unit': 'bpm',
            'category': f'Below 75% of expected max ({int(expected_max_hr)} bpm)',
            'severity': 'moderate',
        })

    return factors


# ---------------------------------------------------------------------------
# Interpretation and recommendations
# ---------------------------------------------------------------------------

def _build_interpretation(prediction, probability, risk_level,
                          risk_factors, user_input):
    lines = [
        f"Risk Level: {risk_level} ({probability:.1%} probability of heart disease)."
    ]

    if prediction == 1:
        lines.append(
            "The clinical parameters suggest an elevated risk of heart disease."
        )
    else:
        lines.append(
            "The clinical parameters do not strongly indicate heart disease "
            "at this time."
        )

    # Key metric summaries
    chol = float(user_input.get('chol', 0))
    trestbps = float(user_input.get('trestbps', 0))
    lines.append(
        f"\nCholesterol: {chol} mg/dL ({_classify_cholesterol(chol)})."
    )
    lines.append(
        f"Resting BP: {trestbps} mm Hg ({_classify_bp(trestbps)})."
    )

    if risk_factors:
        lines.append(f"\nIdentified risk factors ({len(risk_factors)}):")
        for rf in risk_factors:
            lines.append(
                f"  • {rf['factor']}: {rf['value']} {rf['unit']} — {rf['category']}"
            )

    lines.append(
        "\n⚠ This is a screening tool only. Please consult a cardiologist "
        "for proper evaluation, ECG, stress testing, and imaging."
    )
    return "\n".join(lines)


def _generate_recommendations(prediction, risk_level, risk_factors,
                               user_input):
    recs = []

    if prediction == 1 or risk_level in ('Moderate', 'High'):
        recs.append(
            "Schedule a comprehensive cardiac evaluation including ECG, "
            "echocardiogram, and stress test."
        )

    # Cholesterol
    chol = float(user_input.get('chol', 0))
    if chol >= 200:
        recs.append(
            "Work with your doctor on cholesterol management — dietary "
            "changes, exercise, and potentially statin therapy."
        )

    # Blood pressure
    trestbps = float(user_input.get('trestbps', 0))
    if trestbps >= 130:
        recs.append(
            "Monitor blood pressure regularly. Adopt the DASH diet, reduce "
            "sodium, and discuss antihypertensive medications with your doctor."
        )

    # Fasting blood sugar
    fbs = _encode_boolean(user_input.get('fbs', 0))
    if fbs == 1:
        recs.append(
            "Elevated fasting blood sugar may indicate diabetes, which "
            "significantly increases cardiovascular risk. Get an HbA1c test."
        )

    # Exercise-induced angina
    exang = _encode_boolean(user_input.get('exang', 0))
    if exang == 1:
        recs.append(
            "Exercise-induced angina requires immediate cardiac evaluation. "
            "Avoid strenuous activity until cleared by a cardiologist."
        )

    # Lifestyle
    recs.append(
        "Adopt a heart-healthy lifestyle: regular aerobic exercise "
        "(150+ min/week), Mediterranean-style diet, stress management, "
        "and smoking cessation."
    )

    recs.append(
        "Know the warning signs of a heart attack: chest pain or pressure, "
        "shortness of breath, pain radiating to arm/jaw, nausea. "
        "Call emergency services immediately if these occur."
    )

    return recs


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def predict_batch(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Run predictions on an entire preprocessed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns matching ``MODEL_FEATURES``.
    model : sklearn estimator, optional

    Returns
    -------
    pd.DataFrame
        Original DataFrame with ``Prediction``, ``Probability``,
        ``Risk_Level`` columns appended.
    """
    if model is None:
        model = load_model()

    result_df = df.copy()
    features = df[MODEL_FEATURES].copy()

    # Fill NaN with median
    features = features.fillna(features.median())

    result_df['Prediction'] = model.predict(features)

    if hasattr(model, 'predict_proba'):
        result_df['Probability'] = model.predict_proba(features)[:, 1]
    elif hasattr(model, 'decision_function'):
        decisions = model.decision_function(features)
        result_df['Probability'] = 1 / (1 + np.exp(-decisions))
    else:
        result_df['Probability'] = result_df['Prediction'].astype(float)

    result_df['Risk_Level'] = result_df['Probability'].apply(_stratify_risk)

    return result_df


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def get_dataset_statistics(df: pd.DataFrame = None) -> dict:
    """
    Compute summary statistics from the heart disease dataset.

    Returns
    -------
    dict with keys: shape, class_distribution, feature_stats
    """
    if df is None:
        df = load_dataset()

    # Preprocess to get numeric columns
    processed = preprocess_dataset(df)

    stats = {
        'shape': df.shape,
        'class_distribution': processed['target'].value_counts().to_dict(),
        'age_stats': {
            'mean': float(processed['age'].mean()),
            'std': float(processed['age'].std()),
            'min': float(processed['age'].min()),
            'max': float(processed['age'].max()),
        },
        'sex_distribution': processed['sex'].value_counts().to_dict(),
        'feature_stats': processed[MODEL_FEATURES].describe().to_dict(),
    }
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  MedInsight AI — Heart Disease Predictor (CLI Demo)")
    print("=" * 60)

    # High-risk patient
    sample_high_risk = {
        'age': 63,
        'sex': 'Male',
        'cp': 'asymptomatic',
        'trestbps': 145,
        'chol': 233,
        'fbs': True,
        'restecg': 'lv hypertrophy',
        'thalch': 150,
        'exang': False,
        'oldpeak': 2.3,
        'slope': 'downsloping',
        'ca': 0,
        'thal': 'fixed defect',
    }

    print("\n--- High-Risk Patient ---")
    try:
        result = predict(sample_high_risk)
        print(f"Prediction: {'Heart Disease' if result['prediction'] == 1 else 'No Heart Disease'}")
        print(f"Probability: {result['probability']:.1%}")
        print(f"\n{result['interpretation']}")
        print("\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")

    # Low-risk patient
    print("\n" + "-" * 60)
    sample_low_risk = {
        'age': 35,
        'sex': 'Female',
        'cp': 'non-anginal',
        'trestbps': 110,
        'chol': 180,
        'fbs': False,
        'restecg': 'normal',
        'thalch': 175,
        'exang': False,
        'oldpeak': 0.0,
        'slope': 'upsloping',
        'ca': 0,
        'thal': 'normal',
    }

    print("\n--- Low-Risk Patient ---")
    try:
        result = predict(sample_low_risk)
        print(f"Prediction: {'Heart Disease' if result['prediction'] == 1 else 'No Heart Disease'}")
        print(f"Probability: {result['probability']:.1%}")
        print(f"\n{result['interpretation']}")
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
