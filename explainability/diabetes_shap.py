import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load(
    "modules/diabetes/models/diabetes_model.pkl"
)

explainer = shap.TreeExplainer(model)


def explain_prediction(input_df):
    shap_values = explainer(
        input_df,
        check_additivity=False
    )
    return shap_values


def summary_plot(input_df):
    shap_values = explain_prediction(input_df)

    fig = plt.figure(figsize=(8, 5))
    shap.plots.bar(
        shap_values,
        show=False
    )
    return fig


def waterfall_plot(input_df):
    shap_values = explain_prediction(input_df)

    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(
        shap_values[0],
        show=False
    )
    return fig