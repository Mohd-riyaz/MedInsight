import joblib
import shap
import matplotlib.pyplot as plt
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "diabetes_model.pkl"
)

model = joblib.load(MODEL_PATH)

explainer = shap.TreeExplainer(model)


def explain_prediction(input_df):
    shap_values = explainer(
        input_df,
        check_additivity=False
    )
    return shap_values


def summary_plot(input_df):
    shap_values = explain_prediction(input_df)

    plt.close("all")

    if len(shap_values.shape) == 3:
        explanation = shap_values[0, :, 1]
    else:
        explanation = shap_values[0]

    shap.plots.bar(
        explanation,
        show=False
    )

    return plt.gcf()


def waterfall_plot(input_df):
    shap_values = explain_prediction(input_df)

    plt.close("all")

    if len(shap_values.shape) == 3:
        explanation = shap_values[0, :, 1]
    else:
        explanation = shap_values[0]

    shap.plots.waterfall(
        explanation,
        show=False
    )

    return plt.gcf()
