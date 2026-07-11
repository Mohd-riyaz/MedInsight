import joblib
import shap
import matplotlib.pyplot as plt
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "anemia_prediction_model.pkl"
)

model = joblib.load(MODEL_PATH)

explainer = shap.TreeExplainer(model)


def explain_prediction(input_df):
    shap_values = explainer(
        input_df,
        check_additivity=False
    )

    # Select positive class (Anemia)
    return shap_values[:, :, 1]


def summary_plot(input_df):
    shap_values = explain_prediction(input_df)

    plt.close("all")

    shap.plots.bar(
        shap_values,
        show=False
    )

    return plt.gcf()


def waterfall_plot(input_df):
    shap_values = explain_prediction(input_df)

    plt.close("all")

    shap.plots.waterfall(
        shap_values[0],
        show=False
    )

    return plt.gcf()
