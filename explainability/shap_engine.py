import shap
import matplotlib.pyplot as plt


class SHAPEngine:

    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain(self, input_df):
        return self.explainer(
            input_df,
            check_additivity=False
        )

    def summary_plot(self, input_df):
        shap_values = self.explain(input_df)

        fig = plt.figure(figsize=(8,5))

        shap.plots.bar(
            shap_values,
            show=False
        )

        return fig

    def waterfall_plot(self, input_df):
        shap_values = self.explain(input_df)

        fig = plt.figure(figsize=(10,6))

        shap.plots.waterfall(
            shap_values[0],
            show=False
        )

        return fig