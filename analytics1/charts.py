

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Shared theme tokens (mirrors the CSS in app.py)
# ---------------------------------------------------------------------------
_BG_COLOR = "rgba(0,0,0,0)"          # transparent to inherit page bg
_PAPER_COLOR = "rgba(0,0,0,0)"
_FONT_COLOR = "#e2e8f0"
_GRID_COLOR = "rgba(139,92,246,0.12)"

# Accent colours per disease — pulled from the existing card accent bars
_COLORS = {
    "Anemia": "#f87171",        # red-400
    "Diabetes": "#60a5fa",      # blue-400
    "Heart Disease": "#c084fc", # purple-400
}

_PIE_COLORS = ["#f87171", "#60a5fa", "#c084fc"]
_BAR_COLORS = ["#fb923c", "#34d399", "#f472b6"]  # orange, emerald, pink


# ---------------------------------------------------------------------------
# Pie chart — Dataset distribution
# ---------------------------------------------------------------------------
def create_pie_chart(dataset_counts: dict[str, int]) -> go.Figure:
    """Return a Plotly pie chart of dataset record distribution.

    Parameters
    ----------
    dataset_counts : dict
        Mapping of dataset name → record count
        (e.g. {"Anemia": 1421, "Diabetes": 768, "Heart Disease": 920}).
    """
    labels = list(dataset_counts.keys())
    values = list(dataset_counts.values())

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(
                    colors=_PIE_COLORS[: len(labels)],
                    line=dict(color="#1e1b4b", width=2),
                ),
                textinfo="label+percent",
                textfont=dict(size=13, color=_FONT_COLOR),
                hovertemplate="<b>%{label}</b><br>Records: %{value:,}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="Disease Dataset Distribution",
            font=dict(size=16, color=_FONT_COLOR),
            x=0.5,
        ),
        paper_bgcolor=_PAPER_COLOR,
        plot_bgcolor=_BG_COLOR,
        font=dict(family="Inter, sans-serif", color=_FONT_COLOR),
        legend=dict(
            font=dict(color="#94a3b8", size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=50, b=20, l=20, r=20),
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Bar chart — Model accuracy comparison
# ---------------------------------------------------------------------------
def create_accuracy_bar_chart(accuracies: dict[str, float]) -> go.Figure:
    """Return a Plotly bar chart comparing ML model accuracies.

    Parameters
    ----------
    accuracies : dict
        Mapping of model/disease name → accuracy percentage
        (e.g. {"Anemia": 99.0, "Diabetes": 88.31, "Heart Disease": 84.24}).
    """
    names = list(accuracies.keys())
    values = list(accuracies.values())

    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=values,
                marker=dict(
                    color=_BAR_COLORS[: len(names)],
                    line=dict(color="rgba(255,255,255,0.1)", width=1),
                    cornerradius=6,
                ),
                text=[f"{v:.2f}%" for v in values],
                textposition="outside",
                textfont=dict(size=13, color=_FONT_COLOR),
                hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.2f}%<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="Model Accuracy Comparison",
            font=dict(size=16, color=_FONT_COLOR),
            x=0.5,
        ),
        xaxis=dict(
            color="#94a3b8",
            showgrid=False,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            color="#94a3b8",
            range=[0, 110],
            showgrid=True,
            gridcolor=_GRID_COLOR,
            ticksuffix="%",
            tickfont=dict(size=11),
        ),
        paper_bgcolor=_PAPER_COLOR,
        plot_bgcolor=_BG_COLOR,
        font=dict(family="Inter, sans-serif", color=_FONT_COLOR),
        margin=dict(t=50, b=20, l=50, r=20),
        height=380,
        bargap=0.35,
    )
    return fig
import plotly.express as px


def create_histogram(df, feature):
    """Histogram for a selected numeric feature."""

    fig = px.histogram(
        df,
        x=feature,
        nbins=30,
        color_discrete_sequence=["#60a5fa"]
    )

    fig.update_layout(
        title=f"{feature} Distribution",
        paper_bgcolor=_PAPER_COLOR,
        plot_bgcolor=_BG_COLOR,
        font=dict(color=_FONT_COLOR),
        xaxis_title=feature,
        yaxis_title="Count",
        height=400,
    )

    return fig

def create_box_plot(df, feature):

    fig = px.box(
        df,
        y=feature,
        color_discrete_sequence=["#c084fc"]
    )

    fig.update_layout(
        title=f"{feature} Box Plot",
        paper_bgcolor=_PAPER_COLOR,
        plot_bgcolor=_BG_COLOR,
        font=dict(color=_FONT_COLOR),
        height=400,
    )

    return fig

def create_correlation_heatmap(corr):

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r"
    )

    fig.update_layout(
        title="Correlation Heatmap",
        paper_bgcolor=_PAPER_COLOR,
        plot_bgcolor=_BG_COLOR,
        font=dict(color=_FONT_COLOR),
        height=650,
    )

    return fig

def create_missing_values_chart(missing):

    missing = missing[missing > 0]

    fig = px.bar(
        x=missing.index,
        y=missing.values,
        color=missing.values,
        color_continuous_scale="Reds"
    )

    fig.update_layout(
        title="Missing Values",
        paper_bgcolor=_PAPER_COLOR,
        plot_bgcolor=_BG_COLOR,
        font=dict(color=_FONT_COLOR),
        xaxis_title="Feature",
        yaxis_title="Missing",
        height=350,
    )

    return fig

def create_class_distribution(distribution):

    fig = px.bar(
        x=distribution.index.astype(str),
        y=distribution.values,
        color=distribution.values,
        color_continuous_scale="Viridis"
    )

    fig.update_layout(
        title="Target Class Distribution",
        paper_bgcolor=_PAPER_COLOR,
        plot_bgcolor=_BG_COLOR,
        font=dict(color=_FONT_COLOR),
        xaxis_title="Class",
        yaxis_title="Count",
        height=350,
    )

    return fig
def create_feature_importance(feature_scores: dict):

    names = list(feature_scores.keys())
    values = list(feature_scores.values())

    fig = px.bar(
        x=values,
        y=names,
        orientation="h",
        color=values,
        color_continuous_scale="Turbo"
    )

    fig.update_layout(
        title="Feature Importance",
        paper_bgcolor=_PAPER_COLOR,
        plot_bgcolor=_BG_COLOR,
        font=dict(color=_FONT_COLOR),
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=450,
    )

    return fig