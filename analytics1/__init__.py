"""
Public API for the analytics1 package.
"""

from .statistics import (
    load_all_datasets,
    get_kpi_summary,
    get_dataset_stats,
    get_ai_insights,
    MODEL_ACCURACIES,
)

from .charts import (
    create_pie_chart,
    create_accuracy_bar_chart,
)

__all__ = [
    "load_all_datasets",
    "get_kpi_summary",
    "get_dataset_stats",
    "get_ai_insights",
    "MODEL_ACCURACIES",
    "create_pie_chart",
    "create_accuracy_bar_chart",
]