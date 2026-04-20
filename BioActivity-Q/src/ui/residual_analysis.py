"""Four-panel residual analysis for model diagnostics."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def residual_dashboard(y_true, y_pred, mw_values=None):
    """
    Four-panel residual analysis:
    1. Residual vs Predicted (heteroscedasticity check)
    2. Residual Distribution (normality check)
    3. Residual vs Molecular Weight (MW bias check)
    4. Absolute Error by Activity Class
    """
    residuals = y_true - y_pred

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Residuals vs Predicted",
            "Residual Distribution",
            "Residuals vs Molecular Weight",
            "Absolute Error by Activity Class",
        ],
    )

    # Panel 1: Residual vs Predicted
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode="markers",
        marker=dict(size=5, opacity=0.5, color="steelblue"),
        name="Residuals",
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Panel 2: Histogram
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=40, marker_color="steelblue",
        name="Distribution",
    ), row=1, col=2)

    # Panel 3: Residual vs MW
    if mw_values is not None and len(mw_values) == len(residuals):
        fig.add_trace(go.Scatter(
            x=mw_values, y=residuals, mode="markers",
            marker=dict(size=5, opacity=0.5, color="coral"),
            name="MW vs Residual",
        ), row=2, col=1)
    else:
        fig.add_annotation(
            text="MW data not provided", row=2, col=1,
            xref="x3", yref="y3", x=0.5, y=0.5, showarrow=False,
        )

    # Panel 4: Error by activity class
    classes = pd.cut(
        y_true, bins=[0, 5, 6, 7, 12],
        labels=["Inactive", "Moderate", "Active", "Highly Active"],
    )
    abs_err = np.abs(residuals)
    for cls in ["Inactive", "Moderate", "Active", "Highly Active"]:
        mask = classes == cls
        if mask.sum() > 0:
            fig.add_trace(go.Box(
                y=abs_err[mask], name=cls, boxmean=True,
            ), row=2, col=2)

    fig.update_layout(height=800, showlegend=False, title_text="Residual Analysis")
    return fig


# Convenience alias for backward compatibility
plot_residuals = residual_dashboard
