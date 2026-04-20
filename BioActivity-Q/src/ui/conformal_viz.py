"""Conformal prediction interval visualization."""
import numpy as np
import plotly.graph_objects as go


def plot_conformal_intervals(
    y_true, y_pred, lo, hi, sort_by="width", max_points=100
):
    """
    Horizontal interval plot: each molecule is a row.
    Blue dot = prediction, red X = true value, light blue bar = 90% CI.
    Sorted by interval width to show confident vs uncertain predictions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    widths = hi - lo

    if sort_by == "width":
        order = np.argsort(widths)
    elif sort_by == "error":
        order = np.argsort(np.abs(y_true - y_pred))
    else:
        order = np.arange(len(y_true))

    if len(order) > max_points:
        step = len(order) // max_points
        order = order[::step]

    fig = go.Figure()

    # CI bars
    for i, idx in enumerate(order):
        fig.add_trace(go.Scatter(
            x=[lo[idx], hi[idx]], y=[i, i],
            mode="lines", line=dict(color="lightblue", width=4),
            showlegend=(i == 0), name="90% CI",
            hoverinfo="skip",
        ))

    # Predicted values
    fig.add_trace(go.Scatter(
        x=y_pred[order], y=list(range(len(order))),
        mode="markers", marker=dict(color="steelblue", size=6),
        name="Predicted",
    ))

    # True values
    fig.add_trace(go.Scatter(
        x=y_true[order], y=list(range(len(order))),
        mode="markers", marker=dict(color="red", size=6, symbol="x"),
        name="Experimental",
    ))

    covered = ((y_true[order] >= lo[order]) & (y_true[order] <= hi[order])).mean()

    fig.update_layout(
        title=f"Conformal Prediction Intervals (Coverage: {covered:.1%})",
        xaxis_title="pIC50",
        yaxis_title="Molecule Index (sorted by CI width)",
        height=max(400, len(order) * 8),
        yaxis=dict(showticklabels=False),
    )
    return fig
