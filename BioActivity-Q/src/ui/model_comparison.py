"""Side-by-side model comparison and CV fold visualization."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compare_models(metrics_list, labels):
    """Grouped bar chart comparing R2, RMSE, MAE across models/splits."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=["R²", "RMSE", "MAE"])
    colors = ["steelblue", "coral", "seagreen", "gold"]

    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        for j, metric_name in enumerate(["R2", "RMSE", "MAE"]):
            fig.add_trace(go.Bar(
                x=["Train", "Test"],
                y=[metrics["train"][metric_name], metrics["test"][metric_name]],
                name=label, marker_color=colors[i % len(colors)],
                showlegend=(j == 0),
            ), row=1, col=j + 1)

    fig.update_layout(title="Model Comparison", barmode="group", height=400)
    return fig


def plot_cv_folds(cv_scores, model_name="XGBoost"):
    """Bar chart of per-fold R2 with mean +/- std annotation."""
    cv_scores = np.asarray(cv_scores)
    fig = go.Figure(go.Bar(
        x=[f"Fold {i+1}" for i in range(len(cv_scores))],
        y=cv_scores, marker_color="steelblue",
        text=[f"{s:.3f}" for s in cv_scores], textposition="outside",
    ))
    fig.add_hline(
        y=np.mean(cv_scores), line_dash="dash", line_color="red",
        annotation_text=f"Mean: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}",
    )
    fig.update_layout(
        title=f"5-Fold CV R² ({model_name})",
        yaxis_title="R²",
        yaxis_range=[0, max(1, float(cv_scores.max()) + 0.05)],
        height=400,
    )
    return fig


# Backward-compatible alias
plot_model_comparison = compare_models
