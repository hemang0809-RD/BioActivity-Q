"""Interactive bioactivity map — replaces static matplotlib scatter."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_interactive_bioactivity_map(
    y_true, y_pred, smiles=None, similarities=None, metrics=None
):
    """
    Plotly scatter: hover shows SMILES, color by AD zone or abs error.
    Includes y=x reference line and +/-1 log unit band.
    """
    df_plot = pd.DataFrame({
        "Experimental pIC50": y_true,
        "Predicted pIC50": y_pred,
        "Residual": np.round(y_true - y_pred, 3),
        "Abs Error": np.round(np.abs(y_true - y_pred), 3),
    })

    if smiles is not None:
        df_plot["SMILES"] = smiles

    hover_extra = ["Residual", "Abs Error"]
    if similarities is not None:
        df_plot["Tanimoto"] = np.round(similarities, 3)
        df_plot["AD Zone"] = pd.cut(
            similarities,
            bins=[0, 0.25, 0.40, 1.0],
            labels=["Out of Domain", "Borderline", "In Domain"],
        )
        color_col = "AD Zone"
        color_map = {
            "In Domain": "#2ecc71",
            "Borderline": "#f39c12",
            "Out of Domain": "#e74c3c",
        }
        hover_extra += ["SMILES", "Tanimoto"] if smiles is not None else ["Tanimoto"]
    else:
        color_col = "Abs Error"
        color_map = None
        if smiles is not None:
            hover_extra.append("SMILES")

    fig = px.scatter(
        df_plot,
        x="Experimental pIC50",
        y="Predicted pIC50",
        color=color_col,
        color_discrete_map=color_map,
        hover_data=hover_extra,
        title="BioActivity-Q — EGFR Bioactivity Map",
        opacity=0.7,
    )

    # y = x reference line
    lims = [
        min(y_true.min(), y_pred.min()) - 0.3,
        max(y_true.max(), y_pred.max()) + 0.3,
    ]
    fig.add_trace(go.Scatter(
        x=lims, y=lims, mode="lines",
        line=dict(color="red", dash="dash", width=1.5),
        name="Ideal (y = x)", showlegend=True,
    ))

    # +/-1 log unit shaded band
    fig.add_trace(go.Scatter(
        x=lims + lims[::-1],
        y=[v + 1 for v in lims] + [v - 1 for v in lims[::-1]],
        fill="toself", fillcolor="rgba(255,0,0,0.05)",
        line=dict(color="rgba(255,0,0,0)"),
        name="±1 log unit", showlegend=True,
    ))

    if metrics:
        r2 = metrics["test"]["R2"]
        rmse = metrics["test"]["RMSE"]
        fig.add_annotation(
            x=0.05, y=0.95, xref="paper", yref="paper",
            text=f"R² = {r2:.3f}<br>RMSE = {rmse:.3f}",
            showarrow=False, bgcolor="white", bordercolor="black",
            font=dict(size=14),
        )

    fig.update_layout(height=650, width=700)
    return fig
