"""Molecular property space explorer — MW vs LogP with Ro5 boundaries."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


def plot_property_space(df, query_smiles=None):
    """
    MW vs LogP scatter colored by pIC50, with Lipinski Ro5 boundaries.
    Optionally overlays a query molecule as a star marker.
    """
    mols = [Chem.MolFromSmiles(s) for s in df["canonical_smiles"]]
    df_plot = pd.DataFrame({
        "MW": [Descriptors.MolWt(m) if m else np.nan for m in mols],
        "LogP": [Descriptors.MolLogP(m) if m else np.nan for m in mols],
        "TPSA": [Descriptors.TPSA(m) if m else np.nan for m in mols],
        "HBD": [Lipinski.NumHDonors(m) if m else 0 for m in mols],
        "HBA": [Lipinski.NumHAcceptors(m) if m else 0 for m in mols],
        "pIC50": df["pIC50"].values,
    }).dropna()

    fig = px.scatter(
        df_plot, x="MW", y="LogP", color="pIC50",
        color_continuous_scale="RdYlGn",
        hover_data=["TPSA", "HBD", "HBA"],
        title="Molecular Property Space (MW vs LogP)",
        opacity=0.6,
    )

    fig.add_hline(y=5, line_dash="dash", line_color="gray",
                  annotation_text="LogP = 5 (Ro5)")
    fig.add_vline(x=500, line_dash="dash", line_color="gray",
                  annotation_text="MW = 500 (Ro5)")

    if query_smiles:
        mol = Chem.MolFromSmiles(query_smiles)
        if mol:
            fig.add_trace(go.Scatter(
                x=[Descriptors.MolWt(mol)],
                y=[Descriptors.MolLogP(mol)],
                mode="markers",
                marker=dict(
                    size=16, color="black", symbol="star",
                    line=dict(width=2, color="white"),
                ),
                name="Query",
            ))

    fig.update_layout(height=600)
    return fig
