"""Drug-likeness radar chart — Ro5 + Veber's rules."""
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


def plot_druglikeness_radar(smiles):
    """
    Radar chart of 6 druglikeness properties normalized against limits.
    Green shaded zone = drug-like region. Blue polygon = query molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    props = {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
        "TPSA": Descriptors.TPSA(mol),
    }

    limits = {
        "MW": 500, "LogP": 5, "HBD": 5,
        "HBA": 10, "RotBonds": 10, "TPSA": 140,
    }

    normed = {k: min(props[k] / limits[k], 1.5) for k in props}

    categories = list(normed.keys())
    values = list(normed.values())
    values.append(values[0])  # close polygon
    categories.append(categories[0])

    fig = go.Figure()

    # Drug-like safe zone (boundary = 1.0 on normalized scale)
    safe = [1.0] * (len(limits) + 1)
    fig.add_trace(go.Scatterpolar(
        r=safe, theta=categories,
        fill="toself", fillcolor="rgba(46,204,113,0.15)",
        line=dict(color="green", dash="dash"),
        name="Drug-like Zone",
    ))

    # Query molecule polygon
    hover_texts = [
        f"{k}: {props[k]:.1f} / {limits[k]}"
        for k in list(normed.keys())
    ]
    hover_texts.append(hover_texts[0])  # close

    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories,
        fill="toself", fillcolor="rgba(52,152,219,0.25)",
        line=dict(color="steelblue", width=2),
        name="Query",
        text=hover_texts, hoverinfo="text",
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.5])),
        title="Drug-likeness Profile (Ro5 + Veber)",
        height=450,
        showlegend=True,
    )
    return fig
