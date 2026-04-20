"""Scaffold diversity sunburst — train/test split by Murcko scaffold."""
import pandas as pd
import plotly.express as px
from collections import Counter
from src.engine.scaffold_split import generate_scaffold


def plot_scaffold_sunburst(smiles_list, train_idx, test_idx, top_n=20):
    """
    Sunburst chart: outer ring = scaffold groups, inner ring = train/test.
    Shows the top N most common scaffolds; groups the rest as 'Other'.
    """
    scaffolds = [generate_scaffold(s) for s in smiles_list]
    scaffold_counts = Counter(scaffolds)
    top_scaffolds = {s for s, _ in scaffold_counts.most_common(top_n)}

    train_set = set(
        train_idx.tolist() if hasattr(train_idx, "tolist") else list(train_idx)
    )

    records = []
    for i, (smi, scaf) in enumerate(zip(smiles_list, scaffolds)):
        scaf_label = scaf[:40] if scaf in top_scaffolds else "Other scaffolds"
        split = "Train" if i in train_set else "Test"
        records.append({"Scaffold": scaf_label, "Split": split, "Count": 1})

    df = pd.DataFrame(records)
    df_agg = df.groupby(["Scaffold", "Split"]).sum().reset_index()

    fig = px.sunburst(
        df_agg, path=["Split", "Scaffold"], values="Count",
        title=f"Scaffold Distribution (Top {top_n} + Other)",
        color="Split",
        color_discrete_map={"Train": "steelblue", "Test": "coral"},
    )
    fig.update_layout(height=600)
    return fig
