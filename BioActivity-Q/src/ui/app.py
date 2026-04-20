"""
BioActivity-Q — Streamlit Dashboard
Run: streamlit run src/ui/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.inference.predict import BioActivityPredictor, smiles_to_query
from src.ui.mol_renderer import smiles_to_svg
from src.ui.radar_druglikeness import plot_druglikeness_radar
from src.ui.chemical_space import project_query
from src.inference.activity_cliffs import find_activity_cliffs

st.set_page_config(page_title="BioActivity-Q", page_icon="🧬", layout="wide")

# ── Cached loaders ──────────────────────────────────────────────────

@st.cache_resource
def load_predictor():
    return BioActivityPredictor(
        model_path=str(ROOT / "models" / "bioactivity_q.ubj"),
        train_fp_path=str(ROOT / "models" / "X_morgan.npy"),
        conformal_path=str(ROOT / "models" / "conformal.npz"),
    )

@st.cache_data
def load_training_data():
    return pd.read_csv(ROOT / "data" / "egfr_clean.csv")

@st.cache_data
def load_train_embedding():
    path = ROOT / "models" / "train_embedding.npy"
    return np.load(str(path)) if path.exists() else None

predictor = load_predictor()
df_train = load_training_data()

# ── Sidebar ─────────────────────────────────────────────────────────

st.sidebar.title("🧬 BioActivity-Q")
st.sidebar.caption("EGFR pIC50 Predictor with Uncertainty Quantification")

mode = st.sidebar.radio(
    "Mode", ["Single Prediction", "Batch Upload", "Data Explorer"], index=0
)

# =====================================================================
# MODE 1: SINGLE PREDICTION
# =====================================================================
if mode == "Single Prediction":
    smiles_input = st.sidebar.text_input(
        "Enter SMILES",
        value="COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
        help="Paste any valid SMILES string",
    )
    run = st.sidebar.button("Predict", type="primary")

    if run:
        result = predictor.predict(smiles_input)
        if "error" in result:
            st.error(f"Invalid SMILES: {result['error']}")
            st.stop()

        # ── Row 1: Structure + Key metrics ──
        col_struct, col_metrics = st.columns([1, 2])

        with col_struct:
            st.subheader("Structure")
            svg = smiles_to_svg(result["canonical_smiles"], width=350, height=250)
            st.components.v1.html(svg, height=270)

        with col_metrics:
            st.subheader("Prediction Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("pIC50", f"{result['predicted_pIC50']:.2f}")
            m2.metric("IC50 (nM)", f"{result['predicted_IC50_nM']:.1f}")
            m3.metric("Activity", result["activity_class"])

            ad = result["applicability_domain"]
            ad_icons = {"IN DOMAIN": "🟢", "BORDERLINE": "🟡", "OUT OF DOMAIN": "🔴"}
            icon = "🟡" if "novel scaffold" in ad else ad_icons.get(ad.split(" (")[0], "⚪")
            m4.metric("Domain", f"{icon} {ad}")

        st.divider()

        # ── Row 2: Confidence Interval + Tanimoto + Scaffold ──
        col_ci, col_tan = st.columns(2)

        with col_ci:
            if "pIC50_90CI" in result:
                lo, hi = result["pIC50_90CI"]
                pred = result["predicted_pIC50"]
                fig_ci = go.Figure(go.Indicator(
                    mode="gauge+number", value=pred,
                    title={"text": "pIC50 with 90% CI"},
                    gauge={
                        "axis": {"range": [3, 11]},
                        "bar": {"color": "darkblue"},
                        "steps": [{"range": [lo, hi], "color": "lightblue"}],
                        "threshold": {
                            "line": {"color": "red", "width": 2},
                            "thickness": 0.75, "value": pred,
                        },
                    },
                ))
                fig_ci.update_layout(height=280, margin=dict(t=50, b=20))
                st.plotly_chart(fig_ci, use_container_width=True)
            else:
                st.info("Conformal intervals not available. Run the full pipeline first.")

        with col_tan:
            sim = result["nearest_neighbor_similarity"]
            fig_sim = go.Figure(go.Bar(
                x=[sim], y=["Tanimoto"], orientation="h",
                marker_color="steelblue",
                text=[f"{sim:.3f}"], textposition="outside",
            ))
            fig_sim.update_xaxes(range=[0, 1], title="Max Tanimoto to Training Set")
            fig_sim.update_layout(
                height=150, margin=dict(t=10, b=10, l=10, r=10),
                shapes=[
                    dict(type="line", x0=0.40, x1=0.40, y0=-0.5, y1=0.5,
                         line=dict(color="green", dash="dash")),
                    dict(type="line", x0=0.25, x1=0.25, y0=-0.5, y1=0.5,
                         line=dict(color="red", dash="dash")),
                ],
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            # Scaffold info
            if "scaffold" in result:
                status = "✅ Known" if result.get("scaffold_in_training") else "⚠️ Novel"
                st.caption(f"Scaffold: `{result['scaffold'][:60]}` — {status}")

        st.divider()

        # ── Row 3: Drug-likeness radar ──
        st.subheader("Drug-likeness Profile")
        fig_radar = plot_druglikeness_radar(result["canonical_smiles"])
        if fig_radar:
            st.plotly_chart(fig_radar, use_container_width=True)

        st.divider()

        # ── Row 4: Chemical space (UMAP) ──
        train_emb = load_train_embedding()
        if train_emb is not None:
            st.subheader("Position in Chemical Space (UMAP)")
            fp_arr, _, _, _ = smiles_to_query(result["canonical_smiles"])
            if fp_arr is not None:
                try:
                    query_coord = project_query(
                        fp_arr,
                        reducer_path=str(ROOT / "models" / "umap_transformer.joblib"),
                    )
                    n_pts = min(len(train_emb), len(df_train))
                    df_emb = pd.DataFrame({
                        "UMAP 1": train_emb[:n_pts, 0],
                        "UMAP 2": train_emb[:n_pts, 1],
                        "pIC50": df_train["pIC50"].values[:n_pts],
                    })
                    fig_space = px.scatter(
                        df_emb, x="UMAP 1", y="UMAP 2", color="pIC50",
                        color_continuous_scale="RdYlGn", opacity=0.5,
                        title="Training Set Chemical Space",
                    )
                    fig_space.add_trace(go.Scatter(
                        x=[query_coord[0]], y=[query_coord[1]],
                        mode="markers",
                        marker=dict(size=16, color="black", symbol="star",
                                    line=dict(width=2, color="white")),
                        name="Your Molecule",
                    ))
                    fig_space.update_layout(height=500)
                    st.plotly_chart(fig_space, use_container_width=True)
                except Exception as e:
                    st.warning(f"UMAP projection failed: {e}")

        st.divider()

        # ── Row 5: Activity cliff check ──
        st.subheader("Activity Cliff Check")
        fp_arr, _, bv, _ = smiles_to_query(result["canonical_smiles"])
        if fp_arr is not None and bv is not None:
            cliffs = find_activity_cliffs(
                query_bv=bv,
                query_pIC50=result["predicted_pIC50"],
                train_bvs=predictor.ad._bitvects,
                train_pIC50s=df_train["pIC50"].values,
                train_smiles=df_train["canonical_smiles"].values,
                sim_threshold=0.80,
                delta_threshold=1.0,
            )
            if cliffs:
                st.warning(
                    f"Found {len(cliffs)} activity cliff(s) — structurally similar "
                    f"but biologically different training compounds:"
                )
                st.dataframe(pd.DataFrame(cliffs), use_container_width=True)
            else:
                st.success(
                    "No activity cliffs detected — prediction is consistent "
                    "with similar training compounds."
                )

# =====================================================================
# MODE 2: BATCH UPLOAD
# =====================================================================
elif mode == "Batch Upload":
    uploaded = st.sidebar.file_uploader(
        "Upload CSV with a 'smiles' column", type="csv"
    )
    if uploaded:
        df_input = pd.read_csv(uploaded)
        if "smiles" not in df_input.columns:
            st.error("CSV must contain a column named `smiles`.")
            st.stop()

        with st.spinner(f"Predicting {len(df_input)} molecules..."):
            results = predictor.predict_batch(df_input["smiles"].tolist())

        st.subheader(f"Predictions ({len(results)} molecules)")
        st.dataframe(results, use_container_width=True, height=400)

        # Summary plots
        col1, col2 = st.columns(2)
        with col1:
            if "activity_class" in results.columns:
                fig_class = px.pie(
                    results, names="activity_class",
                    title="Activity Class Distribution",
                    color="activity_class",
                    color_discrete_map={
                        "Highly Active": "#2ecc71", "Active": "#3498db",
                        "Moderately Active": "#f39c12", "Inactive": "#e74c3c",
                    },
                )
                st.plotly_chart(fig_class, use_container_width=True)

        with col2:
            if "nearest_neighbor_similarity" in results.columns:
                fig_ad = px.histogram(
                    results, x="nearest_neighbor_similarity",
                    color="applicability_domain",
                    title="Applicability Domain Distribution",
                    nbins=30,
                )
                st.plotly_chart(fig_ad, use_container_width=True)

        # pIC50 distribution of batch
        if "predicted_pIC50" in results.columns:
            fig_batch = px.histogram(
                results, x="predicted_pIC50", nbins=30,
                title="Predicted pIC50 Distribution",
                color_discrete_sequence=["steelblue"],
            )
            st.plotly_chart(fig_batch, use_container_width=True)

        # Download
        st.download_button(
            "📥 Download Predictions CSV",
            results.to_csv(index=False),
            "predictions.csv", "text/csv",
        )

# =====================================================================
# MODE 3: DATA EXPLORER
# =====================================================================
else:
    st.header("Training Data Explorer")
    st.caption(f"{len(df_train)} molecules in the training set")

    tab1, tab2, tab3 = st.tabs(["Distributions", "Property Space", "Scaffolds"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(
                df_train, x="pIC50", nbins=50,
                title="pIC50 Distribution",
                color_discrete_sequence=["steelblue"],
            )
            for val, label, color in [
                (7, "Highly Active", "green"),
                (6, "Active", "orange"),
                (5, "Moderate", "red"),
            ]:
                fig_hist.add_vline(
                    x=val, line_dash="dash", line_color=color,
                    annotation_text=label,
                )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            fig_box = px.box(
                df_train, y="pIC50", title="pIC50 Box Plot",
                color_discrete_sequence=["steelblue"],
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Activity class breakdown
        classes = pd.cut(
            df_train["pIC50"],
            bins=[0, 5, 6, 7, 12],
            labels=["Inactive", "Moderately Active", "Active", "Highly Active"],
        )
        fig_pie = px.pie(
            names=classes.value_counts().index,
            values=classes.value_counts().values,
            title="Activity Class Breakdown",
            color=classes.value_counts().index,
            color_discrete_map={
                "Highly Active": "#2ecc71", "Active": "#3498db",
                "Moderately Active": "#f39c12", "Inactive": "#e74c3c",
            },
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        from src.ui.property_space import plot_property_space
        st.plotly_chart(plot_property_space(df_train), use_container_width=True)

    with tab3:
        from src.ui.scaffold_viz import plot_scaffold_sunburst
        from src.engine.scaffold_split import scaffold_split as _ss
        train_idx, test_idx = _ss(df_train["canonical_smiles"].tolist())
        fig_sun = plot_scaffold_sunburst(
            df_train["canonical_smiles"].tolist(), train_idx, test_idx
        )
        st.plotly_chart(fig_sun, use_container_width=True)
