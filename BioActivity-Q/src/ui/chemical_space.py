"""
Chemical space embedding — UMAP with persistent transformer.
Fit once after training, project queries into stable coordinates.

IMPORTANT: Do NOT use t-SNE here. t-SNE has no .transform() method,
so re-running it per query shifts the entire map. UMAP fit once +
.transform() for new points keeps the manifold stable.
"""
from __future__ import annotations
import numpy as np
import joblib


def fit_and_save_embedding(
    X_fp: np.ndarray,
    save_reducer: str = "models/umap_transformer.joblib",
    save_coords: str = "models/train_embedding.npy",
):
    """
    Run ONCE after training. Fits UMAP on training fingerprints,
    saves the transformer and the 2D coordinates.
    """
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="jaccard",
        random_state=42,
    )
    train_coords = reducer.fit_transform(X_fp.astype(np.float32))

    joblib.dump(reducer, save_reducer)
    np.save(save_coords, train_coords)

    print(f"[umap] saved transformer -> {save_reducer}")
    print(f"[umap] saved coordinates  -> {save_coords}")
    return reducer, train_coords


# Backward-compatible alias
fit_umap = fit_and_save_embedding


def project_query(
    query_fp: np.ndarray,
    reducer_path: str = "models/umap_transformer.joblib",
):
    """
    Project a single query molecule into the pre-fitted UMAP space.
    Returns (x, y) coordinates as a 1D array of length 2.
    """
    reducer = joblib.load(reducer_path)
    coord = reducer.transform(
        np.asarray(query_fp).reshape(1, -1).astype(np.float32)
    )
    return coord[0]
