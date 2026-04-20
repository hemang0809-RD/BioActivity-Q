"""
Unified model serialization — bundle all artifacts into one file.
Run after make_bioactivity_q.py to create a single deployment artifact.

Usage:
    python save_model_bundle.py
"""
from pathlib import Path
import joblib
import numpy as np
import xgboost as xgb


def save_bundle(
    model, conformal, train_fps,
    umap_reducer=None, train_scaffolds=None, cv_residuals=None,
    path="models/bioactivity_q_bundle.joblib",
):
    """Save all model artifacts into a single compressed file."""
    bundle = {
        "model": model,
        "conformal_q_hat": conformal.q_hat if conformal else None,
        "conformal_alpha": conformal.alpha if conformal else None,
        "train_fps": train_fps,
        "umap_reducer": umap_reducer,
        "train_scaffolds": train_scaffolds,
        "cv_residuals": cv_residuals,
        "version": "1.1.0",
    }
    joblib.dump(bundle, path, compress=3)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"[bundle] saved {path} ({size_mb:.1f} MB)")


def load_bundle(path="models/bioactivity_q_bundle.joblib"):
    """Load the unified bundle. Returns a dict."""
    bundle = joblib.load(path)
    print(f"[bundle] loaded v{bundle.get('version', 'unknown')}")
    return bundle


def main():
    """Build bundle from individual saved artifacts."""
    root = Path(__file__).parent

    model = xgb.XGBRegressor()
    model.load_model(str(root / "models" / "bioactivity_q.ubj"))

    train_fps = None
    fp_path = root / "models" / "X_morgan.npy"
    if fp_path.exists():
        train_fps = np.load(str(fp_path))

    conformal_data = None
    conf_path = root / "models" / "conformal.npz"
    if conf_path.exists():
        conformal_data = np.load(str(conf_path))

    umap_reducer = None
    umap_path = root / "models" / "umap_transformer.joblib"
    if umap_path.exists():
        umap_reducer = joblib.load(str(umap_path))

    cv_residuals = None
    cv_path = root / "models" / "cv_residuals.npy"
    if cv_path.exists():
        cv_residuals = np.load(str(cv_path))

    bundle = {
        "model": model,
        "conformal": conformal_data,
        "train_fps": train_fps,
        "umap_reducer": umap_reducer,
        "cv_residuals": cv_residuals,
        "version": "1.1.0",
    }

    out = root / "models" / "bioactivity_q_bundle.joblib"
    joblib.dump(bundle, str(out), compress=3)
    size_mb = out.stat().st_size / 1e6
    print(f"[bundle] saved -> {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
