"""
BioActivity-Q — end-to-end pipeline runner.

Examples:
    # Train from scratch with the recommended scaffold split
    python make_bioactivity_q.py --csv data/egfr_chembl.csv

    # Train with a (less honest) random split for comparison
    python make_bioactivity_q.py --csv data/egfr_chembl.csv --split random

    # Predict on a new molecule using the persisted model
    python make_bioactivity_q.py --predict "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"

    # Predict + show top SHAP fragment contributions
    python make_bioactivity_q.py --predict "<SMILES>" --explain
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import joblib
import xgboost as xgb
from rdkit import DataStructs
from sklearn.model_selection import train_test_split

from src.engine.preprocess import load_and_clean_chembl, featurize_dataframe
from src.engine.train import (
    _build_xgb, train_model, evaluate,
    plot_bioactivity_map, cross_validate,
)
from src.engine.scaffold_split import scaffold_split
from src.inference.conformal import ConformalRegressor
from src.inference.predict import BioActivityPredictor, _print_result
from src.inference.explain import explain_prediction


def _rows_to_bv(rows: np.ndarray, n_bits: int = 2048):
    """Convert (n, n_bits) int8 matrix to a list of RDKit ExplicitBitVects."""
    out = []
    for r in rows:
        bv = DataStructs.ExplicitBitVect(n_bits)
        bv.SetBitsFromList(np.where(r == 1)[0].tolist())
        out.append(bv)
    return out


def run_full_pipeline(csv_path: str, split: str = "scaffold"):
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    print(f"\n{'=' * 64}")
    print(f"  BioActivity-Q Pipeline  |  split={split}")
    print(f"{'=' * 64}\n")

    # --- Stage 1: clean + featurize -------------------------------------
    df = load_and_clean_chembl(csv_path)
    X_combined, X_fp, y, df = featurize_dataframe(df)
    np.save("models/X_features.npy", X_combined)
    np.save("models/X_morgan.npy",   X_fp)
    np.save("models/y_pic50.npy",    y)
    df.to_csv("data/egfr_clean.csv", index=False)

    # For modelling we use the richer (fp + descriptor) matrix.
    # For AD / Tanimoto we use the pure fingerprint matrix.
    X = X_combined

    # --- Stage 2: split + train -----------------------------------------
    if split == "scaffold":
        tr, te = scaffold_split(df["canonical_smiles"].tolist(), test_frac=0.20)
        X_train, X_test = X[tr], X[te]
        X_fp_train, X_fp_test = X_fp[tr], X_fp[te]
        y_train, y_test = y[tr], y[te]
        model = _build_xgb(early_stopping_rounds=80)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        print(f"[train] best_iteration: {model.best_iteration}")
    else:
        model, (X_train, X_test, y_train, y_test) = train_model(X, y)
        # Derive matching fp slices for AD (using the same split indices
        # requires re-splitting; simplest: featurize splits separately)
        X_fp_train, X_fp_test = train_test_split(
            X_fp, test_size=0.20, random_state=42
        )

    # --- Stage 3: evaluate ----------------------------------------------
    metrics, y_pred_te = evaluate(model, X_train, X_test, y_train, y_test)
    cv_mean, cv_std = cross_validate(X, y, k=5)
    metrics["cv5_r2_mean"] = cv_mean
    metrics["cv5_r2_std"]  = cv_std
    metrics["split_strategy"] = split

    # Save per-compound CV residuals for enhanced conformal difficulty
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_residuals = np.zeros(len(y), dtype=np.float32)
    for fold_tr, fold_te in kf.split(X):
        fold_model = _build_xgb(n_estimators=800)
        fold_model.fit(X[fold_tr], y[fold_tr])
        fold_pred = fold_model.predict(X[fold_te])
        cv_residuals[fold_te] = np.abs(y[fold_te] - fold_pred)
    np.save("models/cv_residuals.npy", cv_residuals)
    print(f"[cv] saved per-compound residuals -> models/cv_residuals.npy")

    plot_bioactivity_map(
        y_test, y_pred_te, metrics,
        out_path=f"outputs/bioactivity_map_{split}.png",
    )
    model.save_model("models/bioactivity_q.ubj")

    # --- Stage 4: conformal calibration ---------------------------------
    # Carve a calibration slice out of the training set so it's never
    # been seen during model fitting.
    X_prop, X_calib, X_fp_prop, X_fp_calib, y_prop, y_calib = train_test_split(
        X_train, X_fp_train, y_train, test_size=0.20, random_state=42
    )
    model_calib = _build_xgb(early_stopping_rounds=80)
    model_calib.fit(
        X_prop, y_prop,
        eval_set=[(X_calib, y_calib)],
        verbose=False,
    )

    # Conformal needs RDKit bit-vectors for Tanimoto-based difficulty
    train_bv = _rows_to_bv(X_fp_prop)
    calib_bv = _rows_to_bv(X_fp_calib)
    test_bv  = _rows_to_bv(X_fp_test)

    conf = ConformalRegressor(model_calib, alpha=0.10)
    conf.calibrate(X_calib, y_calib, calib_bv, train_bv)
    conf.save("models/conformal.npz")

    # Validate empirical coverage on the test set
    _, lo, hi = conf.predict_interval(X_test, test_bv)
    covered = float(((y_test >= lo) & (y_test <= hi)).mean())
    avg_width = float((hi - lo).mean())
    print(f"[conformal] test coverage: {covered:.1%}  "
          f"(target: {1 - conf.alpha:.0%})")
    print(f"[conformal] avg interval width: +/- {avg_width / 2:.2f} pIC50")

    metrics["conformal"] = {
        "alpha": conf.alpha,
        "empirical_coverage": covered,
        "avg_half_width":     avg_width / 2,
    }

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Stage 5: UMAP embedding for chemical space viz ---
    from src.ui.chemical_space import fit_and_save_embedding
    fit_and_save_embedding(X_fp)

    print(f"\n[pipeline] complete — artifacts saved in {Path.cwd()}")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     help="ChEMBL CSV to train from")
    p.add_argument("--split",   choices=["random", "scaffold"], default="scaffold")
    p.add_argument("--predict", help="SMILES to predict (uses existing model)")
    p.add_argument("--explain", action="store_true",
                   help="Show SHAP fragment contributions for --predict")
    p.add_argument("--render-fragments", action="store_true",
                   help="Save PNG images of top SHAP fragments")
    args = p.parse_args()

    if args.csv:
        run_full_pipeline(args.csv, split=args.split)

    if args.predict:
        predictor = BioActivityPredictor()
        result = predictor.predict(args.predict)
        _print_result(result)
        if args.explain and "error" not in result:
            explain_prediction(
                predictor.model, args.predict,
                render_fragments=args.render_fragments,
            )

    if not args.csv and not args.predict:
        p.print_help()


if __name__ == "__main__":
    main()
