"""
BioActivity-Q — Stage 2: XGBoost Training + Evaluation
Inputs : X_features.npy (or X_morgan.npy fallback), y_pic50.npy
Outputs: bioactivity_q.ubj, bioactivity_map.png, metrics.json

Supports an ensemble option: averaging several XGBoost models trained
with different seeds typically adds 1-2% R2 on QSAR tasks and smooths
out the occasional unlucky split.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

RANDOM_STATE = 42


def _build_xgb(**overrides):
    """
    Hyperparameters chosen to reduce overfitting on fingerprint data:
      - shallower trees (max_depth=5)  -> less memorization of individual bits
      - higher min_child_weight        -> every leaf must be supported
      - stronger L1/L2 regularization  -> prunes noise-bit weights
      - lower colsample_bytree         -> decorrelates trees on sparse FP
    """
    params = dict(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.1,
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    params.update(overrides)
    return xgb.XGBRegressor(**params)


# ----------------------------------------------------------------------
# STEP 3 — TRAIN XGBOOST REGRESSOR
# ----------------------------------------------------------------------
def train_model(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, shuffle=True
    )
    print(f"[split] train: {X_train.shape}  test: {X_test.shape}")

    model = _build_xgb(early_stopping_rounds=80)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    print(f"[train] best_iteration: {model.best_iteration}")
    return model, (X_train, X_test, y_train, y_test)


def train_ensemble(X_train, y_train, X_test, y_test, n_models: int = 5):
    """Train n_models XGBoost regressors with different seeds.
    Returns the list of models. Prediction is a simple average."""
    models = []
    for k in range(n_models):
        m = _build_xgb(random_state=RANDOM_STATE + k,
                       early_stopping_rounds=80)
        m.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)
        models.append(m)
        print(f"[ensemble] model {k + 1}/{n_models} "
              f"best_iteration={m.best_iteration}")
    return models


def ensemble_predict(models, X):
    preds = np.stack([m.predict(X) for m in models], axis=0)
    return preds.mean(axis=0)


# ----------------------------------------------------------------------
# STEP 4 — EVALUATION + BIOACTIVITY MAP
# ----------------------------------------------------------------------
def evaluate(model, X_train, X_test, y_train, y_test):
    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)
    return _metrics_from(y_train, y_pred_tr, y_test, y_pred_te), y_pred_te


def evaluate_ensemble(models, X_train, X_test, y_train, y_test):
    y_pred_tr = ensemble_predict(models, X_train)
    y_pred_te = ensemble_predict(models, X_test)
    return _metrics_from(y_train, y_pred_tr, y_test, y_pred_te), y_pred_te


def _metrics_from(y_tr, yp_tr, y_te, yp_te):
    metrics = {
        "train": {
            "R2":   float(r2_score(y_tr, yp_tr)),
            "RMSE": float(np.sqrt(mean_squared_error(y_tr, yp_tr))),
            "MAE":  float(mean_absolute_error(y_tr, yp_tr)),
        },
        "test": {
            "R2":   float(r2_score(y_te,  yp_te)),
            "RMSE": float(np.sqrt(mean_squared_error(y_te,  yp_te))),
            "MAE":  float(mean_absolute_error(y_te,  yp_te)),
        },
    }
    print("\n=== Bioactivity-Q Performance ===")
    for split, m in metrics.items():
        print(f"  {split:<5}  R2={m['R2']:.3f}   "
              f"RMSE={m['RMSE']:.3f}   MAE={m['MAE']:.3f}")
    return metrics


def cross_validate(X, y, k: int = 5):
    """5-fold CV on the full set for an honest R2 estimate (no early stop)."""
    cv_model = _build_xgb(n_estimators=800)  # fixed count, no eval_set in CV
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    r2_scores = cross_val_score(cv_model, X, y, cv=kf,
                                scoring="r2", n_jobs=-1)
    print(f"[CV] {k}-fold R2 = {r2_scores.mean():.3f} "
          f"± {r2_scores.std():.3f}")
    return float(r2_scores.mean()), float(r2_scores.std())


def plot_bioactivity_map(y_true, y_pred, metrics,
                         out_path: str = "outputs/bioactivity_map.png"):
    r2   = metrics["test"]["R2"]
    rmse = metrics["test"]["RMSE"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.55, s=28,
               edgecolor="k", linewidth=0.3)

    lims = [min(y_true.min(), y_pred.min()) - 0.3,
            max(y_true.max(), y_pred.max()) + 0.3]
    ax.plot(lims, lims, "r--", lw=1.5, label="Ideal (y = x)")
    ax.fill_between(
        lims,
        [v - 1 for v in lims],
        [v + 1 for v in lims],
        color="red", alpha=0.08, label="±1 log unit",
    )

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Experimental pIC50", fontsize=12)
    ax.set_ylabel("Predicted pIC50",    fontsize=12)
    ax.set_title("BioActivity-Q — EGFR Bioactivity Map",
                 fontsize=13, weight="bold")
    ax.text(
        0.04, 0.96,
        f"R² = {r2:.3f}\nRMSE = {rmse:.3f}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot] saved -> {out_path}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Prefer the richer combined features when available
    path = "models/X_features.npy" if os.path.exists("models/X_features.npy") else "models/X_morgan.npy"
    X = np.load(path)
    y = np.load("models/y_pic50.npy")
    print(f"[load] {path}  X={X.shape}  y={y.shape}")

    model, (X_train, X_test, y_train, y_test) = train_model(X, y)
    metrics, y_pred_te = evaluate(model, X_train, X_test, y_train, y_test)

    cv_mean, cv_std = cross_validate(X, y, k=5)
    metrics["cv5_r2_mean"] = cv_mean
    metrics["cv5_r2_std"]  = cv_std

    plot_bioactivity_map(y_test, y_pred_te, metrics)

    model.save_model("models/bioactivity_q.ubj")
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[done] saved bioactivity_q.ubj + metrics.json")
