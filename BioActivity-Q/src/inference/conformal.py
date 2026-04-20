"""
Inductive Conformal Prediction (ICP) for QSAR regression.

Theory in one paragraph:
  Hold out a 'calibration set' the model never trained on. For each
  calibration point, compute the absolute residual |y - y_hat|. The
  (1 - alpha) quantile of those residuals becomes a universal error
  bar: for any new molecule, the interval [y_hat - q, y_hat + q]
  covers the true value with probability >= 1 - alpha. No
  distributional assumptions — only exchangeability.

We use the *normalized* variant: error bars scale with a per-molecule
difficulty estimate (here: 1 - max Tanimoto similarity to training).
Confident predictions get tight intervals; out-of-domain ones widen.
"""
import numpy as np
from rdkit import DataStructs


class ConformalRegressor:
    def __init__(self, model, alpha: float = 0.10):
        """alpha=0.10 -> 90% prediction intervals."""
        self.model = model
        self.alpha = alpha
        self.q_hat = None
        self._train_bitvects = None

    def _difficulty(self, query_bitvects, reference_bitvects,
                    reference_residuals=None, k=5):
        """
        Higher score = harder / more uncertain molecule.
        Blends Tanimoto distance with local neighborhood error if available.
        """
        diffs = np.empty(len(query_bitvects), dtype=np.float32)
        for i, q in enumerate(query_bitvects):
            sims = DataStructs.BulkTanimotoSimilarity(q, reference_bitvects)
            sims_arr = np.array(sims)
            max_sim = float(sims_arr.max()) if len(sims_arr) > 0 else 0.0
            base_difficulty = 1.0 - max_sim

            if reference_residuals is not None and len(reference_residuals) == len(sims_arr):
                # Find k nearest neighbors by Tanimoto
                top_k_idx = np.argsort(sims_arr)[-k:]
                local_error = float(np.mean(reference_residuals[top_k_idx]))
                # Blend: 70% distance-based + 30% local error
                diffs[i] = 0.7 * base_difficulty + 0.3 * local_error
            else:
                diffs[i] = base_difficulty

        # Floor at 0.25 — prevent zero-width intervals
        return 0.25 + diffs

    def calibrate(self, X_calib, y_calib,
                  calib_bitvects, train_bitvects):
        """Fit q_hat on held-out calibration set."""
        y_pred = self.model.predict(X_calib)
        residuals = np.abs(y_calib - y_pred)
        difficulty = self._difficulty(calib_bitvects, train_bitvects)

        scores = residuals / difficulty
        n = len(scores)

        # Finite-sample-corrected quantile (Vovk et al.)
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        k = min(k, n)
        self.q_hat = float(np.sort(scores)[k - 1])
        self._train_bitvects = train_bitvects

        print(f"[conformal] calibrated on n={n}  alpha={self.alpha}  "
              f"q_hat={self.q_hat:.3f}")
        return self

    def predict_interval(self, X, query_bitvects):
        if self.q_hat is None:
            raise RuntimeError("Call .calibrate() first.")
        y_pred = self.model.predict(X)
        difficulty = self._difficulty(query_bitvects, self._train_bitvects)
        half_width = self.q_hat * difficulty
        return y_pred, y_pred - half_width, y_pred + half_width

    # ---- persistence ----
    def save(self, path: str):
        np.savez(path, q_hat=self.q_hat, alpha=self.alpha)

    @classmethod
    def load(cls, path: str, model, train_bitvects):
        d = np.load(path)
        obj = cls(model, alpha=float(d["alpha"]))
        obj.q_hat = float(d["q_hat"])
        obj._train_bitvects = train_bitvects
        return obj
