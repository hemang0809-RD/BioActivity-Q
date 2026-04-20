"""
BioActivity-Q — Stage 3: Inference
Loads the trained model and predicts pIC50 for new SMILES, with:
  - Tanimoto applicability-domain check
  - Optional conformal prediction intervals (90% CI)
  - Activity classification + physchem context
  - Combined fingerprint + descriptor features (richer than FP alone)

Usage:
    python predict.py "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    python predict.py --batch molecules.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors
import xgboost as xgb

# Re-use the training featurizer so train/test stay perfectly aligned
from src.engine.preprocess import (
    MORGAN_NBITS,
    smiles_to_features,
    standardize_smiles as _preproc_standardize,
)
from src.engine.scaffold_split import generate_scaffold

RDLogger.DisableLog("rdApp.*")

MODEL_PATH     = "models/bioactivity_q.ubj"
MODEL_PATH_OLD = "bioactivity_q.model"   # backwards compatibility
TRAIN_FP_PATH  = "models/X_morgan.npy"
CONFORMAL_PATH = "models/conformal.npz"


# ----------------------------------------------------------------------
# preprocess.standardize_smiles imports SaltRemover + charge neutralization.
# We wrap it here so callers don't have to know about the preprocess module.
# ----------------------------------------------------------------------
def standardize_smiles(smiles: str):
    return _preproc_standardize(smiles)


def smiles_to_query(smiles: str):
    """
    Returns (X_fp_int8, X_features_f32, fp_bitvect, canonical_smiles)
    or (None, None, None, None) on an invalid SMILES.
    """
    canon = standardize_smiles(smiles)
    if canon is None:
        return None, None, None, None
    fp_arr, combined = smiles_to_features(canon)
    if fp_arr is None:
        return None, None, None, None
    # RDKit bit vector for Tanimoto / AD check
    bv = DataStructs.ExplicitBitVect(MORGAN_NBITS)
    bv.SetBitsFromList(np.where(fp_arr == 1)[0].tolist())
    return fp_arr, combined, bv, canon


# ----------------------------------------------------------------------
# Applicability Domain — Tanimoto similarity to nearest training neighbor
# ----------------------------------------------------------------------
class ApplicabilityDomain:
    """
    A prediction is only trustworthy if the query molecule resembles
    at least one training compound. We use max Tanimoto similarity
    to the training set as the confidence indicator.
        >= 0.40   -> IN DOMAIN     (reliable)
        0.25-0.40 -> BORDERLINE    (use with caution)
        <  0.25   -> OUT OF DOMAIN (extrapolation — don't trust)
    """
    def __init__(self, train_fp_matrix: np.ndarray):
        self._bitvects = []
        for row in train_fp_matrix:
            bv = DataStructs.ExplicitBitVect(MORGAN_NBITS)
            on_bits = np.where(row == 1)[0].tolist()
            bv.SetBitsFromList(on_bits)
            self._bitvects.append(bv)

    def max_similarity(self, query_fp) -> float:
        sims = DataStructs.BulkTanimotoSimilarity(query_fp, self._bitvects)
        return float(max(sims)) if sims else 0.0

    @staticmethod
    def label(sim: float) -> str:
        if sim >= 0.40: return "IN DOMAIN"
        if sim >= 0.25: return "BORDERLINE"
        return "OUT OF DOMAIN"


# ----------------------------------------------------------------------
# Core predictor
# ----------------------------------------------------------------------
def _resolve_model_path(user_path: str) -> str:
    """Prefer .ubj; fall back to legacy .model if present."""
    if Path(user_path).exists():
        return user_path
    if Path(MODEL_PATH_OLD).exists():
        return MODEL_PATH_OLD
    raise FileNotFoundError(
        f"No model file found. Train first with:\n"
        f"  python make_bioactivity_q.py --csv data/egfr_chembl.csv"
    )


class BioActivityPredictor:
    def __init__(self,
                 model_path: str = MODEL_PATH,
                 train_fp_path: str = TRAIN_FP_PATH,
                 conformal_path: str = CONFORMAL_PATH):
        self.model = xgb.XGBRegressor()
        self.model.load_model(_resolve_model_path(model_path))

        train_X = np.load(train_fp_path)
        self.ad = ApplicabilityDomain(train_X)

        # Optional conformal calibrator (loaded if file exists)
        self.conformal = None
        if Path(conformal_path).exists():
            from src.inference.conformal import ConformalRegressor
            self.conformal = ConformalRegressor.load(
                conformal_path, self.model, self.ad._bitvects
            )

        # Scaffold-aware applicability domain
        try:
            df_train = pd.read_csv(
                str(Path(__file__).parent.parent.parent / "data" / "egfr_clean.csv")
            )
            self._train_scaffolds = set(
                generate_scaffold(s) for s in df_train["canonical_smiles"]
            )
        except FileNotFoundError:
            self._train_scaffolds = None

    def predict(self, smiles: str) -> dict:
        fp_arr, features, bv, canon = smiles_to_query(smiles)
        if fp_arr is None:
            return {"input_smiles": smiles, "error": "Invalid SMILES"}

        pIC50 = float(self.model.predict(features.reshape(1, -1))[0])
        # Clamp obviously impossible predictions
        pIC50 = float(np.clip(pIC50, 2.0, 12.0))
        IC50_nM = 10 ** (9 - pIC50)
        sim = self.ad.max_similarity(bv)

        mol = Chem.MolFromSmiles(canon)
        result = {
            "input_smiles":                smiles,
            "canonical_smiles":            canon,
            "predicted_pIC50":             round(pIC50, 3),
            "predicted_IC50_nM":           round(IC50_nM, 2),
            "activity_class":              self._classify(pIC50),
            "nearest_neighbor_similarity": round(sim, 3),
            "applicability_domain":        ApplicabilityDomain.label(sim),
            "MW":                          round(Descriptors.MolWt(mol), 2),
            "LogP":                        round(Descriptors.MolLogP(mol), 2),
        }

        # Scaffold novelty check
        if self._train_scaffolds is not None:
            query_scaffold = generate_scaffold(canon)
            result["scaffold"] = query_scaffold
            scaffold_known = query_scaffold in self._train_scaffolds
            result["scaffold_in_training"] = scaffold_known
            if not scaffold_known and result["applicability_domain"] == "IN DOMAIN":
                result["applicability_domain"] = "IN DOMAIN (novel scaffold)"

        # ---- conformal interval if calibrator is loaded ----
        if self.conformal is not None:
            _, lo, hi = self.conformal.predict_interval(
                features.reshape(1, -1), [bv]
            )
            lo = float(np.clip(lo[0], 2.0, 12.0))
            hi = float(np.clip(hi[0], 2.0, 12.0))
            result["pIC50_90CI"]   = [round(lo, 2), round(hi, 2)]
            result["IC50_nM_90CI"] = [
                round(10 ** (9 - hi), 2),
                round(10 ** (9 - lo), 2),
            ]
        return result

    def predict_batch(self, smiles_list) -> pd.DataFrame:
        return pd.DataFrame([self.predict(s) for s in smiles_list])

    @staticmethod
    def _classify(pIC50: float) -> str:
        if pIC50 >= 7.0: return "Highly Active"      # IC50 <= 100 nM
        if pIC50 >= 6.0: return "Active"             # IC50 <= 1 uM
        if pIC50 >= 5.0: return "Moderately Active"  # IC50 <= 10 uM
        return "Inactive"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _print_result(r: dict):
    print("\n" + "=" * 64)
    print("  BioActivity-Q  |  EGFR pIC50 Prediction")
    print("=" * 64)
    if "error" in r:
        print(f"  SMILES : {r['input_smiles']}")
        print(f"  ERROR  : {r['error']}")
        return
    print(f"  Input SMILES        : {r['input_smiles']}")
    print(f"  Canonical SMILES    : {r['canonical_smiles']}")
    print(f"  Predicted pIC50     : {r['predicted_pIC50']}")
    if "pIC50_90CI" in r:
        ci = r["pIC50_90CI"]
        print(f"  pIC50 90% CI        : [{ci[0]}, {ci[1]}]")
    print(f"  Predicted IC50      : {r['predicted_IC50_nM']} nM")
    if "IC50_nM_90CI" in r:
        ic = r["IC50_nM_90CI"]
        print(f"  IC50 90% CI (nM)    : [{ic[0]}, {ic[1]}]")
    print(f"  Activity Class      : {r['activity_class']}")
    print(f"  Nearest-NN Tanimoto : {r['nearest_neighbor_similarity']}")
    print(f"  Applicability       : {r['applicability_domain']}")
    print(f"  MW / LogP           : {r['MW']} / {r['LogP']}")
    print("=" * 64 + "\n")


def main():
    parser = argparse.ArgumentParser(description="BioActivity-Q inference")
    parser.add_argument("smiles", nargs="?", help="SMILES string to predict")
    parser.add_argument("--batch", help="CSV file with a 'smiles' column")
    args = parser.parse_args()

    predictor = BioActivityPredictor()

    if args.batch:
        df = pd.read_csv(args.batch)
        out = predictor.predict_batch(df["smiles"].tolist())
        out_path = "predictions.csv"
        out.to_csv(out_path, index=False)
        print(f"[done] wrote {len(out)} predictions -> {out_path}")
        return

    if not args.smiles:
        # Demo: caffeine — should land OUT OF DOMAIN for an EGFR model
        args.smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        print("[demo] no SMILES given — running caffeine as example")

    _print_result(predictor.predict(args.smiles))


if __name__ == "__main__":
    main()
