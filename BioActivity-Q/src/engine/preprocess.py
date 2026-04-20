"""
BioActivity-Q — Stage 1: Preprocessing & Featurization
Target: EGFR (ChEMBL bioactivity data)

Reads a ChEMBL export CSV, cleans it, converts IC50 -> pIC50,
and generates a combined feature vector: Morgan fingerprints
(radius=2, 2048 bits) + RDKit physchem descriptors.
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, SaltRemover, Descriptors, Lipinski, rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray

# Silence RDKit parse warnings (ChEMBL has messy entries)
RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS = 2
MORGAN_NBITS  = 2048

# RDKit physchem descriptors appended to the fingerprint. Chosen so that
# each captures an orthogonal aspect of drug-likeness/kinase-fit and is
# fast + well-defined for any valid small molecule.
DESCRIPTOR_FNS = [
    ("MolWt",               Descriptors.MolWt),
    ("MolLogP",             Descriptors.MolLogP),
    ("TPSA",                Descriptors.TPSA),
    ("NumHDonors",          Lipinski.NumHDonors),
    ("NumHAcceptors",       Lipinski.NumHAcceptors),
    ("NumRotatableBonds",   Lipinski.NumRotatableBonds),
    ("NumAromaticRings",    rdMolDescriptors.CalcNumAromaticRings),
    ("NumAliphaticRings",   rdMolDescriptors.CalcNumAliphaticRings),
    ("NumSaturatedRings",   rdMolDescriptors.CalcNumSaturatedRings),
    ("FractionCSP3",        Lipinski.FractionCSP3),
    ("NumHeteroatoms",      Lipinski.NumHeteroatoms),
    ("HeavyAtomCount",      Lipinski.HeavyAtomCount),
    ("RingCount",           Lipinski.RingCount),
    ("NumAromaticHeterocycles", rdMolDescriptors.CalcNumAromaticHeterocycles),
    ("LabuteASA",           Descriptors.LabuteASA),
    ("BalabanJ",            Descriptors.BalabanJ),
    ("BertzCT",             Descriptors.BertzCT),
    ("MolMR",               Descriptors.MolMR),
]
DESCRIPTOR_NAMES = [n for n, _ in DESCRIPTOR_FNS]
N_DESCRIPTORS    = len(DESCRIPTOR_FNS)


# ----------------------------------------------------------------------
# Shared SMILES standardizer (used by both training and inference)
# ----------------------------------------------------------------------
_SALT_REMOVER = SaltRemover.SaltRemover()
_NEUTRALIZE_PATTERN = Chem.MolFromSmarts(
    "[+1!H0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
)


def _neutralize(mol):
    """Neutralize +/- charges where chemically reasonable."""
    at_matches = mol.GetSubstructMatches(_NEUTRALIZE_PATTERN)
    for (idx,) in at_matches:
        atom = mol.GetAtomWithIdx(idx)
        chg = atom.GetFormalCharge()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + chg)
        atom.UpdatePropertyCache()
    return mol


def standardize_smiles(smi: str):
    """Canonicalize + strip salts + neutralize charges + reject junk."""
    if smi is None or (isinstance(smi, float) and np.isnan(smi)):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = _SALT_REMOVER.StripMol(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    # Filter out fragments / covalent-warhead-only hits / giant peptides
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < 6 or n_heavy > 80:
        return None
    try:
        mol = _neutralize(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol, canonical=True)


# ----------------------------------------------------------------------
# STEP 1 — DATA CLEANING
# ----------------------------------------------------------------------
def load_and_clean_chembl(csv_path: str) -> pd.DataFrame:
    """
    Load a ChEMBL export and return a cleaned dataframe with columns:
      ['canonical_smiles', 'standard_value', 'pIC50']
    """
    # ChEMBL CSVs are typically semicolon-delimited, but REST pulls
    # produce comma-delimited. Try both.
    try:
        df = pd.read_csv(csv_path, sep=";")
        if len(df.columns) == 1:
            df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path)

    required = {"canonical_smiles", "standard_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    n0 = len(df)

    # --- Drop obvious junk ---
    df = df.dropna(subset=["canonical_smiles", "standard_value"]).copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])

    # Keep only IC50 (if the column exists)
    if "standard_type" in df.columns:
        df = df[df["standard_type"].astype(str).str.upper() == "IC50"]
    if "standard_units" in df.columns:
        df = df[df["standard_units"].astype(str).str.lower() == "nm"]

    # Drop censored values (> and <) if a relation column is present.
    # "=" alone makes the pIC50 values comparable — censored readings
    # pollute regression training.
    if "standard_relation" in df.columns:
        df = df[df["standard_relation"].astype(str).str.strip("' ").eq("=")]

    # Prefer binding assays (B) and functional (F); drop ADMET (A) and
    # physchem (P) which use very different conditions. If the column
    # is missing we skip this filter.
    if "assay_type" in df.columns:
        df = df[df["assay_type"].astype(str).str.upper().isin({"B", "F"})]

    # Activity values must be > 0 for a log transform.
    # Also clip extremes — ChEMBL often has dummy 1e9 sentinel values.
    df = df[(df["standard_value"] > 0) & (df["standard_value"] < 1e8)]

    # --- Standardize SMILES (canonicalize + strip salts + neutralize) ---
    df["canonical_smiles"] = df["canonical_smiles"].map(standardize_smiles)
    df = df.dropna(subset=["canonical_smiles"])

    # Aggregate duplicates: same molecule, multiple measurements.
    # Median is robust to the occasional bad IC50. We also drop
    # compounds with wildly inconsistent readings (>1.5 log units spread).
    df["pIC50_raw"] = 9.0 - np.log10(df["standard_value"])
    spread = df.groupby("canonical_smiles")["pIC50_raw"].agg(
        lambda s: s.max() - s.min()
    )
    inconsistent = set(spread[spread > 1.5].index)
    if inconsistent:
        print(f"[clean] dropping {len(inconsistent)} inconsistent molecules "
              f"(replicate spread > 1.5 log units)")
        df = df[~df["canonical_smiles"].isin(inconsistent)]

    df = (
        df.groupby("canonical_smiles", as_index=False)["standard_value"]
          .median()
    )

    # --- IC50 (nM) -> pIC50 ---
    # pIC50 = -log10(IC50 in Molar) = 9 - log10(IC50 in nM)
    df["pIC50"] = 9.0 - np.log10(df["standard_value"])

    # Reasonable QSAR window: trim fliers far outside kinase-relevant range
    df = df[(df["pIC50"] > 3) & (df["pIC50"] < 11)].reset_index(drop=True)

    print(f"[clean] {n0} raw rows -> {len(df)} unique cleaned molecules")
    print(df["pIC50"].describe().round(2))
    return df


# ----------------------------------------------------------------------
# STEP 2 — FEATURIZATION (Morgan fingerprints + physchem descriptors)
# ----------------------------------------------------------------------
def compute_descriptors(mol) -> np.ndarray:
    """Return a numpy float32 vector of the physchem descriptors above.
    Any descriptor that fails (rare — bad kekulization etc.) is set to 0."""
    out = np.zeros(N_DESCRIPTORS, dtype=np.float32)
    for i, (_, fn) in enumerate(DESCRIPTOR_FNS):
        try:
            v = fn(mol)
            if v is None or not np.isfinite(v):
                v = 0.0
            out[i] = float(v)
        except Exception:
            out[i] = 0.0
    return out


def smiles_to_morgan(smiles: str,
                     radius: int = MORGAN_RADIUS,
                     n_bits: int = MORGAN_NBITS):
    """Convert a single SMILES to a Morgan fingerprint numpy vector (int8)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_features(smiles: str):
    """
    Returns (X_fp, X_combined) where:
      X_fp       — (2048,) int8 Morgan fingerprint (pure, for AD check)
      X_combined — (2048 + N_DESCRIPTORS,) float32 feature vector for model
    Returns (None, None) if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
    fp_arr = np.zeros((MORGAN_NBITS,), dtype=np.int8)
    ConvertToNumpyArray(fp, fp_arr)
    desc = compute_descriptors(mol)
    combined = np.concatenate(
        [fp_arr.astype(np.float32), desc]
    ).astype(np.float32)
    return fp_arr, combined


def featurize_dataframe(df: pd.DataFrame):
    """
    Returns:
      X_combined — (n_samples, 2048 + N_DESCRIPTORS) float32 matrix for ML
      X_fp       — (n_samples, 2048) int8 fingerprint matrix (for AD check)
      y          — (n_samples,) pIC50 targets
      df         — surviving rows aligned with X/y
    """
    fps, combs, keep_idx = [], [], []
    for i, smi in enumerate(df["canonical_smiles"].values):
        fp, comb = smiles_to_features(smi)
        if fp is not None:
            fps.append(fp)
            combs.append(comb)
            keep_idx.append(i)

    X_fp       = np.vstack(fps).astype(np.int8)
    X_combined = np.vstack(combs).astype(np.float32)
    df = df.iloc[keep_idx].reset_index(drop=True)
    y = df["pIC50"].values.astype(np.float32)

    print(f"[featurize] X_combined: {X_combined.shape}  "
          f"X_fp: {X_fp.shape}  y: {y.shape}")
    return X_combined, X_fp, y, df


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    CSV_PATH = "data/egfr_chembl.csv"      # <-- your ChEMBL export
    df_clean = load_and_clean_chembl(CSV_PATH)
    X_combined, X_fp, y, df_final = featurize_dataframe(df_clean)

    np.save("models/X_features.npy", X_combined)
    np.save("models/X_morgan.npy",   X_fp)
    np.save("models/y_pic50.npy",    y)
    df_final.to_csv("data/egfr_clean.csv", index=False)
    print("[done] saved X_features.npy, X_morgan.npy, y_pic50.npy, egfr_clean.csv")
