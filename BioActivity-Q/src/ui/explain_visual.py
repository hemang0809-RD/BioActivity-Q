"""Visual SHAP explanations — waterfall and beeswarm plots."""
import shap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from src.engine.preprocess import DESCRIPTOR_NAMES, MORGAN_NBITS, compute_descriptors


def _build_feature_names(mol, bit_info):
    """Map ON Morgan bits to human-readable substructure labels."""
    fp_names = [f"Bit_{i}" for i in range(MORGAN_NBITS)]
    for bit_idx in bit_info:
        envs = bit_info[bit_idx]
        atom_idx, radius = envs[0]
        try:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
            amap = {}
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            if submol and submol.GetNumAtoms() > 0:
                frag_smi = Chem.MolToSmiles(submol)
                fp_names[bit_idx] = f"Bit_{bit_idx} ({frag_smi})"
        except Exception:
            pass
    return fp_names + list(DESCRIPTOR_NAMES)


def explain_waterfall(model, smiles, max_display=15):
    """
    SHAP waterfall plot for a single molecule prediction.
    Shows the top features pushing prediction up/down from the mean.
    Returns a matplotlib figure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, bitInfo=bit_info)
    arr = np.zeros((2048,), dtype=np.int8)
    ConvertToNumpyArray(fp, arr)

    desc = compute_descriptors(mol)
    combined = np.concatenate([arr.astype(np.float32), desc])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(combined.reshape(1, -1))
    shap_values.feature_names = _build_feature_names(mol, bit_info)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(shap_values[0], max_display=max_display, show=False)
    plt.title(f"SHAP: {smiles[:60]}{'...' if len(smiles) > 60 else ''}")
    plt.tight_layout()
    return fig


def explain_beeswarm(model, X, max_display=20, feature_names=None):
    """
    Global beeswarm: feature importance across all test molecules.
    Each dot = one molecule. Position = SHAP value. Color = feature value.
    Returns a matplotlib figure.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    if feature_names is not None:
        shap_values.feature_names = feature_names

    fig, ax = plt.subplots(figsize=(12, 10))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title("Global Feature Importance (Beeswarm)")
    plt.tight_layout()
    return fig
