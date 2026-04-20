"""
SHAP explanations on Morgan fingerprints — maps the most influential
fingerprint bits back to substructures (atom environments) that set
them. Answers: "which fragments made the model predict active?"
"""
import numpy as np
import shap
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.DataStructs import ConvertToNumpyArray


def explain_prediction(model, smiles: str, top_k: int = 5,
                       render_fragments: bool = False,
                       out_dir: str = "shap_fragments"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, 2, nBits=2048, bitInfo=bit_info
    )
    arr = np.zeros((2048,), dtype=np.int8)
    ConvertToNumpyArray(fp, arr)

    # TreeSHAP — exact Shapley values, fast on XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(arr.reshape(1, -1))[0]

    # Focus only on bits that are ON in this molecule
    on_bits = np.where(arr == 1)[0]
    contribs = [(int(b), float(shap_values[b])) for b in on_bits]
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)

    base = float(np.array(explainer.expected_value).flatten()[0])
    print(f"\nTop {top_k} contributing fragments for: {smiles}")
    print(f"Base value (dataset mean pIC50): {base:.2f}")
    print("-" * 64)
    for rank, (bit, contrib) in enumerate(contribs[:top_k], 1):
        direction = "active   (+)" if contrib > 0 else "inactive (-)"
        n_occ = len(bit_info.get(bit, []))
        print(f"  #{rank:<2} bit {bit:<5} delta={contrib:+.3f}  "
              f"{direction}  (occurs {n_occ}x in molecule)")

    # Optional: render the substructures behind each top bit
    if render_fragments:
        from pathlib import Path
        Path(out_dir).mkdir(exist_ok=True)
        for rank, (bit, contrib) in enumerate(contribs[:top_k], 1):
            if bit not in bit_info:
                continue
            try:
                img = Draw.DrawMorganBit(mol, bit, bit_info, useSVG=False)
                fname = f"{out_dir}/rank{rank}_bit{bit}_{contrib:+.2f}.png"
                img.save(fname)
                print(f"  [render] {fname}")
            except Exception as e:
                print(f"  [render] skipped bit {bit}: {e}")

    return contribs[:top_k]
