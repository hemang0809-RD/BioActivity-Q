"""
Bemis-Murcko scaffold split — groups molecules by core scaffold, then
splits so that no scaffold appears in both train and test. This is the
OECD-aligned way to evaluate QSAR generalization to novel chemotypes.
"""
from collections import defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality
    )


def scaffold_split(smiles_list, test_frac: float = 0.20, seed: int = 42):
    """
    Returns (train_idx, test_idx). Large scaffold groups go to train first;
    smaller / singleton groups fill the test set. This mimics the real
    scenario of predicting on *new chemistry*.
    """
    scaffolds = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffolds[generate_scaffold(smi)].append(i)

    rng = np.random.default_rng(seed)
    groups = sorted(
        scaffolds.values(),
        key=lambda g: (len(g), rng.random()),
        reverse=True,
    )

    n_total = len(smiles_list)
    n_test_target = int(n_total * test_frac)
    train_idx, test_idx = [], []

    for group in groups:
        if len(test_idx) + len(group) <= n_test_target:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    print(f"[scaffold-split] {len(scaffolds)} unique scaffolds | "
          f"train={len(train_idx)}  test={len(test_idx)}")
    return np.array(train_idx), np.array(test_idx)
