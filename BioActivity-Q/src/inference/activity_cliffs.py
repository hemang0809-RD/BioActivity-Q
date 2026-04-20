"""Activity cliff detection utilities."""
from __future__ import annotations

from typing import List, Dict, Sequence
from rdkit import DataStructs


def find_activity_cliffs(
    query_bv,
    query_pIC50: float,
    train_bvs: Sequence,
    train_pIC50s: Sequence[float],
    train_smiles: Sequence[str],
    sim_threshold: float = 0.80,
    delta_threshold: float = 1.0,
) -> List[Dict]:
    """Return structurally similar compounds with large activity deltas."""
    sims = DataStructs.BulkTanimotoSimilarity(query_bv, train_bvs)
    cliffs = []
    for i, sim in enumerate(sims):
        delta = abs(float(query_pIC50) - float(train_pIC50s[i]))
        if sim >= sim_threshold and delta >= delta_threshold:
            cliffs.append(
                {
                    "train_smiles": train_smiles[i],
                    "similarity": round(float(sim), 3),
                    "train_pIC50": round(float(train_pIC50s[i]), 3),
                    "delta_pIC50": round(delta, 3),
                }
            )
    cliffs.sort(key=lambda x: (-x["delta_pIC50"], -x["similarity"]))
    return cliffs
