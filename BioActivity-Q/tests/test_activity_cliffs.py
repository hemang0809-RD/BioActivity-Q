"""Tests for activity cliff detection."""
import numpy as np
from rdkit import DataStructs
from src.inference.activity_cliffs import find_activity_cliffs


def _make_bv(bits, nbits=2048):
    bv = DataStructs.ExplicitBitVect(nbits)
    bv.SetBitsFromList(bits)
    return bv


def test_no_cliffs_when_similar_activity():
    """Identical structure + similar pIC50 => no cliff."""
    bv = _make_bv([1, 5, 10, 50, 100])
    train_bv = _make_bv([1, 5, 10, 50, 100])
    cliffs = find_activity_cliffs(
        query_bv=bv, query_pIC50=7.0,
        train_bvs=[train_bv], train_pIC50s=np.array([7.1]),
        train_smiles=np.array(["CCO"]),
        sim_threshold=0.80, delta_threshold=1.0,
    )
    assert len(cliffs) == 0


def test_cliff_detected():
    """Identical structure + large pIC50 delta => cliff."""
    bv = _make_bv([1, 5, 10, 50, 100])
    train_bv = _make_bv([1, 5, 10, 50, 100])
    cliffs = find_activity_cliffs(
        query_bv=bv, query_pIC50=8.5,
        train_bvs=[train_bv], train_pIC50s=np.array([5.0]),
        train_smiles=np.array(["CCO"]),
        sim_threshold=0.80, delta_threshold=1.0,
    )
    assert len(cliffs) == 1
    assert cliffs[0]["delta_pIC50"] == 3.5


def test_no_cliff_when_dissimilar():
    """Dissimilar structure => no cliff regardless of delta."""
    bv = _make_bv([1, 2, 3])
    train_bv = _make_bv([500, 600, 700])
    cliffs = find_activity_cliffs(
        query_bv=bv, query_pIC50=9.0,
        train_bvs=[train_bv], train_pIC50s=np.array([4.0]),
        train_smiles=np.array(["CCO"]),
        sim_threshold=0.80, delta_threshold=1.0,
    )
    assert len(cliffs) == 0


def test_empty_training_set():
    bv = _make_bv([1, 5, 10])
    cliffs = find_activity_cliffs(
        query_bv=bv, query_pIC50=7.0,
        train_bvs=[], train_pIC50s=np.array([]),
        train_smiles=np.array([]),
    )
    assert cliffs == []
