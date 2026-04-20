"""Sanity tests for the preprocessing pipeline."""
import numpy as np
from src.engine.preprocess import standardize_smiles, smiles_to_features


def test_standardize_valid():
    # Benzene: 6 heavy atoms — passes the minimum size filter
    result = standardize_smiles("c1ccccc1")
    assert result is not None


def test_standardize_canonical():
    # Different input forms should produce same canonical SMILES
    result1 = standardize_smiles("c1ccccc1")
    result2 = standardize_smiles("C1=CC=CC=C1")
    assert result1 == result2


def test_standardize_salt_strip():
    # Aniline + sodium salt — should strip salt, keep aniline (7 heavy atoms)
    result = standardize_smiles("c1ccc(N)cc1.[Na+]")
    assert result is not None
    assert "Na" not in result


def test_standardize_invalid():
    assert standardize_smiles("not_a_molecule") is None
    assert standardize_smiles(None) is None
    assert standardize_smiles(float("nan")) is None


def test_standardize_too_small():
    # Ethanol: only 3 heavy atoms — below the 6-atom minimum
    assert standardize_smiles("CCO") is None
    assert standardize_smiles("C") is None


def test_features_shape():
    fp, combined = smiles_to_features("c1ccccc1")  # benzene
    assert fp is not None
    assert fp.shape == (2048,)
    assert combined.shape == (2066,)  # 2048 FP bits + 18 descriptors
    assert fp.dtype == np.int8
    assert combined.dtype == np.float32


def test_features_invalid():
    fp, combined = smiles_to_features("NOT_REAL")
    assert fp is None
    assert combined is None


def test_features_bits_are_binary():
    fp, _ = smiles_to_features("c1ccc2[nH]c(-c3ccncc3)cc2c1")
    assert set(np.unique(fp)).issubset({0, 1})
