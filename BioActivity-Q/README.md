# BioActivity-Q

A QSAR model that predicts pIC50 of small molecules against **EGFR** (a major
cancer target) from ChEMBL bioactivity data. Built with RDKit + XGBoost.

## Pipeline overview

| Stage | File | Purpose |
|------:|------|---------|
| 1 | `preprocess.py` | Clean ChEMBL CSV, IC50 -> pIC50, Morgan fingerprints (r=2, 2048 bits) |
| 2 | `train.py` | XGBoost regressor + bioactivity scatter plot + 5-fold CV |
| 2.5 | `scaffold_split.py` | Bemis-Murcko scaffold split for honest generalization |
| 3 | `predict.py` | Inference with applicability-domain check + conformal CI |
| 4 | `explain.py` | SHAP fragment-level explanations |
| 5 | `conformal.py` | Inductive conformal prediction (90% intervals) |
| - | `make_bioactivity_q.py` | One-command runner that wires everything together |

## Setup

```bash
# Python 3.10+
pip install -r requirements.txt
```

## Train end-to-end

Place a ChEMBL EGFR export named `egfr_chembl.csv` in this directory
(must contain `canonical_smiles` and `standard_value` columns).

```bash
python make_bioactivity_q.py --csv egfr_chembl.csv
```

Outputs:

- `bioactivity_q.model` — trained XGBoost model
- `conformal.npz` — calibrated conformal regressor
- `bioactivity_map_scaffold.png` — actual vs. predicted plot
- `metrics.json` — R², RMSE, MAE, CV scores, conformal coverage
- `egfr_clean.csv`, `X_morgan.npy`, `y_pic50.npy` — intermediates

## Predict on a new molecule

```bash
python make_bioactivity_q.py --predict "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"
```

Add `--explain` to also see which fingerprint bits drove the prediction:

```bash
python make_bioactivity_q.py --predict "<SMILES>" --explain
```

## Programmatic API

```python
from predict import BioActivityPredictor

bq = BioActivityPredictor()
print(bq.predict("CN1C=NC2=C1C(=O)N(C(=O)N2C)C"))   # caffeine
```

## Interpreting the output

| Field | Meaning |
|-------|---------|
| `predicted_pIC50` | -log10 of predicted IC50 in molar |
| `pIC50_90CI` | 90% conformal prediction interval (mathematically guaranteed coverage) |
| `nearest_neighbor_similarity` | Max Tanimoto to training set — confidence proxy |
| `applicability_domain` | `IN DOMAIN` / `BORDERLINE` / `OUT OF DOMAIN` |
| `activity_class` | Highly Active (>=7), Active (>=6), Moderately Active (>=5), Inactive |

A wide CI **and** an `OUT OF DOMAIN` label means the model is extrapolating —
treat the point estimate as unreliable.

## Why the choices

- **Scaffold split** over random — prevents near-duplicate leakage between train/test
- **Morgan r=2 (ECFP4)** — QSAR-standard featurizer; strong baseline
- **XGBoost (`hist`)** — fast on sparse 2048-bit vectors; handles bit features well
- **Conformal intervals** — distribution-free, calibrated uncertainty per molecule
- **Tanimoto AD check** — independent OOD signal that corroborates the CI width
- **SHAP on bits** — turns the model into something a medicinal chemist can read
