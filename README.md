# AstroSense · Astronomy / ML

An executable feature-engineering pipeline for **asteroid diameter prediction** using public NASA/JPL Small-Body Database (SBDB) orbital and astronomical data.

The target is `log10(diameter in km)`. AstroSense compares magnitude-based reference models with ridge regression using engineered orbital features. It predicts catalog diameter estimates; it does not measure sizes independently or assess collision risk.

| Deliverable | Location |
|---|---|
| Feature definitions, units, formulas, and exclusions | [docs/FEATURES.md](docs/FEATURES.md) |
| Data cleaning, splits, missing values, and scaling | [docs/PREPROCESSING.md](docs/PREPROCESSING.md) |
| Baseline models, evaluation, and interpretation | [docs/BASELINE.md](docs/BASELINE.md) |
| Snapshot provenance and exact rerun instructions | [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) |
| Feature pipeline | [astrosense/data.py](astrosense/data.py) |
| Models and portable inference | [astrosense/model.py](astrosense/model.py) |
| Offline integrity and leakage checks | [tests/test_pipeline.py](tests/test_pipeline.py) |
| Five-baseline comparison, orbit-class imbalance, errors, and limitations | [docs/RESULTS.md](docs/RESULTS.md) |

## Verified first run

The saved JPL snapshot contains **139,204 asteroids**. The baseline used 97,333 for training, 20,963 for validation, and 20,908 for testing. The engineered ridge model achieved test log10-RMSE **0.12957**, versus **0.15472** for H-only ridge; median absolute percentage error was **21.81%**, versus **26.21%**.

Kilometer MAE increased from 1.679 to 1.777, and some rare outer-system objects have severe overestimates. The [results report](docs/RESULTS.md) documents these limitations and the paired uncertainty interval. The updated comparison passes all eight offline tests. Its numerical artifacts are byte-identical to the original baseline; the original prediction-only inference and full replay also succeeded.

## Imbalance and error analysis

This is **regression**, so orbit classes describe sample coverage rather than predicted labels. Main-belt asteroids account for **91.35%** of the 139,204 eligible objects. The test set contains only **11 Centaurs and one trans-Neptunian object**. Engineered ridge improves test log10-RMSE by **16.26%** over H-only ridge, but kilometer RMSE worsens from **5.533 to 37.259**. For example, Varuna has a catalog diameter of 900 km and predictions of 327.919 km (H-only) and 6,186.922 km (engineered).

The [complete report](docs/RESULTS.md) compares all five approaches, separates train/validation/test metrics, tabulates every orbit class, and shows named overestimates and underestimates. Shareable CSV evidence contains [class counts](docs/evidence/class_distribution.csv), [subgroup metrics](docs/evidence/orbit_class_metrics.csv), and [error examples](docs/evidence/error_examples.csv). These diagnostics are produced automatically on every training run.

## Run

From the AstroSense directory, using Python 3.13 (the tested environment):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt

# Download once. Supply your own valid contact URL or email.
.venv/bin/python -m astrosense fetch \
  --output data/raw/jpl_sbdb_2026-10-18 \
  --contact https://github.com/AaravTandon582

# All subsequent work uses the cached snapshot, with no network calls.
.venv/bin/python -m astrosense train \
  --snapshot data/raw/jpl_sbdb_2026-10-18 \
  --output runs/baseline

.venv/bin/python -m astrosense predict \
  --model runs/baseline/model.json \
  --input runs/baseline/example_input.csv \
  --output runs/baseline/example_predictions.csv

.venv/bin/python -m pytest -q
```

For the delivered checkout, reuse the saved snapshot and run rather than downloading or overwriting them. Choose a new output directory to replay training. A download date in a folder name is a label; the manifest's UTC timestamp is authoritative.

## Pipeline

```text
JPL JSON response + query + checksum
  → identity/schema/physical-range validation
  → stateless features and missingness flags
  → stable asteroid-ID train/validation/test assignment
  → training-only imputation and scaling
  → validation-selected ridge penalty
  → held-out metrics, uncertainty, and portable JSON model
```

`train` writes `features.csv`, `split_manifest.csv`, `excluded.csv`, `audit.json`, predictions for all partitions, validation-search scores, a model, class counts, all-model subgroup metrics, deterministically selected test error examples, and a human-readable results report. Train metrics are marked separately from test metrics. `predict` accepts raw predictors without a diameter field and returns estimates in km, keyed by SPK ID, plus an exclusion file.

Data and run directories are git-ignored to avoid accidentally committing large snapshots. Keep or archive them when sharing results; source alone cannot reconstruct a changing catalog snapshot. A pinned environment is in [requirements.lock.txt](requirements.lock.txt).

## Scientific scope

Asteroids with known catalog diameters are a selected population. Their measurements come from heterogeneous sources and may share assumptions with magnitude estimates. Random-ID validation assesses similar catalog objects, not future discoveries or unseen surveys. The project excludes albedo because it can be jointly inferred with diameter and magnitude; row-level physical-data provenance would be needed for a stronger independent measurement study.

Sources: [JPL query API](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html), [filter definitions](https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html), and [CNEOS size estimator](https://cneos.jpl.nasa.gov/tools/ast_size_est.html).
