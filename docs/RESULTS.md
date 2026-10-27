# AstroSense: baseline comparison and error analysis

AstroSense predicts **log10 catalog asteroid diameter (km)**. This is regression: class imbalance below refers to orbit-class representation, not classification targets. Accuracy, F1, and confusion matrices are not applicable.

Engineered ridge has 16.26% lower test log10-RMSE than H-only ridge. Kilometer MAE is 1.777 versus 1.679, and kilometer RMSE is 37.259 versus 5.533. Evaluate both relative and absolute error before choosing a model.

## Data and evaluation

Eligible objects: **139,204**; excluded by local cleaning: 0. Train / validation / test: **97,333 / 20,963 / 20,908**.

A stable SPK-ID hash assigns approximately 70/15/15 partitions. Imputation, scaling, and fitted coefficients use training data only. Each ridge model selects alpha from {0.1, 1, 10, 100} using validation log10-RMSE; there is no train+validation refit. The test set is used for evaluation and descriptive diagnostics, never tuning. Albedo, diameter uncertainty, object IDs, and orbit-class labels are excluded from predictors.

## Baseline approaches

| Model | Approach |
|---|---|
| median_log | Constant median training log-diameter; no-information reference |
| fixed_albedo_0.14 | Brightness relation: log10(D) = 3.1236 − 0.5 log10(0.14) − 0.2H; no fitting |
| fitted_H_relation | Same −0.2H slope, training-fitted intercept |
| H_only_ridge | Standardized absolute magnitude H; fitted linear slope and intercept |
| engineered_ridge | Ridge using H plus orbital transforms and missingness flags (13 features total) |

The fixed-albedo comparator uses an assumed albedo of 0.14, not a measured value. The physical relation is documented by [NASA/JPL CNEOS](https://cneos.jpl.nasa.gov/tools/ast_size_est.html).

Selected alpha: H-only **0.1**; engineered **100**.

## Held-out metrics

| Model | Log10 RMSE ↓ | Log10 R² ↑ | MAE (km) ↓ | RMSE (km) ↓ | Median absolute % error ↓ | Within factor 2 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| median_log | 0.28251 | -0.0035 | 2.742 | 10.193 | 35.44% | 78.60% |
| fixed_albedo_0.14 | 0.21008 | 0.4451 | 1.900 | 4.047 | 37.88% | 87.58% |
| fitted_H_relation | 0.16351 | 0.6639 | 1.701 | 3.775 | 28.08% | 97.02% |
| H_only_ridge | 0.15472 | 0.6990 | 1.679 | 5.533 | 26.21% | 97.09% |
| engineered_ridge | 0.12957 | 0.7889 | 1.777 | 37.259 | 21.81% | 98.88% |

Log10-RMSE is the selection objective; km-RMSE emphasizes large absolute misses. Median percentage error is a median, not mean MAPE. Within factor 2 means 0.5 ≤ predicted/actual ≤ 2.

| Model | Train log10 RMSE | Validation log10 RMSE | Test log10 RMSE |
|---|---:|---:|---:|
| median_log | 0.28181 | 0.29067 | 0.28251 |
| fixed_albedo_0.14 | 0.21006 | 0.21119 | 0.21008 |
| fitted_H_relation | 0.16193 | 0.16268 | 0.16351 |
| H_only_ridge | 0.15373 | 0.15563 | 0.15472 |
| engineered_ridge | 0.12951 | 0.13207 | 0.12957 |

Engineered minus H-only test log10-RMSE: **-0.025151**; paired bootstrap 95% interval **[-0.026237, -0.023962]** (1,000 resamples, seed 20261019). This quantifies test-object sampling conditional on the fitted models, not retraining variation, measurement uncertainty, or survey shift.

## Orbit-class imbalance

| Class | Train | Validation | Test | Total | Share of eligible sample |
|---|---:|---:|---:|---:|---:|
| MBA | 88,881 | 19,131 | 19,147 | 127,159 | 91.3472% |
| OMB | 5,534 | 1,147 | 1,112 | 7,793 | 5.5983% |
| TJN | 1,280 | 306 | 293 | 1,879 | 1.3498% |
| APO | 520 | 141 | 108 | 769 | 0.5524% |
| MCA | 433 | 79 | 84 | 596 | 0.4281% |
| IMB | 311 | 67 | 73 | 451 | 0.3240% |
| AMO | 236 | 50 | 61 | 347 | 0.2493% |
| ATE | 86 | 26 | 16 | 128 | 0.0920% |
| CEN | 38 | 9 | 11 | 58 | 0.0417% |
| TNO | 9 | 3 | 1 | 13 | 0.0093% |
| AST | 5 | 4 | 2 | 11 | 0.0079% |

The largest class, **MBA**, contributes **91.35%** of the eligible sample; largest/smallest nonempty class ratio: **11,559.9:1**. Each training object has equal weight; no class weighting, oversampling, or undersampling is applied. The ID split is not stratified, so rare classes can have very few or zero test examples. Overall metrics primarily describe the dominant population.

Codes follow [JPL orbit-class definitions](https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html): MBA = main-belt asteroid, OMB = outer main-belt asteroid, IMB = inner main-belt asteroid, MCA = Mars-crossing asteroid, TJN = Jupiter Trojan, CEN = Centaur, TNO = trans-Neptunian object; AMO/APO/ATE = Amor/Apollo/Aten, AST = other asteroid, IEO = Atira. UNKNOWN, if present, denotes a missing catalog label.

### Test performance by orbit class

| Class | Test n | H-only log10 RMSE | Engineered log10 RMSE | H-only within factor 2 | Engineered within factor 2 |
|---|---:|---:|---:|---:|---:|
| MBA | 19,147 | 0.15121 | 0.12768 | 97.64% | 99.13% |
| OMB | 1,112 | 0.14568 | 0.09927 | 98.74% | 99.10% |
| TJN | 293 | 0.17052 | 0.15354 | 97.95% | 99.66% |
| APO | 108 | 0.31494 | 0.25521 | 60.19% | 75.00% |
| MCA | 84 | 0.21206 | 0.16467 | 79.76% | 96.43% |
| IMB | 73 | 0.35358 | 0.19285 | 35.62% | 94.52% |
| AMO | 61 | 0.25386 | 0.19304 | 77.05% | 90.16% |
| ATE | 16 | 0.36759 | 0.32533 | 31.25% | 75.00% |
| CEN | 11 | 0.24336 | 0.60376 | 63.64% | 9.09% |
| TNO | 1 | 0.43848 | 0.83723 | 0.00% | 0.00% |
| AST | 2 | 0.09094 | 0.36934 | 100.00% | 50.00% |

These exploratory subgroup results were added after the original evaluation. Small groups cannot support reliable population claims; R² is undefined for one-object or constant-target groups. Full subgroup metrics for all five approaches are in `orbit_class_metrics.csv`.

## Traceable error examples

For each model, `error_examples.csv` selects the two largest absolute km errors, the two largest positive log residuals (overestimates), and the two most negative log residuals (underestimates), using SPK ID to break ties. Only held-out objects are eligible. A case may meet multiple rules. These are extreme failures, not a representative sample. Below are the rank-one examples for the two ridge models; every estimate is in km.

| Model / selection | SPK ID / catalog name | Class | Actual | Predicted | Absolute error | Predicted/actual |
|---|---|---|---:|---:|---:|---:|
| H_only_ridge / largest_km_error | 20020000 / 20000 Varuna (2000 WR106) | TNO | 900.0000 | 327.9186 | 572.0814 | 0.364× |
| H_only_ridge / largest_relative_overestimate | 50760982 / (2016 TM56) | APO | 0.0080 | 0.0519 | 0.0439 | 6.483× |
| H_only_ridge / largest_relative_underestimate | 20409973 / 409973 (2006 VF69) | MBA | 9.8530 | 2.1057 | 7.7473 | 0.214× |
| engineered_ridge / largest_km_error | 20020000 / 20000 Varuna (2000 WR106) | TNO | 900.0000 | 6186.9216 | 5286.9216 | 6.874× |
| engineered_ridge / largest_relative_overestimate | 50404117 / (2007 VH305) | CEN | 23.8000 | 168.9850 | 145.1850 | 7.100× |
| engineered_ridge / largest_relative_underestimate | 20363505 / 363505 (2003 UC20) | ATE | 1.8760 | 0.2886 | 1.5874 | 0.154× |

Ratios below one are underestimates. These diagnostics do not identify physical causes for individual failures. No test outliers were removed and no model was retuned after inspection.

## Limitations

- **Selected population:** only catalog asteroids with positive diameter, defined H, positive semimajor axis, and bound orbits (0 ≤ e < 1) enter this sample. Results do not establish performance on unknown-size or undiscovered asteroids.
- **Imbalanced coverage:** dominant orbit classes drive the objective. Sparse classes and extreme diameters need separate validation; good overall relative error can coexist with severe absolute errors.
- **Target provenance:** catalog diameters come from heterogeneous methods and can share assumptions with H. Excluding albedo limits direct circularity but does not establish independent size measurement.
- **Holdout scope:** random-ID partitions do not separate dynamical families, discovery surveys, or time. Related objects may cross partitions; no common-epoch propagation is performed.
- **Model scope:** simple linear models in log space can miss nonlinear behavior and extrapolate badly. Exponentiating log predictions does not provide calibrated conditional mean diameters or prediction intervals.
- **Uncertainty:** one snapshot and one split are evaluated. The paired interval excludes refitting variation, catalog measurement uncertainty, and domain shift; subgroup comparisons are descriptive.
- **Next study:** assess nonlinear baselines, development-only feature ablations and weighting experiments, then evaluate on a fresh survey/family/time holdout. Reusing this inspected test set for tuning would invalidate its holdout role.

## Provenance and reproduction

Source: [NASA/JPL SBDB Query API](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html). Snapshot retrieved UTC: `2026-10-19T02:24:42.178983+00:00`. SHA-256: `76d739bdcaebb7ba4cd2d2f222f5370b1e8ccad3920bcfccde77a124e64eb000`. Split seed: `20261018`.

The run includes `metrics.json` (all partition metrics and validation search), `class_distribution.csv`, `orbit_class_metrics.csv`, `error_examples.csv`, `predictions.csv`, and `reproducibility.json` (source, environment, and artifact hashes). Use the cached snapshot and a new output directory to reproduce; no live download is required.

## Verified local run

Re-executed offline in `runs/comparison_2026-10-26` using the original snapshot, seed, and model settings. Eight tests passed, including held-out-only error selection, missing/absent class handling, pooled subgroup metric consistency, deterministic reporting, and artifact integrity. Metrics, both model artifacts, features, splits, predictions, audit, and exclusions were byte-identical to the original baseline. All hashes recorded in the new run manifest were verified. The original data and run directories were preserved.

Shareable evidence: [class counts](evidence/class_distribution.csv), [all-model subgroup metrics](evidence/orbit_class_metrics.csv), [error cases](evidence/error_examples.csv), and [run provenance](evidence/reproducibility.json). Aggregate scores remain in [BASELINE_RUN.json](BASELINE_RUN.json). Full local artifacts: [comparison run](../runs/comparison_2026-10-26/RESULTS.md). Raw data and full runs are git-ignored; archive them separately when sharing.

```bash
# From the project root, using the saved environment and snapshot.
# Pick an output directory that does not already exist.
.venv/bin/python -m astrosense train \
  --snapshot data/raw/jpl_sbdb_2026-10-18 \
  --output runs/comparison_replay --seed 20261018 --bootstrap 1000
.venv/bin/python -m pytest -q
```
