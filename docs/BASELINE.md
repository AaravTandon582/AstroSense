# Baseline model and evaluation

## Models delivered

All models predict log10(diameter in km). Comparators and feature choices are fixed before the initial test run.

| Model | Definition | What it checks |
|---|---|---|
| `median_log` | Median training log-diameter for every object | No-information reference |
| `fixed_albedo_0.14` | 3.1236 − 0.5 log10(0.14) − 0.2H | Physical size–brightness relation under one explicit assumed albedo |
| `fitted_H_relation` | b − 0.2H; b = mean_train(log10(D)+0.2H) | Same slope with an intercept learned from the sample |
| `H_only_ridge` | Standardized H → ridge regression | A strong fitted magnitude-only comparator |
| `engineered_ridge` | All 13 engineered features → ridge regression | Whether orbital representations improve this simple baseline |

The fixed-albedo equation follows the [CNEOS size relation](https://cneos.jpl.nasa.gov/tools/ast_size_est.html); 0.14 is a declared demonstration assumption, not an asteroid-specific albedo measurement. The fitted intercept is not interpreted as a measured population albedo.

Ridge is chosen for a transparent, inexpensive baseline with correlated derived inputs. It minimizes squared log-residuals plus alpha times the squared coefficient norm, with an unpenalized intercept. Both ridge models independently search alpha in {0.1, 1, 10, 100} using validation log-RMSE. Exact score ties favor larger alpha. An SVD solver and one BLAS thread are used in the CLI; all feature scaling is fitted on training only. The package lock records the tested implementation. [Estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

## Evaluation outputs

Primary metric: **RMSE in log10(km)**, which measures errors on a relative size scale. Also report log-MAE, km-MAE and km-RMSE, R² in log space, median absolute percentage error, median multiplicative error, and fraction within a factor of two. A median multiplicative error of 1.3 means the median prediction differs from the catalog diameter by a factor of 1.3 in either direction.

Kilometer errors can be dominated by large asteroids; percentage metrics emphasize relative errors. R² is marked null for constant-target/single-object subgroups. Train, validation, and test metrics are separated. The report includes exploratory orbit-class diagnostics for every baseline with sample counts, not class-specific model tuning. `class_distribution.csv` records all split counts and population shares, including classes absent from test. `orbit_class_metrics.csv` records all-model test subgroup metrics. `error_examples.csv` selects held-out extreme absolute errors, relative overestimates, and relative underestimates with SPK IDs, catalog names, actual/predicted diameters, and ratios. The selection rules and class-imbalance limitations are stated in the generated report; no resampling or class weighting is applied.

Use 1,000 paired bootstrap resamples of test IDs to estimate a 95% percentile interval for engineered-minus-H-only log-RMSE. Negative values favor engineered features. This measures test-object sampling uncertainty conditional on the fitted models and catalog sample, not orbit/diameter measurement uncertainty, retraining variation, or future-survey performance. Test results are not used to choose a different feature set or reselect a penalty.

## Reusable model

`model.json` contains the feature-schema version, feature order, imputer medians, scaling parameters, ridge coefficients/intercept, target transform, and source snapshot hash. It is a portable numeric artifact, without executable pickle payloads. The inference command reconstructs the same feature matrix and applies those saved parameters. Training verifies its predictions against the sklearn pipeline on every accepted row before accepting the run.

Inference requires SPK ID, H, a, e, i, om, w, and equinox. A diameter column is unnecessary. Returned predictions are in km and sorted by SPK ID. Rejected IDs are written separately. Predictions outside the training population deserve independent validation; this baseline does not expose calibrated prediction intervals.

## Interpretation

The target is a heterogeneous catalog estimate, not a uniformly independent physical ground truth. Brightness/diameter relationships, selection into the measured-size subset, orbital-family structure, and common data-reduction assumptions can all affect scores. Excluding albedo reduces a direct circularity risk but does not remove every shared assumption. A good fit is not evidence that orbital elements cause diameter, nor that the model is accurate for unmeasured or undiscovered asteroids.

The delivered engineered model remains a baseline even if H-only wins. A subsequent improvement study should use development-only ablations, provenance-aware diameter subsets, and survey/family/time holdouts before seeking stronger scientific claims.
