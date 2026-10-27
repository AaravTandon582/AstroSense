# Reproducibility notes

## What must be preserved

- `data/raw/<snapshot>/sbdb.json`: exact downloaded source bytes.
- `data/raw/<snapshot>/manifest.json`: query/filter, UTC retrieval time, API version, count, and checksum.
- `requirements.lock.txt`: exact installed package versions, including transitive dependencies and the CA bundle.
- The `astrosense/` source files and tests; `runs/<run>/reproducibility.json` records source hashes, Python/platform, package versions, lockfile hash, and hashes of the core artifacts written before the summary report.
- The whole run directory: split manifest, feature table, exclusions, preprocessing statistics, alpha search, model, predictions, metrics, bootstrap settings, `class_distribution.csv`, `orbit_class_metrics.csv`, `error_examples.csv`, and the generated report. New runs include the report and diagnostic CSV hashes in `reproducibility.json`.

Python 3.13 on macOS arm64 is the initial tested environment. The lock is a version pin set, not a cross-platform wheel archive or dependency-hash lock. Different operating systems/BLAS libraries may introduce tiny floating-point differences. The CLI limits BLAS execution to one thread. Exact source, input hash, split algorithm, seed, feature order, and package versions are required for a meaningful replay.

## Replay without network

From the project root after installing the lockfile:

```bash
.venv/bin/python -m astrosense train \
  --snapshot data/raw/jpl_sbdb_2026-10-18 \
  --output runs/replay --seed 20261018 --bootstrap 1000

.venv/bin/python -m pytest -q
```

Compare `metrics.json`, `split_manifest.csv`, model parameters, and predictions with the original run. The offline integration test independently repeats a synthetic-fixture run and verifies identical reports plus agreement of prediction-only inference with the training pipeline. Synthetic fixtures exist only in tests and are not reported as astronomy results.

The delivered real-data run was also replayed in full. Metrics, both ridge artifacts, features, split manifest, predictions, audit, and exclusions were byte-identical between `runs/baseline` and `runs/replay`. A compact, shareable record is [BASELINE_RUN.json](BASELINE_RUN.json); the interpretation is in [RESULTS.md](RESULTS.md). This verified replay applies to the saved environment and snapshot, not every future platform or catalog version.

Do not call `fetch` to reproduce an old result. The live catalog can change; the same query later may return new or revised solutions. Existing output directories are refused instead of overwritten. To refresh data, choose a new snapshot path, retain both manifests, and treat it as a new dataset version.

## Fetch behavior and service policy

The downloader makes one API request at a time, caches the full response, and includes an application-specific User-Agent with a caller-supplied contact. It retries only transient HTTP/network failures, at most three attempts, with backoff; long or date-form Retry-After responses stop the run for a later retry. TLS certificates are verified using certifi, including on Python installations without a system CA bundle. Verification is never disabled. [JPL service policy](https://ssd-api.jpl.nasa.gov/doc/).

No pagination or limit is applied. Unknown response versions, missing/reordered fields, empty/truncated responses, checksum changes, missing identities, and duplicate IDs fail explicitly. Authentication and paid APIs are unnecessary. The exact retrieval timestamp, not the folder name, establishes when the data were acquired.

## Audit and test scope

Tests cover orbital formulas and angle wrapping; target/metadata exclusion; invalid and missing values; repeated-ID rejection; stable split membership; training-only imputation; all-missing features; portable serialization; response truncation; checksum tampering; and offline training/prediction replay. Normal runs save every excluded ID and reason, rather than hiding data loss.

No common-epoch propagation is performed, and no temporal independence is claimed. Retain epoch, frame, orbit solution ID, and solution date for downstream work. Frame mismatches are rejected; missing optional angles are imputed. Catalog diameters are not rederived from H/albedo to fabricate training labels.

## Attribution and sharing

Cite NASA/JPL Solar System Dynamics SBDB as the data source and preserve the API documentation links and retrieval manifest. This project does not assert a license for NASA/JPL material beyond the provider's terms. Source code and small reports are separate from ignored local data/run artifacts; archive the latter explicitly when publishing a reproducible experiment.

Source documentation: [SBDB Query API](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html), [filter syntax](https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html), [physical size relation](https://cneos.jpl.nasa.gov/tools/ast_size_est.html), [imputer behavior](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html).

## Expanded comparison verification (2026-10-26)

`runs/comparison_2026-10-26` re-executes the original five approaches without changing the snapshot, splits, features, hyperparameters, or metrics. Eight core numerical artifacts match the original run byte for byte. Eight offline tests pass; new coverage verifies rare/missing class accounting, subgroup-to-pooled metric agreement, exclusion of training and validation objects from error examples, deterministic tie handling, report replay, and all recorded artifact hashes. The expanded report and small diagnostic CSVs are preserved under `docs/`; full raw data and runs remain ignored.

The new diagnostic tables are post-evaluation analysis. They do not retroactively make the inspected test set suitable for further tuning. For a replay with the expanded reports, use the command in [RESULTS.md](RESULTS.md) with a new output directory.
