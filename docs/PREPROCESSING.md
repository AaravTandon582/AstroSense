# Preprocessing and data quality

## Acquisition and eligible population

Request asteroids only from JPL's SBDB Query API, with a positive catalog diameter, defined H, positive semimajor axis, and 0≤e<1. Request full numerical precision and retrieve the complete matching response in one call. No first-N truncation or paging is used; pagination could combine different database states. API constraints are recorded in the source manifest. [JPL filter syntax](https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html).

Save the exact JSON bytes, retrieval time in UTC, URL, parameters, API signature, response headers, record count, and SHA-256. Training checks the checksum and API schema before reading rows. The `count` must equal the number returned, and field order must match the expected schema. A schema/version mismatch fails explicitly. The fetch command refuses to overwrite an existing snapshot.

## Cleaning policy

1. Require nonempty, unique SPK IDs. Repeated identities stop the run rather than crossing partitions or silently choosing an epoch.
2. Convert numeric strings explicitly. Null, unparsable, and nonfinite values become missing; malformed numeric counts are saved.
3. Exclude rows missing finite H, missing/nonpositive a, invalid eccentricity, or non-J2000 reference frame. Training additionally excludes missing/nonpositive diameter. Inference does not require diameter.
4. Invalid optional angles become missing, with counts. No diameter-tail clipping, arbitrary magnitude cutoff, or statistical outlier deletion is used.
5. Save excluded IDs and all applicable reasons; preserve the immutable raw snapshot. Sort accepted IDs before downstream output to stabilize row order.

The same cleaning and feature functions serve training and inference. Predictor files must include columns for i, om, and w, but their values may be missing. Required numeric features cannot be imputed because the orbital transforms and comparison models require them. Unsupported frames must be converted upstream, not merely relabeled.

## Partitions and leakage boundaries

For each ID compute SHA-256 of `20261018:<SPK ID>`. Convert the first 16 hexadecimal digits to an unsigned integer and divide by 2⁶⁴. Assign [0,.70) to training, [.70,.85) to validation, and [.85,1) to testing. These are approximate 70/15/15 fractions, not exact counts. Membership is independent of file order and target values and remains stable when new IDs are appended. At least ten objects per partition are required.

Feature formulas and physical validity rules are stateless, so they may be applied before splitting. **Learned medians and scaling statistics are fitted on training only.** For each candidate ridge model, the sklearn pipeline fits a median `SimpleImputer`, a `StandardScaler`, and the estimator on the training partition. Entirely missing training columns are retained with a zero imputation value. The three explicit missing-angle flags always exist, including if training has no missing angles.

Imputing sine/cosine coordinates need not produce a point on the unit circle. These values represent missing observations, accompanied by flags; they are not reconstructed orbital angles. Missingness patterns can encode selection effects and may change in a future catalog.

Select regularization using validation log-RMSE only. Retain the fitted training model: do not refit on train+validation or tune on test. No target standardization, resampling, feature selection, or uncertainty-based weighting is applied. Separate artifacts record partition membership, preprocessing parameters, and every validation candidate.

The split prevents repeated-identity overlap but does not separate asteroid dynamical families, discovery surveys, or epochs. It is a same-catalog object holdout, not a prospective or population-shift validation.
