"""Command line: fetch once, train offline, and predict from raw feature CSVs."""

import argparse
from importlib.metadata import version
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .data import clean, features, fetch, read_snapshot, sha256, split_ids, write_json, FEATURE_SCHEMA
from .model import export_model, fit_model, metrics, paired_interval, predict_exported
from .reporting import diagnostics, render_report


def train(snapshot, output, seed=20261018, bootstrap=1000):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Run directory exists: {output}; choose a new output path")
    raw, manifest = read_snapshot(snapshot)
    data, excluded, audit = clean(raw)
    X = features(data)
    y = np.log10(data.diameter)
    splits = split_ids(data.spkid, seed)
    masks = {name: splits.eq(name) for name in ["train", "validation", "test"]}
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    if min(counts.values()) < 10:
        raise ValueError(f"At least 10 objects per partition required; got {counts}")
    if np.var(y[masks["train"]]) == 0:
        raise ValueError("Training target is constant")
    if bootstrap < 100:
        raise ValueError("At least 100 bootstrap resamples required")
    tr, va, te = (masks[name] for name in ["train", "validation", "test"])
    h_model, h_search, h_alpha = fit_model(X.loc[tr, ["H"]], y[tr], X.loc[va, ["H"]], y[va])
    full_model, full_search, full_alpha = fit_model(X[tr], y[tr], X[va], y[va])
    fitted_h_intercept = float(np.mean(y[tr] + .2 * data.H[tr]))
    predictions = {
        "median_log": np.full(len(data), float(np.median(y[tr]))),
        "fixed_albedo_0.14": 3.1236 - .5 * np.log10(.14) - .2 * data.H.to_numpy(),
        "fitted_H_relation": fitted_h_intercept - .2 * data.H.to_numpy(),
        "H_only_ridge": h_model.predict(X[["H"]]),
        "engineered_ridge": full_model.predict(X),
    }
    scores = {name: {part: metrics(y[mask], predicted[mask]) for part, mask in masks.items()}
              for name, predicted in predictions.items()}
    comparison = paired_interval(y[te], predictions["engineered_ridge"][te],
                                 predictions["H_only_ridge"][te], seed + 1, bootstrap)
    model_json = export_model(full_model, X.columns, manifest["sha256"])
    # Verify the portable artifact on every row before writing an accepted run.
    np.testing.assert_allclose(predict_exported(model_json, X), predictions["engineered_ridge"], rtol=1e-10, atol=1e-10)
    subgroups = {}
    for label, group in data[te].groupby("class", dropna=False):
        ix = group.index
        subgroups[str(label)] = metrics(y.loc[ix], predictions["engineered_ridge"][ix])
    report = {
        "task": "Catalog diameter regression; not independent diameter measurement",
        "feature_schema": FEATURE_SCHEMA, "snapshot_sha256": manifest["sha256"],
        "seed": seed, "split_counts": counts, "audit": audit, "metrics": scores,
        "selection": {"metric": "validation RMSE in log10(km)",
                      "H_only": {"alpha": h_alpha, "candidates": h_search},
                      "engineered": {"alpha": full_alpha, "candidates": full_search}},
        "fitted_H_intercept": fitted_h_intercept, "paired_test_comparison": comparison,
        "test_orbit_class_diagnostics": subgroups,
        "fit_policy": "Train-only fitting; no train+validation refit; test not used for selection",
    }
    output.mkdir(parents=True)
    write_json(output / "metrics.json", report)
    write_json(output / "model.json", model_json)
    write_json(output / "H_only_model.json", export_model(h_model, ["H"], manifest["sha256"]))
    write_json(output / "audit.json", audit)
    write_json(output / "source_manifest.json", manifest)
    excluded.to_csv(output / "excluded.csv", index=False)
    data[["spkid", "pdes", "epoch", "equinox", "orbit_id", "soln_date", "class"]].assign(split=splits).to_csv(output / "split_manifest.csv", index=False)
    X.assign(spkid=data.spkid, split=splits, target_log10_diameter_km=y).to_csv(output / "features.csv", index=False)
    prediction_table = data[["spkid", "diameter", "class"]].assign(split=splits)
    for name, values in predictions.items():
        prediction_table[name + "_km"] = 10 ** values
    prediction_table.to_csv(output / "predictions.csv", index=False)
    # An input-only example verifies inference without a diameter column.
    data.loc[te, ["spkid", "H", "a", "e", "i", "om", "w", "equinox"]].head(10).to_csv(output / "example_input.csv", index=False)
    class_counts, class_scores, error_cases = diagnostics(data, splits, predictions)
    class_counts.to_csv(output / "class_distribution.csv", index=False)
    class_scores.to_csv(output / "orbit_class_metrics.csv", index=False)
    error_cases.to_csv(output / "error_examples.csv", index=False)
    (output / "RESULTS.md").write_text(render_report(report, manifest, class_counts, class_scores, error_cases))
    source_root = Path(__file__).resolve().parent
    write_json(output / "reproducibility.json", {
        "python": sys.version, "platform": platform.platform(),
        "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "threadpoolctl"]},
        "source_sha256": {p.name: sha256(p) for p in sorted(source_root.glob("*.py"))},
        "requirements_lock_sha256": sha256(source_root.parent / "requirements.lock.txt") if (source_root.parent / "requirements.lock.txt").exists() else None,
        "artifacts_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    })
    return report


def predict(model_path, input_path, output):
    dest = Path(output)
    if dest.exists() or dest.with_suffix(".excluded.csv").exists():
        raise FileExistsError("Prediction output exists; choose a new path")
    payload = json.loads(Path(model_path).read_text())
    raw = pd.read_csv(input_path, dtype={"spkid": "string"})
    data, excluded, _ = clean(raw, require_target=False)
    log_prediction = predict_exported(payload, features(data))
    diameter = 10.0 ** log_prediction
    if not np.isfinite(diameter).all() or (diameter <= 0).any():
        raise ValueError("Predicted diameter overflow/underflow")
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"spkid": data.spkid, "predicted_diameter_km": diameter}).to_csv(dest, index=False)
    excluded.to_csv(dest.with_suffix(".excluded.csv"), index=False)
    return len(data), len(excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    download = subs.add_parser("fetch", help="Download one cached JPL snapshot")
    download.add_argument("--output", required=True)
    download.add_argument("--contact", required=True, help="Contact URL/email for the JPL User-Agent")
    fitting = subs.add_parser("train", help="Clean, engineer, split, fit, and evaluate offline")
    fitting.add_argument("--snapshot", required=True)
    fitting.add_argument("--output", required=True)
    fitting.add_argument("--seed", type=int, default=20261018)
    fitting.add_argument("--bootstrap", type=int, default=1000)
    inference = subs.add_parser("predict", help="Predict from raw fields without target diameter")
    inference.add_argument("--model", required=True)
    inference.add_argument("--input", required=True)
    inference.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        print(fetch(args.output, args.contact))
    elif args.command == "train":
        # Keep BLAS execution reproducible and avoid excessive parallel overhead.
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            report = train(args.snapshot, args.output, args.seed, args.bootstrap)
        print(json.dumps({"split_counts": report["split_counts"], "test": report["metrics"]["engineered_ridge"]["test"]}, indent=2))
    else:
        accepted, rejected = predict(args.model, args.input, args.output)
        print(f"Predicted {accepted} objects; excluded {rejected}; output: {args.output}")


if __name__ == "__main__":
    main()
