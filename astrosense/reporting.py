"""Deterministic, descriptive diagnostics; never used for fitting or selection."""

import numpy as np
import pandas as pd

from .model import metrics

PARTITIONS = ("train", "validation", "test")


def diagnostics(data, splits, predictions):
    """Return population counts, all-model test subgroup scores, and error cases."""
    classes = data["class"].fillna("UNKNOWN").astype(str)
    counts = pd.crosstab(classes, splits).reindex(columns=PARTITIONS, fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts["percent"] = 100 * counts.total / len(data)
    counts = counts.sort_values(["total", "class"], ascending=[False, True]).reset_index()
    te = splits.eq("test")
    y = np.log10(data.diameter)
    subgroup_rows = []
    for label in counts["class"]:
        mask = te & classes.eq(label)
        for model, predicted in predictions.items():
            # Retain classes absent from test with n=0, rather than hiding them.
            row = {"class": label, "model": model, "n": int(mask.sum())}
            if mask.any():
                row.update(metrics(y[mask], np.asarray(predicted)[mask]))
            subgroup_rows.append(row)
    examples = []
    for model, predicted in predictions.items():
        cases = data.loc[te, ["spkid", "full_name", "diameter", "H", "a", "e"]].copy()
        cases["class"] = classes[te]
        cases["model"] = model
        cases["predicted_km"] = 10.0 ** np.asarray(predicted)[te]
        cases["signed_log10_error"] = np.asarray(predicted)[te] - y[te]
        cases["absolute_error_km"] = abs(cases.predicted_km - cases.diameter)
        cases["prediction_to_actual_ratio"] = cases.predicted_km / cases.diameter
        # Fixed descriptive rules, with SPK ID breaking ties reproducibly.
        for rule, column, ascending in [
            ("largest_km_error", "absolute_error_km", False),
            ("largest_relative_overestimate", "signed_log10_error", False),
            ("largest_relative_underestimate", "signed_log10_error", True),
        ]:
            eligible = cases
            if rule == "largest_relative_overestimate":
                eligible = cases[cases.signed_log10_error > 0]
            elif rule == "largest_relative_underestimate":
                eligible = cases[cases.signed_log10_error < 0]
            selected = eligible.sort_values([column, "spkid"], ascending=[ascending, True]).head(2)
            examples.append(selected.assign(selection_rule=rule))
    return counts, pd.DataFrame(subgroup_rows), pd.concat(examples, ignore_index=True)


def render_report(report, manifest, counts, subgroups, examples):
    """Render the same auditable report on every training run."""
    scores = report["metrics"]
    h, full = (scores[m]["test"] for m in ("H_only_ridge", "engineered_ridge"))
    improvement = 100 * (h["rmse_log10_km"] - full["rmse_log10_km"]) / h["rmse_log10_km"]
    lines = [
        "# AstroSense: baseline comparison and error analysis", "",
        "AstroSense predicts **log10 catalog asteroid diameter (km)**. This is regression: "
        "class imbalance below refers to orbit-class representation, not classification targets. "
        "Accuracy, F1, and confusion matrices are not applicable.", "",
        f"Engineered ridge has {improvement:.2f}% lower test log10-RMSE than H-only ridge. "
        f"Kilometer MAE is {full['mae_km']:.3f} versus {h['mae_km']:.3f}, and kilometer RMSE "
        f"is {full['rmse_km']:.3f} versus {h['rmse_km']:.3f}. "
        "Evaluate both relative and absolute error before choosing a model.", "",
        "## Data and evaluation", "",
        f"Eligible objects: **{report['audit']['accepted_rows']:,}**; "
        f"excluded by local cleaning: {report['audit']['excluded_rows']:,}. "
        f"Train / validation / test: **{report['split_counts']['train']:,} / "
        f"{report['split_counts']['validation']:,} / {report['split_counts']['test']:,}**.", "",
        "A stable SPK-ID hash assigns approximately 70/15/15 partitions. Imputation, scaling, "
        "and fitted coefficients use training data only. Each ridge model selects alpha from "
        "{0.1, 1, 10, 100} using validation log10-RMSE; there is no train+validation refit. "
        "The test set is used for evaluation and descriptive diagnostics, never tuning. "
        "Albedo, diameter uncertainty, object IDs, and orbit-class labels are excluded from predictors.", "",
        "## Baseline approaches", "",
        "| Model | Approach |", "|---|---|",
        "| median_log | Constant median training log-diameter; no-information reference |",
        "| fixed_albedo_0.14 | Brightness relation: log10(D) = 3.1236 − 0.5 log10(0.14) − 0.2H; no fitting |",
        "| fitted_H_relation | Same −0.2H slope, training-fitted intercept |",
        "| H_only_ridge | Standardized absolute magnitude H; fitted linear slope and intercept |",
        "| engineered_ridge | Ridge using H plus orbital transforms and missingness flags (13 features total) |", "",
        "The fixed-albedo comparator uses an assumed albedo of 0.14, not a measured value. "
        "The physical relation is documented by [NASA/JPL CNEOS](https://cneos.jpl.nasa.gov/tools/ast_size_est.html).", "",
        f"Selected alpha: H-only **{report['selection']['H_only']['alpha']:g}**; "
        f"engineered **{report['selection']['engineered']['alpha']:g}**.", "",
        "## Held-out metrics", "",
        "| Model | Log10 RMSE ↓ | Log10 R² ↑ | MAE (km) ↓ | RMSE (km) ↓ | Median absolute % error ↓ | Within factor 2 ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, parts in scores.items():
        m = parts["test"]
        r2 = "—" if m["r2_log10"] is None else f"{m['r2_log10']:.4f}"
        lines.append(f"| {model} | {m['rmse_log10_km']:.5f} | {r2} | {m['mae_km']:.3f} | "
                     f"{m['rmse_km']:.3f} | {m['median_absolute_percentage_error']:.2f}% | "
                     f"{100*m['fraction_within_factor_2']:.2f}% |")
    lines += ["", "Log10-RMSE is the selection objective; km-RMSE emphasizes large absolute misses. "
              "Median percentage error is a median, not mean MAPE. Within factor 2 means 0.5 ≤ predicted/actual ≤ 2.", "",
              "| Model | Train log10 RMSE | Validation log10 RMSE | Test log10 RMSE |",
              "|---|---:|---:|---:|"]
    for model, parts in scores.items():
        lines.append(f"| {model} | " + " | ".join(f"{parts[p]['rmse_log10_km']:.5f}" for p in PARTITIONS) + " |")
    comparison = report["paired_test_comparison"]
    lo, hi = comparison["paired_bootstrap_95_interval"]
    lines += ["", f"Engineered minus H-only test log10-RMSE: **{comparison['delta_rmse_log10_km']:.6f}**; "
              f"paired bootstrap 95% interval **[{lo:.6f}, {hi:.6f}]** "
              f"({comparison['resamples']:,} resamples, seed {comparison['seed']}). "
              "This quantifies test-object sampling conditional on the fitted models, not retraining "
              "variation, measurement uncertainty, or survey shift.", "",
              "## Orbit-class imbalance", "",
              "| Class | Train | Validation | Test | Total | Share of eligible sample |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in counts.to_dict("records"):
        lines.append(f"| {row['class']} | {row['train']:,} | {row['validation']:,} | {row['test']:,} | "
                     f"{row['total']:,} | {row['percent']:.4f}% |")
    largest = counts.iloc[0]
    lines += ["", f"The largest class, **{largest['class']}**, contributes **{largest['percent']:.2f}%** "
              f"of the eligible sample; largest/smallest nonempty class ratio: "
              f"**{largest['total'] / counts.total.min():,.1f}:1**. "
              "Each training object has equal weight; no class weighting, oversampling, or undersampling is applied. "
              "The ID split is not stratified, so rare classes can have very few or zero test examples. "
              "Overall metrics primarily describe the dominant population.", "",
              "Codes follow [JPL orbit-class definitions](https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html): "
              "MBA = main-belt asteroid, OMB = outer main-belt asteroid, IMB = inner main-belt asteroid, "
              "MCA = Mars-crossing asteroid, TJN = Jupiter Trojan, CEN = Centaur, TNO = trans-Neptunian object; "
              "AMO/APO/ATE = Amor/Apollo/Aten, AST = other asteroid, IEO = Atira. "
              "UNKNOWN, if present, denotes a missing catalog label.", "",
              "### Test performance by orbit class", "",
              "| Class | Test n | H-only log10 RMSE | Engineered log10 RMSE | H-only within factor 2 | Engineered within factor 2 |",
              "|---|---:|---:|---:|---:|---:|"]
    for label in counts["class"]:
        group = subgroups[subgroups["class"].eq(label)].set_index("model")
        left, right = (group.loc[m] for m in ("H_only_ridge", "engineered_ridge"))
        if not left["n"]:
            lines.append(f"| {label} | 0 | — | — | — | — |")
        else:
            lines.append(f"| {label} | {int(left['n']):,} | {left['rmse_log10_km']:.5f} | "
                         f"{right['rmse_log10_km']:.5f} | {100*left['fraction_within_factor_2']:.2f}% | "
                         f"{100*right['fraction_within_factor_2']:.2f}% |")
    lines += ["", "These exploratory subgroup results were added after the original evaluation. "
              "Small groups cannot support reliable population claims; R² is undefined for one-object "
              "or constant-target groups. Full subgroup metrics for all five approaches are in `orbit_class_metrics.csv`.", "",
              "## Traceable error examples", "",
              "For each model, `error_examples.csv` selects the two largest absolute km errors, "
              "the two largest positive log residuals (overestimates), and the two most negative log residuals "
              "(underestimates), using SPK ID to break ties. Only held-out objects are eligible. "
              "A case may meet multiple rules. These are extreme failures, not a representative sample. "
              "Below are the rank-one examples for the two ridge models; every estimate is in km.", "",
              "| Model / selection | SPK ID / catalog name | Class | Actual | Predicted | Absolute error | Predicted/actual |",
              "|---|---|---|---:|---:|---:|---:|"]
    shown = examples[examples.model.isin(["H_only_ridge", "engineered_ridge"])].groupby(
        ["model", "selection_rule"], sort=False).head(1)
    for row in shown.to_dict("records"):
        name = str(row["full_name"]).strip().replace("|", "\\|")
        lines.append(f"| {row['model']} / {row['selection_rule']} | {row['spkid']} / {name} | "
                     f"{row['class']} | {row['diameter']:.4f} | {row['predicted_km']:.4f} | "
                     f"{row['absolute_error_km']:.4f} | {row['prediction_to_actual_ratio']:.3f}× |")
    lines += ["", "Ratios below one are underestimates. These diagnostics do not identify physical causes "
              "for individual failures. No test outliers were removed and no model was retuned after inspection.", "",
              "## Limitations", "",
              "- **Selected population:** only catalog asteroids with positive diameter, defined H, positive semimajor axis, "
              "and bound orbits (0 ≤ e < 1) enter this sample. Results do not establish performance on unknown-size or undiscovered asteroids.",
              "- **Imbalanced coverage:** dominant orbit classes drive the objective. Sparse classes and extreme diameters "
              "need separate validation; good overall relative error can coexist with severe absolute errors.",
              "- **Target provenance:** catalog diameters come from heterogeneous methods and can share assumptions "
              "with H. Excluding albedo limits direct circularity but does not establish independent size measurement.",
              "- **Holdout scope:** random-ID partitions do not separate dynamical families, discovery surveys, or time. "
              "Related objects may cross partitions; no common-epoch propagation is performed.",
              "- **Model scope:** simple linear models in log space can miss nonlinear behavior and extrapolate badly. "
              "Exponentiating log predictions does not provide calibrated conditional mean diameters or prediction intervals.",
              "- **Uncertainty:** one snapshot and one split are evaluated. The paired interval excludes refitting "
              "variation, catalog measurement uncertainty, and domain shift; subgroup comparisons are descriptive.",
              "- **Next study:** assess nonlinear baselines, development-only feature ablations and weighting experiments, "
              "then evaluate on a fresh survey/family/time holdout. Reusing this inspected test set for tuning would invalidate its holdout role.", "",
              "## Provenance and reproduction", "",
              f"Source: [NASA/JPL SBDB Query API](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html). "
              f"Snapshot retrieved UTC: `{manifest.get('retrieved_utc', 'not supplied')}`. "
              f"SHA-256: `{report['snapshot_sha256']}`. Split seed: `{report['seed']}`.", "",
              "The run includes `metrics.json` (all partition metrics and validation search), "
              "`class_distribution.csv`, `orbit_class_metrics.csv`, `error_examples.csv`, "
              "`predictions.csv`, and `reproducibility.json` (source, environment, and artifact hashes). "
              "Use the cached snapshot and a new output directory to reproduce; no live download is required.", ""]
    return "\n".join(lines)
