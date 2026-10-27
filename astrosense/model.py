"""Training-only preprocessing, fixed validation search, and portable prediction."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURE_SCHEMA

ALPHAS = (0.1, 1.0, 10.0, 100.0)


def make_model(alpha):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, solver="svd")),
    ])


def fit_model(X_train, y_train, X_validation, y_validation):
    candidates = []
    models = {}
    for alpha in ALPHAS:
        fitted = make_model(alpha).fit(X_train, y_train)
        score = float(np.sqrt(mean_squared_error(y_validation, fitted.predict(X_validation))))
        candidates.append({"alpha": alpha, "validation_rmse_log10_km": score})
        models[alpha] = fitted
    chosen = min(candidates, key=lambda item: (item["validation_rmse_log10_km"], -item["alpha"]))
    return models[chosen["alpha"]], candidates, chosen["alpha"]


def metrics(y_log, prediction_log):
    y_log, prediction_log = np.asarray(y_log), np.asarray(prediction_log)
    actual, predicted = 10.0 ** y_log, 10.0 ** prediction_log
    if not np.isfinite(predicted).all() or (predicted <= 0).any():
        raise ValueError("Nonfinite or underflowed diameter predictions")
    return {
        "n": len(y_log),
        "rmse_log10_km": float(np.sqrt(mean_squared_error(y_log, prediction_log))),
        "mae_log10_km": float(mean_absolute_error(y_log, prediction_log)),
        "mae_km": float(mean_absolute_error(actual, predicted)),
        "rmse_km": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2_log10": float(r2_score(y_log, prediction_log)) if len(y_log) > 1 and np.var(y_log) > 0 else None,
        "median_absolute_percentage_error": float(100 * np.median(np.abs(predicted - actual) / actual)),
        "median_multiplicative_error": float(10 ** np.median(np.abs(prediction_log - y_log))),
        "fraction_within_factor_2": float(np.mean(np.abs(prediction_log - y_log) <= np.log10(2))),
    }


def export_model(model, feature_names, snapshot_sha256):
    return {
        "format": "astrosense-ridge-v1", "feature_schema": FEATURE_SCHEMA,
        "target": "log10(diameter_km)", "snapshot_sha256": snapshot_sha256,
        "feature_names": list(feature_names), "alpha": float(model.named_steps["ridge"].alpha),
        "imputer_medians": model.named_steps["impute"].statistics_.tolist(),
        "scaler_mean": model.named_steps["scale"].mean_.tolist(),
        "scaler_scale": model.named_steps["scale"].scale_.tolist(),
        "coefficients": model.named_steps["ridge"].coef_.tolist(),
        "intercept": float(model.named_steps["ridge"].intercept_),
    }


def predict_exported(payload, X):
    if payload["format"] != "astrosense-ridge-v1" or payload["feature_schema"] != FEATURE_SCHEMA:
        raise ValueError("Unsupported model or feature schema")
    values = X.loc[:, payload["feature_names"]].to_numpy(dtype=float)
    values = np.where(np.isnan(values), np.asarray(payload["imputer_medians"]), values)
    scaled = (values - payload["scaler_mean"]) / payload["scaler_scale"]
    result = scaled @ np.asarray(payload["coefficients"]) + payload["intercept"]
    if not np.isfinite(result).all():
        raise ValueError("Invalid transformed input or model")
    return result


def paired_interval(y, prediction, comparator, seed, repeats=1000):
    rng = np.random.default_rng(seed)
    a = (np.asarray(y) - np.asarray(prediction)) ** 2
    b = (np.asarray(y) - np.asarray(comparator)) ** 2
    values = np.empty(repeats)
    for n in range(repeats):
        ix = rng.integers(0, len(a), len(a))
        values[n] = np.sqrt(a[ix].mean()) - np.sqrt(b[ix].mean())
    return {
        "contrast": "engineered ridge minus H-only ridge; negative favors engineered",
        "delta_rmse_log10_km": float(np.sqrt(a.mean()) - np.sqrt(b.mean())),
        "paired_bootstrap_95_interval": np.quantile(values, [.025, .975]).tolist(),
        "resamples": repeats, "seed": seed,
        "scope": "Test-object sampling conditional on fitted models; not survey-shift uncertainty",
    }
