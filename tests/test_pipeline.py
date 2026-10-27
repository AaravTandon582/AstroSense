"""Offline checks of scientific transforms, leakage boundaries, and replay."""

import json

import numpy as np
import pandas as pd
import pytest

from astrosense.data import FIELDS, clean, features, read_snapshot, sha256, split_ids, validate_payload, write_json
from astrosense.model import export_model, make_model, predict_exported
from astrosense.__main__ import predict, train
from astrosense.reporting import diagnostics


def frame(n=4):
    return pd.DataFrame({"spkid": [str(2000000 + i) for i in range(n)], "H": np.linspace(10, 20, n),
                         "a": np.linspace(1, 4, n), "e": np.full(n, .2), "i": np.full(n, 20.),
                         "om": np.full(n, 30.), "w": np.full(n, 40.), "equinox": ["J2000"] * n,
                         "diameter": np.linspace(1, 20, n)})


def test_geometry_and_angle_wrap():
    data = frame(2)
    data.loc[0, ["a", "e", "i", "om", "w"]] = [2, .25, 0, 0, 360]
    data.loc[1, ["om", "w"]] = [360, 0]
    X = features(data)
    assert X.loc[0, "log10_perihelion_au"] == pytest.approx(np.log10(1.5))
    assert X.loc[0, "log10_aphelion_au"] == pytest.approx(np.log10(2.5))
    assert X.loc[0, "cos_inclination"] == pytest.approx(1)
    for name in ["sin_node", "cos_node", "sin_perihelion_argument", "cos_perihelion_argument"]:
        assert X.loc[0, name] == pytest.approx(X.loc[1, name], abs=1e-14)


def test_target_and_metadata_cannot_change_features():
    data = frame()
    first = features(data)
    data["diameter"] *= 1e6
    data["diameter_sigma"] = 123
    data["albedo"] = .9
    data["class"] = "TNO"
    pd.testing.assert_frame_equal(first, features(data))


def test_quality_rules_and_duplicate_refusal():
    data = frame()
    data.loc[0, "e"] = 1
    data.loc[1, "a"] = -1
    data.loc[2, "i"] = 190
    data.loc[3, "H"] = np.inf
    good, excluded, audit = clean(data)
    assert len(excluded) == 3 and len(good) == 1
    assert np.isnan(good.loc[0, "i"])
    assert audit["invalid_optional_angle_counts"]["i"] == 1
    assert features(good).loc[0, "missing_i"] == 1
    duplicated = pd.concat([frame(), frame().iloc[:1]])
    with pytest.raises(ValueError, match="Repeated"):
        clean(duplicated)


def test_split_independent_of_order_target_and_extension():
    ids = frame(200).spkid
    original = dict(zip(ids, split_ids(ids)))
    shuffled = ids.sample(frac=1, random_state=2)
    assert dict(zip(shuffled, split_ids(shuffled))) == original
    extended = pd.concat([ids, pd.Series(["new-object"])], ignore_index=True)
    assert all(original[i] == s for i, s in zip(extended, split_ids(extended)) if i in original)
    assert set(original.values()) == {"train", "validation", "test"}


def test_train_only_imputation_and_portable_model():
    X = pd.DataFrame({"H": [1., 2., 3., 4.], "angle": [1., np.nan, 3., 5.], "empty": [np.nan] * 4})
    model = make_model(1).fit(X, np.array([0., 1., 2., 3.]))
    assert model.named_steps["impute"].statistics_[1] == 3
    heldout = pd.DataFrame({"H": [100., 2.], "angle": [999., np.nan], "empty": [np.nan, 1.]})
    before = model.named_steps["impute"].statistics_.copy()
    prediction = model.predict(heldout)
    np.testing.assert_array_equal(before, model.named_steps["impute"].statistics_)
    payload = json.loads(json.dumps(export_model(model, X.columns, "fixture")))
    np.testing.assert_allclose(predict_exported(payload, heldout), prediction, atol=1e-12)


def snapshot_fixture(path, n=300):
    path.mkdir()
    data = frame(n)
    data["diameter"] = 10 ** (3.5 - .2 * data.H + .04 * np.sin(data.H))
    for col in FIELDS:
        if col not in data:
            data[col] = "MBA" if col == "class" else None
    payload = {"signature": {"version": "1.0", "source": "synthetic test fixture"},
               "fields": FIELDS, "count": str(n), "data": data[FIELDS].values.tolist()}
    write_json(path / "sbdb.json", payload)
    write_json(path / "manifest.json", {"sha256": sha256(path / "sbdb.json")})
    return payload


def test_truncated_schema_and_checksum_rejected(tmp_path):
    path = tmp_path / "snapshot"
    payload = snapshot_fixture(path)
    payload["count"] = "1"
    with pytest.raises(ValueError, match="truncated"):
        validate_payload(payload)
    (path / "sbdb.json").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        read_snapshot(path)


def test_offline_end_to_end_and_replay(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot_fixture(snapshot)
    report = train(snapshot, tmp_path / "run1", bootstrap=100)
    replay = train(snapshot, tmp_path / "run2", bootstrap=100)
    assert report == replay
    for filename in ["RESULTS.md", "class_distribution.csv", "orbit_class_metrics.csv", "error_examples.csv"]:
        assert (tmp_path / "run1" / filename).read_bytes() == (tmp_path / "run2" / filename).read_bytes()
    provenance = json.loads((tmp_path / "run1/reproducibility.json").read_text())
    for filename, expected in provenance["artifacts_sha256"].items():
        assert sha256(tmp_path / "run1" / filename) == expected
    model = tmp_path / "run1/model.json"
    accepted, rejected = predict(model, tmp_path / "run1/example_input.csv", tmp_path / "predictions.csv")
    assert accepted == 10 and rejected == 0
    got = pd.read_csv(tmp_path / "predictions.csv", dtype={"spkid": str})
    reference = pd.read_csv(tmp_path / "run1/predictions.csv", dtype={"spkid": str}).set_index("spkid")
    np.testing.assert_allclose(got.predicted_diameter_km, reference.loc[got.spkid, "engineered_ridge_km"], rtol=1e-12)
    with pytest.raises(FileExistsError):
        train(snapshot, tmp_path / "run1", bootstrap=100)


def test_diagnostics_counts_metrics_and_extreme_errors():
    data = frame(8)
    data["diameter"] = 10.0
    data["full_name"] = data.spkid
    data["class"] = ["RARE", "MBA", "MBA", "MBA", "MBA", "MBA", None, "MBA"]
    splits = pd.Series(["train", "train", "validation", "test", "test", "test", "test", "test"])
    # Enormous training/validation errors must never be selected as test examples.
    prediction = np.array([8., 8., 8., 2., 0., 1., 1.5, 2.])
    counts, scores, errors = diagnostics(data, splits, {"fixture": prediction})
    assert counts.total.sum() == len(data)
    assert counts.percent.sum() == pytest.approx(100)
    assert counts[list(("train", "validation", "test"))].sum().tolist() == [2, 1, 5]
    assert counts.set_index("class").loc["UNKNOWN", "test"] == 1
    assert scores.set_index("class").loc["RARE", "n"] == 0
    assert pd.isna(scores.set_index("class").loc["RARE", "rmse_log10_km"])
    # Pooled squared subgroup RMSE must reproduce the test squared error.
    tested = scores[scores.n > 0]
    assert np.average(tested.rmse_log10_km ** 2, weights=tested.n) == pytest.approx(
        np.mean((prediction[3:] - 1) ** 2))
    assert set(errors.spkid).issubset(set(data.loc[splits.eq("test"), "spkid"]))
    over = errors[errors.selection_rule.eq("largest_relative_overestimate")]
    assert over.spkid.tolist() == [data.spkid[3], data.spkid[7]]  # ID tie break
    under = errors[errors.selection_rule.eq("largest_relative_underestimate")]
    assert under.spkid.tolist() == [data.spkid[4]]
    assert under.iloc[0].prediction_to_actual_ratio == pytest.approx(.1)
    assert under.iloc[0].absolute_error_km == pytest.approx(9)
