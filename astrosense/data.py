"""Cached public JPL data and explicit, inspectable quality controls."""

import hashlib
import json
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import certifi

API = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
FIELDS = [
    "spkid", "pdes", "full_name", "epoch", "equinox", "orbit_id", "soln_date",
    "a", "e", "i", "om", "w", "H", "diameter", "diameter_sigma",
    "data_arc", "n_obs_used", "condition_code", "class",
]
NUMERIC = ["epoch", "a", "e", "i", "om", "w", "H", "diameter",
           "diameter_sigma", "data_arc", "n_obs_used", "condition_code"]
FEATURE_SCHEMA = "astrosense-orbits-v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fetch(destination, contact):
    """One full filtered response; no paging, silent refresh, or partial snapshot."""
    dest = Path(destination)
    if dest.exists():
        raise FileExistsError(f"Snapshot already exists: {dest}. Reuse it or choose a new path.")
    if not contact.strip() or any(c in contact for c in "\r\n"):
        raise ValueError("Supply a valid contact URL or email for JPL's User-Agent policy")
    constraints = {"AND": ["diameter|GT|0", "H|DF", "a|GT|0", "e|GE|0", "e|LT|1"]}
    query = {"fields": ",".join(FIELDS), "sb-kind": "a", "full-prec": "true",
             "sb-cdata": json.dumps(constraints, separators=(",", ":"))}
    url = API + "?" + urlencode(query)
    request = Request(url, headers={"User-Agent": f"AstroSense/0.1.0 ({contact})"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=120, context=ssl.create_default_context(cafile=certifi.where())) as response:
                body = response.read()
                headers = dict(response.headers.items())
            break
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
            # Stop rather than retry earlier than a server's long Retry-After.
            delay = exc.headers.get("Retry-After")
            if delay is not None and (not delay.isdigit() or int(delay) > 30):
                raise
            time.sleep(max(2 ** (attempt + 1), int(delay or 0)))
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
    payload = json.loads(body)
    validate_payload(payload)
    dest.mkdir(parents=True)
    (dest / "sbdb.json").write_bytes(body)
    write_json(dest / "manifest.json", {
        "url": url, "parameters": query, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256(dest / "sbdb.json"), "rows": len(payload["data"]),
        "signature": payload["signature"], "response_headers": headers,
        "scope": "All matching catalog asteroids in one response; not all asteroids",
    })
    return dest


def validate_payload(payload):
    if payload.get("signature", {}).get("version") != "1.0":
        raise ValueError("Unexpected API version; review JPL docs before accepting a new schema")
    if payload.get("fields") != FIELDS:
        raise ValueError("Unexpected fields or field order from JPL")
    rows = payload.get("data", [])
    if not rows or int(payload["count"]) != len(rows):
        raise ValueError("Empty or truncated snapshot; count must match all returned rows")
    if any(len(row) != len(FIELDS) for row in rows):
        raise ValueError("Malformed row in API response")


def read_snapshot(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha256(directory / "sbdb.json") != manifest["sha256"]:
        raise ValueError("Snapshot checksum mismatch")
    payload = json.loads((directory / "sbdb.json").read_text())
    validate_payload(payload)
    return pd.DataFrame(payload["data"], columns=payload["fields"]), manifest


def clean(frame, require_target=True):
    """Required invalid values exclude rows; optional invalid angles become missing."""
    required = ["spkid", "H", "a", "e", "i", "om", "w", "equinox"]
    if require_target:
        required.append("diameter")
    absent = set(required) - set(frame.columns)
    if absent:
        raise ValueError(f"Missing required columns: {sorted(absent)}")
    data = frame.copy()
    ids = data["spkid"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError("Missing object identity")
    data["spkid"] = ids.astype(str).str.strip()
    if data["spkid"].duplicated().any():
        raise ValueError("Repeated SPK IDs: resolve snapshots before splitting")
    malformed = {}
    for col in NUMERIC:
        if col in data:
            original = data[col]
            numeric = pd.to_numeric(original, errors="coerce").replace([np.inf, -np.inf], np.nan)
            malformed[col] = int((original.notna() & numeric.isna()).sum())
            data[col] = numeric
    reasons = pd.Series("", index=data.index, dtype="object")
    rules = {
        "invalid_H": data.H.isna(),
        "invalid_a": data.a.isna() | (data.a <= 0),
        "invalid_e": data.e.isna() | (data.e < 0) | (data.e >= 1),
        "unsupported_equinox": data.equinox != "J2000",
    }
    if require_target:
        rules["invalid_diameter"] = data.diameter.isna() | (data.diameter <= 0)
    for reason, mask in rules.items():
        reasons.loc[mask] += reason + ";"
    optional_invalid = {}
    for col, upper in [("i", 180), ("om", 360), ("w", 360)]:
        invalid = data[col].notna() & ~data[col].between(0, upper)
        optional_invalid[col] = int(invalid.sum())
        data.loc[invalid, col] = np.nan
    excluded = data.loc[reasons.ne(""), ["spkid"]].copy()
    excluded["reasons"] = reasons.loc[reasons.ne("")]
    accepted = data.loc[reasons.eq("")].sort_values("spkid").reset_index(drop=True)
    audit = {
        "input_rows": len(data), "accepted_rows": len(accepted), "excluded_rows": len(excluded),
        "invalid_numeric_counts": malformed, "invalid_optional_angle_counts": optional_invalid,
        "exclusion_reason_counts": {k: int(v.sum()) for k, v in rules.items()},
        "optional_missing_counts": {c: int(accepted[c].isna().sum()) for c in ["i", "om", "w"]},
    }
    if accepted.empty:
        raise ValueError("No eligible rows remain")
    return accepted, excluded, audit


def features(data):
    """Stateless transforms only; target and metadata cannot enter this allowlist."""
    result = pd.DataFrame(index=data.index)
    result["H"] = data.H
    result["log10_a_au"] = np.log10(data.a)
    result["eccentricity"] = data.e
    result["cos_inclination"] = np.cos(np.deg2rad(data.i))
    for column, name in [("om", "node"), ("w", "perihelion_argument")]:
        radians = np.deg2rad(data[column] % 360)
        result[f"sin_{name}"] = np.sin(radians)
        result[f"cos_{name}"] = np.cos(radians)
    result["log10_perihelion_au"] = np.log10(data.a * (1 - data.e))
    result["log10_aphelion_au"] = np.log10(data.a * (1 + data.e))
    for col in ["i", "om", "w"]:
        result[f"missing_{col}"] = data[col].isna().astype(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Feature overflow; inspect orbital inputs")
    return result


def split_ids(ids, seed=20261018):
    """Stable ID-only 70/15/15 assignment, independent of order and target."""
    def assign(identifier):
        value = int(hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()[:16], 16) / 2**64
        return "train" if value < .70 else "validation" if value < .85 else "test"
    return pd.Series([assign(i) for i in ids], index=ids.index, name="split")
