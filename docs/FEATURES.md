# Feature definitions

One row represents one unique asteroid SPK ID in a saved JPL SBDB response. The task is regression to the catalog's **effective diameter**, in km; training uses `y = log10(diameter_km)` to reduce size-scale domination. Exponentiating the fitted log prediction gives a positive size estimate, not necessarily an unbiased conditional mean in km.

The extraction includes `H`, `a`, `e`, `i`, `om`, and `w`. Orbital angles use the catalog's J2000 ecliptic reference convention; each solution's osculation epoch is preserved as metadata. J2000 is a reference frame, not a claim that every orbit has the same osculation time. Units and raw field meanings come from the [JPL data dictionary](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html).

## Model inputs: 13 columns

| Engineered column | Formula / encoding | Raw units | Purpose |
|---|---|---|---|
| `H` | Absolute magnitude as supplied | magnitude | Brightness-related size information, conditional on reflectivity |
| `log10_a_au` | log10(a / 1 au) | a in au | Compresses semimajor-axis range |
| `eccentricity` | e | dimensionless | Orbit shape; requires 0≤e<1 |
| `cos_inclination` | cos(i × π/180) | i in degrees | Smooth plane-orientation feature; distinguishes prograde and retrograde inclinations |
| `sin_node`, `cos_node` | sin/cos((om mod 360) × π/180) | om in degrees | Two-coordinate representation of ascending-node angle |
| `sin_perihelion_argument`, `cos_perihelion_argument` | sin/cos((w mod 360) × π/180) | w in degrees | Two-coordinate representation of perihelion argument |
| `log10_perihelion_au` | log10(a(1−e) / 1 au) | derived distance in au | Closest heliocentric point of the osculating ellipse |
| `log10_aphelion_au` | log10(a(1+e) / 1 au) | derived distance in au | Farthest heliocentric point of the osculating ellipse |
| `missing_i`, `missing_om`, `missing_w` | 1 if the cleaned angle is missing, otherwise 0 | dimensionless | Distinguish unobserved/invalid angles from observed values |

Angles are converted to radians only for trigonometric functions. The 0° and 360° encodings coincide. Inclination is restricted to [0°,180°]; node and argument accept [0°,360°]. Out-of-range optional angles become missing and are counted. Angles outside this range are not silently wrapped into plausible observations.

Perihelion and aphelion are calculated from the same a/e pair, rather than mixing catalog rounding conventions. These are deterministic transforms, not independent measurements. Ridge regularization handles their redundancy; fitted coefficients must not be interpreted as causal effects or independent physical discoveries. Angular elements can be poorly determined for nearly circular or near-zero-inclination orbits. They are demonstration features whose scientific utility must be established by comparison; no invariance or causal relevance is assumed.

## Explicit exclusions

| Field(s) | Treatment |
|---|---|
| `diameter` | Target only; never included in the predictor allowlist |
| `diameter_sigma` | Archived metadata; excluded as target-associated information; not used as a sample weight |
| Albedo | Not requested or used; may encode information jointly derived with diameter and H |
| `spkid`, designation, name | Identity/provenance only; no numeric encoding as a predictor |
| Orbit class | Descriptive subgroup reporting only; no class prediction task |
| Epoch, orbit ID, solution date, equinox | Provenance; equinox is validated but not encoded as a predictor |
| Observation count, observation arc, condition code | Archived for inspection, excluded to avoid a baseline driven by survey follow-up patterns |
| Orbital period, mean motion, catalog q/Q | Not requested as additional predictors; avoid extra redundant representations |

Feature creation is a fixed allowlist in `features()`. Adding arbitrary target-related columns to the raw frame cannot change the model matrix; a test enforces this. Any new feature set should receive a new schema identifier and a fresh evaluation protocol.
