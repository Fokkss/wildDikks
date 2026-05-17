# Beta 0.10 summary

We stopped using teacher/CSV calibration as the main solution and switched to a reproducible production pipeline:

`train data → XGBoost ensemble → deterministic physics correction → prediction`.

The best production family so far is a conservative blend of two XGBoost models:
- `legacy1700`: shallow legacy regressor, robust on hidden validation;
- `depth4_1892`: slightly deeper companion model;
- blend around `75% legacy + 25% depth4`.

The correction layer is not a hidden-target lookup. It uses only meteorological features and physical regimes found during validation diagnostics:
- low pressure / temperature regimes: air-density and boundary-layer stability effects;
- wind-direction sectors: wake/terrain-like effects;
- `ws5_7` and `ws10_11`: sensitive ramp / near-rated zones of the power curve;
- cloudy regimes: systematic high-power correction;
- a small positive bias: compensates a stable Q1-like underprediction found by production models.

Physics checklist:
- hub height corrected to 84 m;
- direction interpreted as degrees / 1000;
- density proxy `rho = p/(R*T)` is included;
- wind-speed squared/cubed and power-curve proxies are included;
- shear / veer / direction sectors are included;
- lag / rolling wind features are included;
- beta 0.10 additionally adds the turbine-rated-speed proxy around 10.3 m/s while keeping the empirically useful 12 m/s legacy curve.
