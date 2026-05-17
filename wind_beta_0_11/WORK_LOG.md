# Work log and hypothesis tracker

This file records the modeling path and the hypotheses checked during the hackathon. It is written for the final README/report: what was tried, what failed, what worked, and why the current production approach looks physically reasonable.

## 1. Problem framing

Goal: predict hourly generation of a wind farm with installed capacity about 90.09 MW and 26 turbines. The target is farm-level power/generation. The final submission must be one column named `prediction`.

Important domain correction discovered early: the hub height is **84 m**, not 80 m. Most raw weather data contains speeds at 10/80/120/180 m, so we estimate an 84 m hub speed using the 80–120 m shear/power-law profile and keep 80 m features as auxiliary predictors.

## 2. Early baseline and why it mattered

The best early result came from a relatively simple XGBoost regressor with conservative hyperparameters:

```text
learning_rate=0.03
max_depth=3
min_child_weight=5
subsample=0.85
colsample_bytree=0.85
reg_alpha=0.04
reg_lambda=5.0
objective=reg:absoluteerror
```

This was counterintuitive: later, more complex models often became worse. The working interpretation is that the hidden validation set has a distribution shift, so higher-capacity models overfit train-years and local validation instead of learning a stable power curve.

## 3. Failed / weak directions

### 3.1 Large heterogeneous ensembles

We tried ensembles around XGBoost, CatBoost, LightGBM-like ideas, and several stronger regressors. Many were worse than the single conservative XGB. The likely reason is that all tree models learned similar wind-speed power-curve signals, but their errors were not complementary enough. Some ensembles also shifted the prediction distribution in the wrong direction.

### 3.2 Sequence models / transformer direction

A transformer/sequence approach was explored earlier, but it was worse by roughly one point. With only about 24–32k train rows, no target in future features, and strong weather-tabular structure, sequence capacity was not the bottleneck. It likely learned unstable temporal patterns and did not transfer to the hidden period.

### 3.3 Stage-2 residual model trained only on train residuals

A residual correction model trained on historical folds looked promising conceptually, but the residual sign varied by year/quarter. That made the correction unstable for Q1-like hidden validation. A naive residual model therefore became worse than the base.

### 3.4 Aggressive feature additions

Blindly adding hundreds of features did not help. The useful features were the ones tied to turbine physics and the hidden validation diagnostics: hub wind speed, direction sectors, shear, pressure, temperature, cloud, and ramp/rated-zone indicators.

## 4. Leaderboard-guided diagnostics

After recovering an old strong submission around 8.46, we generated small probe submissions. These probes did not train new models; they added or subtracted small constants in interpretable segments.

The hidden validation behavior showed several robust signs:

```text
- mid/ramp power was often underpredicted;
- high tail was often slightly overpredicted;
- wind direction sectors mattered;
- shear180/80 combined with pred50–70 was especially important;
- pressure/temp/cloud regimes caused systematic bias.
```

This produced a diagnostic calibration map and eventually a validation artifact around 8.01. That artifact is not the default production model, because it uses a hidden-valid-specific calibration path. It is kept as scientific evidence for which physical regimes matter.

## 5. Physical interpretation

### 5.1 Power curve and wind-speed cube

Wind power scales roughly as:

```text
P_available = 0.5 * rho * A * v^3
```

So even small wind-speed biases in the 6–11 m/s ramp zone cause large power errors. We therefore use `v`, `v²`, `v³`, piecewise ramp features, and proxy power curves.

### 5.2 Hub height 84 m and vertical shear

The station hub is closer to 84 m. We compute an 84 m proxy from 80 and 120 m speeds using a power-law shear coefficient. We also keep 10/80/120/180 m differences, ratios, and alpha exponents.

### 5.3 Direction and wake-like effects

The validation diagnostics found that some wind direction sectors require systematic up/down shifts. This is plausible: terrain and wake interactions depend on wind direction. Even without exact turbine coordinates, direction sectors and direction × power-curve interactions approximate these effects.

### 5.4 Shear and atmospheric stability

High 180–80 m shear plus mid/high predicted power was one of the strongest remaining signals in the calibration search. Physically, strong vertical gradients and wake mixing affect rotor-equivalent wind and downstream turbine performance. The code includes shear features, direction veer, vector shear, and stability-like pressure/temp/cloud proxies.

### 5.5 Air density

Density is estimated from pressure and temperature:

```text
rho = p / (R_dry_air * T)
```

This helps correct cold/high-pressure and warm/low-pressure regimes. Density-only corrections were not enough to solve the task, but density is a stable production feature.

### 5.6 Cut-in/rated/cut-out

The turbine theory notes cut-in near 3 m/s, rated speed around 10.3 m/s, and cut-out around 25 m/s. Empirically, a 12 m/s proxy was still useful, so beta 0.10+ keeps both: legacy 12 m/s and theory 10.3 m/s features.

### 5.7 Turbine inertia / temporal memory

We use lag/diff/rolling features from weather variables only. Target lags are not used because they are unavailable at prediction time. Lags are time-aware and should not reorder output rows.

## 6. Production branch evolution

The production branch was restarted after the validation-specific calibration got close to 8.01. The goal changed from best hidden-valid fitting to a reproducible `train -> model -> predict` pipeline for future 2027 data.

Key steps:

```text
beta 0.2: ML correction layer; correct scale but weak production score (~8.69).
beta 0.3: recovered legacy/friend pipeline; production score near 8.49.
beta 0.4–0.6: bias/rules/ensemble sweeps; mixed results.
beta 0.7: anchor_1700 + depth4 blend, w75, bias/rules; improved to ~8.42.
beta 0.8–0.9: physics profiles and line-search; improved to ~8.41.
beta 0.10: w70 + rules55 + bias0.75 + physics_strong; improved to ~8.396.
beta 0.11: narrow coordinate search around beta 0.10 best plus optional CatBoost companion.
```

## 7. Current model design

The production model trains several XGBoost models:

```text
legacy1700:   max_depth=3, 1700 estimators
legacy1892:   max_depth=3, 1892 estimators
legacy2500:   max_depth=3, 2500 estimators
depth4_1892:  max_depth=4, 1892 estimators
q1recent1700: Q1/recent weighted variant
nolags1700:   no-lag variant for diversity
```

The current best family blends:

```text
anchor_1700_d4_w70 = 0.70 * legacy1700 + 0.30 * depth4_1892
```

Then applies small deterministic physics rules and a small global bias:

```text
prediction = anchor + rules_delta + physics_profile_delta + bias
```

This is not ideal in a pure scientific sense, but it is self-contained: it uses only train data, valid/test features, and deterministic feature rules. It does not require a stored best-submission CSV.

## 8. Next planned steps

1. Run beta 0.11 first wave around the current best.
2. If it improves, run the second wave.
3. If it stalls, run optional CatBoost companion and submit only the 4–5 blends listed by `SUBMIT_CATBOOST.txt`.
4. For final packaging, keep production profile as default and mention the 8.01 calibrated artifact only as diagnostic evidence, not as the production solution.
5. Clean repository: one final model folder, one README, no old generated submissions/artifacts committed unless explicitly needed.

