## ***Notes***
###### some notes maybe??



PD_:NOTE:

Current implementation:
- adds modular model structure in models/
- implements CatBoostRegressor wrapper
- implements XGBoostRegressor wrapper
- implements weighted ensemble of CatBoost + XGBoost predictions
- adds shared preprocessing for numeric features
- normalizes target and repair-count column names
- adds minimal demo time features: hour_sin and hour_cos
- clips predictions to physically possible station output
- adds train.py for training and local chronological validation
- adds predict.py for generating submission.csv
- saves model artifacts, metrics and feature importances

The model currently uses available numeric weather/features columns from the dataset,
without heavy custom feature engineering. Physical/advanced features are intentionally
kept out for now and will be added later in a controlled way.

PD_:NOTE: predict.py, train.py, config.py all in right place (project root)

SX_:NOTE: not in right place. no logic in such placement