### ***If not already***
```commandline
python -m venv venv
pip install -r requirements.txt
```

### ***Cut Outliers***
```commandline
python -m new_model.cut_outliers \
      --train_path data/train_dataset.csv \
      --output_path data/train_dataset_clean.csv \
      --mode clip
      
python -m new_model.cut_outliers `
    --train_path data/train_dataset.csv `
    --output_path data/train_dataset_clean.csv `
    --mode clip
```

### ***Optuna tuning***
```commandline
python -m new_model.tune_features_optuna \
      --train_path data/train_dataset.csv \
      --output_dir artifacts/optuna \
      --n_trials 40 \
      --valid_size 0.2 \
      --model xgb
      
python -m new_model.tune_features_optuna `
      --train_path data/train_dataset.csv `
      --output_dir artifacts/optuna `
      --n_trials 20 `
      --valid_size 0.2 `
      --model xgb
```
```commandline
python -m new_model.tune_features_optuna \
      --train_path data/train_dataset.csv \
      --output_dir artifacts/optuna \
      --n_trials 20 \
      --valid_size 0.2 \
      --model ensemble
      
python -m new_model.tune_features_optuna `
      --train_path data/train_dataset.csv `
      --output_dir artifacts/optuna `
      --n_trials 40 `
      --valid_size 0.2 `
      --model ensemble
```

### ***Training | Gymnastics | LightWeightBaby***
```commandline
python -m new_model.train ^
  --train_path data/train_dataset.csv ^
  --model_dir artifacts ^
  --seed 42 ^
  --valid_size 0.2 ^
  --catboost_weight 0.5 ^
  --xgboost_weight 0.5
  
python -m new_model.train `
  --train_path data/train_dataset.csv `
  --model_dir artifacts `
  --seed 42 `
  --valid_size 0.2 `
  --catboost_weight 0.5 `
  --xgboost_weight 0.5
```

### Prediction
```commandline
python -m new_model.predict ^
  --features_path data/valid_features.csv ^
  --model_path artifacts/ensemble.pkl ^
  --output_path submission.csv
  
python -m new_model.predict `
  --features_path data/valid_features.csv `
  --model_path artifacts/ensemble.pkl `
  --output_path submission.csv
```

### May be done easier
```commandline
python -m new_model.train --train_path data/train_dataset.csv --model_dir artifacts
python -m new_model.predict --features_path data/valid_features.csv --model_path artifacts/ensemble.pkl --output_path submission.csv
```