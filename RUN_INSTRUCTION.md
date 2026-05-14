### ***If not already***
```commandline
python -m venv venv
pip install -r requirements.txt
```

### ***Training | Gymnastics | LightWeightBaby***
```commandline
python train.py ^
  --train_path dataset/train_dataset.csv ^
  --model_dir artifacts ^
  --seed 42 ^
  --valid_size 0.2 ^
  --catboost_weight 0.5 ^
  --xgboost_weight 0.5
```

### Prediction
```commandline
python predict.py ^
  --features_path dataset/valid_features.csv ^
  --model_path artifacts/ensemble.pkl ^
  --output_path submission.csv
```

### May be done easier
```commandline
python train.py --train_path dataset/train_dataset.csv --model_dir artifacts
python predict.py --features_path dataset/valid_features.csv --model_path artifacts/ensemble.pkl --output_path submission.csv
```