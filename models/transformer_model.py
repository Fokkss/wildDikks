from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for TransformerWindModel. Install it with: pip install torch"
    ) from exc

from .base import BaseWindModel, INSTALLED_CAPACITY_MW, ModelPreprocessor


class _TransformerRegressor(nn.Module):
    """
    Small Transformer Encoder for tabular time-series windows.

    Input shape:  (batch, seq_len, n_features)
    Output shape: (batch,)
    """

    def __init__(
        self,
        n_features: int,
        sequence_length: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.input_projection = nn.Linear(n_features, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x + self.position_embedding[:, : x.shape[1], :]
        x = self.encoder(x)

        # Use the last token because it represents current hour after seeing history.
        last_token = x[:, -1, :]
        return self.head(last_token).squeeze(-1)


@dataclass
class TransformerParams:
    sequence_length: int = 24
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 40
    patience: int = 8


class TransformerWindModel(BaseWindModel):
    """
    Transformer model for hourly wind-farm generation forecasting.

    This wrapper follows the same interface as your existing model wrappers:
        - fit(train_df, valid_df=None)
        - predict(features_df)
        - save/load inherited from BaseWindModel style via joblib

    It uses ModelPreprocessor to build tabular features, then converts rows into
    rolling time windows of length `sequence_length`.
    """

    def __init__(
        self,
        preprocessor: ModelPreprocessor | None = None,
        random_seed: int = 42,
        clip_predictions: bool = True,
        params: dict[str, Any] | TransformerParams | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(
            preprocessor=preprocessor,
            random_seed=random_seed,
            clip_predictions=clip_predictions,
        )

        if params is None:
            self.params = TransformerParams()
        elif isinstance(params, TransformerParams):
            self.params = params
        else:
            self.params = TransformerParams(**params)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model: _TransformerRegressor | None = None
        self.best_valid_loss_: float | None = None

    def _set_seed(self) -> None:
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

    def _prepare_X_train(self, df: pd.DataFrame) -> np.ndarray:
        X = self.preprocessor.fit_transform(df)
        X_imp = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imp)
        return np.asarray(X_scaled, dtype=np.float32)

    def _prepare_X_inference(self, df: pd.DataFrame) -> np.ndarray:
        X = self.preprocessor.transform(df)
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        return np.asarray(X_scaled, dtype=np.float32)

    def _make_sequences(self, X: np.ndarray) -> np.ndarray:
        seq_len = self.params.sequence_length

        if len(X) == 0:
            return np.empty((0, seq_len, X.shape[1]), dtype=np.float32)

        # Left-pad with the first row, so prediction count remains equal to input row count.
        pad = np.repeat(X[[0]], repeats=seq_len - 1, axis=0)
        X_pad = np.vstack([pad, X])

        sequences = np.empty((len(X), seq_len, X.shape[1]), dtype=np.float32)
        for i in range(len(X)):
            sequences[i] = X_pad[i : i + seq_len]

        return sequences

    def _make_loader(
        self,
        X_seq: np.ndarray,
        y: np.ndarray | None = None,
        shuffle: bool = False,
    ) -> DataLoader:
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)

        if y is None:
            dataset = TensorDataset(X_tensor)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, y_tensor)

        return DataLoader(
            dataset,
            batch_size=self.params.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _build_model(self, n_features: int) -> _TransformerRegressor:
        return _TransformerRegressor(
            n_features=n_features,
            sequence_length=self.params.sequence_length,
            d_model=self.params.d_model,
            n_heads=self.params.n_heads,
            n_layers=self.params.n_layers,
            dim_feedforward=self.params.dim_feedforward,
            dropout=self.params.dropout,
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
    ) -> "TransformerWindModel":
        self._validate_input_dataframe(train_df)
        self._set_seed()

        X_train = self._prepare_X_train(train_df)
        y_train = self.preprocessor.get_target(train_df).to_numpy(dtype=np.float32)

        train_mask = np.isfinite(y_train)
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]

        X_train_seq = self._make_sequences(X_train)

        valid_loader = None
        if valid_df is not None:
            self._validate_input_dataframe(valid_df)
            X_valid = self._prepare_X_inference(valid_df)
            y_valid = self.preprocessor.get_target(valid_df).to_numpy(dtype=np.float32)
            valid_mask = np.isfinite(y_valid)
            X_valid = X_valid[valid_mask]
            y_valid = y_valid[valid_mask]
            X_valid_seq = self._make_sequences(X_valid)
            valid_loader = self._make_loader(X_valid_seq, y_valid, shuffle=False)

        self.model = self._build_model(n_features=X_train.shape[1]).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params.learning_rate,
            weight_decay=self.params.weight_decay,
        )
        loss_fn = nn.L1Loss()

        train_loader = self._make_loader(X_train_seq, y_train, shuffle=True)

        best_state = None
        best_valid_loss = float("inf")
        bad_epochs = 0

        for epoch in range(1, self.params.epochs + 1):
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                pred = self.model(batch_X)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()

                train_losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

            if valid_loader is not None:
                valid_loss = self._evaluate_loader(valid_loader, loss_fn)
                print(
                    f"[Transformer] epoch={epoch:03d} "
                    f"train_mae={train_loss:.4f} valid_mae={valid_loss:.4f}"
                )

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if bad_epochs >= self.params.patience:
                    print(f"[Transformer] early stopping at epoch {epoch}")
                    break
            else:
                print(f"[Transformer] epoch={epoch:03d} train_mae={train_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.best_valid_loss_ = best_valid_loss

        return self

    def _evaluate_loader(self, loader: DataLoader, loss_fn: nn.Module) -> float:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")

        self.model.eval()
        losses = []

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                pred = self.model(batch_X)
                loss = loss_fn(pred, batch_y)
                losses.append(float(loss.detach().cpu().item()))

        return float(np.mean(losses)) if losses else float("nan")

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        self._validate_input_dataframe(features_df)

        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")

        X = self._prepare_X_inference(features_df)
        X_seq = self._make_sequences(X)
        loader = self._make_loader(X_seq, y=None, shuffle=False)

        self.model.eval()
        preds = []

        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                batch_pred = self.model(batch_X)
                preds.append(batch_pred.detach().cpu().numpy())

        predictions = np.concatenate(preds, axis=0) if preds else np.array([], dtype=float)
        return self._clip(predictions)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TransformerWindModel":
        return joblib.load(path)


class TransformerEnsembleModel:
    """
    Simple ensemble of several TransformerWindModel instances.

    Diversity comes from different random seeds and optionally different parameters.
    Final prediction is the mean prediction clipped to physical installed capacity.
    """

    def __init__(
        self,
        models: list[TransformerWindModel] | None = None,
        clip_predictions: bool = True,
    ) -> None:
        self.models = models or []
        self.clip_predictions = clip_predictions

    @classmethod
    def from_config(
        cls,
        n_models: int = 3,
        seed: int = 42,
        params: dict[str, Any] | None = None,
        preprocessor_factory: Any | None = None,
    ) -> "TransformerEnsembleModel":
        models: list[TransformerWindModel] = []

        for i in range(n_models):
            preprocessor = preprocessor_factory() if preprocessor_factory is not None else ModelPreprocessor()
            model = TransformerWindModel(
                preprocessor=preprocessor,
                random_seed=seed + i,
                params=params,
            )
            models.append(model)

        return cls(models=models)

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
    ) -> "TransformerEnsembleModel":
        if not self.models:
            raise ValueError("TransformerEnsembleModel has no base models.")

        for i, model in enumerate(self.models, start=1):
            print(f"[TransformerEnsemble] training model {i}/{len(self.models)}")
            model.fit(train_df=train_df, valid_df=valid_df)

        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise ValueError("TransformerEnsembleModel has no base models.")

        preds = [model.predict(features_df) for model in self.models]
        prediction = np.mean(np.vstack(preds), axis=0)

        if self.clip_predictions:
            prediction = np.clip(prediction, 0.0, INSTALLED_CAPACITY_MW)

        return prediction

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TransformerEnsembleModel":
        return joblib.load(path)
