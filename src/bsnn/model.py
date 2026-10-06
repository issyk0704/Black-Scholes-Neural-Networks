"""Neural-network option pricers, trained on option-chain data and scored against Black-Scholes.

Two kinds of model:

- ``"smile"`` (default): the network predicts each contract's implied volatility
  relative to a reference vol (30-day historical, or the expiry's at-the-money
  implied vol), and Black-Scholes turns that vol into a price. The smile's shape
  is far more stable over time than price levels, so this generalises to market
  conditions the network hasn't seen.
- ``"price"``: the network maps the features in :mod:`bsnn.features` straight
  to price / strike. Kept as the research baseline; it fits the days it was
  trained on well but doesn't carry over to later dates.

Rows are split into train and test sets by whole expiries (or by snapshot date),
so the test set never shares an expiry with the data the model learned from.

TensorFlow is imported only when a model is built or loaded, so the rest of the
package (and the GUI) starts quickly.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from bsnn import paths, pricing
from bsnn.features import ATM_IV_FEATURE, feature_columns
from bsnn.market_data import market_date

MODEL_KINDS = {"smile": "Volatility smile → Black-Scholes", "price": "Direct price"}
SPLITS = {"expiry": "Random expiries", "date": "Newest snapshot date"}
SMILE_FEATURES = ["log_moneyness", "std_moneyness", "time_to_expiry", "is_call"]
PRICE_SCALE = 100.0  # the direct-price network predicts price as a percentage of strike

NN = "Neural network"
BS_HIST = "Black-Scholes (30d hist. vol)"
BS_ATM = "Black-Scholes (ATM implied vol)"


def _keras():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    import keras

    return keras


@dataclass
class TrainConfig:
    kind: str = "smile"
    use_atm_iv: bool = True
    hidden_layers: tuple[int, ...] = (64, 64, 64)
    epochs: int = 300
    batch_size: int = 256
    learning_rate: float = 1e-3
    patience: int = 30
    split: str = "expiry"
    test_size: float = 0.2
    seed: int = 42


def reference_vol(use_atm_iv: bool) -> str:
    return ATM_IV_FEATURE if use_atm_iv else "hist_vol"


def model_inputs(df: pd.DataFrame, kind: str, reference: str) -> tuple[list[str], np.ndarray]:
    """(feature names, input matrix) for a model of ``kind`` using ``reference`` vol."""
    if kind == "smile":
        std_moneyness = df["log_moneyness"] / (df[reference] * np.sqrt(df["time_to_expiry"]))
        X = np.column_stack([df["log_moneyness"], std_moneyness, df["time_to_expiry"], df["is_call"]])
        return SMILE_FEATURES, X.astype(np.float32)
    if kind == "price":
        names = feature_columns(reference == ATM_IV_FEATURE)
        return names, df[names].to_numpy(dtype=np.float32)
    raise ValueError(f"Unknown model kind {kind!r}; choose from {', '.join(MODEL_KINDS)}")


def model_target(df: pd.DataFrame, kind: str, reference: str) -> np.ndarray:
    if kind == "smile":
        return np.log(df["market_iv"] / df[reference]).to_numpy(dtype=np.float32)
    return (df["target"] * PRICE_SCALE).to_numpy(dtype=np.float32)


# --- Splitting ---------------------------------------------------------------

def _expiry_groups(df: pd.DataFrame) -> pd.Series:
    return df["ticker"].astype(str) + "|" + df["snapshot"].astype(str) + "|" + df["expiry"].astype(str)


def _snapshot_dates(df: pd.DataFrame) -> pd.Series:
    return df["snapshot"].map(market_date)


def split_dataset(df: pd.DataFrame, split: str = "expiry", test_size: float = 0.2,
                  seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train, test) without letting an expiry straddle the two sets.

    ``"expiry"`` holds out a random ``test_size`` share of expiries. ``"date"``
    holds out every contract from the newest snapshot date, which tests whether
    the model still works on a later day.
    """
    if split == "date":
        dates = _snapshot_dates(df)
        if dates.nunique() < 2:
            raise ValueError("Splitting by date needs snapshots from at least two different days")
        test_mask = (dates == dates.max()).to_numpy()
    elif split == "expiry":
        groups = _expiry_groups(df)
        if groups.nunique() < 2:
            raise ValueError("Need at least two expiries to make a train/test split")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        _, test_idx = next(splitter.split(df, groups=groups))
        test_mask = np.zeros(len(df), dtype=bool)
        test_mask[test_idx] = True
    else:
        raise ValueError(f"Unknown split {split!r}; choose from {', '.join(SPLITS)}")
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


# --- Model -------------------------------------------------------------------

@dataclass
class OptionPricer:
    """A trained network plus what it needs to turn its output into a price."""

    model: object
    kind: str
    reference: str
    metadata: dict = field(default_factory=dict)

    @property
    def uses_atm_iv(self) -> bool:
        return self.reference == ATM_IV_FEATURE

    @property
    def features(self) -> list[str]:
        return SMILE_FEATURES if self.kind == "smile" else feature_columns(self.uses_atm_iv)

    @property
    def description(self) -> str:
        vol = "ATM implied vol" if self.uses_atm_iv else "30d historical vol"
        return f"{MODEL_KINDS[self.kind]}, using {vol}"

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        _, X = model_inputs(df, self.kind, self.reference)
        # Calling the model directly (not .predict) avoids a TensorFlow retrace per new input size.
        return np.asarray(self.model(X, training=False)).ravel()

    def predict_vol(self, df: pd.DataFrame) -> np.ndarray:
        """Implied vol the model assigns each contract (smile models only)."""
        if self.kind != "smile":
            raise ValueError("Only smile models predict volatility")
        return df[self.reference].to_numpy(dtype=float) * np.exp(self._raw(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted option prices in dollars."""
        if self.kind == "smile":
            return np.asarray(pricing.price(df["S"], df["K"], df["time_to_expiry"], df["risk_free_rate"],
                                            self.predict_vol(df), df["dividend_yield"], df["is_call"] > 0))
        return self._raw(df) / PRICE_SCALE * df["K"].to_numpy(dtype=float)

    def save(self, path: Path | None = None) -> Path:
        if path is None:
            path = paths.MODELS_DIR / f"{self.kind}_{dt.datetime.now():%Y%m%d_%H%M%S}.keras"
        path = Path(path).with_suffix(".keras")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        meta = {"kind": self.kind, "reference": self.reference, "features": self.features, **self.metadata}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: Path) -> OptionPricer:
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        kind, reference = meta.pop("kind"), meta.pop("reference")
        meta.pop("features", None)
        return cls(_keras().saving.load_model(path), kind, reference, meta)


def build_network(train_X: np.ndarray, config: TrainConfig):
    keras = _keras()
    normalise = keras.layers.Normalization()
    normalise.adapt(train_X)
    inputs = keras.Input(shape=(train_X.shape[1],))
    x = normalise(inputs)
    for units in config.hidden_layers:
        x = keras.layers.Dense(units, activation="silu")(x)
    # Prices are never negative; log vol ratios can be.
    outputs = keras.layers.Dense(1, activation="softplus" if config.kind == "price" else None)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(config.learning_rate), loss="mse")
    return model


# --- Evaluation --------------------------------------------------------------

def black_scholes_prices(df: pd.DataFrame, vol_column: str) -> np.ndarray:
    return np.asarray(pricing.price(df["S"], df["K"], df["time_to_expiry"], df["risk_free_rate"],
                                    df[vol_column], df["dividend_yield"], df["is_call"] > 0))


def price_metrics(predicted: np.ndarray, df: pd.DataFrame) -> dict:
    err = predicted - df["mid"].to_numpy()
    inside = (predicted >= df["bid"].to_numpy()) & (predicted <= df["ask"].to_numpy())
    return {
        "MAE ($)": float(np.mean(np.abs(err))),
        "RMSE ($)": float(np.sqrt(np.mean(err**2))),
        "Median abs error (%)": float(np.median(np.abs(err) / df["mid"].to_numpy()) * 100),
        "Inside bid-ask (%)": float(inside.mean() * 100),
    }


def evaluate(pricer: OptionPricer, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Price ``df`` with the network and both Black-Scholes benchmarks.

    Returns (rows with a column per model, one metrics row per model).
    """
    predictions = df.assign(**{
        NN: pricer.predict(df),
        BS_HIST: black_scholes_prices(df, "hist_vol"),
        BS_ATM: black_scholes_prices(df, ATM_IV_FEATURE),
    })
    metrics = pd.DataFrame({name: price_metrics(predictions[name].to_numpy(), df)
                            for name in (NN, BS_HIST, BS_ATM)}).T
    return predictions, metrics


@dataclass
class TrainingResult:
    pricer: OptionPricer
    history: dict
    predictions: pd.DataFrame
    metrics: pd.DataFrame


def train(dataset: pd.DataFrame, config: TrainConfig = TrainConfig(),
          on_epoch: Callable[[int, int, dict], None] | None = None,
          should_stop: Callable[[], bool] | None = None) -> TrainingResult:
    """Train on part of ``dataset`` and score the network on the held-out rest.

    ``on_epoch(epoch, total_epochs, logs)`` is called after every epoch; training
    ends early when ``should_stop()`` returns True or validation loss stops improving.
    """
    keras = _keras()
    keras.utils.set_random_seed(config.seed)
    reference = reference_vol(config.use_atm_iv)
    train_df, test_df = split_dataset(dataset, config.split, config.test_size, config.seed)
    fit_df, val_df = split_dataset(train_df, "expiry", 0.15, config.seed)

    def xy(df):
        return model_inputs(df, config.kind, reference)[1], model_target(df, config.kind, reference)

    (fit_X, fit_y), val_data = xy(fit_df), xy(val_df)
    model = build_network(fit_X, config)

    class Progress(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if on_epoch:
                on_epoch(epoch + 1, config.epochs, dict(logs or {}))
            if should_stop and should_stop():
                self.model.stop_training = True

    history = model.fit(
        fit_X, fit_y, validation_data=val_data, epochs=config.epochs, batch_size=config.batch_size,
        verbose=0, callbacks=[
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=max(config.patience // 3, 1), min_lr=1e-5),
            keras.callbacks.EarlyStopping(patience=config.patience, restore_best_weights=True),
            Progress(),
        ])

    pricer = OptionPricer(model, config.kind, reference)
    predictions, metrics = evaluate(pricer, test_df)
    pricer.metadata = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "config": asdict(config),
        "tickers": sorted(dataset["ticker"].unique()),
        "snapshot_dates": sorted({str(d.date()) for d in _snapshot_dates(dataset)}),
        "rows": {"train": len(fit_df), "validation": len(val_df), "test": len(test_df)},
        "epochs_run": len(history.history["loss"]),
        "test_metrics": metrics.round(4).to_dict(orient="index"),
    }
    return TrainingResult(pricer, history.history, predictions, metrics)
