"""The price forecaster, run with NumPy.

The model is trained in PyTorch (scripts/train_forecaster.py or the notebook) and its weights
are saved to models/forecaster.npz. This file runs the same network with NumPy, so the website
works even on computers where PyTorch can't be installed.

    last 10 daily returns -> Linear(10, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 1)
    -> predicted next-day return

Honest note for the write-up: next-day stock returns are close to unpredictable, so this model
barely beats the baseline. The satire is that the dashboard presents it as "AI-powered" and
"highly confident" anyway, while the footnote shows the real metrics.
"""
import json
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
WEIGHTS = MODEL_DIR / "forecaster.npz"
META = MODEL_DIR / "forecaster_meta.json"
WINDOW = 10


def make_windows(returns, window=WINDOW):
    """Turn a 1-D array of returns into (X, y): each row of X is `window` days, y is the next day."""
    returns = np.asarray(returns, dtype=np.float32)
    X = np.stack([returns[i:i + window] for i in range(len(returns) - window)])
    y = returns[window:].reshape(-1, 1)
    return X, y


_cache = {}


def load_forecaster():
    """Load the saved weights once. Returns (weights, meta) or (None, None) if not trained yet."""
    if "w" in _cache:
        return _cache["w"], _cache["meta"]
    if not WEIGHTS.exists() or not META.exists():
        return None, None
    _cache["w"] = dict(np.load(WEIGHTS))
    _cache["meta"] = json.loads(META.read_text())
    return _cache["w"], _cache["meta"]


def _predict_scaled(w, x):
    #the same three layers as the PyTorch model (weights are stored as [out, in], like nn.Linear)
    h = np.maximum(0, x @ w["net.0.weight"].T + w["net.0.bias"])
    h = np.maximum(0, h @ w["net.2.weight"].T + w["net.2.bias"])
    return (h @ w["net.4.weight"].T + w["net.4.bias"]).item()


def forecast_prices(closes, days=10):
    """Predict the next `days` closing prices by feeding each prediction back in.

    Returns a list of floats, or [] if the model hasn't been trained yet.
    """
    w, meta = load_forecaster()
    if w is None:
        return []
    closes = np.asarray(closes, dtype=np.float32)
    returns = list(np.diff(closes) / closes[:-1])
    mean, std, window = meta["mean"], meta["std"], meta["window"]
    price = float(closes[-1])
    out = []
    for _ in range(days):
        #scale the last `window` returns the same way as in training
        x = (np.array(returns[-window:], dtype=np.float32) - mean) / std
        pred = _predict_scaled(w, x) * std + mean   #undo the scaling
        returns.append(pred)
        price = price * (1 + pred)
        out.append(round(price, 2))
    return out
