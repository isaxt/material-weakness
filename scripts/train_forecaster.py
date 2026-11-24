"""Train the PyTorch forecaster on every CSV in data/prices/ and save it to models/.

start.py runs this for you. The weights are saved as models/forecaster.npz so the website can
run the model with NumPy (ml/forecast.py), even where PyTorch isn't installed.

Same steps as class: tensors -> scale -> DataLoader -> model -> loss + optimizer -> training loop
-> evaluate on a held-out test set -> compare to a baseline -> save.
"""
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    #Intel Macs on Python 3.13+ (and some other setups) can't install PyTorch
    print("      PyTorch isn't available on this computer, so the model that ships with the project "
          "(already trained in PyTorch) will be used.")
    sys.exit(0)

#let this script import from the ml/ folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.data import list_tickers, load_prices, daily_returns  # noqa: E402
from ml.forecast import make_windows, WINDOW, WEIGHTS, META  # noqa: E402


class Forecaster(nn.Module):
    #regression network: WINDOW inputs -> 64 -> 32 -> 1 output (no activation at the end)
    def __init__(self, window=WINDOW):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(window, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


torch.manual_seed(22)

#1. build windows for every ticker, splitting each one by time (first 80% train, last 20% test)
#   splitting by time matters: shuffling before splitting would let the model peek at the future
X_train, y_train, X_test, y_test = [], [], [], []
for ticker in list_tickers():
    r = daily_returns(load_prices(ticker)).values
    X, y = make_windows(r, WINDOW)
    cut = int(len(X) * 0.8)
    X_train.append(X[:cut]); y_train.append(y[:cut])
    X_test.append(X[cut:]); y_test.append(y[cut:])
    print(f"      {ticker}: {len(X)} windows")

X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
X_test, y_test = np.concatenate(X_test), np.concatenate(y_test)

#2. scale with the TRAINING mean and std only (returns are tiny numbers like 0.004)
mean, std = float(X_train.mean()), float(X_train.std())
scale = lambda a: (a - mean) / std  # noqa: E731

Xtr = torch.tensor(scale(X_train), dtype=torch.float32)
ytr = torch.tensor(scale(y_train), dtype=torch.float32)
Xte = torch.tensor(scale(X_test), dtype=torch.float32)
yte = torch.tensor(scale(y_test), dtype=torch.float32)

#3. batches, shuffled for training
train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)

#4. model, loss, optimizer
model = Forecaster(WINDOW)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

#5. training loop
losses = []
for epoch in range(30):
    running_loss = 0.0
    for xb, yb in train_loader:
        #pass the batch through the model
        yhat = model(xb)
        #compare predictions to the real next-day returns
        loss = loss_fn(yhat, yb)
        #clear old gradients, backpropagate, step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(xb)
    losses.append(running_loss / len(Xtr))
    if epoch % 5 == 0:
        print(f"      epoch {epoch:>2}  train MSE {losses[-1]:.4f}")

#6. evaluate on the held-out test set
model.eval()
with torch.no_grad():
    test_pred = model(Xte)
test_mse = loss_fn(test_pred, yte).item()
#baseline: always predict the average training return (0 after scaling)
baseline_mse = loss_fn(torch.zeros_like(yte), yte).item()
#direction accuracy: did we at least get up vs. down right?
direction_acc = ((test_pred > (0 - mean) / std) == (yte > (0 - mean) / std)).float().mean().item()
up_share = (yte > (0 - mean) / std).float().mean().item()

print(f"      test MSE {test_mse:.4f} vs. baseline {baseline_mse:.4f} | "
      f"direction accuracy {direction_acc:.1%} vs. {max(up_share, 1 - up_share):.1%} for always guessing")

#7. save weights (as NumPy arrays) + everything the app needs to use them
WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
np.savez(WEIGHTS, **{k: v.detach().numpy() for k, v in model.state_dict().items()})
META.write_text(json.dumps({
    "window": WINDOW,
    "mean": mean,
    "std": std,
    "test_mse": round(test_mse, 4),
    "baseline_mse": round(baseline_mse, 4),
    "direction_accuracy": round(direction_acc, 4),
    "up_day_share": round(up_share, 4),
    "tickers": list_tickers(),
    "train_losses": [round(x, 4) for x in losses],
}, indent=2))
print(f"      saved {WEIGHTS.name}")
