"""Write a FICTIONAL sample ticker (ACME) so the app runs before you download real data.

ACME is made up: prices are simulated with a random walk. Use scripts/fetch_prices.py for real tickers.

Run from the project folder:
    python scripts/make_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "prices" / "ACME.csv"

rng = np.random.default_rng(22)
dates = pd.bdate_range("2023-01-02", periods=750)  #about 3 years of trading days

#random walk with a small upward drift, a crash in the middle, and a recovery
returns = rng.normal(0.0004, 0.018, len(dates))
returns[380:420] -= 0.012   #a bad stretch to spin
returns[600:640] += 0.006   #a rally to cherry-pick
close = 100 * np.cumprod(1 + returns)

df = pd.DataFrame({
    "date": dates.strftime("%Y-%m-%d"),
    "open": np.round(close * (1 + rng.normal(0, 0.004, len(dates))), 2),
    "high": np.round(close * (1 + np.abs(rng.normal(0, 0.01, len(dates)))), 2),
    "low": np.round(close * (1 - np.abs(rng.normal(0, 0.01, len(dates)))), 2),
    "close": np.round(close, 2),
    "volume": rng.integers(800_000, 5_000_000, len(dates)),
})
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Wrote {len(df)} rows to {OUT}")
