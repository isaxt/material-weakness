"""Loading price data from data/prices/<TICKER>.csv.

Every CSV needs at least two columns: `date` and `close`.
scripts/make_sample_data.py writes a fictional ACME.csv so the app runs out of the box;
scripts/fetch_prices.py downloads real tickers with yfinance.
"""
from pathlib import Path

import pandas as pd

PRICE_DIR = Path(__file__).resolve().parent.parent / "data" / "prices"


def list_tickers():
    #every csv in data/prices is one ticker
    return sorted(p.stem for p in PRICE_DIR.glob("*.csv"))


def load_prices(ticker):
    path = PRICE_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No price file for {ticker}. Run scripts/fetch_prices.py first.")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").dropna(subset=["close"]).reset_index(drop=True)
    return df


def daily_returns(df):
    #percent change from one day to the next (0.01 = +1%)
    return df["close"].pct_change().dropna()
