"""Download real daily prices with yfinance and save one CSV per ticker in data/prices/.

start.py runs this for you. To use other companies, edit TICKERS below and run start.py again.
"""
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "prices"
TICKERS = ["AAPL", "NVDA", "TSLA", "JPM", "PTON"]

try:
    import logging
    import yfinance as yf
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  #our own messages are enough
except ImportError:
    print("      yfinance isn't installed, so the sample ticker (ACME) will be used.")
    sys.exit(0)

OUT_DIR.mkdir(parents=True, exist_ok=True)
saved = 0
for ticker in TICKERS:
    try:
        #3 years of daily prices
        df = yf.download(ticker, period="3y", interval="1d", auto_adjust=True, progress=False)
    except Exception as err:
        print(f"      {ticker}: download failed ({err.__class__.__name__})")
        continue
    if df is None or df.empty:
        print(f"      {ticker}: no data returned")
        continue
    #newer yfinance returns two-level column names; keep the first level
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index().rename(columns=str.lower)
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(OUT_DIR / f"{ticker}.csv", index=False)
    saved += 1
    print(f"      {ticker}: {len(df)} days")

#real data replaces the fictional sample
if saved and (OUT_DIR / "ACME.csv").exists():
    (OUT_DIR / "ACME.csv").unlink()
if not saved:
    print("      couldn't download prices (offline?), so the sample ticker (ACME) will be used.")
