"""Numbers for the Terminal vs. App page."""
import numpy as np

from ml.data import load_prices
from ml.forecast import forecast_prices, load_forecaster


def _chg(series, days):
    if len(series) <= days:
        return None
    return round((series.iloc[-1] / series.iloc[-1 - days] - 1) * 100, 2)


def build_stats(ticker):
    df = load_prices(ticker)
    close = df["close"].astype(float)
    r = close.pct_change().dropna()
    year = close.tail(252)
    drawdown = (close / close.cummax() - 1).min()
    forecast = forecast_prices(close.tolist(), 1)
    _, meta = load_forecaster()

    changes = {"1D": _chg(close, 1), "1W": _chg(close, 5), "1M": _chg(close, 21),
               "3M": _chg(close, 63), "1Y": _chg(close, 252), "ALL": _chg(close, len(close) - 1)}
    #the App view shows only the most flattering period: that's the joke
    best_period = max((k for k, v in changes.items() if v is not None), key=lambda k: changes[k])

    return {
        "ticker": ticker,
        "as_of": df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "last": round(close.iloc[-1], 2),
        "changes": changes,
        "best_period": best_period,
        "best_change": changes[best_period],
        "vol_annual_pct": round(float(r.std() * np.sqrt(252) * 100), 1),
        "max_drawdown_pct": round(float(drawdown * 100), 1),
        "high_52w": round(float(year.max()), 2),
        "low_52w": round(float(year.min()), 2),
        "avg_volume": int(df["volume"].tail(21).mean()) if "volume" in df else None,
        "up_days_pct": round(float((r > 0).mean() * 100), 1),
        "forecast_next": forecast[0] if forecast else None,
        "model": meta and {k: meta[k] for k in ("test_mse", "baseline_mse", "direction_accuracy")},
        "spark": close.tail(60).round(2).tolist(),
        "table": [{"date": d.strftime("%Y-%m-%d"), "close": round(c, 2)}
                  for d, c in zip(df["date"].tail(12)[::-1], close.tail(12)[::-1])],
    }
