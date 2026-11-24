"""The Spin Dashboard: the same price data, restyled four ways.

Each mood returns a "chart spec" the front end draws with Chart.js. Every spec includes a
`truth` block with the unspun numbers, shown in the "What actually happened" footnote.
"""
import numpy as np

from ml.data import load_prices
from ml.forecast import forecast_prices, load_forecaster

MOODS = {
    "honest": "Honest",
    "investor_day": "Investor Day",
    "earnings_miss": "Earnings Miss",
    "layoff_week": "Layoff Week",
}

#which mood each FinBERT label picks when the user types a headline
MOOD_FOR_SENTIMENT = {"positive": "investor_day", "negative": "earnings_miss", "neutral": "honest"}


def _pct(a, b):
    return round((b / a - 1) * 100, 1)


def _truth(df):
    first, last = df["close"].iloc[0], df["close"].iloc[-1]
    recent = df.tail(21)  #about one trading month
    return {
        "start_date": df["date"].iloc[0].strftime("%b %d, %Y"),
        "end_date": df["date"].iloc[-1].strftime("%b %d, %Y"),
        "start_price": round(float(first), 2),
        "end_price": round(float(last), 2),
        "total_change_pct": _pct(first, last),
        "last_month_change_pct": _pct(recent["close"].iloc[0], recent["close"].iloc[-1]),
        "max_price": round(float(df["close"].max()), 2),
        "min_price": round(float(df["close"].min()), 2),
    }


def _best_window(closes, lookback=252, min_len=20):
    """Find the start/end inside the last `lookback` days with the biggest gain (the cherry-pick)."""
    closes = closes[-lookback:]
    offset = len(closes)
    best = (0, min_len, -np.inf)
    running_min_idx = 0
    for end in range(min_len, len(closes)):
        #track the lowest point at least min_len days before `end`
        if closes[end - min_len] < closes[running_min_idx]:
            running_min_idx = end - min_len
        gain = closes[end] / closes[running_min_idx] - 1
        if gain > best[2]:
            best = (running_min_idx, end, gain)
    return best[0] - offset, best[1] - offset  #negative indexes from the end


def build_spin(ticker, mood):
    df = load_prices(ticker)
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    closes = df["close"].astype(float).tolist()
    truth = _truth(df)
    _, meta = load_forecaster()

    if mood == "investor_day":
        start, end = _best_window(np.array(closes))
        end_abs = len(closes) + end          #turn the negative index into a normal one
        d, c = dates[start:end_abs + 1], closes[start:end_abs + 1]
        #the "AI outlook" continues from the cherry-picked high, not from today
        forecast = forecast_prices(closes[:end_abs + 1], 10)
        gain = _pct(c[0], c[-1])
        stops_early = end_abs < len(closes) - 1
        spec = {
            "headline": f"Up {gain}% since {df['date'].iloc[start]:%B} {df['date'].iloc[start].day}",
            "subhead": "Momentum you can feel. Outlook powered by AI.",
            "labels": d, "values": c,
            "forecast": forecast,
            "y_min": round(min(c) * 0.995, 2),  #truncated axis: starts just under the lowest point
            "y_max": None,
            "palette": "bull",
            "kpis": [
                {"label": "Growth", "value": f"+{gain}%"},
                {"label": "AI Outlook", "value": "Very Bullish ✨"},
                {"label": "Momentum Score", "value": "98 / 100"},
            ],
            "tricks": ["Cherry-picked start date (the lowest point of the past year)",
                       "Y-axis starts just under the low, so small moves look huge",
                       "Untested AI forecast shown as a solid line"]
                      + ([f"Chart stops at the high on {d[-1]} instead of today, "
                          "and the forecast starts from that high"] if stops_early else []),
        }

    elif mood == "earnings_miss":
        #30-day rolling average smooths away the drops; a tall y-axis flattens what's left
        smooth = df["close"].rolling(30, min_periods=1).mean().round(2).tolist()
        since = _pct(closes[0], closes[-1])
        spec = {
            "headline": (f"Up {since}% since inception" if since > 0
                         else "Resilient through the cycle"),
            "subhead": "We remain focused on long-term value creation.",
            "labels": dates, "values": smooth,
            "forecast": [],
            "y_min": 0,
            "y_max": round(max(smooth) * 2.2, 2),  #lots of empty sky makes the drop look tiny
            "palette": "pastel",
            "kpis": [
                {"label": "Long-term trend", "value": "Stable"},
                {"label": "Customer love", "value": "Strong"},
                {"label": "Last month", "value": "See appendix"},
            ],
            "tricks": ["30-day average hides the worst days",
                       "Y-axis runs to more than double the peak, so drops look flat",
                       "The longest possible time range, so recent losses shrink"],
        }

    elif mood == "layoff_week":
        forecast = forecast_prices(closes, 10)
        future_labels = [f"Day +{i}" for i in range(1, len(forecast) + 1)]
        spec = {
            "headline": "Right-sizing for the future",
            "subhead": "We're looking forward, not back.",
            "labels": ["Today"] + future_labels,
            "values": [closes[-1]] + [None] * len(forecast),
            "forecast": forecast,
            "forecast_only": True,
            "y_min": None, "y_max": None,
            "palette": "calm",
            "kpis": [
                {"label": "Forecast confidence", "value": "Very High"},
                {"label": "Model", "value": "Proprietary AI"},
                {"label": "History", "value": "Not shown"},
            ],
            "tricks": ["All past data hidden; only the forecast is shown",
                       "\"Very High\" confidence label not based on any metric",
                       "Forecast comes from a model that barely beats guessing (see footnote)"],
        }
        if not forecast:
            spec["subhead"] = "Train the forecaster first: python scripts/train_forecaster.py"

    else:  #honest
        mood = "honest"
        spec = {
            "headline": f"{ticker} closed at ${truth['end_price']}, "
                        f"{truth['total_change_pct']:+}% over the full period",
            "subhead": f"{truth['start_date']} to {truth['end_date']}. Full range, axis from zero.",
            "labels": dates, "values": closes,
            "forecast": [],
            "y_min": 0, "y_max": None,
            "palette": "honest",
            "kpis": [
                {"label": "Total change", "value": f"{truth['total_change_pct']:+}%"},
                {"label": "Last month", "value": f"{truth['last_month_change_pct']:+}%"},
                {"label": "Range", "value": f"${truth['min_price']} to ${truth['max_price']}"},
            ],
            "tricks": [],
        }

    spec.update({"ticker": ticker, "mood": mood, "mood_name": MOODS[mood], "truth": truth,
                 "model": meta and {k: meta[k] for k in
                                    ("test_mse", "baseline_mse", "direction_accuracy", "up_day_share")}})
    return spec
