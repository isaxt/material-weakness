"""Financial sentiment with FinBERT (ProsusAI/finbert), run in NumPy by ml/finbert.py.

get_sentiment(text) returns:
    {"label": "positive" | "negative" | "neutral",
     "score": 0.93,                       # confidence in that label
     "scores": {"positive": .., "negative": .., "neutral": ..},
     "source": "finbert" | "word-list"}

start.py downloads FinBERT the first time you run it. If the download couldn't happen (no
internet), a small finance word list answers instead, and every page says which one answered.
"""
import re

from ml import finbert

_model = {"loaded": False, "model": None}

POSITIVE = {"growth", "grew", "record", "strong", "beat", "beats", "exceeded", "profit", "profitable",
            "gain", "gains", "surge", "surged", "rally", "upgrade", "improved", "improvement", "raised",
            "outperform", "momentum", "expansion", "robust", "accelerating", "bullish", "dividend"}
NEGATIVE = {"loss", "losses", "decline", "declined", "miss", "missed", "layoff", "layoffs", "cut", "cuts",
            "downgrade", "lawsuit", "investigation", "risk", "weak", "weaker", "slump", "plunge", "plunged",
            "bankruptcy", "default", "impairment", "headwinds", "uncertain", "uncertainty", "recall",
            "restructuring", "bearish", "fell", "drop", "dropped"}


def _load():
    if not _model["loaded"]:
        _model["loaded"] = True
        if finbert.is_downloaded():
            try:
                _model["model"] = finbert.FinBERT()
            except Exception as err:
                print(f"[sentiment] FinBERT files look damaged, using word list ({err})")
    return _model["model"]


def _word_list_sentiment(text):
    words = re.findall(r"[a-z']+", text.lower())
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    total = pos + neg
    if total == 0:
        scores = {"positive": 0.1, "negative": 0.1, "neutral": 0.8}
    else:
        scores = {"positive": 0.9 * pos / total, "negative": 0.9 * neg / total, "neutral": 0.1}
    label = max(scores, key=scores.get)
    return {"label": label, "score": round(scores[label], 3),
            "scores": {k: round(v, 3) for k, v in scores.items()}, "source": "word-list"}


def get_sentiment(text):
    text = (text or "").strip()[:2000]
    if not text:
        return {"label": "neutral", "score": 1.0,
                "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}, "source": "empty"}
    model = _load()
    if model is None:
        return _word_list_sentiment(text)
    scores = model.predict(text)
    label = max(scores, key=scores.get)
    return {"label": label, "score": round(scores[label], 3),
            "scores": {k: round(v, 3) for k, v in scores.items()}, "source": "finbert"}
