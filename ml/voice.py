"""Voice & Tone checker: scores how "on brand" a sentence is for corporate finance.

On-brand = hedged, jargon-heavy and gently positive. The score mixes:
    - FinBERT sentiment (positive helps, negative hurts)
    - hedge words ("may", "anticipate") that make any claim unfalsifiable
    - corporate jargon ("synergy", "headwinds")
"""
import re

from ml.sentiment import get_sentiment

HEDGES = ["may", "might", "could", "approximately", "anticipate", "anticipates", "believe", "believes",
          "expect", "expects", "potentially", "intend", "intends", "seek", "seeks", "estimate",
          "forward-looking", "uncertain", "substantially", "generally", "subject to", "no assurance",
          "we cannot guarantee", "in part"]
JARGON = ["synergy", "synergies", "headwinds", "tailwinds", "right-size", "right-sizing", "leverage",
          "optimize", "optimization", "streamline", "alignment", "scalable", "ecosystem", "stakeholders",
          "value creation", "best-in-class", "transformational", "paradigm", "disciplined",
          "capital allocation", "growth story", "pivot", "agentic", "ai-powered", "resilience",
          "resilient", "operational excellence", "go-forward"]


def _find(terms, text):
    lower = text.lower()
    found = []
    for t in terms:
        if re.search(rf"(?<![a-z-]){re.escape(t)}(?![a-z-])", lower):
            found.append(t)
    return found


def check_voice(text):
    s = get_sentiment(text)
    hedges, jargon = _find(HEDGES, text), _find(JARGON, text)
    words = max(len(text.split()), 1)

    score = 40
    score += {"positive": 20, "neutral": 10, "negative": -25}[s["label"]]
    score += min(len(hedges) * 8, 24)
    score += min(len(jargon) * 7, 21)
    score -= 10 if words < 6 else 0           #too short to be properly vague
    score -= 15 if re.search(r"\d+%", text) and not hedges else 0  #a bare number is a commitment
    score = max(0, min(100, score))

    if score >= 80:
        verdict, note = "On brand", "Legal would approve this without reading it."
    elif score >= 55:
        verdict, note = "Nearly there", "Add a hedge or two so nobody can hold you to it."
    elif score >= 30:
        verdict, note = "Off brand", "This sounds like a person. Consider \"we remain focused on\"."
    else:
        verdict, note = "Off brand", "This is just true. Please route through Communications."

    return {"score": score, "verdict": verdict, "note": note, "sentiment": s,
            "hedges": hedges, "jargon": jargon}
