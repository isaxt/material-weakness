"""Material Weakness: a design language with known deficiencies.

Run from the project folder:
    python -m flask --app app run --debug
then open http://127.0.0.1:5000

Pages render HTML templates; /api/... routes return JSON that the page's JavaScript draws.
All the data and model work lives in the ml/ folder, so this file stays short.
"""
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from ml.data import list_tickers
from ml.sentiment import get_sentiment
from ml.spin import MOODS, MOOD_FOR_SENTIMENT, build_spin
from ml.stats import build_stats
from ml.voice import check_voice

app = Flask(__name__)
app.json.sort_keys = False  #keep dictionaries in the order we built them
PALETTE_FILE = Path(__file__).parent / "data" / "palette.json"

NAV = [("index", "Overview"), ("foundations", "Foundations"), ("voice", "Voice & Tone"),
       ("spin", "Spin Dashboard"), ("neobank", "Neobank Mode"), ("terminal", "Terminal vs. App")]


@app.context_processor
def inject_nav():
    #makes `nav` available in every template (for the sidebar)
    return {"nav": NAV}


def default_ticker():
    tickers = list_tickers()
    return tickers[0] if tickers else None


# ---------- pages ----------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/foundations")
def foundations():
    palette = json.loads(PALETTE_FILE.read_text()) if PALETTE_FILE.exists() else None
    return render_template("foundations.html", palette=palette)


@app.route("/voice")
def voice():
    return render_template("voice.html")


@app.route("/spin")
def spin():
    return render_template("spin.html", tickers=list_tickers(), moods=MOODS)


@app.route("/neobank")
def neobank():
    return render_template("neobank.html")


@app.route("/terminal")
def terminal():
    return render_template("terminal.html", tickers=list_tickers())


@app.errorhandler(404)
def not_found(err):
    return render_template("404.html"), 404


# ---------- JSON APIs (called by the JavaScript in static/js) ----------

@app.route("/api/spin")
def api_spin():
    ticker = request.args.get("ticker") or default_ticker()
    mood = request.args.get("mood", "honest")
    headline = request.args.get("headline", "").strip()
    sentiment = None
    if headline:
        #a typed headline overrides the dial: FinBERT picks the mood
        sentiment = get_sentiment(headline)
        mood = MOOD_FOR_SENTIMENT[sentiment["label"]]
    try:
        spec = build_spin(ticker, mood)
    except FileNotFoundError as err:
        return jsonify({"error": str(err)}), 404
    spec["sentiment"] = sentiment
    return jsonify(spec)


@app.route("/api/sentiment", methods=["POST"])
def api_sentiment():
    text = (request.get_json(silent=True) or {}).get("text", "")
    return jsonify(get_sentiment(text))


@app.route("/api/voice", methods=["POST"])
def api_voice():
    text = (request.get_json(silent=True) or {}).get("text", "")
    return jsonify(check_voice(text))


@app.route("/api/stats")
def api_stats():
    ticker = request.args.get("ticker") or default_ticker()
    try:
        return jsonify(build_stats(ticker))
    except FileNotFoundError as err:
        return jsonify({"error": str(err)}), 404


if __name__ == "__main__":
    app.run(debug=True)
