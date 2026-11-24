"""Build the Foundations color palette from FinBERT sentiment on 10-K sentences.

start.py runs this for you. The share of positive / neutral / negative sentences sets each
color's swatch size, and the most confident sentence for each label becomes that color's example.

data/sample_10k_sentences.txt holds 40 sentences written in the style of 10-K filings (not quotes
from real filings). Paste real 10-K sentences into that file, one per line, to use real filings.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ml.sentiment import get_sentiment  # noqa: E402

sentences = [s.strip() for s in (ROOT / "data" / "sample_10k_sentences.txt").read_text().splitlines()
             if s.strip()]
source = "Sample sentences written in 10-K style"

print(f"      scoring {len(sentences)} sentences...")
results = [(s, get_sentiment(s)) for s in sentences]
engine = results[0][1]["source"] if results else "none"

#the satirical token names for each sentiment
TOKENS = {
    "positive": {"name": "Guidance Green", "hex": "#00B37E",
                 "usage": "Headlines, growth figures, anything above last quarter."},
    "neutral": {"name": "Cautious Optimism Blue", "hex": "#2337E6",
                "usage": "Body copy, risk factors, forward-looking statements."},
    "negative": {"name": "Write-Down Red", "hex": "#E5484D",
                 "usage": "Reserved for competitors. Use sparingly, at 8pt or smaller."},
}

palette = []
for label, token in TOKENS.items():
    matches = [(s, r["scores"].get(label, 0)) for s, r in results if r["label"] == label]
    share = len(matches) / len(results) if results else 0
    example = max(matches, key=lambda m: m[1])[0] if matches else None
    palette.append({**token, "label": label, "share": round(share, 3),
                    "count": len(matches), "example": example})

out = {"source": source, "engine": engine, "n": len(results), "palette": palette}
(ROOT / "data" / "palette.json").write_text(json.dumps(out, indent=2))
for p in palette:
    print(f"      {p['name']:<24} {p['share']:.0%}")
