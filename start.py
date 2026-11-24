"""Material Weakness: one command sets everything up and opens the site.

    python3 start.py

The first run installs packages, downloads stock prices, trains the forecaster, downloads
FinBERT (~440 MB) and builds the color palette. Later runs skip anything already done and start
in a few seconds. Press Ctrl+C to stop the site.
"""
import importlib
import importlib.util
import json
import platform
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRICES = ROOT / "data" / "prices"
MODEL = ROOT / "models" / "forecaster.npz"
META = ROOT / "models" / "forecaster_meta.json"
PALETTE = ROOT / "data" / "palette.json"
CORE = ["flask", "pandas", "numpy", "yfinance", "certifi"]
# PyTorch stopped making Intel Mac versions after 2.2, so it's skipped there
TORCH_SUPPORTED = not (sys.platform == "darwin" and platform.machine() == "x86_64")


def step(n, msg):
    print(f"\n[{n}/5] {msg}")


def has(module):
    importlib.invalidate_caches()
    return importlib.util.find_spec(module) is not None


def run_script(name):
    subprocess.run([sys.executable, str(ROOT / "scripts" / name)], cwd=ROOT)


def free_port(start=5050):
    # port 5000 is taken by AirPlay on Macs, so start at 5050 and find an open one
    for port in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


# 1. packages ---------------------------------------------------------------
step(1, "Packages")
missing = [m for m in CORE if not has(m)] + (["torch"] if TORCH_SUPPORTED and not has("torch") else [])
if missing:
    print(f"      installing {', '.join(missing)} (first run only, this can take a few minutes)...")
    reqs = [line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    pip = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
    # the site's packages first, then PyTorch on its own, so a PyTorch problem can't block the site
    subprocess.run(pip + [r for r in reqs if not r.startswith("torch")])
    if TORCH_SUPPORTED and not has("torch"):
        subprocess.run(pip + ["torch>=2.2"])
still_missing = [m for m in CORE if not has(m)]
if still_missing:
    sys.exit(f"\nCouldn't install {', '.join(still_missing)}. Check your internet connection and run "
             f"python3 start.py again.")
print("      ready" + ("" if has("torch") else " (PyTorch isn't available on this computer; "
                                              "the site uses the forecaster that ships already trained)"))

# 2. price data -------------------------------------------------------------
step(2, "Stock prices")
real = [p for p in PRICES.glob("*.csv") if p.stem != "ACME"]
if real:
    print(f"      using {', '.join(sorted(p.stem for p in real))}")
else:
    run_script("fetch_prices.py")
    if not list(PRICES.glob("*.csv")):
        run_script("make_sample_data.py")

# 3. forecaster -------------------------------------------------------------
step(3, "Forecaster (PyTorch)")
tickers = sorted(p.stem for p in PRICES.glob("*.csv"))
trained_on = json.loads(META.read_text()).get("tickers", []) if META.exists() else []
if has("torch") and (not MODEL.exists() or trained_on != tickers):
    print("      training on " + ", ".join(tickers))
    run_script("train_forecaster.py")
elif MODEL.exists():
    print("      using the trained model in models/forecaster.npz")
else:
    print("      no trained model yet; the AI outlook line will be hidden")

# 4. FinBERT ----------------------------------------------------------------
step(4, "FinBERT")
sys.path.insert(0, str(ROOT))
from ml import finbert  # noqa: E402

if finbert.is_downloaded():
    print("      ready")
else:
    print("      downloading from Hugging Face (first run only, about 440 MB)...")
    try:
        finbert.download_finbert()
        print("      ready")
    except Exception as err:
        print(f"      couldn't download FinBERT ({err}).\n      The site will use a simple word list "
              "instead; run python3 start.py again when you're online.")

# 5. palette ----------------------------------------------------------------
step(5, "Color palette")
engine = json.loads(PALETTE.read_text()).get("engine") if PALETTE.exists() else None
wanted = "finbert" if finbert.is_downloaded() else "word-list"
if engine != wanted:
    run_script("build_palette.py")
else:
    print("      ready")

# launch --------------------------------------------------------------------
from app import app  # noqa: E402

port = free_port()
url = f"http://127.0.0.1:{port}"
print(f"\nMaterial Weakness is running at {url}  (press Ctrl+C to stop)\n")
threading.Timer(1.5, lambda: webbrowser.open(url)).start()
app.run(port=port, debug=False, use_reloader=False)
