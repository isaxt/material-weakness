# Material Weakness

*A design language with known deficiencies.*

A satirical design-system website about how dashboards, brand systems and fintech apps
frame financial data. Built with Flask, PyTorch and FinBERT for a Python: Data Science final project.

## Run it

From this folder:

```bash
python3 start.py
```

That's the whole setup. It installs the packages, downloads stock prices, trains the forecaster,
downloads FinBERT, builds the color palette, and opens the site in your browser. The first run
takes a few minutes (FinBERT is about 440 MB); after that it starts in seconds. Press Ctrl+C to stop.

## Pages

| URL | Page | Data | Model |
| --- | --- | --- | --- |
| `/` | Overview | none | none |
| `/foundations` | Color palette | 10-K-style sentences | FinBERT |
| `/voice` | Voice & Tone checker | typed text | FinBERT + word lists |
| `/spin` | Spin Dashboard | daily prices | PyTorch forecaster + FinBERT |
| `/neobank` | Neobank Mode | typed headline | FinBERT |
| `/terminal` | Terminal vs. App | daily prices | PyTorch forecaster |

## How the models run

- **Forecaster:** trained in PyTorch (`scripts/train_forecaster.py`, or step by step in
  `notebooks/train_forecaster.ipynb`) and saved as NumPy arrays, so the site runs it without PyTorch.
  PyTorch has no versions for Intel Macs; there, the site uses the forecaster that ships already trained.
- **FinBERT** (`ProsusAI/finbert`): run in plain NumPy by `ml/finbert.py`, so it works on any computer.
