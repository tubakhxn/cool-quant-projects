# Correlation Regime Dynamics

A 3D ribbon wall visualization of rolling cross-asset correlations across 6 assets over 2 years of simulated data, with a Markov regime-switching model that generates alternating calm and crisis periods. During crisis regimes, all ribbons turn deep red simultaneously — the famous "correlation goes to 1" phenomenon that destroys diversification at the worst possible time.

---

## Developer / Creator

## tubakhxn

GitHub: [github.com/tubakhxn](https://github.com/tubakhxn)

---

## What This Project Is

Correlation is not constant — it shifts dramatically depending on market regime. This project simulates 6 assets (SPY, QQQ, TLT, GLD, VIX-inverse, HYG) under a two-state hidden Markov model: a calm regime with moderate correlations and a crisis regime where equity-equity correlations spike above 0.90 while bonds and gold diverge.

15 asset pairs are each rendered as a colored 3D ribbon stretching across the full time horizon. Each ribbon's color encodes correlation value: blue for negative, white for zero, deep red for strongly positive. Crisis periods are highlighted with a red glow wall behind all ribbons. The "correlation goes to 1" moment is visible as every ribbon shifts to red simultaneously — the exact phenomenon that blew up LTCM, destroyed 2008 risk models, and continues to catch portfolio managers off guard.

Side panels show the average correlation time series with regime coloring (blue = calm, red = crisis) and the most recent rolling correlation matrix rendered as a 3D bar heatmap.

---

## What You Learn Building This

- Markov chain regime switching with transition probability matrices
- Cholesky decomposition for sampling from correlated multivariate normal distributions
- Rolling correlation matrix computation over 3D arrays
- Poly3DCollection for rendering filled 3D polygon ribbons
- Regime detection and event marking on time series
- Building diverging custom colormaps centered at zero

---

## Setup

**Requirements:** Python 3.8+

Install dependencies:

```bash
pip install numpy matplotlib scipy
```

Run the project:

```bash
python p3_correlations.py
```

The output image `p3_correlation_regimes.png` is saved automatically in the same directory as the script.

---

## How to Fork It

1. Fork or clone the repository
2. Install dependencies with `pip install numpy matplotlib scipy`
3. Open `p3_correlations.py` and modify the regime parameters:
   - `trans` — the 2x2 transition matrix controls how sticky each regime is
   - `corr_calm()` and `corr_crisis()` — edit the correlation matrices for each regime
   - `vols_calm` and `vols_crisis` — per-asset volatility in each regime
4. Add more assets by expanding `N_ASSETS`, `ASSETS`, and both correlation matrices
5. Replace synthetic returns with real historical data from Yahoo Finance using `yfinance`
6. Add a third regime (e.g. "euphoria" with elevated but positive correlations) by extending the Markov chain to a 3x3 transition matrix
7. Use `hmmlearn` to fit the hidden Markov model to real data and detect regimes in-sample

---

## Using Real Data

To replace the simulation with real market data:

```python
import yfinance as yf
import pandas as pd

tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'HYG', 'LQD']
data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']
returns = data.pct_change().dropna().values
```

Then pass `returns` directly into the rolling correlation loop. The rest of the visualization code works unchanged.

---

## Output

A 22x13 inch figure at 180 DPI saved as `p3_correlation_regimes.png`. Three subplots: the main 15-ribbon correlation wall with crisis glow overlays, the average correlation time series with regime coloring, and the current correlation matrix as a 3D bar heatmap. Dark background (#050508) with a blue-white-red diverging colormap.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | any | Markov simulation, Cholesky, rolling corr |
| matplotlib | any | Poly3DCollection, bar3d, GridSpec |
| scipy | any | gaussian_filter for ribbon smoothing |

---

## License

MIT — free to use, modify, and distribute.