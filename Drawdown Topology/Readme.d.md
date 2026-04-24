# Drawdown Topology Map

A 3D probability density visualization of portfolio drawdowns across 3,000 simulated fat-tailed GBM paths. The density landscape is rendered as a radar-green topographic mountain range — showing exactly where portfolios go to die over a 252-day trading year.

---

## Developer / Creator

## tubakhxn

GitHub: [github.com/tubakhxn](https://github.com/tubakhxn)

---

## What This Project Is

Standard risk reports show a single number — VaR at 95%, or maximum drawdown. This project renders the entire drawdown probability landscape as a 3D surface across both time and depth dimensions, so you can see how the distribution evolves day by day through the trading year.

3,000 paths are simulated using a fat-tailed GBM: 93% standard normal returns mixed with 7% Student-t shocks at 4 degrees of freedom, mimicking the heavy tails observed in real equity returns. The running drawdown is computed for every path at every timestep, then binned into a 2D histogram and smoothed with a Gaussian filter.

The result is a terrain map where peaks indicate where the probability mass concentrates. You can see the slow left-drift of the distribution as time passes, the rare deep canyons below -50%, and the 5th percentile red spine carving through the landscape. Side panels show a fan chart of 80 individual paths colored by final P&L, and a 3D terminal value distribution bar chart.

---

## What You Learn Building This

- Geometric Brownian Motion with fat-tailed (Student-t) shock mixing
- Rolling maximum drawdown computation over 2D arrays with `np.maximum.accumulate`
- 2D histogram density estimation across time and drawdown space
- Gaussian smoothing of density surfaces with `scipy.ndimage.gaussian_filter`
- Multi-panel figure layouts with GridSpec
- Dirichlet-style path sampling and percentile fan charts

---

## Setup

**Requirements:** Python 3.8+

Install dependencies:

```bash
pip install numpy matplotlib scipy
```

Run the project:

```bash
python p2_drawdown.py
```

The output image `p2_drawdown_topology.png` is saved automatically in the same directory as the script.

---

## How to Fork It

1. Fork or clone the repository
2. Install dependencies with `pip install numpy matplotlib scipy`
3. Open `p2_drawdown.py` and adjust the simulation parameters:
   - `N_PATHS` — number of simulated paths (more = smoother density, slower)
   - `mu_ann` — annualized drift (expected return)
   - `sig_ann` — annualized volatility
   - `df_t` — degrees of freedom for Student-t shocks (lower = fatter tails)
4. Change the shock probability from `0.07` to higher values to simulate crises
5. Swap GBM for a mean-reverting Ornstein-Uhlenbeck process to model rates or spreads
6. Add a second portfolio (e.g. 60/40 bonds overlay) and subtract drawdowns to compare strategies
7. Replace synthetic paths with real historical return data loaded from a CSV

---

## Output

A 22x12 inch figure at 180 DPI saved as `p2_drawdown_topology.png`. Three subplots: the main drawdown topology mountain range with median and 5th percentile spines, the path fan chart, and the terminal value distribution. Dark background (#020810) with a radar-green colormap.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | any | GBM simulation, running max, histograms |
| matplotlib | any | 3D surface, bar3d, GridSpec |
| scipy | any | gaussian_kde, gaussian_filter |

---

## License

MIT — free to use, modify, and distribute.