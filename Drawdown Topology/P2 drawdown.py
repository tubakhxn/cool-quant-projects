import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

np.random.seed(7)

# ── Simulate portfolio paths (GBM + fat tails) ───────────────────────────────
N_PATHS  = 3000
N_DAYS   = 252
dt       = 1/252
mu_ann   = 0.12
sig_ann  = 0.22
rf       = 0.05

# Fat-tailed returns: 90% normal + 10% t-distribution shocks
def fat_gbm(n_paths, n_days, mu, sigma, df_t=4):
    norm_ret = np.random.normal((mu - 0.5*sigma**2)*dt,
                                 sigma*np.sqrt(dt),
                                 (n_paths, n_days))
    t_shocks = np.random.standard_t(df_t, (n_paths, n_days)) * sigma * np.sqrt(dt) * 1.5
    mask     = np.random.random((n_paths, n_days)) < 0.07
    returns  = np.where(mask, t_shocks, norm_ret)
    prices   = np.cumprod(1 + returns, axis=1)
    return np.hstack([np.ones((n_paths, 1)), prices])

paths = fat_gbm(N_PATHS, N_DAYS, mu_ann, sig_ann)

# ── Compute running drawdown for each path ───────────────────────────────────
running_max = np.maximum.accumulate(paths, axis=1)
drawdowns   = (paths - running_max) / running_max  # always <= 0

# ── Build 2D density: (day, drawdown_depth) ──────────────────────────────────
days_arr = np.arange(N_DAYS + 1)
dd_bins  = np.linspace(-0.70, 0.0, 100)
density  = np.zeros((len(days_arr), len(dd_bins) - 1))

for i, day in enumerate(days_arr):
    dd_today = drawdowns[:, day]
    h, _ = np.histogram(dd_today, bins=dd_bins, density=True)
    density[i] = h

# Smooth the density landscape
density = gaussian_filter(density, sigma=[3, 2])

DD_mid = (dd_bins[:-1] + dd_bins[1:]) / 2
D_g, T_g = np.meshgrid(DD_mid * 100, days_arr)  # DD in %, days on x

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 12), facecolor='#020810')
gs  = gridspec.GridSpec(2, 3, figure=fig,
                         left=0.03, right=0.97,
                         top=0.90, bottom=0.06,
                         wspace=0.06, hspace=0.40)

# ── MAIN: Drawdown density mountain ───────────────────────────────────────────
ax = fig.add_subplot(gs[:, :2], projection='3d')
ax.set_facecolor('#020810')

# Custom colormap: black → dark green → lime → yellow → white (like a radar)
radar_colors = ['#000000', '#001a00', '#003300', '#006600',
                '#00cc00', '#99ff00', '#ffff00', '#ffffff']
cmap_radar = mcolors.LinearSegmentedColormap.from_list('radar', radar_colors, N=512)

# Normalize log-density for dramatic peaks
log_density = np.log1p(density * 20)

surf = ax.plot_surface(T_g, D_g, log_density,
                        cmap=cmap_radar,
                        alpha=0.90, linewidth=0,
                        antialiased=True, shade=True)

# Max drawdown spine (median path)
median_dd = np.median(drawdowns, axis=0) * 100
p5_dd     = np.percentile(drawdowns, 5, axis=0)  * 100
p95_dd    = np.percentile(drawdowns, 95, axis=0) * 100

# Get z-height for these lines
def get_z_at(day, dd_val):
    dd_pct = np.clip(dd_val, dd_bins[0]*100, dd_bins[-1]*100)
    idx_d  = np.argmin(np.abs(DD_mid*100 - dd_pct))
    return log_density[day, idx_d]

z_med = [get_z_at(d, median_dd[d]) + 0.05 for d in range(len(days_arr))]
z_p5  = [get_z_at(d, p5_dd[d])    + 0.05 for d in range(len(days_arr))]

ax.plot(days_arr, median_dd, z_med, color='#00ffcc', linewidth=2.5,
        label='Median Drawdown', zorder=10)
ax.plot(days_arr, p5_dd, z_p5, color='#ff4444', linewidth=2.0,
        linestyle='--', label='5th Percentile (Worst 5%)', zorder=10)

# Curtain walls (vertical lines to zero)
for day in range(0, N_DAYS, 25):
    z_floor = np.zeros(len(dd_bins)-1)
    ax.plot([day]*len(dd_bins[:-1]), DD_mid*100, z_floor,
            color='#003300', linewidth=0.3, alpha=0.4)

# ── VaR plane ─────────────────────────────────────────────────────────────────
var_95 = np.percentile(drawdowns[:, -1], 5) * 100
ax.plot([0, N_DAYS], [var_95, var_95], [0, 0],
        color='#ff0000', linewidth=2, alpha=0.7, linestyle=':',
        label=f'Annual VaR 95% = {var_95:.1f}%')

# ── Styling main ──────────────────────────────────────────────────────────────
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('#0a1a0a')
ax.grid(color='#051005', linewidth=0.5, alpha=0.8)
ax.tick_params(colors='#66aa66', labelsize=8)
ax.set_xlabel('Trading Day',       color='#88dd88', fontsize=11, labelpad=12)
ax.set_ylabel('Drawdown (%)',      color='#88dd88', fontsize=11, labelpad=12)
ax.set_zlabel('Probability Density', color='#88dd88', fontsize=9, labelpad=10)
ax.set_ylim(-70, 5)
ax.set_title('DRAWDOWN TOPOLOGY\nμ=12%  σ=22%  Fat-Tails  |  3,000 Paths',
             color='#ccffcc', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', framealpha=0.15, facecolor='#001a00',
          edgecolor='#006600', labelcolor='white', fontsize=9)
ax.view_init(elev=35, azim=210)

cbar = fig.colorbar(surf, ax=ax, shrink=0.3, aspect=10, pad=0.06)
cbar.ax.tick_params(colors='#66aa66', labelsize=7)
cbar.set_label('log(1+density)', color='#88dd88', fontsize=9)

# ── RIGHT TOP: Fan chart ──────────────────────────────────────────────────────
ax_fan = fig.add_subplot(gs[0, 2], projection='3d')
ax_fan.set_facecolor('#020810')

# Show 100 actual paths as semi-transparent ribbons
sample_idx = np.random.choice(N_PATHS, 80, replace=False)
for idx in sample_idx:
    dd_path = drawdowns[idx] * 100
    final_val = paths[idx, -1]
    color_val = (final_val - 0.5) / 2.0  # normalize
    col = plt.cm.RdYlGn(np.clip(color_val, 0, 1))
    ax_fan.plot(days_arr, dd_path, np.zeros_like(days_arr),
                color=col, linewidth=0.6, alpha=0.35)

# Percentile bands
for p, col, lw in [(10,'#ff3333',2),(25,'#ff9900',1.5),(50,'#00ff88',2.5),(75,'#ff9900',1.5),(90,'#ff3333',2)]:
    pct = np.percentile(drawdowns, p, axis=0) * 100
    ax_fan.plot(days_arr, pct, np.zeros_like(days_arr),
                color=col, linewidth=lw, alpha=0.9)

ax_fan.set_title('PATH FAN CHART\n(colored by final P&L)', color='#aaddaa', fontsize=9, fontweight='bold')
ax_fan.set_xlabel('Day', color='#88dd88', fontsize=7, labelpad=5)
ax_fan.set_ylabel('DD %', color='#88dd88', fontsize=7, labelpad=5)
for pane in [ax_fan.xaxis.pane, ax_fan.yaxis.pane, ax_fan.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('#051005')
ax_fan.tick_params(colors='#66aa66', labelsize=6)
ax_fan.view_init(elev=60, azim=-80)

# ── RIGHT BOTTOM: Terminal distribution ──────────────────────────────────────
ax_td = fig.add_subplot(gs[1, 2], projection='3d')
ax_td.set_facecolor('#020810')

terminal_vals = paths[:, -1]
val_bins = np.linspace(0.1, 4.0, 80)
val_mid  = (val_bins[:-1] + val_bins[1:]) / 2

hist_vals, _ = np.histogram(terminal_vals, bins=val_bins, density=True)
hist_smooth  = gaussian_filter(hist_vals, sigma=2)

# 3D bar chart as vertical ribbon
xs = val_mid
ys = np.zeros_like(xs)
zs = np.zeros_like(xs)
dx = (val_bins[1] - val_bins[0]) * 0.85
dy = 0.5
colors_bar = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(xs)))

ax_td.bar3d(xs, ys, zs, dx, dy, hist_smooth,
            color=colors_bar, alpha=0.85, shade=True)
ax_td.axvline(1.0, color='white', linewidth=1.5, alpha=0.6)

ax_td.set_title('TERMINAL VALUE\nDistribution after 1yr', color='#aaddaa', fontsize=9, fontweight='bold')
ax_td.set_xlabel('Portfolio Value', color='#88dd88', fontsize=7, labelpad=5)
ax_td.set_zlabel('Density', color='#88dd88', fontsize=7, labelpad=5)
for pane in [ax_td.xaxis.pane, ax_td.yaxis.pane, ax_td.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('#051005')
ax_td.tick_params(colors='#66aa66', labelsize=6)
ax_td.view_init(elev=25, azim=-50)

fig.text(0.5, 0.96,
         '🏔  PROJECT 2  ──  DRAWDOWN TOPOLOGY  ──  3,000 MONTE CARLO PATHS',
         ha='center', va='top', fontsize=13, color='#88dd88',
         fontfamily='monospace', fontweight='bold')

import os as _os
_out = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'p2_drawdown_topology.png')
plt.savefig(_out, dpi=180, bbox_inches='tight', facecolor='#020810')
print(f"Saved -> {_out}")
plt.show()