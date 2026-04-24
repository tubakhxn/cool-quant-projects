import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import norm
from matplotlib.colors import LightSource

np.random.seed(42)

# ── Black-Scholes machinery ──────────────────────────────────────────────────
def bs(S, K, T, r, sigma, cp=1):
    T = np.maximum(T, 1e-9)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    price = cp*(S*norm.cdf(cp*d1) - K*np.exp(-r*T)*norm.cdf(cp*d2))
    delta = cp*norm.cdf(cp*d1)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - cp*r*K*np.exp(-r*T)*norm.cdf(cp*d2))/365
    vega  = S*norm.pdf(d1)*np.sqrt(T)/100
    return price, delta, gamma, theta, vega

# Parameters
K      = 100.0
r      = 0.05
sigma  = 0.25
T0     = 1.0        # initial time to expiry
entry_price, *_ = bs(K, K, T0, r, sigma)  # ATM entry

# Grid: spot × time
spots = np.linspace(70, 135, 120)
times = np.linspace(0.01, T0, 100)
S_g, T_g = np.meshgrid(spots, times)

# P&L surface for long call
price_g, delta_g, gamma_g, theta_g, vega_g = bs(S_g, K, T_g, r, sigma)
pnl_g = price_g - entry_price

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 13), facecolor='#060010')
gs  = gridspec.GridSpec(2, 3, figure=fig,
                         left=0.04, right=0.96,
                         top=0.90, bottom=0.06,
                         wspace=0.05, hspace=0.35)

# ── MAIN: 3D PnL surface ──────────────────────────────────────────────────────
ax_main = fig.add_subplot(gs[:, :2], projection='3d')
ax_main.set_facecolor('#060010')

# Lighting for 3D depth
ls = LightSource(azdeg=315, altdeg=45)
rgb = ls.shade(pnl_g, cmap=plt.cm.RdYlGn,
               blend_mode='overlay', vert_exag=0.4,
               vmin=-entry_price, vmax=entry_price*3)

surf = ax_main.plot_surface(S_g, T_g*365, pnl_g,
                             facecolors=rgb,
                             linewidth=0, antialiased=True,
                             alpha=0.93, shade=False)

# Zero P&L plane (breakeven)
S_be, T_be = np.meshgrid(spots, times)
zero_plane  = np.zeros_like(pnl_g)
ax_main.plot_surface(S_be, T_be*365, zero_plane,
                      color='#ffffff', alpha=0.06,
                      linewidth=0, antialiased=False)

# Breakeven contour lifted slightly above the surface
ax_main.contour(S_g, T_g*365, pnl_g, levels=[0],
                zdir='z', colors=['#00ffcc'], linewidths=2.5,
                linestyles='--', offset=None)

# Delta ribbons — vertical slices at key Delta values
for delta_target, col in [(0.25, '#ff6b6b'), (0.50, '#ffd700'), (0.75, '#69b3ff')]:
    # find spot where delta ≈ target at mid-time
    T_mid = T0 / 2
    s_arr = np.linspace(70, 135, 2000)
    _, d_arr, *_ = bs(s_arr, K, T_mid, r, sigma)
    idx = np.argmin(np.abs(d_arr - delta_target))
    s_val = s_arr[idx]
    t_line = np.linspace(0.01, T0, 80)
    p_line, *_ = bs(s_val, K, t_line, r, sigma)
    pnl_line = p_line - entry_price
    ax_main.plot([s_val]*80, t_line*365, pnl_line,
                 color=col, linewidth=2.2, alpha=0.9,
                 label=f'Δ={delta_target:.2f}  S≈{s_val:.0f}')

# Expiry payoff profile on back wall
t_expiry = np.array([0.001]*len(spots))
p_expiry, *_ = bs(spots, K, t_expiry, r, sigma)
pnl_expiry = p_expiry - entry_price
ax_main.plot(spots, np.ones_like(spots)*0.5, pnl_expiry,
             color='#ff4500', linewidth=3, alpha=1.0, label='Expiry Payoff', zorder=10)

# ── Styling main ──────────────────────────────────────────────────────────────
for pane in [ax_main.xaxis.pane, ax_main.yaxis.pane, ax_main.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('#1a0040')
ax_main.grid(color='#150030', linewidth=0.5, alpha=0.8)
ax_main.tick_params(colors='#9999cc', labelsize=8)
ax_main.set_xlabel('Spot Price ($)',      color='#cc99ff', fontsize=11, labelpad=12)
ax_main.set_ylabel('Days to Expiry',      color='#cc99ff', fontsize=11, labelpad=12)
ax_main.set_zlabel('P&L ($)',             color='#cc99ff', fontsize=11, labelpad=12)
ax_main.set_title('LONG CALL  P&L MANIFOLD\nK=100  σ=25%  r=5%',
                  color='#eee0ff', fontsize=14, fontweight='bold')
ax_main.legend(loc='upper left', framealpha=0.15, facecolor='#1a0040',
               edgecolor='#6633cc', labelcolor='white', fontsize=8)
ax_main.view_init(elev=24, azim=-50)

# ── RIGHT TOP: Gamma surface ──────────────────────────────────────────────────
ax_g = fig.add_subplot(gs[0, 2], projection='3d')
ax_g.set_facecolor('#060010')
gamma_capped = np.minimum(gamma_g, np.percentile(gamma_g, 98))
surf_g = ax_g.plot_surface(S_g, T_g*365, gamma_capped,
                            cmap='hot', alpha=0.9,
                            linewidth=0, antialiased=True)
ax_g.set_title('GAMMA SURFACE', color='#ffcc88', fontsize=10, fontweight='bold')
ax_g.set_xlabel('Spot', color='#ffcc88', fontsize=7, labelpad=6)
ax_g.set_ylabel('DTE',  color='#ffcc88', fontsize=7, labelpad=6)
ax_g.set_zlabel('Γ',    color='#ffcc88', fontsize=7, labelpad=6)
for pane in [ax_g.xaxis.pane, ax_g.yaxis.pane, ax_g.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('#2a1500')
ax_g.tick_params(colors='#996633', labelsize=6)
ax_g.view_init(elev=30, azim=-60)

# ── RIGHT BOTTOM: Theta bleed heatmap ────────────────────────────────────────
ax_t = fig.add_subplot(gs[1, 2], projection='3d')
ax_t.set_facecolor('#060010')
theta_surf = theta_g * 100  # in cents
surf_t = ax_t.plot_surface(S_g, T_g*365, theta_surf,
                            cmap='Blues_r', alpha=0.9,
                            linewidth=0, antialiased=True)
ax_t.set_title('THETA BLEED  (¢/day)', color='#88ccff', fontsize=10, fontweight='bold')
ax_t.set_xlabel('Spot', color='#88aaff', fontsize=7, labelpad=6)
ax_t.set_ylabel('DTE',  color='#88aaff', fontsize=7, labelpad=6)
ax_t.set_zlabel('Θ (¢)', color='#88aaff', fontsize=7, labelpad=6)
for pane in [ax_t.xaxis.pane, ax_t.yaxis.pane, ax_t.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('#001530')
ax_t.tick_params(colors='#335577', labelsize=6)
ax_t.view_init(elev=30, azim=-60)

# ── Title banner ──────────────────────────────────────────────────────────────
fig.text(0.5, 0.96,
         '⚡  PROJECT 1  ──  OPTION GREEKS MANIFOLD  ──  BUILD IN 12 HRS',
         ha='center', va='top', fontsize=13, color='#cc99ff',
         fontfamily='monospace', fontweight='bold')

plt.savefig('outputs/p1_greeks_manifold.png',
            dpi=180, bbox_inches='tight', facecolor='#060010')
print(" Project 1 done → outputs/p1_greeks_manifold.png")
plt.show()