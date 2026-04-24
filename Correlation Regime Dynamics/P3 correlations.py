import os as _os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
np.random.seed(99)
N_DAYS=504; N_ASSETS=6
ASSETS=['SPY','QQQ','TLT','GLD','VIX_inv','HYG']
def corr_calm():
    return np.array([[1.00,0.85,-0.30,0.05,-0.60,0.70],[0.85,1.00,-0.25,0.10,-0.55,0.65],[-0.30,-0.25,1.00,0.20,0.40,-0.20],[0.05,0.10,0.20,1.00,-0.10,0.00],[-0.60,-0.55,0.40,-0.10,1.00,-0.55],[0.70,0.65,-0.20,0.00,-0.55,1.00]])
def corr_crisis():
    return np.array([[1.00,0.95,-0.75,0.35,-0.85,0.88],[0.95,1.00,-0.72,0.32,-0.82,0.85],[-0.75,-0.72,1.00,0.60,0.70,-0.68],[0.35,0.32,0.60,1.00,-0.25,0.28],[-0.85,-0.82,0.70,-0.25,1.00,-0.80],[0.88,0.85,-0.68,0.28,-0.80,1.00]])
vols_calm=np.array([0.15,0.18,0.10,0.12,0.60,0.08])
vols_crisis=np.array([0.35,0.40,0.20,0.22,1.20,0.28])
trans=np.array([[0.985,0.015],[0.05,0.95]])
regime=0; regimes=[]; returns=np.zeros((N_DAYS,N_ASSETS))
cc,ck=corr_calm(),corr_crisis()
for t in range(N_DAYS):
    regime=np.random.choice(2,p=trans[regime]); regimes.append(regime)
    vc=vols_calm if regime==0 else vols_crisis
    cr=cc if regime==0 else ck
    cov=np.outer(vc,vc)*cr+1e-7*np.eye(N_ASSETS)
    try:
        L=np.linalg.cholesky(cov); returns[t]=(L@np.random.normal(0,1,N_ASSETS))/np.sqrt(252)
    except:
        returns[t]=np.random.normal(0,vc/np.sqrt(252))
regimes=np.array(regimes)
WIN=30; n_roll=N_DAYS-WIN
corr_ts=np.zeros((n_roll,N_ASSETS,N_ASSETS))
for t in range(n_roll):
    corr_ts[t]=np.corrcoef(returns[t:t+WIN].T)
pairs=[(i,j) for i in range(N_ASSETS) for j in range(i+1,N_ASSETS)]
pair_labels=[f'{ASSETS[i]}x{ASSETS[j]}' for i,j in pairs]
n_pairs=len(pairs)
corr_pair_ts=np.array([[corr_ts[t,i,j] for t in range(n_roll)] for i,j in pairs])
roll_days=np.arange(n_roll)
regime_roll=regimes[WIN:WIN+n_roll]
corr_cmap=mcolors.LinearSegmentedColormap.from_list('corr',['#0033cc','#3366ff','#9999ff','#ffffff','#ffaa77','#ff4400','#cc0000'],N=512)
fig=plt.figure(figsize=(22,13),facecolor='#050508')
gs=gridspec.GridSpec(2,3,figure=fig,left=0.03,right=0.97,top=0.90,bottom=0.06,wspace=0.06,hspace=0.38)
ax=fig.add_subplot(gs[:,0:2],projection='3d'); ax.set_facecolor('#050508')
for pi,(pair_ts,label) in enumerate(zip(corr_pair_ts,pair_labels)):
    smooth=gaussian_filter(pair_ts.astype(float),sigma=2.5)
    y_lo=pi+0.05; y_hi=pi+0.72
    for si in range(0,len(roll_days)-2,2):
        x0,x1=float(roll_days[si]),float(roll_days[si+1]); z0,z1=float(smooth[si]),float(smooth[si+1])
        cv=(z0+z1)/2; rgba=corr_cmap((cv+1)/2)
        verts=[[(x0,y_lo,z0),(x1,y_lo,z1),(x1,y_hi,z1),(x0,y_hi,z0)]]
        ax.add_collection3d(Poly3DCollection(verts,alpha=0.78,facecolor=rgba,edgecolor='none'))
    ax.plot(roll_days,np.full(n_roll,y_hi),smooth,color='white',linewidth=0.5,alpha=0.3)
in_crisis=(regime_roll==1); trans_arr=np.diff(in_crisis.astype(int))
starts=np.where(trans_arr==1)[0]+1; ends=np.where(trans_arr==-1)[0]+1
if in_crisis[0]: starts=np.concatenate([[0],starts])
if in_crisis[-1]: ends=np.concatenate([ends,[n_roll-1]])
for cs,ce in zip(starts[:8],ends[:8]):
    for pi in range(n_pairs):
        verts=[[(float(cs),float(pi),-1.15),(float(ce),float(pi),-1.15),(float(ce),float(pi)+0.72,-1.15),(float(cs),float(pi)+0.72,-1.15)]]
        ax.add_collection3d(Poly3DCollection(verts,alpha=0.07,facecolor='#ff0000',edgecolor='none'))
for pane in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
    pane.fill=False; pane.set_edgecolor('#101025')
ax.grid(color='#0a0a20',linewidth=0.4,alpha=0.8); ax.tick_params(colors='#8888cc',labelsize=7)
ax.set_yticks([i+0.35 for i in range(n_pairs)]); ax.set_yticklabels(pair_labels,fontsize=6,color='#aaaaee')
ax.set_xlabel('Trading Day',color='#aaaaff',fontsize=11,labelpad=12); ax.set_zlabel('Correlation',color='#aaaaff',fontsize=11,labelpad=12)
ax.set_zlim(-1.15,1.15)
ax.set_title('CROSS-ASSET CORRELATION REGIME DYNAMICS\n6 Assets | 30-Day Rolling | Regime-Switching Markov Model',color='#ddddff',fontsize=13,fontweight='bold')
ax.view_init(elev=20,azim=-50)
sm=plt.cm.ScalarMappable(cmap=corr_cmap,norm=plt.Normalize(-1,1)); sm.set_array([])
cbar=fig.colorbar(sm,ax=ax,shrink=0.3,aspect=10,pad=0.06); cbar.ax.tick_params(colors='#8888cc',labelsize=7); cbar.set_label('Correlation',color='#aaaaff',fontsize=9)
ax_avg=fig.add_subplot(gs[0,2],projection='3d'); ax_avg.set_facecolor('#050508')
avg_s=gaussian_filter(corr_pair_ts.mean(axis=0),sigma=3)
for t in range(n_roll-1):
    col='#ff3300' if regime_roll[t]==1 else '#3366ff'
    ax_avg.plot([roll_days[t],roll_days[t+1]],[avg_s[t],avg_s[t+1]],[0,0],color=col,linewidth=1.5,alpha=0.8)
ax_avg.set_title('AVG CORRELATION\nBlue=Calm Red=Crisis',color='#aaaaee',fontsize=9,fontweight='bold')
ax_avg.set_xlabel('Day',color='#aaaaff',fontsize=7,labelpad=5)
for pane in [ax_avg.xaxis.pane,ax_avg.yaxis.pane,ax_avg.zaxis.pane]: pane.fill=False; pane.set_edgecolor('#101025')
ax_avg.tick_params(colors='#6666aa',labelsize=6); ax_avg.view_init(elev=45,azim=-60)
ax_hm=fig.add_subplot(gs[1,2],projection='3d'); ax_hm.set_facecolor('#050508')
final_corr=corr_ts[-1]; xs=np.arange(N_ASSETS)
X,Y=np.meshgrid(xs,xs); Xf,Yf,Zf=X.flatten(),Y.flatten(),final_corr.flatten()
colors_hm=corr_cmap((Zf+1)/2); mask=Xf!=Yf
ax_hm.bar3d(Xf[mask]-0.38,Yf[mask]-0.38,np.zeros(mask.sum()),0.76,0.76,Zf[mask],color=colors_hm[mask],alpha=0.88,shade=True)
ax_hm.set_title('CURRENT CORR MATRIX',color='#aaaaee',fontsize=9,fontweight='bold')
ax_hm.set_xticks(xs); ax_hm.set_xticklabels(ASSETS,fontsize=5,color='#aaaaee',rotation=30)
ax_hm.set_yticks(xs); ax_hm.set_yticklabels(ASSETS,fontsize=5,color='#aaaaee',rotation=-20)
ax_hm.set_zlabel('rho',color='#aaaaff',fontsize=7,labelpad=5)
for pane in [ax_hm.xaxis.pane,ax_hm.yaxis.pane,ax_hm.zaxis.pane]: pane.fill=False; pane.set_edgecolor('#101025')
ax_hm.tick_params(colors='#6666aa',labelsize=5); ax_hm.view_init(elev=35,azim=-55)
fig.text(0.5,0.96,'PROJECT 3  --  CORRELATION REGIME DYNAMICS  --  CRISIS DETECTION ENGINE',ha='center',va='top',fontsize=13,color='#aaaaff',fontfamily='monospace',fontweight='bold')
plt.savefig(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'p3_correlation_regimes.png'),dpi=180,bbox_inches='tight',facecolor='#050508')
print("p3 done")