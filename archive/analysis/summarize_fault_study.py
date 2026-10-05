"""Summarize the Sept 27 delayed-fault diagnostic study; no simulator required."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/fault_diagnosis_matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]/'logs/diagnosis/fr_front_alpha01_20260927'
rows=[]
for folder,label in [('batch300_seed1','Fixed alpha = 0.1'),('play_severe_batch300_seed1','Play: alpha uniform [0, 0.1)')]:
    d=pd.read_csv(root/folder/'analysis/episode_metrics.csv')
    summary=pd.read_csv(root/folder/'analysis/joint_summary.csv').set_index('fault_joint')
    for name,g in d.groupby('fault_joint',sort=False):
        s=summary.loc[name]
        rows.append(dict(protocol=label,joint=name,n=len(g),terminated_by_10s_after_fault=int(round((1-s.survival_10s)*len(g))),height_below_20cm_first2s=int((g.post2s_min_base_height_m<.2).sum()),mean_min_height_cm=g.post2s_min_base_height_m.mean()*100,mean_vx_error_first2s=g.post2s_ate_vx_mps.mean(),detection_latency_s=s.median_detected_latency_s))
res=pd.DataFrame(rows);res.to_csv(root/'comparison.csv',index=False)
names=list(res[res.protocol=='Fixed alpha = 0.1'].joint)
fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
for k,(protocol,g) in enumerate(res.groupby('protocol',sort=False)):
    g=g.set_index('joint').loc[names];x=np.arange(len(names))+(k-.5)*.36
    for ax,field,title,ylabel in [(axes[0,0],'terminated_by_10s_after_fault','Terminated within 10 s after fault','Episodes / 300'),(axes[0,1],'height_below_20cm_first2s','Base height below 20 cm within first 2 s','Episodes / 300'),(axes[1,0],'mean_min_height_cm','Mean minimum base height, first 2 s','cm'),(axes[1,1],'mean_vx_error_first2s','Forward-speed error, first 2 s','Mean absolute error (m/s)')]:
        ax.bar(x,g[field],width=.35,label=protocol);ax.set_title(title);ax.set_ylabel(ylabel);ax.set_xticks(np.arange(len(names)),[n.replace('_joint','').replace('_','\n') for n in names],fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0,0].legend(fontsize=9);fig.suptitle('EquivGCNMLP v8 + randomized friction | random_rough | vx = 0.75 m/s | fault at 5 s\n300 first episodes per joint and 300 healthy controls per protocol; low height is not itself a termination')
fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)
print(res.round(3).to_string(index=False))
