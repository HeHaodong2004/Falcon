"""Export source-bound scientific figures; no inference or model selection."""
import json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eventwm_graph.l2_final.common import OUT,DOC,sha,dump


def save(fig,name):
    fig.savefig(DOC/'figures'/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(DOC/'figures'/(name+'.svg'),bbox_inches='tight');plt.close(fig)


def main():
    (DOC/'figures').mkdir(exist_ok=True);metrics=json.loads((OUT/'metrics.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#506b83','#a06b4b','#267879'];names=['A original','B repaired','C full data']
    primary=[next(r for r in metrics['variants'][arm]['forecasting']['scopes'] if r['cohort']=='native' and r['scope']=='practical')['metrics'] for arm in 'ABC']
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for ax,key,title in zip(axes,['path_es','coverage50','coverage90'],['Path Energy Score (lower better)','Coverage50','Coverage90']):
        for i,m in enumerate(primary):
            r=m[key];ax.errorbar(i,r['mean'],yerr=[[r['mean']-r['lo']],[r['hi']-r['mean']]],fmt='o',capsize=4,color=colors[i])
        ax.set_xticks(range(3));ax.set_xticklabels(names,rotation=15);ax.set_title(title)
        if key.startswith('coverage'):ax.axhline(.5 if key=='coverage50' else .9,ls='--',color='#333',lw=1,label='Nominal');ax.set_ylim(.2,1.)
        ax.grid(axis='y',alpha=.18)
    fig.suptitle('Frozen native gap: age 0.75–1.25 s, horizons ≤2 s · 120 roots / 53 scenes\nMarginal supported scores; 95% scene bootstrap intervals · development reuse',fontsize=11)
    fig.tight_layout();save(fig,'primary_forecasting')
    learning=json.loads((OUT/'training/learning.json').read_text());epochs=[r['epoch'] for r in learning]
    fig,ax=plt.subplots(figsize=(7.4,3.6));ax.plot(epochs,[r['dev_es'] for r in learning],label='Frozen selection ES',color=colors[2]);ax.plot(epochs[1:],[r['train_es'] for r in learning[1:]],label='Training ES (different sampling)',color=colors[0],alpha=.8)
    ax.axvline(52,ls='--',color='#555',lw=1,label='Selected epoch52');ax.set(xlabel='Complete full-cohort epoch',ylabel='Path Energy Score',title='One complete maximum-data fit · 21,872 motion roots per epoch');ax.legend();ax.grid(alpha=.15);fig.tight_layout();save(fig,'training_curve')
    cells=metrics['variants']['C']['forecasting']['age_horizon'];ages=['visible','0-.25','.25-.5','.5-1','1-1.5','1.5-2','over2'];hs=[.5,1.,2.,3.];matrix=np.full((len(ages),len(hs)),np.nan);counts=np.zeros_like(matrix,int)
    for i,age in enumerate(ages):
        for j,h in enumerate(hs):
            part=[r for r in cells if r['cohort']=='native' and r['age_bin']==age and r['H']==h]
            if not part:continue
            r=part[0]['metrics']['coverage90'];counts[i,j]=r['roots']
            if r['roots']>=30 and r['scenes']>=10:matrix[i,j]=r['mean']
    fig,ax=plt.subplots(figsize=(6.8,4.8));im=ax.imshow(matrix,vmin=.4,vmax=1.,cmap='YlGnBu');ax.set_xticks(range(4));ax.set_xticklabels(hs);ax.set_yticks(range(len(ages)));ax.set_yticklabels(ages)
    for i in range(len(ages)):
        for j in range(len(hs)):
            label=f'{matrix[i,j]:.2f}\nn={counts[i,j]}' if np.isfinite(matrix[i,j]) else f'insufficient\nn={counts[i,j]}'
            ax.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if np.isfinite(matrix[i,j]) and matrix[i,j]>.79 else '#222')
    ax.set(xlabel='Prediction horizon (s)',ylabel='Last legal observation age (s)',title='C native Coverage90 · supported cases\nCells <30 roots or <10 scenes are not estimated');fig.colorbar(im,ax=ax,label='Nominal target 0.90');fig.tight_layout();save(fig,'coverage90_age_horizon')
    latency=json.loads((OUT/'reviewer_c_checks/C_final_readout_inclusive_derived.json').read_text());fig,ax=plt.subplots(figsize=(7.2,3.8));x=np.arange(3);width=.19
    for k,(cache,field,label,color) in enumerate([('warm','frozen_complete','Warm core','#506b83'),('dynamic_cold','frozen_complete','Cold core','#8ba0b0'),('warm','readouts_inclusive','Warm inclusive','#267879'),('dynamic_cold','readouts_inclusive','Cold inclusive','#83b3ac')]):
        values=[next(r for r in latency['rows'] if r['cohort']==cohort and r['cache']==cache)[field]['p95_ms'] for cohort in ('representative','expanded','dense')]
        ax.bar(x+(k-1.5)*width,values,width=width,label=label,color=color)
    ax.axhline(100,ls='--',color='#8e4a40',lw=1);ax.set_xticks(x);ax.set_xticklabels(['Representative','Expanded','Dense']);ax.set(ylabel='Complete-call P95 (ms)',title='C fixed 16-world workload · inclusive dense target remains unmet');ax.legend(ncol=2,fontsize=8);fig.tight_layout();save(fig,'latency_p95')
    dump(DOC/'figures/sources.json',dict(metrics_sha256=sha(OUT/'metrics.json'),learning_sha256=sha(OUT/'training/learning.json'),latency_sha256=sha(OUT/'reviewer_c_checks/C_final_readout_inclusive_derived.json'),
        interpretation='development reuse; marginal supported scores; dependent cases averaged within root then scene; insufficient cells masked; no figure establishes release promotion'))


if __name__=='__main__':main()
