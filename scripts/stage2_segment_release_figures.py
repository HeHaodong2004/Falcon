"""Standalone research figures from frozen, already evaluated JSON results."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eventwm_graph.stage2_segment.common import BASE, OUT, dump, sha


def main():
    dest=BASE/'docs/l2/stage2_segment_20261024/figures';dest.mkdir(exist_ok=True)
    paths=dict(science=OUT/'evaluation/final_segment_dev/metrics.json',
               single=OUT/'benchmarks/release_single_v1/metrics.json',
               multi=OUT/'benchmarks/release_multi_v1/metrics.json')
    data={k:json.loads(p.read_text()) for k,p in paths.items()};m=data['science']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    colors={'baseline':'#65758b','candidate':'#007f82'}
    def save(fig,name):
        fig.savefig(dest/(name+'.png'),bbox_inches='tight');fig.savefig(dest/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,6),sharex='col')
    for col,cls in enumerate(('unknown','free','blocked')):
        for model in ('baseline','candidate'):
            bins=m[model]['calibration'][cls]['bins'];p=np.array([b['probability'] if b['probability'] is not None else np.nan for b in bins])
            freq=np.array([b['frequency'] if b['frequency'] is not None else np.nan for b in bins]);mass=np.array([b['mass'] for b in bins])
            axes[0,col].plot(p,freq,'o-',ms=3,color=colors[model],label='Original' if model=='baseline' else 'Final')
            axes[1,col].step([(b['lower']+b['upper'])/2 for b in bins],mass,where='mid',color=colors[model])
        axes[0,col].plot([0,1],[0,1],':',color='black',lw=.8);axes[0,col].set_title(cls.upper());axes[0,col].set_ylim(0,1);axes[0,col].set_xlim(0,1)
        axes[1,col].set_yscale('log');axes[1,col].set_ylim(1e-8,1.1);axes[1,col].set_xlabel('Predicted class probability')
    axes[0,0].set_ylabel('Empirical class frequency');axes[1,0].set_ylabel('Scene-balanced bin mass');axes[0,0].legend()
    fig.suptitle('Frozen segment development: calibration and support\n15 fixed bins; proper-score gains coexist with mixed classwise calibration')
    fig.tight_layout();save(fig,'calibration')
    fig,ax=plt.subplots(figsize=(6,4))
    for model in ('baseline','candidate'):
        curve=m[model]['revelation_precision_recall']
        ax.plot([p['recall'] for p in curve],[p['precision'] for p in curve],'o-',ms=3,color=colors[model],label='Original' if model=='baseline' else 'Final')
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('Revelation recall');ax.set_ylabel('Revelation precision')
    ax.set_title('New-observation probability precision–recall\nFrozen thresholds on P(FREE)+P(BLOCKED); class identity ignored here')
    ax.legend();fig.tight_layout();save(fig,'revelation_precision_recall')
    cohorts=['duration:<=0.5','duration:0.5-1','duration:1-2','duration:>2']
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(4)
    for model,offset,label in (('baseline',-.12,'Original'),('candidate',.12,'Final')):
        values=[];low=[];high=[]
        for name in cohorts:
            d=m['cohorts'][name][model];v=d['scene_balanced']['label_aware_delta_iou'];ci=d['ci95']['label_aware_delta_iou']
            values.append(v);low.append(v-ci[0]);high.append(ci[1]-v)
        ax.errorbar(x+offset,values,yerr=[low,high],fmt='o',capsize=3,color=colors[model],label=label)
    ax.set_xticks(x,['≤0.5','0.5–1','1–2','>2']);ax.set_xlabel('Actual segment duration (s)');ax.set_ylabel('Label-aware GraphDelta IoU');ax.set_ylim(0,None)
    ax.set_title('New legal map content by segment duration\nScene/Root/window balancing; scene-cluster 95% intervals');ax.legend();fig.tight_layout();save(fig,'duration_iou')
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True);ks=[1,2,4,8]
    for ax,condition,title in zip(axes,('map_cold','retained_cache'),('Map cold','Retained cache')):
        for model in ('baseline','candidate'):
            d=data['multi']['workloads'][model][condition]
            ax.plot(ks,[d[str(k)]['p95_ms'] for k in ks],'o-',color=colors[model],label='Original CPU' if model=='baseline' else 'Final CUDA')
        ax.axhline(200,ls=':',color='black',lw=.8);ax.set_xticks(ks);ax.set_title(title);ax.set_xlabel('Distinct segments sharing one graph')
    axes[0].set_ylabel('Complete K-segment call P95 (ms)');axes[0].legend();fig.suptitle('Matched real saved segments; every future graph materialized');fig.tight_layout();save(fig,'multi_latency')
    dump(dest/'manifest.json',dict(inputs={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        outputs={p.name:sha(p) for p in sorted(dest.iterdir()) if p.suffix in ('.png','.pdf')},
        scope='Descriptive plots of frozen results; no model selection, thresholds or metric changes'))


if __name__=='__main__':main()
