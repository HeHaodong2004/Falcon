"""Publication artifacts from already computed intrinsic forecast records."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',action='append',required=True,help='NAME=result.json')
    p.add_argument('--joint',action='append',default=[],help='NAME=uncertainty.json');p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);bench=dict(x.split('=',1) for x in a.benchmark);joint=dict(x.split('=',1) for x in a.joint)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);summary={}
    population=None
    for index,(name,path) in enumerate(bench.items()):
        d=read(path);rows=d['extras']['3.0']['records'];color=plt.get_cmap('tab10')(index)
        ids={r['root_id'] for r in rows}
        if population is not None and ids!=population:raise ValueError('Plot requires matched root populations')
        population=ids;horizons=sorted(d['scores'],key=float)
        axes[0,0].plot([float(h) for h in horizons],[d['scores'][h]['metrics']['energy'] for h in horizons],'-o',color=color,label=name)
        pits=np.array([r['pit_histogram'] for r in rows if 'pit_histogram' in r],float)
        histogram=(pits/pits.sum(1,keepdims=True)).mean(0)
        axes[0,1].plot(np.arange(10)/10+.05,histogram,'-o',color=color,label=name)
        bins=[]
        for i in range(10):
            cells=[v for r in rows for v in r.get('occupancy',{}).get('reliability',[]) if v['bin']==i]
            count=sum(c['n'] for c in cells)
            if count:
                bins.append((sum(c['n']*c['predicted'] for c in cells)/count,
                    sum(c['n']*c['observed'] for c in cells)/count,count))
        axes[1,1].plot([x[0] for x in bins],[x[1] for x in bins],'-o',color=color,label=name)
        summary[name]=dict(benchmark_sha256=sha256(path),pit_roots=len(pits),pit_root_mean=histogram.tolist(),
            horizon_roots={h:d['scores'][h]['trajectory_roots'] for h in horizons},occupancy_reliability=bins)
        if name in joint:
            u=read(joint[name]);rank=np.array([r['joint_density_rank'] for r in u['records']]);nominal=np.arange(1,10)/10
            expected=np.floor(nominal*(u['draws']+1))/(u['draws']+1)
            observed=(rank[:,None]<=nominal).mean(0)
            axes[1,0].plot(expected,observed,'-o',color=color,label=name)
            summary[name].update(joint_sha256=sha256(joint[name]),joint_roots=len(rank),
                finite_rank_expected=expected.tolist(),joint_coverage=observed.tolist())
    axes[0,0].set(xlabel='Forecast horizon (s)',ylabel='Joint trajectory Energy Score',title='Complete identity support at each horizon')
    axes[0,1].axhline(.1,color='black',linestyle='--',linewidth=1)
    axes[0,1].set(xlabel='Predictive marginal CDF at truth',ylabel='Root-balanced probability',title='3 s marginal PIT; uniform is calibrated')
    for ax,title in ((axes[1,0],'3 s joint trajectory density coverage'),(axes[1,1],'3 s foreground center-cell occupancy')):
        ax.plot([0,1],[0,1],'k--',linewidth=1);ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted probability',ylabel='Observed frequency',title=title)
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Exploratory train-internal development; 5 s extrapolates beyond the training horizon',fontsize=12)
    fig.savefig(out/'calibration.png',dpi=180);fig.savefig(out/'calibration.pdf');plt.close(fig)
    save_json(out/'result.json',dict(models=summary,source_sha256=sha256(__file__),
        caveat='Point estimates; clustered intervals are in benchmark and joint diagnostic JSON, not pointwise confidence bands here'))
    print(json.dumps(dict(output=str(out),models=list(bench))))


if __name__=='__main__':main()
