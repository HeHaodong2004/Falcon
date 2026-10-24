"""Supplemental measured scaling/conditioning/calibration curves on fixed dev cohort."""
import argparse,collections,json,pathlib
import numpy as np
from .evaluate import scene_ci
from .reporting import metric

def run(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':240})
    root=pathlib.Path(directory);figdir=root/'figures'
    def save(f,n):f.tight_layout();f.savefig(figdir/(n+'.png'),bbox_inches='tight');f.savefig(figdir/(n+'.pdf'),bbox_inches='tight');plt.close(f)
    stress=json.loads((root/'stress_release.json').read_text());groups=collections.defaultdict(list)
    for r in stress['records']:groups[(r['humans'],r['graph_cells'],r['candidates'],r['worlds'],r['mode'])].append(r['complete_ms'])
    f,ax=plt.subplots(1,3,figsize=(11,3.6))
    for a,dim,xs,label in zip(ax,[0,2,3],[(0,3,6,12,24),(1,4,24),(8,16,32)],['Human slots','Distinct candidate queries','Shared worlds']):
        for mode,c in [('warm','#0072B2'),('version_invalidation','#D55E00')]:
            med=[];tail=[]
            for x in xs:
                key=[6,3025,24,16,mode];key[dim]=x;v=groups[tuple(key)];med.append(np.median(v));tail.append(np.percentile(v,95))
            a.plot(xs,med,'o-',color=c,label=mode+' P50');a.plot(xs,tail,'s--',color=c,label=mode+' P95')
        a.axhline(65,ls=':',color='black');a.set(xlabel=label,ylabel='Full-call ms')
    ax[0].legend(fontsize=7);f.suptitle('Synthetic stress: 3 s; other axes fixed at 6 people / 24 queries / 16 worlds; 15 warm repeats',fontsize=10);save(f,'06b_latency_budget_sweeps')
    results=[]
    for n in (218,436,872):
        d=json.loads((root/('learning_dev'+str(n)+'.json')).read_text());rs=[r for r in d['forecasts'] if r['model']=='structured_learned']
        # Average available horizons per root, then equal scenes. Same supported
        # cohort/horizons for all three model checkpoints.
        es=metric(rs,'energy_score');pairs=d['conditioning_pairs'];gain=scene_ci(pairs,'delta');qs=[r for r in d['queries'] if r['risk_evaluable']]
        briers={}
        for k in ('collision','psc'):
            rr=[{**r,'brier':(r['pred_'+k]-r[k])**2,'bias':r['pred_'+k]-r[k]} for r in qs]
            briers[k]={'brier':metric(rr,'brier'),'bias':metric(rr,'bias')}
        results.append({'training_roots':n,'forecast':es,'conditioning':gain,'risk':briers,'eval_roots':d['roots']})
    (root/'learning_cohort_metrics.json').write_text(json.dumps({'scope':'post-freeze descriptive development evaluation; first200devroots; same final process kernel/event head; no further selection;20/20/30training epochs','records':results},indent=2))
    f,ax=plt.subplots(1,4,figsize=(14,3.6));xs=[r['training_roots'] for r in results]
    for a,key,label in [(ax[0],'forecast','Available-horizon ES (m)'),(ax[1],'conditioning','Posterior−prior ES (m)')]:
        vals=[r[key] for r in results];a.errorbar(xs,[v['mean'] for v in vals],yerr=[[v['mean']-v['lo'] for v in vals],[v['hi']-v['mean'] for v in vals]],fmt='o-',capsize=3);a.set(xlabel='Independent training roots',ylabel=label)
    ax[1].axhline(0,color='gray',ls='--');ax[1].axhline(.02,color='#D55E00',ls=':')
    for k in ('collision','psc'):ax[2].plot(xs,[r['risk'][k]['brier']['mean'] for r in results],'o-',label=k)
    ax[2].set(xlabel='Independent training roots',ylabel='Native Brier score');ax[2].legend();
    for k in ('collision','psc'):ax[3].plot(xs,[r['risk'][k]['bias']['mean'] for r in results],'o-',label=k)
    ax[3].axhline(0,color='gray',ls='--');ax[3].set(xlabel='Independent training roots',ylabel='Predicted − observed probability');ax[3].legend();f.suptitle('Fixed 200-root development cohort; conditioning support is small; no holdout selection',fontsize=10);save(f,'07b_learning_conditioning_calibration')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');run(**vars(p.parse_args()))
