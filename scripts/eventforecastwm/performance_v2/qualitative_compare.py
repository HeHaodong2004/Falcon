"""Matched-scale comparison figures, including typical and difficult cases."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json,occupancy_targets
from eventforecastwm.benchmark import load_model
from eventforecastwm.benchmark_reuse import CooperativeGuard
from eventforecastwm.geometry_data import attach_geometry
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.readouts import tracked_occupancy
from eventforecastwm.sampling import joint_samples
from eventforecastwm.train import load_roots,prepare_roots,prepared_batch,encode


def select_cases(diagnostic,census):
    groups=defaultdict(list)
    for row in read(diagnostic)['agent_records']:
        if row['horizon']==3.:groups[row['root_id'],row['window']].append(row)
    records=[dict(root_id=rid,window=j,error=float(np.mean([r['mean_ade'] for r in rows])),
                  max_age=max(r['age'] for r in rows),visible=all(r['visible'] for r in rows),
                  turning=any(r['turning'] for r in rows),agents=len(rows)) for (rid,j),rows in groups.items()]
    choices=[('Visible',lambda r:r['visible'],.5),('Temporarily lost',lambda r:not r['visible'] and r['max_age']<2.,.5),
        ('Long observation gap',lambda r:r['max_age']>=3.,.5),('Dense tracked scene',lambda r:r['agents']>=3,.5),
        ('Turning',lambda r:r['turning'],.5),('Lower error',lambda r:True,.1),('Higher error',lambda r:True,.9)]
    result=[];used=set()
    for label,criterion,quantile in choices:
        by_root=defaultdict(list)
        for row in records:
            if criterion(row) and row['root_id'] not in used:by_root[row['root_id']].append(row)
        representatives=[sorted(rows,key=lambda r:(r['error'],r['window']))[len(rows)//2] for rows in by_root.values()]
        representatives.sort(key=lambda r:(r['error'],r['root_id']))
        if not representatives:continue
        row=representatives[round((len(representatives)-1)*quantile)]
        result.append(dict(row,label=label,selection_quantile=quantile));used.add(row['root_id'])
    anonymous=[r for r in read(census)['records'] if r['split']=='dev' and r['legal_ever']==0 and
               r['physical_count_target_only']>0 and r['horizons']['3']['physical'] and r['root_id'] not in used]
    if anonymous:
        row=sorted(anonymous,key=lambda r:(r['root_id'],r['window']))[0]
        result.append(dict(root_id=row['root_id'],window=row['window'],label='Never observed',selection='first unused root/window identity'))
    return result


@torch.no_grad()
def run(cfg):
    out=Path(cfg['run']);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Completed figures are immutable')
    guard=CooperativeGuard(out);torch.set_num_threads(1)
    if cfg.get('population','development')!='development':
        from .evaluation_population import population
        memberships=[set(population(dict(cfg,checkpoint=path))[0]) for path in cfg['models'].values()]
        if not memberships or any(ids!=memberships[0] for ids in memberships):
            raise ValueError('Qualitative model populations differ')
        allowed=memberships[0]
    else:allowed=set(read(cfg['protocol'])['development'])
    cases=cfg.get('cases') or select_cases(cfg['baseline_diagnostic'],cfg['census'])
    if any(case['root_id'] not in allowed for case in cases):raise ValueError('Plot population mismatch')
    roots=load_roots(cfg['manifest'],sorted({c['root_id'] for c in cases}),guard)
    attach_geometry(roots,cfg['geometry_manifest'],guard);roots={r['root_id']:r for r in roots}
    models={name:load_model(path).eval() for name,path in cfg['models'].items()}
    states={name:torch.load(path,map_location='cpu') for name,path in cfg['models'].items()}
    receipts=[]
    for index,case in enumerate(cases):
        guard.check();root=roots[case['root_id']];window=root['windows'][case['window']]
        forecasts={};shown=[np.zeros((1,2))]
        for name,model in models.items():
            prepare(model,[root],cfg['manifest'],cfg['prepared_cache'],prepare_roots,guard)
            batch=prepared_batch([window],'cpu')
            if model.recipe.get('variant')!='geometry_occupancy_v2' and not cfg.get('physical_uses_geometry',{}).get(name,False):
                batch=dict(batch,geometry=torch.zeros_like(batch['geometry']),geometry_support=torch.zeros_like(batch['geometry_support']))
            belief=encode(model,batch);forecast=model.imagine(belief,horizon_s=3.)
            paths,_=joint_samples(forecast,6,torch.Generator().manual_seed(7791))
            mean=(forecast.positions*forecast.weights[:,:,None,None,None]).sum(1)[0]
            tracked=tracked_occupancy(forecast)[0]
            anonymous=getattr(forecast,'anonymous_occupancy',states[name]['anonymous_occupancy_prior'][:12][None])[0]
            forecasts[name]=dict(paths=paths[0].numpy(),mean=mean.numpy(),valid=forecast.valid[0].numpy(),
                total=(1-(1-tracked)*(1-anonymous)).numpy(),anonymous=anonymous.numpy())
            for n,valid in enumerate(forecast.valid[0]):
                if valid:shown.extend([paths[0,:,:,n].numpy().reshape(-1,2),mean[:,n].numpy()])
        target,anonymous_target=occupancy_targets(batch,12)
        humans=window['base'].humans.numpy();truth=window['targets']['track_positions'][:12].numpy()
        valid=window['targets']['track_valid'][:12].numpy()
        for n in range(len(humans)):
            arrived=(humans[n,:,4]>.5)&(humans[n,:,6]>.5)&(humans[n,:,10]>.5)
            shown.append(humans[n,arrived,:2])
            if valid[:,n].all():shown.append(truth[:,n])
        points=np.concatenate(shown)
        if not np.isfinite(points).all():raise ValueError('Nonfinite trajectory in qualitative comparison')
        lower=np.minimum(points.min(0)-1.,[-4.,-4.]);upper=np.maximum(points.max(0)+1.,[4.,4.])
        vmax_total=max(.02,max(float(v['total'][-1].max()) for v in forecasts.values()))
        vmax_anonymous=max(.002,max(float(v['anonymous'][-1].max()) for v in forecasts.values()))
        fig,axes=plt.subplots(3,len(models),figsize=(6*len(models),13),squeeze=False,constrained_layout=True)
        for column,(name,value) in enumerate(forecasts.items()):
            ax=axes[0,column]
            for n in range(len(humans)):
                arrived=(humans[n,:,4]>.5)&(humans[n,:,6]>.5)&(humans[n,:,10]>.5)
                ax.scatter(humans[n,arrived,0],humans[n,arrived,1],s=9,c='black',zorder=5)
                if value['valid'][n]:
                    for path in value['paths'][:,:,n]:ax.plot(path[:,0],path[:,1],color='#2978ab',alpha=.35,lw=.8)
                    ax.plot(value['mean'][:,n,0],value['mean'][:,n,1],'--',color='#125080',lw=1.5)
                if valid[:,n].all():ax.plot(truth[:,n,0],truth[:,n,1],color='#cb463b',lw=2,marker='o',markevery=[-1],ms=4)
            ax.scatter([0],[0],marker='^',c='gray');ax.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),aspect='equal',title=name)
            for row,key,vmax,truth_grid in ((1,'total',vmax_total,target[0,-1]),(2,'anonymous',vmax_anonymous,anonymous_target[0,-1])):
                ax=axes[row,column];heat=ax.imshow(value[key][-1],origin='lower',extent=(-8,8,-8,8),vmin=0,vmax=vmax,cmap='viridis')
                y,x=torch.where(truth_grid>.5);ax.scatter((x.numpy()+.5)/2-8,(y.numpy()+.5)/2-8,marker='s',s=35,facecolors='none',edgecolors='#ff554a')
                ax.set(title=key.capitalize()+' center occupancy at 3s',xlim=(-8,8),ylim=(-8,8),aspect='equal')
                fig.colorbar(heat,ax=ax,shrink=.8,label='Cell probability')
            for ax in axes[:,column]:ax.set(xlabel='Local x (m)',ylabel='Local z (m)')
        fig.suptitle(case['label']+' | '+root['root_id']+' | window '+str(case['window'])+
            '\nBlack: arrived locations; red: supported future truth; blue: six coherent samples / dashed mean'+
            '\nMatching axes and probability scales across models. Anonymous truth is shown only as occupied cells.',fontsize=11)
        filename=f'case_{index:02d}';files=[]
        for extension in ('png','pdf'):
            path=out/(filename+'.'+extension);fig.savefig(path,dpi=170);files.append(dict(path=path.name,sha256=sha256(path)))
        plt.close(fig);receipts.append(dict(case,scene=root['scene'],files=files,trajectory_limits=[lower.tolist(),upper.tolist()],
            probability_maxima=dict(total=vmax_total,anonymous=vmax_anonymous),draws=6,seed=7791))
    save_json(out/'result.json',dict(status='qualitative_comparison_complete',config=cfg,cases=receipts,
        model_sha256={name:sha256(path) for name,path in cfg['models'].items()},source_sha256=sha256(__file__),
        selection=(cfg.get('case_selection','Explicit cases supplied in config; see each recorded case rationale') if cfg.get('cases') else
                   'Baseline-only root-representative median cases plus 10th/90th error quantiles; never selected on candidate improvement'),
        limitation='Illustrative cases, not an aggregate performance estimate; no anonymous identity trajectories predicted'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);run(read(parser.parse_args().config))


if __name__=='__main__':main()
