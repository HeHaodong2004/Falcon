"""Export legal-history/physical-future figures with explicit case selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import load_model,raster
from eventforecastwm.train import load_roots,prepare_roots,prepared_batch,encode,evaluate
from eventforecastwm.sampling import joint_samples
from eventforecastwm.benchmark_reuse import CooperativeGuard as Guard


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);p.add_argument('--geometry-manifest')
    a=p.parse_args();out=Path(a.output)
    if (out/'result.json').exists():raise FileExistsError('Completed figures are immutable')
    out.mkdir(parents=True,exist_ok=True);guard=Guard(out)
    model=load_model(a.checkpoint).eval();roots=load_roots(a.manifest,read(a.protocol)['development'],guard)
    if a.geometry_manifest:
        from eventforecastwm.geometry_data import attach_geometry
        attach_geometry(roots,a.geometry_manifest,guard)
    prepare_roots(model,roots,guard)
    scores={r['root_id']:r for r in evaluate(model,roots,'cpu',horizons=(3.,),draws=6)['3.0']['records']}
    categories={}
    for r in roots:
        w=r['windows'][0];h=w['base'].humans;visible=int((h[:,-1,4]>.5).sum()) if len(h) else 0
        categories[r['root_id']]=dict(lost=len(h)>visible,never=len(h)==0 and w['metadata']['physical_humans_target_only']>0,
            dense=w['metadata']['physical_humans_target_only']>=3)
    selected=[];used=set()
    for label,criterion,worst in [('Visible: lower error','visible',False),('Visible: higher error','visible',True),
            ('Lost track: lower error','lost',False),('Lost track: higher error','lost',True),
            ('Dense scene','dense',False),('Never observed','never',False)]:
        candidates=[r for r in roots if r['root_id'] not in used and
            (not categories[r['root_id']]['lost'] if criterion=='visible' else categories[r['root_id']][criterion]) and
            (scores[r['root_id']].get('physical_supported') if criterion=='never' else 'mean_ade' in scores[r['root_id']])]
        if not candidates:continue
        candidates.sort(key=lambda r:(scores[r['root_id']].get('mean_ade',0.),r['root_id']),reverse=worst)
        r=candidates[0];selected.append((label,r));used.add(r['root_id'])
    fig,axes=plt.subplots(2,3,figsize=(13,9),constrained_layout=True);records=[]
    for ax,(label,r) in zip(axes.flat,selected):
        guard.check();w=r['windows'][0];batch=prepared_batch([w],'cpu');f=model.imagine(encode(model,batch),horizon_s=3.)
        paths,_=joint_samples(f,6,torch.Generator().manual_seed(7791));mean=(f.positions*f.weights[:,:,None,None,None]).sum(1)[0]
        heat=raster(f)[0,-1].numpy()
        if w['geometry_support'].all():ax.imshow(w['geometry'][1].numpy(),origin='lower',extent=(-8,8,-8,8),cmap='Greys',alpha=.15,vmin=0,vmax=1)
        ax.imshow(heat,origin='lower',extent=(-8,8,-8,8),cmap='Blues',alpha=.45,vmin=0,vmax=max(.02,float(heat.max())))
        h=w['base'].humans.numpy();truth=w['targets']['track_positions'][:12].numpy();valid=w['targets']['track_valid'][:12].numpy()
        # Bounds contain only things actually drawn. Invalid history slots and
        # unplotted anonymous targets must not silently stretch the figure.
        shown=[np.zeros((1,2))]
        for n in range(len(h)):
            arrived=(h[n,:,4]>.5)&(h[n,:,6]>.5)&(h[n,:,10]>.5)
            ax.plot(h[n,arrived,0],h[n,arrived,1],'o-',c='black',ms=2,lw=1)
            shown.append(h[n,arrived,:2])
            for path in paths[0,:,:,n].numpy():ax.plot(path[:,0],path[:,1],c='#1672b8',lw=.8,alpha=.35)
            shown.extend([paths[0,:,:,n].numpy().reshape(-1,2),mean[:,n].numpy()])
            ax.plot(mean[:,n,0],mean[:,n,1],c='#075288',lw=1.5,ls='--')
            if valid[:,n].all():
                ax.plot(truth[:,n,0],truth[:,n,1],c='#d24a32',lw=2,marker='o',markevery=[-1],ms=4,zorder=8)
                shown.append(truth[:,n])
        if not len(h):
            xy=w['targets']['all_human_positions'][:12].numpy();support=w['targets']['all_human_valid'][:12].numpy()
            for n in range(xy.shape[1]):ax.plot(xy[support[:,n],n,0],xy[support[:,n],n,1],c='#d24a32',lw=2,marker='o',markevery=[-1],ms=4)
            shown.append(xy[support])
            ax.text(.02,.02,'No identity forecast: unsupported',transform=ax.transAxes,fontsize=8)
        points=np.concatenate(shown);lower=np.minimum(points.min(0)-1.,[-4.,-4.]);upper=np.maximum(points.max(0)+1.,[4.,4.])
        ax.plot(0,0,'^',c='#555555',ms=7);ax.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),xlabel='Local x (m)',ylabel='Local z (m)',aspect='equal')
        score=scores[r['root_id']].get('mean_ade');ax.set_title(label+(f' | ADE {score:.2f} m' if score is not None else ''),fontsize=10)
        records.append(dict(case=label,root_id=r['root_id'],scene=r['scene'],score=scores[r['root_id']],
            selection='explicit minimum/maximum development mean ADE within displayed stratum; illustrative, not representative average'))
    for ax in list(axes.flat)[len(selected):]:ax.set_visible(False)
    fig.suptitle('Black: arrived history · Red: supported future truth · Blue: six sampled futures / dashed mean\nHeat: predicted tracked-human center-cell probability at 3 s',fontsize=11)
    fig.savefig(out/'forecasts.png',dpi=180);fig.savefig(out/'forecasts.pdf');plt.close(fig)
    save_json(out/'result.json',dict(checkpoint_sha256=sha256(a.checkpoint),manifest_sha256=sha256(a.manifest),
        cases=records,files=['forecasts.png','forecasts.pdf'],limitations='Identity-associated forecasts only; no exact never-observed trajectories invented'))
    print(json.dumps(dict(output=str(out),cases=len(records))))


if __name__=='__main__':main()
