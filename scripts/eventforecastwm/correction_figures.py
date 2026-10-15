"""Actual-arrival correction figures, with stable-ID reacquisition distinguished.

Selection uses development error extrema and is declared in the artifact. A
future target only scores/draws a forecast; it never enters observe or imagine.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import load_model,scene_interval
from eventforecastwm.train import load_roots,prepare_roots,prepared_batch,encode
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.benchmark_reuse import CooperativeGuard as Guard
from eventforecastwm.sampling import joint_samples


@torch.no_grad()
def cases(model,roots,guard):
    records=[];details={}
    for root in roots:
        guard.check()
        if len(root['windows'])<2:continue
        w,nxt=root['windows'][:2];ids=w['metadata']['track_ids'];new=nxt['metadata']['track_ids']
        if not ids or set(ids)!=set(new):continue
        if w['metadata']['association']['associated']!=len(ids) or nxt['metadata']['association']['associated']!=len(new):continue
        delta=nxt['metadata']['timestamp']-w['metadata']['timestamp'];tick=math.ceil((delta-1e-8)/.25)*.25;j=round(tick/.25)
        if tick<=0 or tick>=3.:continue
        batch=prepared_batch([w],'cpu');mask=batch['targets']['track_valid'][0,:12].all(0)
        if not mask.any():continue
        belief=encode(model,batch);n=belief.core.valid.shape[1]
        pos=torch.full((1,n,2),float('nan'));acq=torch.full((1,n),float('nan'),dtype=torch.float64);observed=torch.zeros(1,n,dtype=torch.bool)
        rotation=lambda yaw:np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        R0=rotation(w['metadata']['yaw']);R1=rotation(nxt['metadata']['yaw'])
        shift=np.asarray(nxt['metadata']['origin'])[[0,2]]-np.asarray(w['metadata']['origin'])[[0,2]]
        for i,identity in enumerate(ids):
            k=new.index(identity);h=nxt['base'].humans[k,-1]
            observed[0,i]=bool(h[4]>.5 and h[6]>.5)
            if observed[0,i]:
                pos[0,i]=torch.tensor((h[:2].numpy()@R1+shift)@R0.T)
                acq[0,i]=nxt['track_acquisition_times'][k,-1]-w['metadata']['timestamp']
        if not observed.any():continue
        prior=model.imagine(belief,horizon_s=tick).next_belief
        posterior=model.observe(prior,dict(positions=pos,observed=observed,acquisition_times=acq,timestamp_s=tick))
        before=model.imagine(prior,horizon_s=3.-tick);after=model.imagine(posterior,horizon_s=3.-tick)
        truth=batch['targets']['track_positions'][0,j:12];means=[];errors=[]
        for f in (before,after):
            mean=(f.positions*f.weights[:,:,None,None,None]).sum(1)[0];means.append(mean)
            errors.append(float((mean-truth).norm(dim=-1)[:,mask].mean()))
        reacquired=observed[0]&(w['base'].humans[:,-1,4]<.5)&mask
        row=dict(root_id=root['root_id'],scene=root['scene'],before_ade=errors[0],after_ade=errors[1],
            difference=errors[1]-errors[0],stable_id_reacquired_agents=int(reacquired.sum()),
            supported_agents=int(mask.sum()),actual_observed_agents=int(observed.sum()),update_time=tick)
        records.append(row);details[root['root_id']]=(w,truth,mask,means,before,after,pos,observed)
    return records,details


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--prepared-cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--geometry-manifest');a=p.parse_args();out=Path(a.output)
    if (out/'result.json').exists():raise FileExistsError('Completed correction figures are immutable')
    out.mkdir(parents=True,exist_ok=True);guard=Guard(out);model=load_model(a.checkpoint).eval()
    roots=load_roots(a.manifest,read(a.protocol)['development'],guard)
    if a.geometry_manifest:
        from eventforecastwm.geometry_data import attach_geometry
        attach_geometry(roots,a.geometry_manifest,guard)
    prepare(model,roots,a.manifest,a.prepared_cache,prepare_roots,guard);rows,details=cases(model,roots,guard)
    selected=[];used=set()
    for label,only_reacquired,worst in [('Reacquired: greatest gain',True,False),('Reacquired: smallest gain',True,True),
            ('Actual correction: greatest gain',False,False),('Actual correction: smallest gain',False,True)]:
        eligible=[r for r in rows if r['root_id'] not in used and (not only_reacquired or r['stable_id_reacquired_agents'])]
        if eligible:
            row=sorted(eligible,key=lambda r:(r['difference'],r['root_id']),reverse=worst)[0]
            used.add(row['root_id']);selected.append((label,row))
    fig,axes=plt.subplots(2,2,figsize=(10,9),constrained_layout=True)
    for ax,(label,row) in zip(axes.flat,selected):
        w,truth,mask,means,before,after,pos,observed=details[row['root_id']]
        samples,_=joint_samples(after,6,torch.Generator().manual_seed(9));points=[]
        for n in torch.where(mask)[0].tolist():
            h=w['base'].humans[n];arrived=(h[:,4]>.5)&(h[:,6]>.5)&(h[:,10]>.5);hist=h[arrived,:2].numpy()
            ax.plot(hist[:,0],hist[:,1],'k.-',lw=1,ms=3)
            gt=truth[:,n].numpy();ax.plot(gt[:,0],gt[:,1],color='#c53b35',lw=2)
            for mean,color in zip(means,['#d6891e','#1765a3']):
                v=mean[:,n].numpy();ax.plot(v[:,0],v[:,1],color=color,lw=1.7,ls='--');points.append(v)
            for sample in samples[0,:,:,n].numpy():ax.plot(sample[:,0],sample[:,1],color='#1765a3',alpha=.2,lw=.8)
            if observed[0,n]:ax.plot(*pos[0,n].numpy(),marker='*',color='#158b51',ms=12)
            points.extend([hist,gt])
        xy=np.concatenate([x for x in points if len(x)]);lo=xy.min(0)-1;hi=xy.max(0)+1
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',xlabel='Local x (m)',ylabel='Local z (m)',
            title=f'{label}\nremaining ADE {row["before_ade"]:.2f} → {row["after_ade"]:.2f} m')
    for ax in list(axes.flat)[len(selected):]:ax.set_visible(False)
    fig.suptitle('Black: legal history · Red: supported future · Orange: prior · Blue: corrected futures\nGreen star: actual newly arrived detection; no target used in correction',fontsize=10)
    fig.savefig(out/'correction.png',dpi=180);fig.savefig(out/'correction.pdf');plt.close(fig)
    save_json(out/'result.json',dict(checkpoint_sha256=sha256(a.checkpoint),manifest_sha256=sha256(a.manifest),
        roots=len(rows),stable_id_reacquisition_roots=sum(r['stable_id_reacquired_agents']>0 for r in rows),
        intervals={k:scene_interval(rows,k) for k in ('before_ade','after_ade','difference')},records=rows,
        selected=[dict(label=label,**row) for label,row in selected],source_sha256=sha256(__file__),
        selection='Explicit error-difference extrema on development roots; illustrative rather than representative',
        limitations='Stable already-associated identities only; changed IDs and births unsupported; delayed update approximate over at most .25s'))


if __name__=='__main__':main()
