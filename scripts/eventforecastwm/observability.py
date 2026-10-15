"""Forecast metrics on explicitly defined legal-observation strata.

Each group retains the joint distribution marginalized to that group's agents.
Missing trajectories are excluded, never imputed. Metrics average roots, so a
crowded root is not silently replicated once per human. Groups can overlap.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import load_model,scene_interval
from eventforecastwm.distribution import energy_score,trajectory_log_probability
from eventforecastwm.train import load_roots,prepare_roots,prepared_batch,encode
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.benchmark_reuse import CooperativeGuard as Guard


def groups(window,belief):
    core=belief.core;valid=core.valid[0];age=core.age[0]
    h=window['base'].humans
    continuous=torch.zeros_like(valid);reacquired=torch.zeros_like(valid)
    for n in range(len(h)):
        seen=(h[n,:,4]>.5)&(h[n,:,6]>.5)&(h[n,:,10]>.5)
        continuous[n]=bool(seen.all())
        indices=seen.nonzero().flatten()
        if len(indices)>1:
            # At least one observed -> unobserved -> observed pattern before
            # this origin. No future observation availability enters a stratum.
            reacquired[n]=bool((~seen[indices[0]:indices[-1]+1]).any())
    return dict(all=valid,continuous_recorded_history=continuous&valid,
        reacquired_in_recorded_history=reacquired&valid,
        age_lt025=valid&(age<.25),age_025_to1=valid&(age>=.25)&(age<1.),
        age_1_to3=valid&(age>=1.)&(age<3.),age_ge3=valid&(age>=3.),
        velocity_unobserved=valid&~core.velocity_valid[0])


@torch.no_grad()
def assess(model,roots,guard=None,horizons=(1.,3.,5.)):
    rows=[]
    for root in roots:
        if guard:guard.check()
        w=root['windows'][0];batch=prepared_batch([w],'cpu');belief=encode(model,batch)
        strata=groups(w,belief)
        for horizon in horizons:
            steps=round(horizon/.25);forecast=model.imagine(belief,horizon_s=horizon)
            truth=batch['targets']['track_positions'][:,:steps];raw=batch['targets']['track_valid'][:,:steps]
            exposed=raw.all(1);mean=(forecast.positions*forecast.weights[:,:,None,None,None]).sum(1)
            distance=(mean-truth).norm(dim=-1)
            for name,group in strata.items():
                eligible=exposed&group[None];mask=eligible[:,None].expand_as(raw)
                if not eligible.any():continue
                generator=torch.Generator().manual_seed(73199+int(horizon*10))
                es,_=energy_score(forecast,truth,mask,draws=max(1,48//forecast.weights.shape[1]),generator=generator)
                logp,count=trajectory_log_probability(forecast,truth,mask)
                rows.append(dict(root_id=root['root_id'],scene=root['scene'],stratum=name,horizon=horizon,
                    agents=int(eligible.sum()),energy=float(es[0]),nll=float(-logp[0]/count[0]),
                    mean_ade=float(distance[mask].mean()),mean_fde=float(distance[:,-1][eligible].mean())))
    summary={}
    for name in sorted({r['stratum'] for r in rows}):
        summary[name]={}
        for h in horizons:
            selected=[r for r in rows if r['stratum']==name and r['horizon']==h]
            summary[name][str(h)]=dict(roots=len(selected),agents=sum(r['agents'] for r in selected),
                metrics={k:float(np.mean([r[k] for r in selected])) if selected else None
                    for k in ('energy','nll','mean_ade','mean_fde')},
                scene_intervals={k:scene_interval(selected,k) for k in ('energy','nll','mean_ade','mean_fde')})
    return dict(summary=summary,records=rows,
        definitions='Root-origin complete identity support at each horizon; legal observation strata only. Groups overlap. Continuous means every recorded history frame, not continuous-time sensor coverage. Reacquisition keeps the recorded ID.',
        limitations='No changed-ID repair or never-observed identity forecast; missing trajectories contribute no score. Observation loss does not identify its physical cause. 5s extrapolates beyond 3s training.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--prepared-cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--geometry-manifest');a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);guard=Guard(out.parent);model=load_model(a.checkpoint).eval()
    roots=load_roots(a.manifest,read(a.protocol)['development'],guard)
    if a.geometry_manifest:
        from eventforecastwm.geometry_data import attach_geometry
        attach_geometry(roots,a.geometry_manifest,guard)
    prepare(model,roots,a.manifest,a.prepared_cache,prepare_roots,guard)
    result=assess(model,roots,guard)
    save_json(out,dict(result,checkpoint_sha256=sha256(a.checkpoint),manifest_sha256=sha256(a.manifest),
        protocol_sha256=sha256(a.protocol),source_sha256=sha256(__file__)))
    print({'output':str(out),'strata':list(result['summary'])})


if __name__=='__main__':main()
