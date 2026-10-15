"""Native free-running composition and actual-arrived correction diagnostics."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import numpy as np
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import load_model
from eventforecastwm.train import load_roots,prepare_roots,prepared_batch,encode
from eventforecastwm.benchmark_reuse import CooperativeGuard as Guard
from eventforecastwm.prepared_cache import prepare


@torch.no_grad()
def assess(model,roots,guard=None):
    rows=[]
    for root in roots:
        if guard:guard.check()
        w=root['windows'][0];batch=prepared_batch([w],'cpu');belief=encode(model,batch)
        full=model.imagine(belief,horizon_s=3.)
        first=model.imagine(belief,horizon_s=1.);rest=model.imagine(first.next_belief,horizon_s=2.)
        g1=torch.Generator().manual_seed(397);g2=torch.Generator().manual_seed(397)
        all_paths,_,_=full.sample(2,g1);left,_,noise=first.sample(2,g2);right,_,_=rest.sample(2,g2,noise)
        query_poison=model.imagine(belief,torch.full((1,33,16,10),float('nan')),horizon_s=3.)
        recurrent=belief;one_step_positions=[]
        for _ in range(12):
            one=model.imagine(recurrent,horizon_s=.25)
            one_step_positions.append(one.positions);recurrent=one.next_belief
        row=dict(root_id=root['root_id'],scene=root['scene'],
            composition_mean_max_m=float((full.positions[:,:,4:]-rest.positions).abs().max()),
            composition_covariance_max=float((full.next_belief.covariance-rest.next_belief.covariance).abs().max()),
            composition_sample_max_m=float((all_paths-torch.cat([left,right],2)).abs().max()),
            repeated_quarter_second_mean_max_m=float((full.positions-torch.cat(one_step_positions,2)).abs().max()),
            repeated_quarter_second_covariance_max=float((full.next_belief.covariance-recurrent.covariance).abs().max()),
            query_invariance_max_m=float((query_poison.positions-full.positions).abs().max()))
        gt=batch['targets']['track_positions'][:,:12];raw=batch['targets']['track_valid'][:,:12]
        mask=raw.all(1)[:,None].expand_as(raw)
        if mask.any():
            mean=(full.positions*full.weights[:,:,None,None,None]).sum(1)
            err=(mean-gt).norm(dim=-1)
            row['free_running_ade']=float(err[mask].mean())
            current=belief.core;errors=[]
            for j in range(12):
                predicted=model.core._step(current)
                point=(predicted.positions*predicted.scenario_weights[:,:,None,None]).sum(1)
                if mask[:,j].any():errors.extend((point-gt[:,j]).norm(dim=-1)[mask[:,j]].tolist())
                vv=raw[:,j]&raw[:,j-1] if j else torch.zeros_like(raw[:,j])
                true_velocity=(gt[:,j]-gt[:,j-1])/.25 if j else torch.zeros_like(gt[:,j])
                current=replace(predicted,
                    positions=torch.where(raw[:,None,j,:,None],gt[:,None,j],predicted.positions),
                    velocities=torch.where(vv[:,None,:,None],true_velocity[:,None],predicted.velocities))
            row['teacher_forced_one_step_ade']=float(np.mean(errors))
            row['teacher_forcing_scope']='diagnostic privileged past target resets; never used by free-running imagine'
        row['correction_supported']=False
        if len(root['windows'])>1:
            next_w=root['windows'][1];old_ids=w['metadata']['track_ids'];new_ids=next_w['metadata']['track_ids']
            valid_identity=(old_ids and set(old_ids)==set(new_ids) and
                w['metadata']['association']['associated']==len(old_ids) and
                next_w['metadata']['association']['associated']==len(new_ids))
            delta=next_w['metadata']['timestamp']-w['metadata']['timestamp']
            tick=math.ceil((delta-1e-8)/.25)*.25;j=int(round(tick/.25))
            if valid_identity and tick<3. and mask.any():
                n=belief.core.valid.shape[1]
                positions=torch.zeros(1,n,2);observed=torch.zeros(1,n,dtype=torch.bool)
                acquisition=torch.zeros(1,n,dtype=torch.float64)
                y0,y1=w['metadata']['yaw'],next_w['metadata']['yaw']
                R0=np.array([[np.cos(y0),-np.sin(y0)],[np.sin(y0),np.cos(y0)]])
                R1=np.array([[np.cos(y1),-np.sin(y1)],[np.sin(y1),np.cos(y1)]])
                o0=np.array(w['metadata']['origin'])[[0,2]];o1=np.array(next_w['metadata']['origin'])[[0,2]]
                for i,identity in enumerate(old_ids):
                    k=new_ids.index(identity);h=next_w['base'].humans[k,-1]
                    observed[0,i]=bool(h[4]>.5 and h[6]>.5)
                    positions[0,i]=torch.tensor((h[:2].numpy()@R1+o1-o0)@R0.T)
                    acquisition[0,i]=next_w['track_acquisition_times'][k,-1]-w['metadata']['timestamp']
                if observed.any():
                    before=model.imagine(belief,horizon_s=tick).next_belief
                    after=model.observe(before,dict(source='actual_arrived',positions=positions,observed=observed,
                        acquisition_times=acquisition,timestamp_s=tick))
                    corrected=model.imagine(after,horizon_s=3.-tick)
                    uncorrected=model.imagine(before,horizon_s=3.-tick)
                    cm=(corrected.positions*corrected.weights[:,:,None,None,None]).sum(1)
                    um=(uncorrected.positions*uncorrected.weights[:,:,None,None,None]).sum(1)
                    row.update(correction_supported=True,observation_time=delta,update_time=tick,
                        actual_acquisition_lag_max_s=float((tick-acquisition)[observed].max()),
                        uncorrected_remaining_ade=float((um-gt[:,j:]).norm(dim=-1)[mask[:,j:]].mean()),
                        corrected_remaining_ade=float((cm-gt[:,j:]).norm(dim=-1)[mask[:,j:]].mean()))
        rows.append(row)
    keys=('composition_mean_max_m','composition_covariance_max','composition_sample_max_m',
          'repeated_quarter_second_mean_max_m','repeated_quarter_second_covariance_max','query_invariance_max_m')
    return dict(roots=len(rows),maxima={k:max(r[k] for r in rows) for k in keys},
        correction_roots=sum(r['correction_supported'] for r in rows),
        correction={k:float(np.mean([r[k] for r in rows if k in r])) if any(k in r for r in rows) else None
                    for k in ('uncorrected_remaining_ade','corrected_remaining_ade')},records=rows,
        limitations='Correction uses stable already-associated legal IDs; births/reassociation unsupported. Delayed update is local linear approximation.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);p.add_argument('--roots',type=int,default=32)
    p.add_argument('--prepared-cache');p.add_argument('--geometry-manifest')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);guard=Guard(out.parent)
    model=load_model(a.checkpoint).eval();roots=load_roots(a.manifest,read(a.protocol)['development'][:a.roots],guard)
    if a.geometry_manifest:
        from eventforecastwm.geometry_data import attach_geometry
        attach_geometry(roots,a.geometry_manifest,guard)
    elif hasattr(model.core,'map_patch_encoder'):
        raise ValueError('Map-conditioned mechanics require the matching legal geometry sidecar')
    if a.prepared_cache:prepare(model,roots,a.manifest,a.prepared_cache,prepare_roots,guard)
    else:prepare_roots(model,roots,guard)
    result=assess(model,roots,guard);guard.check()
    save_json(out,dict(checkpoint_sha256=sha256(a.checkpoint),manifest_sha256=sha256(a.manifest),
        geometry_manifest_sha256=sha256(a.geometry_manifest) if a.geometry_manifest else None,
        source_sha256=sha256(__file__),**result))
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
