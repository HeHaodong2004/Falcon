"""Stable-legal-ID observation correction on actual development arrivals.

Targets are read only after observations are constructed and forecasts made.
No changed-ID association, simulator identity, or hidden position is used.
The existing one-tick delayed observation approximation is retained explicitly.
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from eventbeliefwm.data import read, sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import load_model, scene_interval
from eventforecastwm.benchmark_reuse import CooperativeGuard
from eventforecastwm.distribution import energy_score, trajectory_log_probability
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.train import load_roots, prepare_roots, prepared_batch, encode
from eventforecastwm.window_evaluate import aggregate


def rotation(yaw,like):
    return like.new_tensor([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])


def arrived_observation(old,new,core):
    """Uses legal IDs, acquired positions/times and legal ego coordinate frames."""
    prepared=new['_prepared'];lookup={identity:i for i,identity in enumerate(new['metadata']['track_ids'])}
    positions=torch.zeros_like(core.last_observed_positions);acquired=torch.zeros_like(core.last_observed_times)
    observed=torch.zeros_like(core.valid);origin0=positions.new_tensor(old['metadata']['origin'])[[0,2]]
    origin1=positions.new_tensor(new['metadata']['origin'])[[0,2]]
    r0=rotation(old['metadata']['yaw'],positions);r1=rotation(new['metadata']['yaw'],positions)
    delta=new['metadata']['timestamp']-old['metadata']['timestamp']
    for j,identity in enumerate(old['metadata']['track_ids']):
        k=lookup.get(identity)
        if k is None or not prepared.valid[0,k] or not core.valid[0,j]:continue
        acquired[0,j]=delta+prepared.last_observed_times[0,k]
        lag=float(core.clock[0]-acquired[0,j])
        if not -1e-5<=lag<=.25001 or acquired[0,j]<=core.last_observed_times[0,j]+1e-7:continue
        positions[0,j]=(prepared.positions[0,k]@r1+origin1-origin0)@r0.T
        observed[0,j]=True
    return dict(source='actual_arrived',positions=positions,acquisition_times=acquired,
                observed=observed,track_ids=core.track_ids.clone(),timestamp_s=core.clock.clone())


@torch.no_grad()
def run(cfg):
    out=Path(cfg['run']);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Completed recurrent audit is immutable')
    guard=CooperativeGuard(out);torch.set_num_threads(1);start=time.time()
    model=load_model(cfg['checkpoint']).eval();protocol=read(cfg['protocol'])
    state=torch.load(cfg['checkpoint'],map_location='cpu')
    if set(protocol['development'])&set(state.get('fit_root_ids',[])):raise ValueError('Evaluation leakage')
    roots=load_roots(cfg['manifest'],protocol['development'],guard)
    if cfg.get('geometry_manifest'):
        from eventforecastwm.geometry_data import attach_geometry
        attach_geometry(roots,cfg['geometry_manifest'],guard)
    prepare(model,roots,cfg['manifest'],cfg['prepared_cache'],prepare_roots,guard)
    rows=[];composition=0.;paths_difference=0.;considered=0;anonymous_composition=None
    for root in roots:
        guard.check()
        for j,(old,new) in enumerate(zip(root['windows'],root['windows'][1:])):
            delta=new['metadata']['timestamp']-old['metadata']['timestamp']
            ticks=math.ceil((delta-1e-7)/.25)
            if not 1<=ticks<=16:continue
            batch=prepared_batch([old],'cpu')
            if model.recipe.get('variant')!='geometry_occupancy_v2' and not cfg.get('physical_uses_geometry',False):
                batch=dict(batch,geometry=torch.zeros_like(batch['geometry']),geometry_support=torch.zeros_like(batch['geometry_support']))
            belief=encode(model,batch)
            elapsed=ticks*.25;left=model.imagine(belief,horizon_s=elapsed)
            observation=arrived_observation(old,new,left.next_belief.core);considered+=1
            if not observation['observed'].any():continue
            before=model.imagine(left.next_belief,horizon_s=1.)
            whole=model.imagine(belief,horizon_s=elapsed+1.)
            composition=max(composition,float((whole.positions[:,:,ticks:]-before.positions).abs().max()))
            if hasattr(whole,'anonymous_occupancy'):
                error=float((whole.anonymous_occupancy[:,ticks:]-before.anonymous_occupancy).abs().max())
                anonymous_composition=max(anonymous_composition or 0.,error)
                if not torch.equal(whole.anonymous_occupancy_supported[:,ticks:],before.anonymous_occupancy_supported):
                    raise AssertionError('Anonymous occupancy composition support changed')
            g1=torch.Generator().manual_seed(5133);g2=torch.Generator().manual_seed(5133)
            expected=whole.sample(2,g1)[0];prefix,_,samples=left.sample(2,g2)
            suffix=before.sample(2,g2,samples)[0]
            paths_difference=max(paths_difference,float((expected-torch.cat([prefix,suffix],2)).abs().max()))
            corrected=model.observe(left.next_belief,observation)
            after=model.imagine(corrected,horizon_s=1.)
            # Supervision starts here; it never constructs the observation.
            truth=batch['targets']['track_positions'][:,ticks:ticks+4]
            raw=batch['targets']['track_valid'][:,ticks:ticks+4]
            complete=raw.all(1);mask=complete[:,None].expand_as(raw)
            if not complete.any():continue
            base=dict(root_id=root['root_id'],scene=root['scene'],window=j,delay_to_tick_s=elapsed-delta,
                      updated_tracks=int(observation['observed'].sum()),supported_tracks=int(complete.sum()))
            for name,forecast in (('unobserved',before),('observed',after)):
                lp,count=trajectory_log_probability(forecast,truth,mask)
                generator=torch.Generator().manual_seed(73199+100*j)
                es,_=energy_score(forecast,truth,mask,draws=max(1,48//forecast.weights.shape[1]),generator=generator)
                mean=(forecast.positions*forecast.weights[:,:,None,None,None]).sum(1)
                error=(mean-truth).norm(dim=-1)
                rows.append(dict(**base,variant=name,energy=float(es[0]),nll=float(-lp[0]/count[0]),
                    mean_ade=float((error*mask).sum()/mask.sum()),
                    mean_fde=float((error[:,-1]*complete).sum()/complete.sum())))
    keys=('energy','nll','mean_ade','mean_fde');groups={name:aggregate([r for r in rows if r['variant']==name],keys)
                                                  for name in ('unobserved','observed')}
    other={r['root_id']:r for r in groups['unobserved']}
    result=dict(status='recurrent_audit_complete',config=cfg,runtime_s=time.time()-start,
        checkpoint_sha256=sha256(cfg['checkpoint']),manifest_sha256=sha256(cfg['manifest']),
        protocol_sha256=sha256(cfg['protocol']),source_sha256=sha256(__file__),records=rows,
        considered_window_pairs=considered,supported_window_pairs=len(rows)//2,
        maximum_composition_position_difference=composition,maximum_persistent_path_difference=paths_difference,
        maximum_anonymous_occupancy_composition_difference=anonymous_composition,
        summary={name:{key:scene_interval(values,key) for key in keys} for name,values in groups.items()},
        observed_minus_unobserved={key:scene_interval(groups['observed'],key,other) for key in keys},
        limitation='Stable existing legal IDs only; real arrivals advanced to next .25s boundary with <=.25s delayed correction',
        changed_id_association_attempted=False,confirmation=False)
    if composition!=0. or paths_difference!=0. or anonymous_composition not in (None,0.):raise AssertionError('Recurrent composition changed')
    save_json(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k in ('status','runtime_s','supported_window_pairs')}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);run(read(parser.parse_args().config))


if __name__=='__main__':main()
