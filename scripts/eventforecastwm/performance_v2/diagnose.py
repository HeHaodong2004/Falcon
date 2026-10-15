"""Root-aware age, target-support and occupancy diagnosis on development only."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from eventbeliefwm.data import read, sha256
from eventbeliefwm.experiment import save_json, occupancy_targets
from eventbeliefwm.metrics import binary_summary
from eventforecastwm.benchmark import load_model, scene_interval
from eventforecastwm.benchmark_reuse import CooperativeGuard
from eventforecastwm.distribution import energy_score, trajectory_log_probability
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.readouts import tracked_occupancy
from eventforecastwm.train import load_roots, prepare_roots, prepared_batch, encode
from eventforecastwm.window_evaluate import aggregate
from .evaluation_population import population as resolve_population


def quantiles(values):
    return dict(count=len(values), quantiles=dict(zip(('min','p25','median','p75','p90','p99','max'),
                np.quantile(values, [0,.25,.5,.75,.9,.99,1]).tolist()))) if values else dict(count=0)


def census(cfg, guard):
    protocol = read(cfg['protocol'])
    fit = set(protocol['subsets']['full'])
    dev = set(protocol['development'])
    assert not fit & dev
    model = load_model(cfg['checkpoint']).eval()
    rows = []
    for ids in (protocol['development'], protocol['subsets']['full']):
        for offset in range(0, len(ids), 64):
            roots = load_roots(cfg['manifest'], ids[offset:offset+64], guard)
            prepare(model, roots, cfg['manifest'], cfg['prepared_cache'], prepare_roots, guard)
            for root in roots:
                for j,w in enumerate(root['windows']):
                    p = w['_prepared']; age = -p.last_observed_times[0]
                    h = w['base'].humans; n = h.shape[0]
                    row = dict(root_id=root['root_id'], scene=root['scene'], window=j,
                               split='fit' if root['root_id'] in fit else 'dev',
                               category=w['metadata']['category'], history_s=w['metadata']['history_span_s'],
                               history_frames=w['metadata']['history_steps'],
                               legal_ever=int(p.valid.sum()), tracker_present=int((h[:,-1,6]>.5).sum()),
                               visible=int((h[:,-1,4]>.5).sum()),
                               physical_count_target_only=w['metadata']['physical_humans_target_only'],
                               all_ages=age[p.valid[0]].tolist(), horizons={})
                    for horizon in (1,2,3,5):
                        keep=w['targets']['track_valid'][:4*horizon].all(0)
                        row['horizons'][str(horizon)] = dict(
                            physical=bool(w['targets']['physical_exposure'][:4*horizon].all()),
                            agents=int(keep.sum()), ages=age[:n][keep].tolist(),
                            expired_supported=int((keep & (age[:n]>2.)).sum()))
                    rows.append(row)
            guard.check()
    summaries={}
    for split in ('fit','dev'):
        summaries[split]={}
        for population in ('origin','all_windows'):
            selected=[r for r in rows if r['split']==split and (population=='all_windows' or r['window']==0)]
            horizons={}
            for horizon in (1,2,3,5):
                key=str(horizon); supported=[r for r in selected if r['horizons'][key]['agents']]
                ages=[a for r in supported for a in r['horizons'][key]['ages']]
                horizons[key]=dict(
                    roots=len({r['root_id'] for r in supported}), windows=len(supported),
                    age_occurrences=quantiles(ages),
                    aged_ge1_roots=len({r['root_id'] for r in supported if max(r['horizons'][key]['ages'])>=1}),
                    aged_gt2_roots=len({r['root_id'] for r in supported if max(r['horizons'][key]['ages'])>2}),
                    aged_ge3_roots=len({r['root_id'] for r in supported if max(r['horizons'][key]['ages'])>=3}),
                    physical_roots=len({r['root_id'] for r in selected if r['horizons'][key]['physical']}))
            summaries[split][population]=dict(roots=len({r['root_id'] for r in selected}),
                windows=len(selected),horizons=horizons)
    return dict(status='census_complete', summaries=summaries, records=rows,
        age_definition='Forecast origin minus last positively observed acquisition, including slots absent from final tracker table',
        tracker_retention_s=2., occurrence_quantiles_are_not_independent_samples=True)


def point_grid(points, valid, extent=8., size=32):
    coord=((points+extent)*(size/(2*extent))).floor().long()
    valid=valid & ((coord>=0)&(coord<size)).all(-1)
    index=coord[...,1].clamp(0,size-1)*size+coord[...,0].clamp(0,size-1)
    grid=points.new_zeros(*points.shape[:2],size*size)
    grid.scatter_add_(2,index,valid.to(points.dtype))
    return grid.clamp_max(1).reshape(*points.shape[:2],size,size)


@torch.no_grad()
def metrics(cfg, guard):
    ids,population_identity=resolve_population(cfg)
    roots=load_roots(cfg['manifest'],ids,guard)
    model=load_model(cfg['checkpoint']).eval()
    if cfg.get('geometry_manifest'):
        from eventforecastwm.geometry_data import attach_geometry
        attach_geometry(roots,cfg['geometry_manifest'],guard)
    prepare(model,roots,cfg['manifest'],cfg['prepared_cache'],prepare_roots,guard)
    state=torch.load(cfg['checkpoint'],map_location='cpu'); prior=state['anonymous_occupancy_prior']
    if set(ids)&set(state.get('fit_root_ids',[])):raise ValueError('Diagnostic/fitting root leakage')
    windows=[(r,j,w) for r in roots for j,w in enumerate(r['windows'])]
    rows=[]; occupancy=[]; agent_rows=[]
    for offset in range(0,len(windows),16):
        guard.check(); selected=windows[offset:offset+16]
        batch=prepared_batch([w for r,j,w in selected],'cpu')
        physical=batch if model.recipe.get('variant')=='geometry_occupancy_v2' or cfg.get('physical_uses_geometry',False) else dict(
            batch,geometry=torch.zeros_like(batch['geometry']),geometry_support=torch.zeros_like(batch['geometry_support']))
        belief=encode(model,physical)
        age=belief.core.age; n=age.shape[1]
        visible=torch.zeros_like(age,dtype=torch.bool)
        for i,(root,j,w) in enumerate(selected):
            visible[i,:len(w['base'].humans)]=w['base'].humans[:,-1,4]>.5
        for horizon in (1.,2.,3.):
            length=round(horizon*4);f=model.imagine(belief,horizon_s=horizon)
            truth=batch['targets']['track_positions'][:,:length]
            complete=batch['targets']['track_valid'][:,:length].all(1)
            mask=complete[:,None].expand(-1,length,-1)
            mean=(f.positions*f.weights[:,:,None,None,None]).sum(1)
            distance=(mean-truth).norm(dim=-1); displacement=truth[:,-1]-truth[:,0]
            speed=displacement.norm(dim=-1)/max(.25,horizon-.25)
            if length>=8:
                first=truth[:,3]-truth[:,0];last=truth[:,-1]-truth[:,-4]
                turn=(first*last).sum(-1)/(first.norm(dim=-1)*last.norm(dim=-1)).clamp_min(1e-6)
                turning=(turn<math.cos(math.pi/4))&(first.norm(dim=-1)>.2)&(last.norm(dim=-1)>.2)
            else:turning=torch.zeros_like(complete)
            strata=dict(all=complete,visible=complete&visible,lost=complete&~visible,
                age_lt025=complete&(age<.25),age_025_to1=complete&(age>=.25)&(age<1),
                age_1_to2=complete&(age>=1)&(age<2),age_2_to3=complete&(age>=2)&(age<3),
                age_ge3=complete&(age>=3),slow=complete&(speed<.25),fast=complete&(speed>=1.25),
                turning=complete&turning,velocity_unknown=complete&~belief.core.velocity_valid)
            for name,eligible in strata.items():
                if not eligible.any():continue
                support=eligible[:,None].expand_as(mask)
                generator=torch.Generator().manual_seed(73199+int(horizon*10)+offset)
                es,_=energy_score(f,truth,support,draws=max(1,48//f.weights.shape[1]),generator=generator)
                lp,count=trajectory_log_probability(f,truth,support)
                for i,(root,j,w) in enumerate(selected):
                    if not eligible[i].any():continue
                    error=(mean[i]-truth[i])[:,eligible[i]]
                    rows.append(dict(root_id=root['root_id'],scene=root['scene'],window=j,
                        category=w['metadata']['category'],horizon=horizon,stratum=name,
                        agents=int(eligible[i].sum()),energy=float(es[i]),nll=float(-lp[i]/count[i]),
                        mean_ade=float(error.norm(dim=-1).mean()),mean_fde=float(error[-1].norm(dim=-1).mean()),
                        bias_x=float(error[...,0].mean()),bias_y=float(error[...,1].mean()),
                        history_s=w['metadata']['history_span_s'],physical_count_target_only=w['metadata']['physical_humans_target_only']))
            total_target,anonymous=occupancy_targets(batch,length)
            tracked=tracked_occupancy(f)
            anonymous_probability=getattr(f,'anonymous_occupancy',prior[:length][None].expand(len(selected),-1,-1,-1))
            combined=1-(1-tracked)*(1-anonymous_probability)
            supported_target=point_grid(truth,mask)
            supported_prediction=tracked_occupancy(replace(f,valid=complete))
            for i,(root,j,w) in enumerate(selected):
                for agent in complete[i].nonzero().flatten().tolist():
                    agent_rows.append(dict(root_id=root['root_id'],scene=root['scene'],window=j,agent_slot=agent,
                        horizon=horizon,age=float(age[i,agent]),visible=bool(visible[i,agent]),
                        mean_ade=float(distance[i,:,agent].mean()),mean_fde=float(distance[i,-1,agent]),
                        target_chord_speed=float(speed[i,agent]),turning=bool(turning[i,agent])))
                base=dict(root_id=root['root_id'],scene=root['scene'],window=j,horizon=horizon)
                if bool(batch['targets']['physical_exposure'][i,:length].all()):
                    occupancy.append(dict(**base,population='total',
                        **binary_summary(total_target[i].numpy(),combined[i].numpy())))
                    occupancy.append(dict(**base,population='anonymous',
                        **binary_summary(anonymous[i].numpy(),anonymous_probability[i].numpy())))
                if complete[i].any():
                    occupancy.append(dict(**base,population='supported_tracked_marginal',
                        **binary_summary(supported_target[i].numpy(),supported_prediction[i].numpy())))
    summary={}
    for population in ('origin','all_windows'):
        summary[population]={}
        for horizon in (1.,2.,3.):
            scores={}
            for name in sorted({r['stratum'] for r in rows}):
                selected=[r for r in rows if r['horizon']==horizon and r['stratum']==name and (population=='all_windows' or r['window']==0)]
                root_rows=aggregate(selected,('energy','nll','mean_ade','mean_fde','bias_x','bias_y'))
                scores[name]={k:scene_interval(root_rows,k) for k in ('energy','nll','mean_ade','mean_fde','bias_x','bias_y')}
            occ={}
            for name in ('total','anonymous','supported_tracked_marginal'):
                selected=[r for r in occupancy if r['horizon']==horizon and r['population']==name and (population=='all_windows' or r['window']==0)]
                root_rows=aggregate(selected,('average_precision','brier','nll','positive_nll'))
                occ[name]={k:scene_interval(root_rows,k) for k in ('average_precision','brier','nll','positive_nll')}
            summary[population][str(horizon)]=dict(trajectory=scores,occupancy=occ)
    return dict(status='diagnosis_complete',summary=summary,trajectory_records=rows,occupancy_records=occupancy,
        agent_records=agent_rows,age_definition='last arrived positive observation; ever-seen belief, not tracker TTL',
        support_definition='Total/anonymous use complete physical exposure; supported-tracked marginal restricts both predicted and target population for diagnosis only',
        population=population_identity,population_averaging='Each metric averages supported windows within each root first',
        secondary_rng='48 paths, reset by horizon and window batch; do not substitute these numbers for primary benchmark RNG')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();cfg=read(args.config)
    run=Path(cfg['run']);run.mkdir(parents=True,exist_ok=True)
    if (run/'result.json').exists():raise FileExistsError('Completed diagnostic is immutable')
    guard=CooperativeGuard(run);torch.set_num_threads(1);start=time.time()
    result=(census if cfg['task']=='census' else metrics)(cfg,guard)
    result.update(runtime_s=time.time()-start,config=cfg,source_sha256=sha256(__file__),
                  checkpoint_sha256=sha256(cfg['checkpoint']),manifest_sha256=sha256(cfg['manifest']),protocol_sha256=sha256(cfg['protocol']))
    save_json(run/'result.json',result)
    print(json.dumps(dict(status=result['status'],runtime_s=result['runtime_s'],summary=result.get('summaries'))),flush=True)


if __name__=='__main__':main()
