"""Bounded native positive-observation replay for changed-ID reacquisition.

Reads small sealed prefix histories, not query branch truth. Physical identity
and state truth are used only after inference to score duplication and diagnostic
retirement. Public imagine/observe receive legal history, actual acquired human
positions, legal tracker IDs, and arrived ego odometry. Assimilation is delayed
to canonical .25s ticks; no future detection is used early. Stationary queries
between ticks isolate belief maintenance and are never assigned native outcomes.
"""
import argparse
from dataclasses import replace
import gzip
import json
from pathlib import Path
import time
import numpy as np
import torch

from eventquerywm.inputs import ModelInput, collate
from eventbeliefwm.data import (RootTransitionDataset, sha256, read, planar_frame,
                                compact_truth, interpolate)
from eventbeliefwm.model import BeliefWorldModel, BeliefConfig
from eventbeliefwm.experiment import save_json


def stationary_query(belief, duration):
    q=belief.positions.new_zeros(len(belief.clock),2,10)
    q[:,:,:2]=belief.ego_position[:,None]
    q[:,1,2]=duration
    q[:,:,5]=belief.ego_yaw.sin()[:,None]
    q[:,:,6]=belief.ego_yaw.cos()[:,None]
    q[:,1,9]=1.
    return q


def truth_associations(history, truth, origin, rotation):
    """Evaluation bookkeeping only; retain duplicate mappings to measure them."""
    matches={}
    evidence={}
    for h in history:
        xy,valid,_,support=interpolate(truth,[float(h['timestamp'])])
        if not support[0]:continue
        for track in h['human_tracks']:
            if not (track['visible_now'] and track['valid']):continue
            point=rotation@(np.asarray(track['position'])-origin)[[0,2]]
            close=np.flatnonzero((np.linalg.norm(xy[0]-point,axis=-1)<1e-4)&valid[0])
            identity=track['track_id']
            matches.setdefault(identity,set())
            evidence.setdefault(identity,[]).append(float(track['last_seen_time']))
            if len(close)==1:matches[identity].add(int(truth['ids'][close[0]]))
    mapping={k:next(iter(v)) for k,v in matches.items() if len(v)==1}
    return mapping,{k:dict(first_acquisition_s=min(v),last_acquisition_s=max(v),positive_frames=len(v))
                    for k,v in evidence.items()}


def load_native(sample):
    row=sample['metadata']['rows'][0]
    root=Path(row['dataset'])/row['root_path']
    seal=Path(row['dataset'])/'root_seals'/f"{sample['metadata']['root_id']}.json"
    if sha256(seal)!=sample['extraction']['seal_sha256'] or sha256(root/'COMPLETE.json')!=sample['extraction']['complete_sha256']:
        raise ValueError('Frozen native root provenance mismatch')
    seals=read(seal)
    files=('legal_observations/history.json','roots/prefix_truth.json.gz')
    for filename in files:
        if sha256(root/filename)!=seals[filename]:raise ValueError('Native prefix changed: '+filename)
    history=read(root/files[0])
    origin,yaw,rotation=planar_frame(history)
    with gzip.open(root/files[1],'rt') as stream:
        truth=compact_truth(json.load(stream),origin,yaw,rotation)
    return history,truth,origin,yaw,rotation,dict(root=str(root),
        files={name:dict(sha256=seals[name],bytes=(root/name).stat().st_size) for name in files})


def initial_belief(model,sample,history):
    base=sample['base']
    all_ids=sample['metadata']['track_ids']
    arrived={tr['track_id'] for tr in history[0]['human_tracks']}
    indices=[i for i,name in enumerate(all_ids) if name in arrived]
    humans=base.humans[indices,:1].clone()
    ego=base.robot[:1].clone()
    start=ego[-1,7].clone()
    humans[...,8]-=start;ego[...,7]-=start
    query=base.query.clone()  # ignored by encode_history
    value=collate([ModelInput(humans,torch.zeros(len(indices),dtype=torch.bool),ego,
                              torch.zeros(1,dtype=torch.bool),query)])
    ids=torch.tensor([indices if indices else [0]],dtype=torch.long)
    acquisition=sample['track_acquisition_times'][indices,:1]
    if not indices:acquisition=torch.zeros(1,1,dtype=torch.float64)
    belief=model.encode_history(value,sensor_config=sample['sensor_config'],robot_geometry=sample['robot_geometry'],
        history_timestamps=sample['history_timestamps'][None,:1],track_acquisition_times=acquisition[None],track_ids=ids)
    return replace(belief,ego_position=ego[-1:,5:7].clone())


@torch.no_grad()
def replay(model,sample,native):
    history,truth,origin,root_yaw,rotation,_=native
    mapping,evidence=truth_associations(history,truth,origin,rotation)
    all_ids=sample['metadata']['track_ids']
    id_index={name:i for i,name in enumerate(all_ids)}
    start=float(history[0]['timestamp'])
    duration=float(history[-1]['timestamp'])-start
    belief=initial_belief(model,sample,history)
    model.eval()
    direct=model.encode_history(collate([sample['base']]),sensor_config=sample['sensor_config'],
        robot_geometry=sample['robot_geometry'],history_timestamps=sample['history_timestamps'][None],
        track_acquisition_times=sample['track_acquisition_times'][None],
        track_ids=torch.arange(len(all_ids))[None])
    direct_objects=[mapping[all_ids[i]] for i in direct.valid[0].nonzero().flatten().tolist() if all_ids[i] in mapping]
    rows=[];cursor=1;latest=history[0]
    positive_updates=0;max_delay=0.
    # One final canonical tick may be <dt beyond the native prefix end so every
    # actual arrival is assimilated. No physical truth is extrapolated there;
    # only identity duplication and model predictions are reported on that tick.
    for step in range(1,int(np.ceil(duration/model.config.dt))+1):
        clock=step*model.config.dt
        belief=model.imagine(belief,stationary_query(belief,model.config.dt),model.config.dt,rasterize=False).next_belief
        arrived={}
        while cursor<len(history) and float(history[cursor]['timestamp'])<=start+clock+1e-8:
            latest=history[cursor]
            for tr in latest['human_tracks']:
                if tr['visible_now'] and tr['valid']:
                    arrived[tr['track_id']]=tr
            cursor+=1
        if arrived:
            names=sorted(arrived)
            positions=np.stack([rotation@(np.asarray(arrived[name]['position'])-origin)[[0,2]] for name in names])
            acquired=np.asarray([float(arrived[name]['last_seen_time'])-start for name in names])
            if np.any(acquired>clock+1e-8):raise ValueError('Future detection entered replay')
            max_delay=max(max_delay,float((clock-acquired).max()))
            positive_updates+=len(names)
            belief=model.observe(belief,dict(positions=torch.tensor(positions,dtype=torch.float32)[None],
                observed=torch.ones(1,len(names),dtype=torch.bool),acquisition_times=torch.tensor(acquired)[None],
                timestamp_s=clock,track_ids=torch.tensor([[id_index[name] for name in names]]),source='actual_arrived'))
        # Legal arrived ego odometry correction; it is never taken from future truth.
        ego=rotation@(np.asarray(latest['robot_position'])-origin)[[0,2]]
        belief=replace(belief,ego_position=torch.tensor(ego,dtype=torch.float32)[None],
                       ego_yaw=torch.tensor([float(latest['robot_yaw'])-root_yaw]))
        grouped={};valid_slots=belief.valid[0].nonzero().flatten().tolist()
        for slot in valid_slots:
            index=int(belief.track_ids[0,slot])
            name=all_ids[index]
            if name in mapping:grouped.setdefault(mapping[name],[]).append(slot)
        duplicate={str(obj):[all_ids[int(belief.track_ids[0,j])] for j in slots]
                   for obj,slots in grouped.items() if len(slots)>1}
        physical,physical_valid,_,physical_support=interpolate(truth,[start+clock])
        diagnostic=dict(clock_s=clock,retained_native_tracker_ids=[tr['track_id'] for tr in latest['human_tracks'] if tr['valid']],
            wm_valid_identities=len(valid_slots),evaluation_unique_physical_identities=len(grouped),
            evaluation_actual_physical_humans=int(physical_valid[0].sum()) if physical_support[0] else None,
            physical_truth_exposed=bool(physical_support[0]),
            beyond_initial_training_horizon=clock>model.config.trained_horizon_s,
            duplicate_identity_groups=duplicate)
        if duplicate:
            # Target-based retirement is an explicit privileged diagnostic, not
            # an inference algorithm or a model input in the evaluated replay.
            keep=belief.valid.clone();retired=[]
            mean=(belief.positions*belief.scenario_weights[:,:,None,None]).sum(1)[0]
            for obj,slots in grouped.items():
                if len(slots)<2:continue
                newest=max(slots,key=lambda slot:float(belief.last_observed_times[0,slot]))
                physical_index=truth['ids'].index(obj)
                for slot in slots:
                    if slot==newest:continue
                    keep[0,slot]=False
                    retired.append(dict(track_id=all_ids[int(belief.track_ids[0,slot])],
                        last_actual_observation_age_s=float(belief.age[0,slot]),
                        mean_distance_to_actual_human_m=float((mean[slot]-torch.tensor(physical[0,physical_index])).norm())
                        if physical_support[0] and physical_valid[0,physical_index] else None))
            # Compare the same legal ego pose on a fresh event window; past
            # absorbing mass must not conceal extra instantaneous ghost hazard.
            fresh=replace(belief,survival_mass=belief.scenario_weights.clone(),collision_mass=torch.zeros_like(belief.collision_mass))
            a=model.imagine(fresh,stationary_query(fresh,.25),.25)
            retired_belief=replace(fresh,valid=keep)
            z=model.imagine(retired_belief,stationary_query(retired_belief,.25),.25)
            diagnostic.update(retired_duplicate_hypotheses=retired,
                tracked_occupancy_cell_mass=float(a.physical_occupancy.sum()),
                retired_duplicate_occupancy_cell_mass=float(z.physical_occupancy.sum()),
                synthetic_wait_collision_risk=float(a.cumulative_collision[0,0,-1]),
                retired_duplicate_synthetic_wait_collision_risk=float(z.cumulative_collision[0,0,-1]))
        rows.append(diagnostic)
    duplicates=[row for row in rows if row['duplicate_identity_groups']]
    return dict(model_kind=model.config.kind,scenario_count=model.scenarios,positive_assimilation_updates=positive_updates,
        native_duration_s=duration,
        direct_root_encode=dict(clock_s=0.,valid_identities=int(direct.valid.sum()),
            evaluation_unique_physical_identities=len(set(direct_objects)),
            duplicate_identity_excess=len(direct_objects)-len(set(direct_objects)),
            note='Full arrived root history; this duplication test is at supported initial time, independent of long replay extrapolation'),
        maximum_latest_positive_observation_age_at_tick_s=max_delay,
        mapping_target_only=mapping,positive_observation_evidence=evidence,
        canonical_ticks=len(rows),duplicate_ticks=len(duplicates),
        first_duplicate_clock_s=duplicates[0]['clock_s'] if duplicates else None,
        maximum_valid_identity_excess=max((row['wm_valid_identities']-row['evaluation_unique_physical_identities'] for row in rows),default=0),
        rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint');p.add_argument('--max-roots',type=int,default=2)
    args=p.parse_args();torch.set_num_threads(1)
    if not 1<=args.max_roots<=4:raise ValueError('Diagnostic limited to at most four native roots')
    if Path(args.output).exists():raise FileExistsError(args.output)
    cache=Path(args.cache);manifest=read(cache/'manifest.json')
    selected=[]
    for split in ('validation','train','unseen_evaluation'):
        dataset=RootTransitionDataset(cache,split)
        for i,row in enumerate(dataset.rows):
            if sha256(cache/row['file'])!=row['sha256']:raise ValueError('Cached root changed')
            sample=dataset[i];meta=sample['metadata']
            if meta['physical_humans_target_only']==1 and len(meta['track_ids'])>1:
                selected.append(sample)
            if len(selected)>=args.max_roots:break
        if len(selected)>=args.max_roots:break
    models={'cv':BeliefWorldModel(BeliefConfig(kind='cv'))}
    if args.checkpoint:
        state=torch.load(args.checkpoint,map_location='cpu')
        learned=BeliefWorldModel(BeliefConfig(**state['protocol']['config']));learned.load_state_dict(state['model'])
        models['learned']=learned
    started=time.time();roots=[]
    for sample in selected:
        native=load_native(sample)
        roots.append(dict(root_id=sample['metadata']['root_id'],split=sample['metadata']['split'],
            source=native[-1],model_replays={name:replay(model,sample,native) for name,model in models.items()}))
    result=dict(snapshot_sha256=manifest['snapshot_sha256'],cache_manifest_sha256=sha256(cache/'manifest.json'),
        checkpoint=args.checkpoint,checkpoint_sha256=sha256(args.checkpoint) if args.checkpoint else None,
        roots=roots,seconds=time.time()-started,
        selection=f'First at most {args.max_roots} single-physical-human roots with multiple legal IDs in validation/train/unseen manifest order; targeted failure reproduction, not prevalence estimate',
        truth_boundary='Native prefix physical identities/positions are assessment and privileged-retirement diagnostics only; never model inputs',
        inference_observations='Actual positive detections, exact acquisition clocks, and arrived ego odometry; latest positive per identity in each canonical interval',
        action_scope='Stationary imagination isolates belief maintenance; no synthetic action is assigned a native outcome',
        conclusion='Changed-ID reacquisition gate fails if multiple unit-valid physical hypotheses persist for the same native person',
        emission_scale_interpretation='Gaussian position_scale is currently a conditional emission/model-error score around finite coherent scenarios. It is not integrated into event hazards or used by observe (fixed .20 m likelihood). Report these separately; calibrated Gaussian NLL is not calibrated joint-event uncertainty.',
        minimal_followup='Add legal observation association/existence hypotheses and a changed-ID replay regression before broader belief claims. Tie the documented measurement/predictive scale interpretation to correction and event integration; retain compact dynamics and compare to native tracker/CV.')
    save_json(args.output,result)
    print(json.dumps(dict(roots=[dict(root_id=r['root_id'],models={name:{k:v[k] for k in ('canonical_ticks','duplicate_ticks','first_duplicate_clock_s','maximum_valid_identity_excess')} for name,v in r['model_replays'].items()}) for r in roots],seconds=result['seconds']),indent=2))


if __name__=='__main__':main()
