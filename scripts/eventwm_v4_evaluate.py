#!/usr/bin/env python3
"""Frozen Gaussian CV, planning-bank reuse, and bounded successor diagnostics."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from eventwm_v4_research import (DEFAULT_OUT, PROJECT, Cache, read, save, sha, seed_all,
                                score, infer, ProductionGuard, production_sample)
from eventquerywm.model import EventQueryWM, ModelConfig
from eventquerywm.inputs import collate, ModelInput
from eventquerywm.v2.history import arrived_tracks
from eventquerywm.v2.model import sample_query, interaction_features
from eventquerywm.v2.dynamics import Dynamics
from eventquerywm.planning.calibration import fit_monotone, probability
from eventquerywm.planning.offline import banks, choose, root_summary, paired_selection
from eventquerywm.sr_metrics import ranking_metrics
from eventquerywm.successor import nominal_at_horizons

WORKTREE=Path(__file__).resolve().parents[1]
REFERENCE=WORKTREE/'planning_baselines/registry.json'
LEGACY=PROJECT/'output/eventwm_planning_evaluation_checkout/eventwm_planner_results'
HORIZONS=np.array([.25,.5,1.,2.,3.,5.])


def cv_model(device):
    registry=read(REFERENCE)
    ref=registry['models']['cv']
    path=REFERENCE.parent/ref['checkpoint']
    assert sha(path)==ref['sha256']
    model=Dynamics('cv').to(device)
    model.load_reference(torch.load(path,map_location='cpu')['model'])
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


@torch.no_grad()
def cv_exposure(model,value):
    tracks,valid,sequence,mask=arrived_tracks(value)
    query=sample_query(value.query,64,None)
    forecast=model.forecast(tracks,valid,query[...,2],sequence,mask)
    features=interaction_features(query,tracks,valid,forecast)
    return torch.trapz(features[...,0],query[...,2],dim=1)


def cv(args):
    out=args.output/'cv'
    out.mkdir(parents=True,exist_ok=False)
    cache=Cache(args.output)
    model=cv_model(args.device)
    guard=ProductionGuard(out)
    predictions={}
    for split in ('train','validation','natural_reference','unseen_evaluation'):
        indices=cache.splits[split]
        result=[]
        for start in range(0,len(indices),128):
            ix=indices[start:start+128]
            result.append(cv_exposure(model,cache.batch(ix,device=args.device)).cpu().numpy())
            guard.check()
        predictions[split]=np.concatenate(result)
    labels=np.array([1-cache.rows[i]['sr_label'] for i in cache.splits['train']])
    calibration=fit_monotone(predictions['train'],labels)
    save(out/'calibration.json',calibration)
    reports={}
    for split,exposure in predictions.items():
        ix=cache.splits[split]
        collision=probability(exposure,calibration)
        z=-(calibration['intercept']+calibration['slope']*np.log1p(exposure))
        reports[split]=score(cache,ix,z)
        np.savez_compressed(out/(split+'.npz'),indices=ix,exposure=exposure,logits=z)
    save(out/'results.json',dict(model='Existing frozen Gaussian CV, 64-clock full nominal-query exposure',
         checkpoint=read(REFERENCE)['models']['cv'],calibration_scope='V4 optimization only; no held-out tuning',
         input_contract='Legal arrived track position, estimated velocity, age, and candidate nominal query only',
         reports=reports,production_end=production_sample()))


def gaussian_tracks(base):
    """Exact arrived_tracks motion state, omitting its unused history-GRU sequence.

    Deduplicate observation clocks with the same 1e-5 tolerance and compute the
    last finite-difference estimate over the same .5s maximum reacquisition gap.
    This query-free result can be shared across all 33 candidates in one root.
    """
    n=max(1,len(base.humans))
    tracks=torch.zeros(1,n,8);valid=torch.zeros(1,n,dtype=torch.bool)
    for k in range(len(base.humans)):
        h=base.humans[k];last=h[-1]
        if base.human_padding[k] or last[6]<.5 or last[10]<.5 or last[7]>2:
            continue
        a=h.numpy();ix=np.flatnonzero((a[:,6]>=.5)&(a[:,10]>=.5))
        clocks=a[ix,8].astype(float)-a[ix,7].astype(float)
        distinct=np.r_[True,np.diff(clocks)>1e-5]
        ix=ix[distinct];clocks=clocks[distinct]
        velocity=torch.zeros(2);vvalid=False
        if len(ix)>1:
            gap=float(clocks[-1]-clocks[-2])
            vvalid=gap<=.5+1e-5
            if vvalid:velocity=(h[ix[-1],:2]-h[ix[-2],:2])/gap
        tracks[0,k]=torch.cat([last[:2],velocity,last[7:8],last[4:5],last.new_tensor([float(vvalid),1.])])
        valid[0,k]=True
    return tracks,valid


def cv_fast(args):
    out=args.output/'cv_fast'
    out.mkdir(exist_ok=False)
    cache=Cache(args.output)
    model=cv_model(args.device)
    guard=ProductionGuard(out)
    motion={};max_error=0.;checked=0
    started=time.time()
    for i,b in enumerate(cache.blocks):
        tracks,valid=gaussian_tracks(b['base'])
        # Real-data parity at multiple lengths, legal categories, and epochs.
        if i%20==0:
            expected=arrived_tracks(collate([b['base']]))
            error=float((tracks-expected[0]).abs().max())
            assert torch.equal(valid,expected[1]) and error<1e-6
            max_error=max(max_error,error);checked+=1
        motion[b['root_id']]=(tracks[0],valid[0])
    predictions={}
    for split in ('train','validation','natural_reference','unseen_evaluation'):
        indices=cache.splits[split];result=[]
        for start in range(0,len(indices),128):
            ix=indices[start:start+128]
            n=max(motion[cache.rows[j]['root_id']][0].shape[0] for j in ix)
            tracks=torch.zeros(len(ix),n,8);valid=torch.zeros(len(ix),n,dtype=torch.bool);queries=[]
            for bi,j in enumerate(ix):
                b,k=cache.samples[j];t,v=motion[b['root_id']]
                tracks[bi,:len(t)]=t;valid[bi,:len(v)]=v;queries.append(b['query'][k])
            tracks,valid=tracks.to(args.device),valid.to(args.device)
            query=sample_query(torch.stack(queries).to(args.device),64,None)
            with torch.no_grad():
                forecast=model.forecast(tracks,valid,query[...,2])
                f=interaction_features(query,tracks,valid,forecast)
                result.append(torch.trapz(f[...,0],query[...,2],dim=1).cpu().numpy())
            guard.check()
        predictions[split]=np.concatenate(result)
    calibration=fit_monotone(predictions['train'],np.array([1-cache.rows[i]['sr_label'] for i in cache.splits['train']]))
    reports={}
    for split,exposure in predictions.items():
        ix=cache.splits[split]
        z=-(calibration['intercept']+calibration['slope']*np.log1p(exposure))
        reports[split]=score(cache,ix,z)
        np.savez_compressed(out/(split+'.npz'),indices=ix,exposure=exposure,logits=z)
    save(out/'results.json',dict(model='Existing frozen Gaussian CV, 64-clock full-query exposure',calibration=calibration,
         motion_state_parity_roots=checked,motion_max_absolute_error=max_error,
         implementation='Query-independent legal motion cached once/root; no feature, Gaussian, exposure, or calibration change',
         reports=reports,seconds=time.time()-started,production_end=production_sample()))


def successor_audit(args):
    """Bounded read-only target smoke and whole-snapshot duration coverage audit."""
    out=args.output/'successor_audit'
    out.mkdir(exist_ok=True)
    assert not (out/'results.json').exists()
    cache=Cache(args.output)
    snapshot=read(args.output/'snapshot.json')
    blocks={b['root']['root_id']:b for b in snapshot['blocks']}
    coverage={}
    for split in ('train','validation','natural_reference','unseen_evaluation'):
        rows=[cache.rows[i] for i in cache.splits[split]]
        coverage[split]=[dict(horizon_s=float(h),queries=int(sum(r['observed_duration_s']>=h for r in rows)),total=len(rows)) for h in HORIZONS]
    smoke=[]
    for split in ('train','validation','natural_reference','unseen_evaluation'):
        selected=[b for b in cache.blocks if b['split']==split][:4]
        for b in selected:
            ref=blocks[b['root_id']];path=Path(ref['archive_path'])/'branch_truth/branches.jsonl.gz'
            with gzip.open(path,'rt') as f:branch=json.loads(f.readline())
            assert branch['branch_id']=='b000'
            target=target_branch(branch,b['base'])
            smoke.append(dict(root_id=b['root_id'],split=split,query_id=branch['query_id'],
                 physical_mask=target['physical_mask'].tolist(),observed_mask=target['observed_mask'].tolist(),
                 truth_duration=target['duration'],no_posttermination_targets=True))
    save(out/'results.json',dict(duration_coverage=coverage,actual_future_target_smoke=smoke,
         extraction_checks_passed=True,neural_successor_training_completed=False,
         reason='Optional whole-root successor extraction was stopped conservatively when production publication declined; partial extraction was not committed as a completed cache',
         source='Sealed immutable archived branches already deep-verified by production; bounded smoke does not rehash entire compressed root archives',
         limits='Duration coverage measures physical availability. Observed masks additionally require an arrived future legal observation. Occupancy decoders do not define identity-resolved ADE/FDE or full physical state. Current stock successor supports only <=3s; 5s requires an explicitly trained extension.'))


def planning(args):
    out=args.output/getattr(args,'planning_tag','planning')
    out.mkdir(parents=True,exist_ok=False)
    cache=getattr(args,'cache',None) or Cache(args.output)
    checkpoint=getattr(args,'checkpoint',None) or args.output/'runs/all_full_seed20261012/best.pt'
    ck=torch.load(checkpoint,map_location='cpu')
    model=EventQueryWM(ModelConfig(**ck['config'])).to(args.device)
    model.load_state_dict(ck['model']);model.eval()
    features_path=PROJECT/'output/eventwm_v2_architecture_checkout/output/head_features_20261010/validation.pt'
    raw= torch.load(features_path,map_location='cpu')
    metadata=raw['metadata']
    source_index=PROJECT/'output/first_training_checkout/output/first_training/index.json'
    audit=read(args.output/'audit.json')['reserved_selector_bank']
    assert sha(source_index)==audit['sha256']
    bank_scenes={m['scene'] for m in metadata}
    bank_roots={m['root_id'] for m in metadata}
    assert bank_roots.isdisjoint({cache.rows[i]['root_id'] for i in cache.splits['train']+cache.splits['validation']})
    assert {m['query_id'] for m in metadata}.isdisjoint({cache.rows[i]['query_id'] for i in cache.splits['train']+cache.splits['validation']})
    # Reuse the immutable canonical legal tensor cache in precisely its old order.
    legal_path=PROJECT/'output/first_training_checkout/output/full_baseline_seed20261008/validation_legal_tensors.pt'
    legal=torch.load(legal_path,map_location='cpu')
    index_rows=[r for r in read(source_index)['rows'] if r['split']=='validation']
    identity=__import__('hashlib').sha256(json.dumps([(r['query_id'],r['sr_label']) for r in index_rows]).encode()).hexdigest()
    assert legal['identity']==identity
    mapping={r['query_id']:v for r,(v,y) in zip(index_rows,legal['data'])}
    rows={r['query_id']:r for r in index_rows}
    values=[mapping[m['query_id']] for m in metadata]
    assert len(values)==len(metadata)==4092
    logits=[]
    guard=ProductionGuard(out)
    with torch.no_grad():
        for start in range(0,len(values),128):
            logits.append(model(collate(values[start:start+128]).to(args.device)).cpu().numpy())
            guard.check()
    z=np.concatenate(logits)
    pc=1/(1+np.exp(np.clip(z,-700,700)))
    f=raw['features']
    duration,movement=f['duration'].numpy(),f['movement'].numpy()
    targets=raw['targets']['outcome_label'].numpy()
    ids=[m['query_id'] for m in metadata]
    legacy_predictions=np.load(LEGACY/'validation_predictions.npz')
    # The old saved probabilities/exposures and policies are reused byte for byte.
    save(out/'legacy_prediction_keys.json',list(legacy_predictions.files))
    assert np.array_equal(legacy_predictions['query_ids'],np.asarray([m['query_id'] for m in metadata]))
    old_binary=legacy_predictions['binary_collision']
    exposure=f['exposure'].numpy()
    old_cv_calibration=read(LEGACY/'cv_calibration.json')['final']
    cvp=probability(exposure,old_cv_calibration)
    scores=dict(fastest=duration,old_eventwm=old_binary,new_eventwm=pc,gaussian_cv=exposure,
                old_eventwm_time_risk=duration+10*old_binary,
                new_eventwm_time_risk=duration+10*pc,
                gaussian_cv_time_risk=duration+10*cvp)
    result={}
    old_results=read(LEGACY/'validation.json')['banks']['same_path']['methods']
    for scope,matched in [('same_path',True),('same_root',False)]:
        bank=banks(metadata,movement,matched)
        methods={}
        for name,s in scores.items():
            selected=choose(bank,s,duration,ids)
            methods[name]=root_summary(metadata,selected,targets,duration,movement,bank,10,0)
        if matched:
            for name,old_name in [('fastest','fastest'),('old_eventwm','binary_sr'),('gaussian_cv','cv_exposure')]:
                assert methods[name]['selected_query_ids']==old_results[old_name]['selected_query_ids']
        result[scope]=dict(methods=methods,
            new_minus_gaussian=paired_selection(methods['new_eventwm'],methods['gaussian_cv']),
            time_risk_new_minus_gaussian=paired_selection(methods['new_eventwm_time_risk'],methods['gaussian_cv_time_risk']))
    np.savez_compressed(out/'new_predictions.npz',logits=z,p_collision=pc)
    save(out/'results.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
         bank=dict(roots=124,queries=4092,primary='Existing movement-matched same-path bank; stationary WAIT excluded by unchanged eligibility',
                   candidate_policy='Existing banks/choose/root_summary; score, nominal duration, query-ID tie breaks unchanged',
                   bank_file_sha256=sha(features_path),legal_cache_sha256=sha(legal_path)),
         frozen_rules='Minimum risk and pre-existing duration+10*P(collision); no new selector or validation-selected penalty',
         semantics='Clean-safe = class0; social violation success = class1; collision=class2; social success = class0 among noncollisions',
         limitation='Old model was selected on this development bank. New models exclude all bank roots, queries, episodes and physical-waypoint trajectory fixtures; scene overlap is allowed in the revised study. Offline branch outcomes are not closed-loop social task success.',
         comparisons=result,legacy_selectors_exactly_reproduced=True))


def grid(points):
    g=np.zeros((24,24),np.uint8)
    for p in points:
        x,z=np.asarray(p)[:2]
        if -6<=x<6 and -6<=z<6:
            g[int((z+6)/.5),int((x+6)/.5)]=1
    return g


def target_branch(branch,base):
    # Future truth and legal observations are supervision only. The input base
    # was cached earlier by the public legal loader, without opening this file.
    origin=base.robot[-1,:3].numpy().astype(float)
    yaw=np.arctan2(float(base.robot[-1,3]),float(base.robot[-1,4]))
    rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    def local(p):return rotation@(np.asarray(p,dtype=float)-origin)[[0,2]]
    t0=float(branch['truth'][0]['timestamp'])
    lookup={round(float(f['timestamp'])-t0,8):f for f in branch['truth']}
    times=np.array(sorted(lookup));frames=[lookup[t] for t in times]
    end=float(times[-1])
    legal=branch['legal_observations']
    lt=np.array([f['timestamp']-t0 for f in legal])
    physical,observed,pm,om,positions=[],[],[],[],[]
    for h in HORIZONS:
        active=times[0]-1e-8<=h<=end+1e-8
        points={}
        if active:
            hi=min(int(np.searchsorted(times,h)),len(times)-1);lo=max(0,hi-1)
            a=0 if hi==lo else (h-times[lo])/(times[hi]-times[lo])
            before={p['object_id']:np.asarray(p['position']) for p in frames[lo]['humans']}
            points={p['object_id']:local(before[p['object_id']]+a*(np.asarray(p['position'])-before[p['object_id']])) for p in frames[hi]['humans'] if p['object_id'] in before}
        eligible=np.flatnonzero(lt<=h+1e-9)
        available=active and len(eligible)>0 and h-lt[eligible[-1]]<=.06
        visible=[local(p['position']) for p in legal[eligible[-1]]['human_tracks'] if p['visible_now'] and p['valid']] if available else []
        physical.append(grid(points.values()));observed.append(grid(visible));pm.append(active);om.append(available);positions.append(points)
    assert not any(a and h>end+1e-7 for h,a in zip(HORIZONS,pm))
    return dict(physical=np.stack(physical),observed=np.stack(observed),physical_mask=np.array(pm),observed_mask=np.array(om),
                duration=end,positions=positions)


def successor_prepare(args):
    out=args.output/'successor'
    out.mkdir(parents=True,exist_ok=False)
    cache=Cache(args.output)
    snapshot=read(args.output/'snapshot.json')
    evidence={b['root']['root_id']:b for b in snapshot['blocks']}
    records=[]
    started=time.time()
    guard=ProductionGuard(out)
    for i,b in enumerate(cache.blocks):
        ref=evidence[b['root_id']]
        path=Path(ref['archive_path'])/'branch_truth/branches.jsonl.gz'
        assert sha(path)==ref['evidence']['branch_truth/branches.jsonl.gz']
        # A fixed b000 branch per root is chosen without outcomes, visibility,
        # duration or availability selection; reduce repeated future-data I/O.
        with gzip.open(path,'rt') as f:
            branch=json.loads(f.readline())
        assert branch['branch_id']=='b000' and branch['root_id']==b['root_id']
        j=next(j for j,r in enumerate(b['rows']) if r['query_id']==branch['query_id'])
        target=target_branch(branch,b['base'])
        records.append(dict(root_id=b['root_id'],split=b['split'],query_id=branch['query_id'],query_ordinal=j,target=target))
        if (i+1)%100==0:
            print(json.dumps(dict(stage='successor_targets',roots=i+1,total=len(cache.blocks),seconds=time.time()-started)),flush=True)
            guard.check()
    torch.save(records,out/'targets.pt')
    coverage={}
    for split in cache.splits:
        rs=[r for r in records if r['split']==split]
        coverage[split]=dict(roots=len(rs),physical=[int(sum(r['target']['physical_mask'][h] for r in rs)) for h in range(6)],
                            observed=[int(sum(r['target']['observed_mask'][h] for r in rs)) for h in range(6)])
    save(out/'target_manifest.json',dict(sha256=sha(out/'targets.pt'),horizons_s=HORIZONS.tolist(),coverage=coverage,seconds=time.time()-started,
        selection='Exactly one precommitted b000 branch for every eligible/evaluation root; not all 194832 branch targets',
        target_scope='24x24 human center cells in fixed decision frame; observed = valid visible tracks at latest <= horizon observation, <=.06s age',
        censoring='No future targets beyond actual recorded branch termination; 5s may have little or no support',
        labels_never_inputs=True))


def successor_panel_prepare(args):
    """Outcome-blind fixed resource cap, with resumable research-only partials."""
    from eventwm_v4_research import key
    out=args.output/'successor_bounded'
    out.mkdir(exist_ok=True)
    assert not (out/'targets.pt').exists()
    cache=Cache(args.output)
    snapshot=read(args.output/'snapshot.json')
    evidence={b['root']['root_id']:b for b in snapshot['blocks']}
    selected=[]
    for split,cap in [('train',512),('validation',106),('natural_reference',64),('unseen_evaluation',64)]:
        selected.extend(sorted([b for b in cache.blocks if b['split']==split],key=lambda b:key('successor/'+b['root_id']))[:cap])
    partial=out/'partial_targets.pt'
    records=torch.load(partial,map_location='cpu') if partial.exists() else []
    completed={r['root_id'] for r in records}
    guard=ProductionGuard(out)
    started=time.time()
    for i,b in enumerate(selected):
        if b['root_id'] in completed:continue
        ref=evidence[b['root_id']]
        path=Path(ref['archive_path'])/'branch_truth/branches.jsonl.gz'
        assert sha(path)==ref['evidence']['branch_truth/branches.jsonl.gz']
        with gzip.open(path,'rt') as f:branch=json.loads(f.readline())
        assert branch['branch_id']=='b000' and branch['root_id']==b['root_id']
        j=next(j for j,r in enumerate(b['rows']) if r['query_id']==branch['query_id'])
        target=target_branch(branch,b['base'])
        records.append(dict(root_id=b['root_id'],split=b['split'],query_id=branch['query_id'],query_ordinal=j,target=target))
        if len(records)%32==0:
            torch.save(records,partial)
            print(json.dumps(dict(stage='bounded_successor_targets',roots=len(records),total=len(selected),seconds=time.time()-started)),flush=True)
            guard.check()
    torch.save(records,out/'targets.pt')
    coverage={}
    for split in ('train','validation','natural_reference','unseen_evaluation'):
        rs=[r for r in records if r['split']==split]
        coverage[split]=dict(roots=len(rs),physical=[int(sum(r['target']['physical_mask'][h] for r in rs)) for h in range(6)],
                            observed=[int(sum(r['target']['observed_mask'][h] for r in rs)) for h in range(6)])
    save(out/'target_manifest.json',dict(sha256=sha(out/'targets.pt'),snapshot_sha256=sha(args.output/'snapshot.json'),
         horizons_s=HORIZONS.tolist(),coverage=coverage,seconds=time.time()-started,
         selection='SHA256(20261012/successor/root), <=512 TRAIN, all106 validation, <=64 each natural/unseen; one fixed b000 branch/root, independent of labels/duration/visibility',
         scientific_scope='Bounded decoder adequacy and nested decoder data comparison, not all 194832 successor branches',labels_never_inputs=True))


def occupancy_metrics(p,y,mask):
    selected=p[mask].astype(float);truth=y[mask].astype(float)
    if not len(truth):return dict(valid=0,bce=None,foreground_ap=None,prevalence=None)
    labels=truth.reshape(-1);scores=selected.reshape(-1)
    clipped=np.clip(scores,1e-7,1-1e-7)
    return dict(valid=int(mask.sum()),bce=float(np.mean(-labels*np.log(clipped)-(1-labels)*np.log1p(-clipped))),
                foreground_ap=ranking_metrics(labels,scores)[0],prevalence=float(labels.mean()),
                brier=float(np.mean((labels-scores)**2)))


def successor_train(args):
    source=args.output/('successor_bounded' if args.successor_panel=='bounded' else 'successor')
    out=source/args.subset
    out.mkdir(exist_ok=False)
    records=torch.load(source/'targets.pt',map_location='cpu')
    assert sha(source/'targets.pt')==read(source/'target_manifest.json')['sha256']
    cache=Cache(args.output)
    byroot={b['root_id']:b for b in cache.blocks}
    checkpoint=args.output/'runs/all_full_seed20261012/best.pt'
    ck=torch.load(checkpoint,map_location='cpu')
    backbone=EventQueryWM(ModelConfig(**ck['config'])).to(args.device)
    backbone.load_state_dict(ck['model']);backbone.eval()
    for p in backbone.parameters():p.requires_grad_(False)
    hist=[];queries=[]
    cv=cv_model(args.device)
    cv.horizons=torch.as_tensor(HORIZONS,dtype=torch.float32,device=args.device)
    # Existing Gaussian scalar parameters, evaluated at 5s without pretending
    # that an old 3s neural successor was trained at that horizon.
    cv.background=torch.zeros(6,24,24,device=args.device)
    gp=[]
    point=[];visible_gp=[]
    for start in range(0,len(records),64):
        rs=records[start:start+64]
        values=[]
        for r in rs:
            b=byroot[r['root_id']];v=b['base'];values.append(ModelInput(v.humans,v.human_padding,v.robot,v.history_padding,b['query'][r['query_ordinal']]))
        value=collate(values).to(args.device)
        with torch.no_grad():
            hist.append(backbone.encode_scene(value)[0][:,0].detach())
            queries.append(value.query.detach())
            motion=arrived_tracks(value);tracks,valid,sequence,mask=motion
            gp.append(cv(tracks,valid,sequence,mask).cpu().numpy())
            visible_gp.append(cv(tracks,valid & (tracks[...,5]>=.5),sequence,mask).cpu().numpy())
            age=tracks[...,4,None]+cv.horizons
            positions=tracks[:,:,None,:2]+tracks[:,:,None,2:4]*age[...,None]
            cells=torch.floor((positions+6)/.5).long()
            p=tracks.new_full((len(tracks),len(HORIZONS),24,24),1e-6)
            active=valid[...,None] & (cells>=0).all(-1) & (cells<24).all(-1)
            for bi,ni,hi in active.nonzero().tolist():
                x,z=cells[bi,ni,hi].tolist();p[bi,hi,z,x]=1-1e-6
            point.append(p.cpu().numpy())
    history=torch.cat(hist)
    query=torch.cat(queries)
    gaussian=np.concatenate(gp);point=np.concatenate(point)
    ys=torch.tensor(np.stack([r['target']['physical'] for r in records]),dtype=torch.float32,device=args.device)
    masks=torch.tensor(np.stack([r['target']['physical_mask'] for r in records]),device=args.device)
    allowed=set(read(args.output/'snapshot.json')['subsets'][args.subset])
    train=np.array([i for i,r in enumerate(records) if r['split']=='train' and r['root_id'] in allowed])
    splits={s:np.array([i for i,r in enumerate(records) if r['split']==s]) for s in ('train','validation','natural_reference','unseen_evaluation')}
    splits['train']=train
    oy=torch.tensor(np.stack([r['target']['observed'] for r in records]),dtype=torch.float32,device=args.device)
    om=torch.tensor(np.stack([r['target']['observed_mask'] for r in records]),device=args.device)
    opriors=np.stack([oy[train,h][om[train,h]].mean(0).cpu().numpy() if om[train,h].any() else np.zeros((24,24)) for h in range(6)])
    priors=np.stack([ys[train,h][masks[train,h]].mean(0).cpu().numpy() if masks[train,h].any() else np.zeros((24,24)) for h in range(6)])
    # Same Gaussian CV function with a current TRAIN-only background occupancy
    # prior, rather than handicapping unseen humans with an empty background.
    background=priors.clip(1e-7,.02)
    gaussian=(1-(1-gaussian)*(1-background[None])).clip(1e-7,1-1e-6)
    # Existing physical_head topology, same 107456 parameters; no decoder redesign.
    def head(features):return nn.Sequential(nn.Linear(features,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,576))
    seed_all(20261012)
    net=head(129).to(args.device)
    onet=head(139).to(args.device)
    with torch.no_grad():
        p=torch.tensor(priors[:5].mean(0).clip(1e-5,1-1e-5),device=args.device)
        net[-1].bias.copy_(torch.logit(p).flatten())
        op=torch.tensor(opriors[:5].mean(0).clip(1e-5,1-1e-5),device=args.device)
        onet[-1].bias.copy_(torch.logit(op).flatten())
    clock=torch.tensor(HORIZONS/3,dtype=torch.float32,device=args.device)
    def predict(ix,observed=False):
        h=history[ix,None].expand(-1,6,-1)
        c=clock[None,:,None].expand(len(ix),-1,-1)
        features=torch.cat([h,c],-1)
        if observed:
            features=torch.cat([features,nominal_at_horizons(query[ix],torch.tensor(HORIZONS,dtype=torch.float32,device=args.device))],-1)
        return (onet if observed else net)(features).reshape(-1,6,24,24)
    def report(p,observed=False):
        y=(oy if observed else ys).cpu().numpy();mask=(om if observed else masks).cpu().numpy()
        return {s:[dict(horizon_s=float(h),**occupancy_metrics(p[ix,k],y[ix,k],mask[ix,k])) for k,h in enumerate(HORIZONS)] for s,ix in splits.items()}
    baselines=dict(gaussian_cv=report(gaussian),point_cv=report(point),constant_training_prior=report(np.broadcast_to(priors,gaussian.shape)))
    observed_gaussian=(1-(1-np.concatenate(visible_gp))*(1-opriors[None].clip(1e-7,.02))).clip(1e-7,1-1e-6)
    observed_baselines=dict(gaussian_cv_current_visibility=report(observed_gaussian,True),constant_training_prior=report(np.broadcast_to(opriors,gaussian.shape),True))
    save(out/'baselines.json',baselines)
    save(out/'observed_baselines.json',observed_baselines)
    parameters=list(net.parameters())+list(onet.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=.01)
    generator=np.random.default_rng(20261012)
    curve=[];best=-1;best_step=0
    guard=ProductionGuard(out)
    started=time.time()
    for step in range(1025):
        if step%128==0:
            net.eval();onet.eval()
            with torch.no_grad():
                pieces=[predict(np.arange(start,min(start+128,len(records)))).sigmoid().cpu().numpy() for start in range(0,len(records),128)]
                p=np.concatenate(pieces)
                observed_p=np.concatenate([predict(np.arange(start,min(start+128,len(records))),True).sigmoid().cpu().numpy() for start in range(0,len(records),128)])
            result=report(p)
            observed_result=report(observed_p,True)
            ap=np.mean([r['foreground_ap'] for r in result['validation'][:3] if r['foreground_ap'] is not None])
            curve.append(dict(step=step,metrics=result,observed_metrics=observed_result,selection_ap=float(ap),seconds=time.time()-started))
            save(out/'curve.json',curve)
            if ap>best:
                best=ap;best_step=step
                torch.save(dict(model=net.state_dict(),observed_model=onet.state_dict(),step=step,source_backbone_sha256=sha(checkpoint)),out/'best_physical_head.pt')
                np.savez_compressed(out/'best_predictions.npz',probability=p,observed_probability=observed_p)
            print(json.dumps(dict(stage='successor_train',step=step,train_1s_ap=result['train'][2]['foreground_ap'],val_1s_ap=result['validation'][2]['foreground_ap'],seconds=time.time()-started)),flush=True)
            guard.check()
        if step==1024:break
        ix=generator.choice(train,size=min(64,len(train)),replace=False)
        net.train();onet.train();optimizer.zero_grad(set_to_none=True)
        z=predict(ix);active=masks[ix]
        loss=torch.nn.functional.binary_cross_entropy_with_logits(z[active],ys[ix][active])
        observed_z=predict(ix,True);observed_active=om[ix]
        if observed_active.any():
            loss=.5*(loss+torch.nn.functional.binary_cross_entropy_with_logits(observed_z[observed_active],oy[ix][observed_active]))
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1,error_if_nonfinite=True);optimizer.step()
    save(out/'results.json',dict(best_step=best_step,best=next(r for r in curve if r['step']==best_step),final=curve[-1],
        baselines=baselines,observed_baselines=observed_baselines,trainable_parameters=sum(p.numel() for p in parameters),optimization_roots=len(train),
        source_backbone_sha256=sha(checkpoint),loss='Equal physical/observed masked BCE; existing physical and observed head topologies on the same frozen SR encoder',
        observed_state='Valid currently visible track center occupancy supervised at future observed clocks; future observations never enter inputs',
        ADE_FDE='Not measured: neural decoder outputs unordered occupancy grids without track identities; deterministic CV center errors alone would be a different task',
        limitations=['One b000 branch/root, not 200k successor targets','5s censoring and selection support reported explicitly','Center occupancy is not full physical state, collision footprints, or future visibility','Changed frozen backbone and scene-disjoint data prevent treating old successor numbers as controlled data-only scaling'],
        seconds=time.time()-started))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('cv','cv_fast','planning','successor_prepare','successor_panel_prepare','successor_train','successor_audit','stop_optional'))
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--subset',choices=('30k','all'),default='all')
    p.add_argument('--successor-panel',choices=('bounded','full'),default='bounded')
    args=p.parse_args()
    if args.command=='stop_optional':
        import os,signal
        stopped=[]
        for path in Path('/proc').iterdir():
            if not path.name.isdigit() or int(path.name)==os.getpid():continue
            try:
                cmd=(path/'cmdline').read_bytes().split(b'\0')
                if (b'scripts/eventwm_v4_evaluate.py' in cmd and
                    any(c in cmd for c in (b'cv',b'successor_prepare')) and
                    (path/'cwd').resolve()==WORKTREE):
                    os.kill(int(path.name),signal.SIGTERM)
                    stopped.append(dict(pid=int(path.name),command=[c.decode(errors='replace') for c in cmd if c]))
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        save(args.output/'optional_research_stopped.json',dict(timestamp=time.time(),processes=stopped,
             reason='Conservative stop of optional research after publication slowdown; collector untouched; no causal attribution established',production=production_sample()))
        print(json.dumps(stopped),flush=True)
        return
    torch.set_num_threads(1)
    seed_all(20261012)
    globals()[args.command](args)


if __name__=='__main__':main()
