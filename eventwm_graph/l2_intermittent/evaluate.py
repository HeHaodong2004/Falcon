"""Frozen development science protocol. All physical ancestry stays OFFLINE."""
import argparse
import collections
import gzip
import json
import math
import time
import numpy as np
from scipy.spatial.distance import cdist
from eventwm_graph.l2_partial.state import Hypothesis
from eventwm_graph.l2_partial.structural_state import StructuralBeliefState
from eventwm_graph.l2_120k.model import load_model
from eventwm_graph.l2_120k.evaluate import restore
from .common import OUT, OLD, SNAP, DOC, HORIZONS, age_bin, seed_for, read, centers, canonical, truth_at, dump, sha
from .model import IntermittentWorldModel, freeze_graph
from .baselines import MemoryCV

PATH_TIMES=np.array([0.,.25,.5,.75,1.,1.5,2.,3.])
SCORE_TIMES=np.unique(np.r_[PATH_TIMES,np.array(HORIZONS)-.048])


def energy(x,y):
    x=np.asarray(x).reshape(len(x),-1);y=np.asarray(y).ravel();W=len(x)
    return float(np.linalg.norm(x-y,axis=-1).mean()-cdist(x,x).sum()/(2*W*(W-1)))


def exact_ids(ob,times,truth):
    y=truth_at(times,truth,float(ob['timestamp']));result={}
    if y is None:return result
    for z in centers(ob):
        d=np.linalg.norm(y-z,axis=-1)
        if len(d) and d.min()<2e-4 and np.count_nonzero(d<2e-4)==1:result[int(d.argmin())]=z
    return result


def extend_ancestry(b,ancestry,times,truth,positive_only=True):
    """OFFLINE first-anchor labels; conflicting first assignments are unsupported.

    Existing physical anchors never change, including when a numeric slot is
    reused with a different birth. A new origin/person label requires the same
    genuinely observed (slot,birth) in EVERY positive-weight same-origin fork.
    """
    y=truth_at(times,truth,b.timestamp)
    if y is None:return
    proposals=collections.defaultdict(set);component_rows=collections.defaultdict(list)
    existing={(origin,value[0]) for (origin,slot),value in ancestry.items() if value[0]>=0}
    for c in b.components:
        if c.weight<=0:continue
        seen=collections.defaultdict(list)
        for i,s in enumerate(c.slots):
            if abs(c.last_seen[i]-b.timestamp)>1e-7:continue
            d=np.linalg.norm(y-c.last_position[i],axis=-1)
            if not len(d) or d.min()>=2e-4 or np.count_nonzero(d<2e-4)!=1:continue
            p=int(d.argmin());birth=float(c.birth[i]);slot=int(s)
            seen[p].append((slot,birth))
            if (c.origin,slot) not in ancestry:proposals[(c.origin,slot)].add((p,birth,float(b.timestamp)))
        component_rows[c.origin].append(seen)
    for key,values in proposals.items():
        origin,slot=key;people={v[0] for v in values};births={v[1] for v in values}
        valid=len(people)==len(births)==1
        p=next(iter(people)) if valid else -2
        birth=min(births)
        if valid and (origin,p) not in existing:
            expected=(slot,birth)
            valid=all(len(r.get(p,()))==1 and r[p][0]==expected for r in component_rows[origin])
        ancestry[key]=(p if valid else -2,birth,float(b.timestamp))


def lineage_anchor(ancestry,origin,physical):
    # First-established correspondence stays authoritative. Simultaneous
    # different first anchors are ambiguous, rather than selected by slot ID.
    choices=[((value[2] if len(value)>2 else -float('inf')),value[1],int(slot))
             for (source,slot),value in ancestry.items() if source==origin and value[0]==physical]
    if not choices:return None
    first=min(c[0] for c in choices)
    candidates={c[2] for c in choices if c[0]==first}
    return next(iter(candidates)) if len(candidates)==1 else None


def lineage_slot(c,ancestry,physical):
    preferred=lineage_anchor(ancestry,c.origin,physical)
    if preferred is None:return None
    if preferred is None:return None
    expected_birth=ancestry[(c.origin,preferred)][1]
    found=np.flatnonzero((c.slots==preferred)&np.isclose(c.birth,expected_birth,rtol=0,atol=1e-8))
    return int(found[0]) if len(found)==1 else None


def to_previous(model,b):
    hs=[]
    for c in b.components:
        h=Hypothesis(c.weight,c.mean,c.covariance,c.last_seen,c.slots,c.hidden_pmf,
                     height=np.full(c.n,b.graph.height),visibility_mode=c.visibility_mode,birth_time=c.birth,
                     birth_position=c.last_position)
        h.diagnostic_origin=c.origin;hs.append(h)
    state=StructuralBeliefState(b.graph,hs,b.timestamp,b.robot_pose,model_version=model.model_version,
                               evidence={'belief_role':'actual_continuous'})
    return model.prepare(state)


def score(model,b,ancestry,physical_ids,last_times,times,truth,meta,stream,old=False):
    if not physical_ids:return []
    if times[-1]<b.timestamp+.5-1e-8:
        return [dict(**meta,stream=stream,physical=p,age=b.timestamp-last_times[p],age_bin=age_bin(b.timestamp-last_times[p]),H=H,status='truth_censored') for p in physical_ids for H in HORIZONS]
    if old:
        future=model.predict_shared(to_previous(model,b),3.,32,seed_for(meta['root_id'],stream),times=SCORE_TIMES)
        world_components=[future.source.hypotheses[i] for i in future.world_source_indices]
        if len(future.source.hypotheses)!=len(b.components):raise ValueError('frozen dynamics changed ancestry order')
        origins=[b.components[i].origin for i in future.world_source_indices]
        slots=[c.slot_ids for c in world_components]
    else:
        future=model.predict_shared(b,3.,32,seed_for(meta['root_id'],stream),times=SCORE_TIMES)
        world_components=future.endpoints[0];origins=[c.origin for c in world_components];slots=[c.slots for c in world_components]
    records=[];joint=collections.defaultdict(list)
    for physical in sorted(physical_ids):
        draws=[];vel=[];st=[];missing=0;duplicates=0
        represented_missing_mass=sum(c.weight for c in b.components if c.weight>0 and lineage_slot(c,ancestry,physical) is None)
        conflicting_anchor_mass=sum(c.weight for c in b.components if c.weight>0 and any(ancestry.get((c.origin,int(s)),(-1,))[0]==-2 for s in c.slots))
        for w,(origin,ss,c) in enumerate(zip(origins,slots,world_components)):
            world_birth=c.birth_time if old else c.birth
            found=[(ancestry[(origin,int(s))][1],int(s),i) for i,s in enumerate(ss)
                   if ancestry.get((origin,int(s)),(-1,))[0]==physical
                   and abs(float(world_birth[i])-ancestry[(origin,int(s))][1])<=1e-8]
            duplicates+=len(found)>1
            anchors=[((value[2] if len(value)>2 else -float('inf')),value[1],int(slot))
                     for (anchor_origin,slot),value in ancestry.items() if anchor_origin==origin and value[0]==physical]
            preferred=lineage_anchor(ancestry,origin,physical)
            chosen_anchor=next((anchor for anchor in anchors if anchor[2]==preferred),None) if preferred is not None else None
            selected=np.flatnonzero((ss==chosen_anchor[2])&np.isclose(world_birth,chosen_anchor[1],rtol=0,atol=1e-8)) if chosen_anchor is not None else []
            if len(selected)!=1:missing+=1;continue
            i=int(selected[0])
            draws.append(future.positions[w,:,i]);vel.append(future.velocities[w,:,i])
            if not old:st.append(future.stopped[w,:,i])
        age=max(0.,b.timestamp-last_times[physical]);agegroup=age_bin(age)
        for H in HORIZONS:
            y=truth_at(times,truth,b.timestamp+H)
            if y is None:
                records.append(dict(**meta,stream=stream,physical=physical,age=age,age_bin=agegroup,H=H,status='truth_censored'));continue
            base=dict(**meta,stream=stream,physical=physical,age=age,age_bin=agegroup,H=H,missing_worlds=missing,duplicate_worlds=duplicates,
                      represented_missing_mass=float(represented_missing_mass),conflicting_anchor_mass=float(conflicting_anchor_mass),
                      one_second=.75-1e-8<=age<=1.25+1e-8)
            if missing or represented_missing_mass>0:
                records.append(dict(**base,status='ancestry_unsupported'));continue
            x=np.asarray(draws);vi=np.asarray(vel);ti=int(np.argmin(abs(future.times-(b.timestamp+H))));pi=int(np.argmin(abs(future.times-(b.timestamp+H-.048))))
            path_idx=np.flatnonzero((SCORE_TIMES<=H+1e-8)&np.isin(SCORE_TIMES,PATH_TIMES))
            yt=np.array([truth_at(times,truth,float(future.times[j]))[physical] for j in path_idx]);path=x[:,path_idx]/math.sqrt(len(path_idx))
            mean=x[:,ti].mean(0);error=float(np.linalg.norm(mean-y[physical]));rad=np.linalg.norm(x[:,ti]-mean,axis=-1)
            speed=np.linalg.norm((x[:,ti]-x[:,pi])/.048,axis=-1);y0=truth_at(times,truth,b.timestamp+H-.048)[physical];tv=(y[physical]-y0)/.048;tspeed=float(np.linalg.norm(tv))
            pstop=float(np.mean(speed<.1));dv=(x[:,ti]-x[:,pi])/.048;dmean=dv.mean(0);rvel=np.linalg.norm(dv-dmean,axis=-1)
            direction_error=None;direction_brier=None
            if tspeed>.1:
                trueangle=np.arctan2(tv[1],tv[0]);ang=np.arctan2(dv[:,1],dv[:,0]);diff=np.angle(np.exp(1j*(ang-trueangle)))
                direction_error=float(abs(np.angle(np.exp(1j*(np.arctan2(dmean[1],dmean[0])-trueangle)))))
                bins=np.floor((ang+np.pi)/(2*np.pi)*8).astype(int)%8;yb=int(np.floor((trueangle+np.pi)/(2*np.pi)*8))%8
                prob=np.bincount(bins,minlength=8)/len(bins);target=np.eye(8)[yb];direction_brier=float(((prob-target)**2).sum())
            vals=dict(status='supported',tracked_es=energy(x[:,ti],y[physical]),path_es=energy(path,yt/math.sqrt(len(path_idx))),
                future_path_es=energy(x[:,path_idx[1:]]/math.sqrt(len(path_idx)-1),yt[1:]/math.sqrt(len(path_idx)-1)),
                endpoint_error=error,coverage50=float(error<=np.quantile(rad,.5)),coverage90=float(error<=np.quantile(rad,.9)),
                radius50=float(np.quantile(rad,.5)),radius90=float(np.quantile(rad,.9)),
                stop_probability=pstop,observed_stop=float(tspeed<.1),stop_brier=(pstop-float(tspeed<.1))**2,
                mode_stop_probability=float(np.mean(np.asarray(st)[:,ti])) if st else None,
                velocity_error=float(np.linalg.norm(dmean-tv)),velocity_coverage90=float(np.linalg.norm(dmean-tv)<=np.quantile(rvel,.9)),
                direction_error=direction_error,direction_brier=direction_brier,
                diversity=float(cdist(path.reshape(32,-1),path.reshape(32,-1)).sum()/(32*31)),
                variance_trace=float(np.var(x[:,ti],axis=0).sum()))
            records.append(dict(**base,**vals));joint[H].append((physical,path,yt/math.sqrt(len(path_idx))))
    for H,items in joint.items():
        if len(items)>1:
            records.append(dict(**meta,stream=stream,H=H,status='joint_supported',people=len(items),
                                joint_path_es=energy(np.concatenate([p for _,p,_ in items],axis=-1),np.concatenate([y for _,_,y in items],axis=-1))))
    return records


def mask_advance(model,b,obs,end_index):
    # Consolidate physical propagation using its tested semigroup while applying
    # each independently arrived static overlay in chronological order.
    graph=b.graph
    for ob in obs[:end_index+1]:graph=graph.update(ob.get('static_updates',[]))
    ob=obs[end_index];masked={**ob,'tracks':[],'human_sensor_available':False,'static_updates':[]}
    return model.observe(b,masked).fork(graph=freeze_graph(graph))


def run(limit=0,start_index=0,stop_index=0):
    import torch
    torch.set_num_threads(1)
    if not limit and not (OUT/'training/training.json').exists():raise ValueError('fit must finish before final candidate evaluation')
    inputs=[SNAP/'manifest.json',OLD/'dev.npy',OLD/'motion.json',OLD/'population_prior.json',OLD/'training/checkpoint.pt',
            OUT/'motion.json',OUT/'training/checkpoint.pt',OUT/'training/epoch0.pt',OUT/'training/training.json',
            DOC.parent/'graph_wm_120k_20261021/artifacts/population_prior.json',
            DOC.parent/'graph_wm_120k_20261021/artifacts/motion.json',
            DOC.parent/'graph_wm_120k_20261021/artifacts/checkpoints/frozen_observation_fit.json',
            DOC.parent/'partial_graph_wm_structural_20261021/artifacts/checkpoints/native_events_v5_scene.json']
    inputs_before={str(p):sha(p) for p in inputs}
    entries={r['root_id']:r for r in json.loads((SNAP/'manifest.json').read_text())['rows']
             if r['split']=='dev' and r['source_split']=='train'}
    law=json.loads((OUT/'motion.json').read_text());models={
        'A_CV':MemoryCV(0.,law=law),'B_MemoryCV':MemoryCV(.16,law=law),'B_HistoricalCV':MemoryCV(.64,law=law),
        'D_StopMove':IntermittentWorldModel.load(OUT/'training/checkpoint.pt'),
        'E_Initial':IntermittentWorldModel.load(OUT/'training/epoch0.pt')}
    previous=load_model(OLD/'training/checkpoint.pt',json.loads((OLD/'population_prior.json').read_text()),
                        str(OLD/'training/frozen_observation_fit.json') if (OLD/'training/frozen_observation_fit.json').exists() else 'docs/l2/graph_wm_120k_20261021/artifacts/checkpoints/frozen_observation_fit.json',
                        json.loads((OLD/'motion.json').read_text()))
    bootstrap=load_model();rows=list(np.load(OLD/'dev.npy',allow_pickle=True));total_roots=len(rows)
    stop_index=stop_index or total_roots
    if not 0<=start_index<stop_index<=total_roots:raise ValueError('invalid immutable root shard')
    rows=rows[start_index:stop_index][:limit or None]
    name='evaluation_smoke' if limit else ('evaluation_shard_%04d_%04d'%(start_index,stop_index) if start_index or stop_index!=total_roots else 'evaluation')
    dest=OUT/name;dest.mkdir(exist_ok=False)
    source_paths=[__import__('pathlib').Path(__file__).with_name(name) for name in ('evaluate.py','common.py','model.py','dynamics.py','neural.py','observations.py','baselines.py')]
    source_paths += [DOC/'PROTOCOL.md',DOC/'EVALUATION_FREEZE.md']
    source_before={p.name:sha(p) for p in source_paths};(dest/'sources').mkdir()
    for p in source_paths:(dest/'sources'/p.name).write_bytes(p.read_bytes())
    start=time.perf_counter();failures=[];masks=[];reappearance_eligibility=[];allcounts=collections.Counter()
    with gzip.open(dest/'forecasts.jsonl.gz','wt') as fout,gzip.open(dest/'reobservation.jsonl.gz','wt') as rout,gzip.open(dest/'beliefs.jsonl.gz','wt') as bout,gzip.open(dest/'update_gains.jsonl.gz','wt') as gout:
        def emit(records):
            for r in records:fout.write(json.dumps(r,allow_nan=False)+'\n')
        def emit_failure(meta,targets,last,timestamp,stream):
            emit([dict(**meta,stream=stream,physical=p,age=max(0.,timestamp-last[p]),age_bin=age_bin(max(0.,timestamp-last[p])),H=H,
                       status='model_failed' if timestamp+H<=times[-1]+1e-8 else 'truth_censored') for p in targets for H in HORIZONS])
        def emit_belief(b,ancestry,last,ids,meta,stream,event):
            # Physical count and correspondence are OFFLINE diagnostics only.
            probability=0.;missing=0.;duplicates=0.;preferred_missing=0.
            for c in b.components:
                remaining=truth.shape[1]-c.n
                if 0<=remaining<len(c.hidden_pmf):probability+=c.weight*c.hidden_pmf[remaining]
                for p in last:
                    n=sum(ancestry.get((c.origin,int(s)),(-1,))[0]==p and abs(float(c.birth[i])-ancestry[(c.origin,int(s))][1])<=1e-8 for i,s in enumerate(c.slots))
                    preferred_missing+=c.weight*(lineage_slot(c,ancestry,p) is None)
                    missing+=c.weight*(n==0);duplicates+=c.weight*max(0,n-1)
            exact_errors=[float(np.linalg.norm(c.mean[i,:2]-c.last_position[i])) for c in b.components for i in range(c.n) if abs(c.last_seen[i]-b.timestamp)<1e-7]
            bout.write(json.dumps(dict(**meta,stream=stream,timestamp=b.timestamp,event=event,truth_count=truth.shape[1],
                true_count_probability=float(probability),true_count_support=bool(probability>0),
                count_nll=float(-math.log(probability)) if probability>0 else 'INFINITE_WITH_ZERO_SUPPORT',
                previously_observed_people=len(last),expected_missing_ancestries=float(missing),expected_duplicate_ancestries=float(duplicates),expected_missing_preferred_ancestries=float(preferred_missing),
                ancestry_diagnostic_semantics='any-label counts require matching birth; preferred missing uses fixed first anchor',
                max_arrived_coordinate_error=max(exact_errors,default=None),localized_count_mean=float(sum(c.weight*c.n for c in b.components)),
                components=len(b.components),association_entropy=b.diagnostics.get('association_entropy'),
                resampling_omitted_probability=b.diagnostics.get('resampling_omitted_probability'),
                enumerated_candidates=b.diagnostics.get('enumerated_candidates'),
                beam_retained_fraction_of_enumerated=b.diagnostics.get('beam_retained_fraction_of_enumerated')),allow_nan=False)+'\n')
        def gain_failure(meta,targets,ob,ids,reappeared,stream):
            for p,last_t in targets.items():
                category=('unavailable' if not ob.get('human_sensor_available',True) or not ob.get('observation_valid',True)
                          else 'reappearance' if p in reappeared else 'positive' if p in ids else 'available_no_relevant_detection')
                for H in HORIZONS:
                    status='model_failed' if ob['timestamp']+H<=times[-1]+1e-8 else 'truth_censored'
                    gout.write(json.dumps(dict(**meta,stream=stream,physical=p,H=H,age_before=float(ob['timestamp']-last_t),
                        category=category,frame_has_positive=bool(len(centers(ob))),prior_status=status,posterior_status=status),allow_nan=False)+'\n')
        for ri,row in enumerate(rows):
            root=row['root_id'];folder=SNAP/'roots'/root
            entry=entries[root]
            if entry['scene_id']!=row['scene_id']:raise ValueError('cache scene differs from immutable manifest')
            for name in ('legal_root.json.gz','queries.json.gz','outcomes.json.gz','future_observations.json.gz','privileged_future.npz'):
                if sha(folder/name)!=entry['files'][name]:raise ValueError('immutable payload hash changed: '+root+'/'+name)
            legal=read(folder/'legal_root.json.gz');query,obs=canonical(folder)
            with np.load(folder/'privileged_future.npz') as f:times=f['timestamps'].copy();truth=f['positions'][...,[0,2]].copy()
            initial=restore(bootstrap,row,legal['partial_graph']);rootob=legal['history'][-1];rootids=exact_ids(rootob,times,truth)
            now=initial.timestamp;choices={}
            for age in (.5,1.,2.):
                if obs:
                    j=min(range(len(obs)),key=lambda i:abs(obs[i]['timestamp']-now-age))
                    if abs(obs[j]['timestamp']-now-age)<=.06:choices[age]=j
            metadata=dict(root_id=root,scene_id=row['scene_id'])
            for stream in (41,107):
                states={k:m.from_legacy(initial,seed_for(root,stream+10000)) for k,m in models.items()}
                ancestry={k:{} for k in models}
                for k in models:extend_ancestry(states[k],ancestry[k],times,truth)
                for k in models:emit_belief(states[k],ancestry[k],rootids,rootids,dict(**metadata,model=k),stream,'root')
                for k,m in models.items():
                    try:emit(score(m,states[k],ancestry[k],rootids,{j:now for j in rootids},times,truth,dict(**metadata,model=k,cohort='native',decision='root'),stream))
                    except Exception as e:
                        failures.append(dict(**metadata,stream=stream,model=k,where='root',error=repr(e)));emit_failure(dict(**metadata,model=k,cohort='native',decision='root'),rootids,{p:now for p in rootids},now,stream)
                try:emit(score(previous,states['B_MemoryCV'],ancestry['B_MemoryCV'],rootids,{j:now for j in rootids},times,truth,dict(**metadata,model='C_Previous',cohort='native',decision='root'),stream,True))
                except Exception as e:
                    failures.append(dict(**metadata,stream=stream,model='C_Previous',where='root',error=repr(e)));emit_failure(dict(**metadata,model='C_Previous',cohort='native',decision='root'),rootids,{p:now for p in rootids},now,stream)
                # Separate artificial missing-evidence experiment, exact legal
                # robot/static arrivals retained. No observation is a negative.
                for age,j in choices.items():
                    if not rootids:continue
                    ob=obs[j];last={p:now for p in rootids}
                    if stream==41:masks.append(dict(**metadata,requested_age=age,current_timestamp=ob['timestamp'],masked_timestamps=[o['timestamp'] for o in obs[:j+1]],root_positive_ids=list(rootids),human_sensor_available=False))
                    for k,m in models.items():
                        try:
                            aged=mask_advance(m,states[k],obs,j)
                            emit(score(m,aged,ancestry[k],rootids,last,times,truth,dict(**metadata,model=k,cohort='masked',decision=str(age)),stream))
                        except Exception as e:
                            failures.append(dict(**metadata,stream=stream,model=k,where='mask'+str(age),error=repr(e)));emit_failure(dict(**metadata,model=k,cohort='masked',decision=str(age)),rootids,last,float(ob['timestamp']),stream)
                    try:
                        aged=mask_advance(models['B_MemoryCV'],states['B_MemoryCV'],obs,j)
                        emit(score(previous,aged,ancestry['B_MemoryCV'],rootids,last,times,truth,dict(**metadata,model='C_Previous',cohort='masked',decision=str(age)),stream,True))
                    except Exception as e:
                        failures.append(dict(**metadata,stream=stream,model='C_Previous',where='mask',error=repr(e)));emit_failure(dict(**metadata,model='C_Previous',cohort='masked',decision=str(age)),rootids,last,float(ob['timestamp']),stream)
                # Genuine canonical sequence. Every target ever observed after
                # native root is eligible, even if it never reappears again.
                last={j:now for j in rootids};last_frame=now;seen_pairs=set();seen_events=set();active={k:states[k] for k in models if k not in ('E_Initial','A_CV')}
                # Plain CV is a forecast comparator on the same legal-history
                # kinematic estimates. A zero-acceleration, zero-sensor-noise
                # FILTER becomes singular after two exact centers and cannot
                # assimilate a real turn; that is not the intended CV baseline.
                active['A_CV']=states['A_CV']
                for oi,ob in enumerate(obs):
                    if ob['timestamp']>now+3.+.06:break
                    ids=exact_ids(ob,times,truth);reappeared={j:ob['timestamp']-last[j] for j in ids if j in last and ob['timestamp']-last[j]>ob['timestamp']-last_frame+1e-7}
                    eligible={p:g for p,g in reappeared.items() if g>=.25-1e-8 and (p,age_bin(g)) not in seen_events}
                    for p,g in eligible.items():
                        seen_events.add((p,age_bin(g)))
                        reappearance_eligibility.append(dict(**metadata,stream=stream,physical=p,gap=g,gap_bin=age_bin(g),index=oi,timestamp=ob['timestamp'],future_available={str(H):bool(ob['timestamp']+H<=times[-1]+1e-8) for H in HORIZONS}))
                    completed_pairs=set()
                    priors=dict(active);previous_targets=dict(last);ancestry_before={k:dict(v) for k,v in ancestry.items()}
                    for k in list(active):
                        m=models[k];prior=active[k]
                        try:
                            if k=='A_CV' and 'B_MemoryCV' not in active:raise ValueError('common CV history filter failed')
                            post=(active['B_MemoryCV'].fork(model_version=m.model_version) if k=='A_CV' else m.observe(prior,ob))
                            for physical,gap in eligible.items():
                                key=(physical,age_bin(gap),k)
                                if gap<.25-1e-8 or key in seen_pairs:continue
                                seen_pairs.add(key)
                                before=(models['B_MemoryCV'].observe(priors['B_MemoryCV'],{**ob,'tracks':[],'human_sensor_available':False}).fork(model_version=m.model_version)
                                        if k=='A_CV' else m.observe(prior,{**ob,'tracks':[],'human_sensor_available':False}))
                                targetlast={physical:float(ob['timestamp'])}
                                pairmeta=dict(**metadata,model=k,cohort='reobservation',decision=str(oi))
                                pr=score(m,before,ancestry[k],[physical],targetlast,times,truth,pairmeta,stream)
                                po=score(m,post,ancestry[k],[physical],targetlast,times,truth,pairmeta,stream)
                                good=split=merge=dup=missing_antecedent=0.
                                for c in post.components:
                                    born=[i for i in range(c.n) if c.birth[i]>=post.timestamp-1e-7 and np.linalg.norm(c.last_position[i]-ids[physical])<2e-4]
                                    if born:split+=c.weight
                                    slot=lineage_slot(c,ancestry[k],physical)
                                    if slot is None:
                                        missing_antecedent+=c.weight;continue
                                    if abs(c.last_seen[slot]-post.timestamp)<1e-7:
                                        if np.linalg.norm(c.last_position[slot]-ids[physical])<2e-4:good+=c.weight
                                        else:merge+=c.weight
                                    if born and slot not in born:dup+=c.weight
                                for a,brec in zip(pr,po):
                                    rec=dict(**pairmeta,stream=stream,physical=physical,gap=gap,gap_bin=age_bin(gap),H=a['H'],association_accuracy=good,false_split=split,false_merge=merge,duplicate=dup,
                                        prior_status=a['status'],posterior_status=brec['status'],association_entropy=post.diagnostics.get('association_entropy',0.),
                                        eligible_event=True,target_available=bool(ob['timestamp']+a['H']<=times[-1]+1e-8),reappearance_recovery=good,missing_antecedent_mass=missing_antecedent)
                                    for metric in ('tracked_es','path_es','future_path_es'):
                                        if metric in a and metric in brec:rec[metric+'_gain']=a[metric]-brec[metric]
                                    for metric in ('coverage50','coverage90'):
                                        if metric in a and metric in brec:rec[metric+'_change']=brec[metric]-a[metric]
                                    rout.write(json.dumps(rec,allow_nan=False)+'\n')
                                completed_pairs.add((k,physical))
                            active[k]=post;extend_ancestry(post,ancestry[k],times,truth)
                            diagnostic_last={**last,**{p:float(ob['timestamp']) for p in ids}}
                            event=('unavailable' if not ob.get('human_sensor_available',True) or not ob.get('observation_valid',True)
                                   else 'reappearance' if reappeared else 'positive' if len(centers(ob)) else 'available_empty')
                            emit_belief(post,ancestry[k],diagnostic_last,ids,dict(**metadata,model=k),stream,event)
                        except Exception as e:
                            failures.append(dict(**metadata,stream=stream,model=k,where='native_observe',index=oi,error=repr(e)));del active[k]
                    for k in models:
                        if k=='E_Initial':continue
                        for physical,gap in eligible.items():
                            if (k,physical) in completed_pairs:continue
                            for H in HORIZONS:
                                rout.write(json.dumps(dict(**metadata,stream=stream,model=k,cohort='reobservation',decision=str(oi),physical=physical,gap=gap,gap_bin=age_bin(gap),H=H,
                                    prior_status='model_failed',posterior_status='model_failed',eligible_event=True,target_available=bool(ob['timestamp']+H<=times[-1]+1e-8),reappearance_recovery=0.))+'\n')
                    for j in ids:last[j]=float(ob['timestamp'])
                    if oi in choices.values():
                        for k in models:
                            if k=='E_Initial':continue
                            smeta=dict(**metadata,model=k,cohort='native',decision=str(oi))
                            if k not in active:
                                emit_failure(smeta,last,last,float(ob['timestamp']),stream);gain_failure(smeta,previous_targets,ob,ids,reappeared,stream);continue
                            try:
                                post_records=score(models[k],active[k],ancestry[k],last,last,times,truth,smeta,stream);emit(post_records)
                            except Exception as e:
                                failures.append(dict(**metadata,stream=stream,model=k,where='native_score',index=oi,error=repr(e)));emit_failure(smeta,last,last,float(ob['timestamp']),stream)
                                gain_failure(smeta,previous_targets,ob,ids,reappeared,stream);continue
                            if previous_targets:
                                try:
                                    before=(models['B_MemoryCV'].observe(priors['B_MemoryCV'],{**ob,'tracks':[],'human_sensor_available':False}).fork(model_version=models[k].model_version)
                                            if k=='A_CV' else models[k].observe(priors[k],{**ob,'tracks':[],'human_sensor_available':False}))
                                    prior_records=score(models[k],before,ancestry_before[k],previous_targets,previous_targets,times,truth,smeta,stream)
                                    gain_post_records=post_records if ancestry_before[k]==ancestry[k] else score(models[k],active[k],ancestry_before[k],previous_targets,previous_targets,times,truth,smeta,stream)
                                    posterior={(r.get('physical'),r['H']):r for r in gain_post_records if r['status']!='joint_supported'}
                                    for pr in prior_records:
                                        if pr['status']=='joint_supported':continue
                                        po=posterior[(pr['physical'],pr['H'])];physical=pr['physical']
                                        category=('unavailable' if not ob.get('human_sensor_available',True) or not ob.get('observation_valid',True)
                                                  else 'reappearance' if physical in reappeared else 'positive' if physical in ids else 'available_no_relevant_detection')
                                        gain=dict(**smeta,stream=stream,physical=physical,H=pr['H'],age_before=float(ob['timestamp']-previous_targets[physical]),
                                            category=category,frame_has_positive=bool(len(centers(ob))),prior_status=pr['status'],posterior_status=po['status'])
                                        for metric in ('tracked_es','path_es','future_path_es'):
                                            if metric in pr and metric in po:gain[metric+'_gain']=pr[metric]-po[metric]
                                        for metric in ('coverage50','coverage90'):
                                            if metric in pr and metric in po:gain[metric+'_change']=po[metric]-pr[metric]
                                        gout.write(json.dumps(gain,allow_nan=False)+'\n')
                                except Exception as e:
                                    failures.append(dict(**metadata,stream=stream,model=k,where='same_target_gain',index=oi,error=repr(e)));gain_failure(smeta,previous_targets,ob,ids,reappeared,stream)
                        if 'B_MemoryCV' in active:
                            try:emit(score(previous,active['B_MemoryCV'],ancestry['B_MemoryCV'],last,last,times,truth,dict(**metadata,model='C_Previous',cohort='native',decision=str(oi)),stream,True))
                            except Exception as e:
                                failures.append(dict(**metadata,stream=stream,model='C_Previous',where='native_score',index=oi,error=repr(e)));emit_failure(dict(**metadata,model='C_Previous',cohort='native',decision=str(oi)),last,last,float(ob['timestamp']),stream)
                        else:emit_failure(dict(**metadata,model='C_Previous',cohort='native',decision=str(oi)),last,last,float(ob['timestamp']),stream)
                    last_frame=float(ob['timestamp'])
            if (ri+1)%25==0:
                print(json.dumps(dict(roots=ri+1,failures=len(failures),seconds=round(time.perf_counter()-start,1))),flush=True);fout.flush();rout.flush()
                dump(dest/'progress.json',dict(roots=ri+1,failures=failures,seconds=time.perf_counter()-start))
    dump(dest/'failures.json',failures);dump(dest/'mask_manifest.json',masks);dump(dest/'reappearance_eligibility.json',reappearance_eligibility)
    dump(dest/'configuration.json',dict(roots=len(rows),streams=[41,107],worlds=32,bootstrap_stream_offset=10000,path_times=PATH_TIMES.tolist(),
         root_range=[start_index,stop_index],root_ids=[r['root_id'] for r in rows],total_eligible_dev_roots=total_roots,
         score_times=SCORE_TIMES.tolist(),old_baseline='frozen 43187f24 network+motion, common MemoryCV legal-history filtering states; no checkpoint modification',
         cv_baseline='A: zero-acceleration forecasts on common B MemoryCV legal-history actual estimates; B: .16 and .64 acceleration diffusion filters/forecasts',
         snapshot_sha256=sha(SNAP/'manifest.json'),checkpoint_sha256=sha(OUT/'training/checkpoint.pt'),sources=source_before,
         inputs_before=inputs_before,inputs_stable=inputs_before=={str(p):sha(p) for p in inputs},
         payload_files_hash_verified_per_root=5,
         ancestry_policy='birth-qualified common genuine first explainer in all positive same-origin forks; immutable earliest preferred anchor; no duplicate rescue; pre-arrival gain ancestry',
         annotation_policy='raw post-root identity rows require the separately preserved annotation-accounting rule before final native summaries',
         executed_reference_evaluator_sha256='a278381a264b564eb51683469c89d307a41268805292ac856cec611434f3c184',
         source_stable=all(sha(p)==source_before[p.name] for p in source_paths),seconds=time.perf_counter()-start))
    summarize(dest)


def paired_ci(records,key):
    root=collections.defaultdict(list)
    for r in records:
        value=r.get(key)
        if value is not None and math.isfinite(value):root[(r['scene_id'],r['root_id'])].append(value)
    scene=collections.defaultdict(list)
    for (s,r),v in root.items():scene[s].append(float(np.mean(v)))
    a=np.array([np.mean(v) for _,v in sorted(scene.items())]);n=len(a)
    if not n:return dict(mean=None,lo=None,hi=None,roots=0,scenes=0)
    rng=np.random.default_rng(73);boot=a[rng.integers(0,n,size=(1000,n))].mean(1)
    return dict(mean=float(a.mean()),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)),roots=len(root),scenes=n)


def summarize(dest):
    with gzip.open(dest/'forecasts.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
    with gzip.open(dest/'reobservation.jsonl.gz','rt') as f:pairs=[json.loads(x) for x in f]
    updates=[]
    if (dest/'update_gains.jsonl.gz').exists():
        with gzip.open(dest/'update_gains.jsonl.gz','rt') as f:updates=[json.loads(x) for x in f]
    metrics=['tracked_es','path_es','future_path_es','endpoint_error','coverage50','coverage90','radius90','stop_brier','stop_probability','observed_stop','mode_stop_probability','velocity_error','velocity_coverage90','direction_brier','diversity','variance_trace']
    grouped=collections.defaultdict(list)
    for r in rows:
        if r['status']=='joint_supported':continue
        grouped[(r['model'],r['cohort'],r.get('age_bin'),r['H'])].append(r)
    cells=[]
    for key,part in sorted(grouped.items()):
        cells.append(dict(model=key[0],cohort=key[1],age_bin=key[2],H=key[3],counts=dict(collections.Counter(r['status'] for r in part)),
                          metrics={m:paired_ci(part,m) for m in metrics}))
    # Match EXACT roots, streams, physical lineages, timestamps, and horizons.
    keyed={(r['model'],r['cohort'],r['root_id'],r['stream'],r.get('physical'),r['decision'],r['H']):r for r in rows if r['status']=='supported'}
    contrasts=[];cell_contrasts=[]
    for baseline in ('A_CV','B_MemoryCV','B_HistoricalCV','C_Previous','E_Initial'):
        for cohort in ('native','masked'):
            selected=[]
            for key,d in keyed.items():
                if key[0]!='D_StopMove' or key[1]!=cohort:continue
                other=keyed.get((baseline,)+key[1:])
                if other:
                    selected.append(dict(**{k:d[k] for k in ('root_id','scene_id','age','age_bin','H','one_second')},
                        tracked_es=d['tracked_es']-other['tracked_es'],path_es=d['path_es']-other['path_es']))
            for scope in ('all','visible','one_second'):
                part=[r for r in selected if scope=='all' or (scope=='visible' and r['age']<=1e-7) or (scope=='one_second' and r['one_second'] and r['H']<=2.)]
                contrasts.append(dict(baseline=baseline,cohort=cohort,scope=scope,tracked_es=paired_ci(part,'tracked_es'),path_es=paired_ci(part,'path_es')))
            for agegroup,H in sorted({(r['age_bin'],r['H']) for r in selected}):
                part=[r for r in selected if r['age_bin']==agegroup and r['H']==H]
                cell_contrasts.append(dict(baseline=baseline,cohort=cohort,age_bin=agegroup,H=H,tracked_es=paired_ci(part,'tracked_es'),path_es=paired_ci(part,'path_es')))
    gains=[]
    for model in sorted({r['model'] for r in pairs}):
        for gap in ['all']+sorted({r['gap_bin'] for r in pairs}):
            for H in HORIZONS:
                part=[r for r in pairs if r['model']==model and r['H']==H and (gap=='all' or r['gap_bin']==gap)]
                if part:gains.append(dict(model=model,gap_bin=gap,H=H,statuses=dict(collections.Counter(r['prior_status']+'/'+r['posterior_status'] for r in part)),
                    metrics={m:paired_ci(part,m) for m in ('tracked_es_gain','path_es_gain','future_path_es_gain','coverage50_change','coverage90_change','association_accuracy','false_split','false_merge','duplicate','association_entropy','reappearance_recovery','missing_antecedent_mass')}))
    update_summary=[]
    for model,category,H in sorted({(r['model'],r['category'],r['H']) for r in updates}):
        part=[r for r in updates if (r['model'],r['category'],r['H'])==(model,category,H)]
        update_summary.append(dict(model=model,category=category,H=H,statuses=dict(collections.Counter(r['prior_status']+'/'+r['posterior_status'] for r in part)),
            metrics={m:paired_ci(part,m) for m in ('tracked_es_gain','path_es_gain','future_path_es_gain','coverage50_change','coverage90_change')}))
    dump(dest/'summary.json',dict(cells=cells,contrasts=contrasts,cell_contrasts=cell_contrasts,reobservation=gains,update_gains=update_summary,
         joint=[dict(model=k[0],cohort=k[1],H=k[2],people=k[3],joint_path_es=paired_ci([r for r in rows if r['status']=='joint_supported' and (r['model'],r['cohort'],r['H'],r['people'])==k],'joint_path_es')) for k in sorted({(r['model'],r['cohort'],r['H'],r['people']) for r in rows if r['status']=='joint_supported'})],
         counts=dict(collections.Counter(r['status'] for r in rows)),raw_rows=len(rows),reobservation_rows=len(pairs)))


def merge_shards():
    import shutil
    shards=sorted(OUT.glob('evaluation_shard_*'));configs=[json.loads((p/'configuration.json').read_text()) for p in shards]
    expected=[r['root_id'] for r in np.load(OLD/'dev.npy',allow_pickle=True)]
    if not configs or [r for c in configs for r in c['root_ids']]!=expected:raise ValueError('shards must partition ALL immutable dev roots exactly in order')
    if not all(c['source_stable'] and c['inputs_stable'] and c['sources']==configs[0]['sources'] and
               c['inputs_before']==configs[0]['inputs_before'] and c['checkpoint_sha256']==configs[0]['checkpoint_sha256'] and
               all(c[k]==configs[0][k] for k in ('worlds','streams','path_times','score_times','bootstrap_stream_offset','ancestry_policy'))
               for c in configs):raise ValueError('incompatible or unstable scientific shards')
    destination=OUT/'evaluation';destination.mkdir(exist_ok=False)
    for name in ('forecasts','reobservation','beliefs','update_gains'):
        with gzip.open(destination/(name+'.jsonl.gz'),'wb') as target:
            for folder in shards:
                with gzip.open(folder/(name+'.jsonl.gz'),'rb') as source:shutil.copyfileobj(source,target)
    for name in ('failures','mask_manifest','reappearance_eligibility'):
        dump(destination/(name+'.json'),[r for folder in shards for r in json.loads((folder/(name+'.json')).read_text())])
    shutil.copytree(shards[0]/'sources',destination/'sources')
    dump(destination/'configuration.json',dict(configs[0],roots=len(expected),root_range=[0,len(expected)],root_ids=expected,
        shards=[dict(path=p.name,configuration_sha256=sha(p/'configuration.json')) for p in shards],
        seconds=sum(c['seconds'] for c in configs),shard_semantics='disjoint immutable root ranges; identical laws/streams/targets; no independent sample inflation'))
    summarize(destination)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=0)
    p.add_argument('--merge-shards',action='store_true');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize(OUT/'evaluation')
    elif a.merge_shards:merge_shards()
    else:run(a.limit,a.start,a.stop)
