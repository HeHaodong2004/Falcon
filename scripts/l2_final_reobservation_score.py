"""One fixed native return/root, scored through actual public inference APIs."""
import argparse,copy,gzip,hashlib,json,math,os,sys,time,types
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'output/l2_final_800k_20261023'
# PYTHONPATH selects either the isolated original package or final repaired code.
from eventwm_graph.l2_intermittent.model import IntermittentWorldModel
from eventwm_graph.l2_intermittent.common import read,canonical
name='eventwm_graph.l2_intermittent._frozen_return_scores';frozen=types.ModuleType(name);sys.modules[name]=frozen
source=BASE/'docs/l2/intermittent_wm_20261022/artifacts/verified_execution/sources/eventwm_graph/l2_intermittent/evaluate.py'
frozen.__file__=str(source);exec(compile(source.read_bytes(),str(source),'exec'),frozen.__dict__)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def count_metrics(b,true_count):
    size=max((c.n+len(c.hidden_pmf) for c in b.components),default=1);pmf=np.zeros(size)
    for c in b.components:pmf[c.n:c.n+len(c.hidden_pmf)]+=c.weight*c.hidden_pmf
    p=float(pmf[true_count]) if true_count<len(pmf) else 0.
    return dict(true_count_probability=p,true_count_support=p>0,count_nll=-math.log(p) if p>0 else 'INFINITE_WITH_ZERO_SUPPORT',
                count_brier=float((pmf*pmf).sum()-2*p+1),count_entropy=float(-sum(v*math.log(v) for v in pmf if v)),
                count_mean=float(pmf@np.arange(len(pmf))),normalization_error=float(abs(pmf.sum()-1)))

def physical_ids(ob,times,truth):
    if not ob.get('observation_valid',True) or not ob.get('human_sensor_available',True):return {},False
    t=float(ob['timestamp']);near=np.flatnonzero(abs(times-t)<1e-7)
    if len(near)!=1:return {},False
    mapped={};clean=True
    for track in ob.get('tracks',[]):
        if not track.get('visible_now',False) or not track.get('valid',True):continue
        if abs(float(track.get('last_seen_time',t))-t)>1e-7:return {},False
        near_actor=np.flatnonzero(np.linalg.norm(truth[near[0]]-np.asarray(track['position']),axis=-1)<2e-4)
        if len(near_actor)!=1:clean=False;continue
        actor=int(near_actor[0])
        if actor in mapped:clean=False
        mapped[actor]=track['track_id']
    return mapped,clean

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['A','B','C'],required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(1);manifest=json.loads((OUT/'reobservation_supplement/manifest.json').read_text());selected=manifest['rows'][:a.limit or None];dest=OUT/'reobservation_supplement'/(('smoke_' if a.limit else 'evaluation_')+a.arm);dest.mkdir(exist_ok=False)
    model=IntermittentWorldModel.load(a.checkpoint);checkpoint_sha=sha(a.checkpoint);model_source=Path(sys.modules['eventwm_graph.l2_intermittent.model'].__file__);model_source_sha=sha(model_source);start=time.perf_counter();failures=[];init=[]
    with gzip.open(dest/'records.jsonl.gz','wt') as output:
        for ri,case in enumerate(selected):
            folder=OUT/'reobservation_supplement/roots'/case['root_id']
            for filename,value in case['files'].items():assert sha(folder/filename)==value
            legal=read(folder/'legal_root.json.gz');query,obs=canonical(folder);assert query['query_id']==case['query_id']
            with np.load(folder/'privileged_future.npz',allow_pickle=False) as f:times=f['timestamps'].copy();xyz=f['positions'].copy()
            truth=xyz[...,[0,2]];actor=case['physical'];index=case['index'];arrived=obs[index]
            assert abs(float(arrived['timestamp'])-case['timestamp'])<1e-7
            unclean=0
            for frame in [legal['history'][-1]]+obs[:index+1]:
                mapped,clean=physical_ids(frame,times,xyz)
                if not clean or set(mapped)!=set(frozen.exact_ids(frame,times,truth)):unclean+=1
            meta={k:case[k] for k in ('root_id','scene_id','physical','time_since_last_certified_positive','source_id_changed','discrete_gap_duration_bounds','native_remaining_horizon')}
            meta.update(model=a.arm,cohort='strict_native_return',decision=str(index),gap=case['time_since_last_certified_positive'],gap_bin=frozen.age_bin(case['time_since_last_certified_positive']),unclean_prefix_frames=unclean)
            try:
                tick=time.perf_counter();b=model.initialize(legal['history'],legal['partial_graph']);ancestry={}
                init.append(dict(root_id=case['root_id'],milliseconds=(time.perf_counter()-tick)*1000,fallback=b.diagnostics.get('bootstrap_duplicate_fallback',False)))
                frozen.extend_ancestry(b,ancestry,times,truth)
                for ob in obs[:index]:b=model.observe(b,ob);frozen.extend_ancestry(b,ancestry,times,truth)
                before=model.observe(b,{**arrived,'tracks':[],'human_sensor_available':False})
                after=model.observe(b,arrived)
                source_age=[]
                focal_source_age=None;source_retained=False
                previous_ob=obs[index-1] if index else legal['history'][-1]
                anchors=[ob for ob in [legal['history'][-1]]+obs[:index] if abs(float(ob['timestamp'])-case['last_positive'])<1e-7]
                if len(anchors)==1:
                    anchor_id=physical_ids(anchors[0],times,xyz)[0].get(actor)
                    matches=[track for track in previous_ob.get('tracks',[]) if track.get('track_id')==anchor_id]
                    if len(matches)==1:source_retained=True;focal_source_age=float(previous_ob['timestamp'])-float(matches[0]['last_seen_time'])
                supported=[(c,frozen.lineage_slot(c,ancestry,actor)) for c in before.components]
                supported=[(c,i) for c,i in supported if i is not None and c.weight>0]
                support_mass=sum(c.weight for c,i in supported)
                belief_age=sum(c.weight*(before.timestamp-c.last_seen[i]) for c,i in supported)/support_mass if support_mass else None
                if index:
                    for track in obs[index-1].get('tracks',[]):
                        source_age.append(float(obs[index-1]['timestamp'])-float(track['last_seen_time']))
                prior_count=count_metrics(before,truth.shape[1]);posterior_count=count_metrics(after,truth.shape[1])
                good=split=merge=duplicate=missing=0.
                ids=frozen.exact_ids(arrived,times,truth)
                if actor not in ids:unclean+=1
                point=ids.get(actor,truth[int(np.argmin(abs(times-case['timestamp']))),actor])
                for c in after.components:
                    born=[i for i in range(c.n) if c.birth[i]>=after.timestamp-1e-7 and np.linalg.norm(c.last_position[i]-point)<2e-4]
                    if born:split+=c.weight
                    preferred=frozen.lineage_slot(c,ancestry,actor)
                    if preferred is None:missing+=c.weight;continue
                    if abs(c.last_seen[preferred]-after.timestamp)<1e-7:
                        if np.linalg.norm(c.last_position[preferred]-point)<2e-4:good+=c.weight
                        else:merge+=c.weight
                    if born and preferred not in born:duplicate+=c.weight
                for stream in (41,107):
                    common=dict(root_id=case['root_id'],scene_id=case['scene_id'],model='D_StopMove',cohort='reobservation',decision=str(index))
                    prior=frozen.score(model,before,ancestry,[actor],{actor:case['timestamp']},times,truth,common,stream)
                    posterior=frozen.score(model,after,ancestry,[actor],{actor:case['timestamp']},times,truth,common,stream)
                    for left,right in zip(prior,posterior):
                        if left['status']=='joint_supported':continue
                        record=dict(**meta,stream=stream,H=left['H'],prior_status=left['status'],posterior_status=right['status'],
                                    association_accuracy=good,false_split=split,false_merge=merge,duplicate=duplicate,missing_antecedent_mass=missing,
                                    prior_count=prior_count,posterior_count=posterior_count,all_source_track_ages_before=source_age,focal_source_track_age_before=focal_source_age,focal_source_record_retained_before=source_retained,belief_age_before_conditional=belief_age,preferred_lineage_support_before=support_mass)
                        if unclean:
                            for key in ('association_accuracy','false_split','false_merge','duplicate','missing_antecedent_mass'):record['raw_'+key]=record.pop(key)
                            record.update(prior_status='identity_annotation_censored' if left['status']!='truth_censored' else left['status'],
                                          posterior_status='identity_annotation_censored' if right['status']!='truth_censored' else right['status'])
                        else:
                            for key in ('tracked_es','path_es','future_path_es','coverage50','coverage90','radius90','endpoint_error'):
                                if key in left:record['prior_'+key]=left[key]
                                if key in right:record['posterior_'+key]=right[key]
                                if key in left and key in right:record[key+'_gain']=left[key]-right[key]
                        output.write(json.dumps(record,allow_nan=False)+'\n')
            except Exception as error:
                failures.append(dict(root_id=case['root_id'],error=repr(error)))
                for stream in (41,107):
                    for horizon in (.5,1.,2.,3.):
                        status='model_failed' if case['timestamp']+horizon<=times[-1]+1e-8 else 'truth_censored'
                        output.write(json.dumps(dict(**meta,stream=stream,H=horizon,prior_status=status,posterior_status=status,error=repr(error)))+'\n')
            output.flush()
            if (ri+1)%25==0:print(json.dumps(dict(arm=a.arm,roots=ri+1,failures=len(failures),seconds=time.perf_counter()-start)),flush=True)
    config=dict(arm=a.arm,roots=len(selected),scenes=len({r['scene_id'] for r in selected}),checkpoint_sha256=checkpoint_sha,
                model_source=str(sys.modules['eventwm_graph.l2_intermittent.model'].__file__),model_source_sha256=sha(sys.modules['eventwm_graph.l2_intermittent.model'].__file__),
                frozen_scoring_source_sha256=sha(source),extract_manifest_sha256=sha(OUT/'reobservation_supplement/manifest.json'),
                initialization=init,failures=failures,seconds=time.perf_counter()-start,model_version=model.model_version)
    assert sha(a.checkpoint)==checkpoint_sha and sha(model_source)==model_source_sha
    (dest/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
if __name__=='__main__':main()
