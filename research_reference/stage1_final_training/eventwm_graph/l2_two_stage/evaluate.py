"""Matched native forecasting, probability accounting, and paired scene CIs.

Tracked path scores are conditional on an anonymous episode's existence. Count,
presence and population occupancy are scored separately and block promotion if
the conditional diagnostic conceals missing/duplicated population mass.
"""
import argparse
import collections
import gzip
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import poisson
from scipy.special import logsumexp
from .common import OUT,OLD,DOC,dump,sha,seed_for
from .data import load_records,jobs_for,native_index
from .model import Stage1WorldModel
from .set_score import joint_future_set_kernel_score
from .count import count_log_probability as exact_count_log_probability,certain_sources

HORIZONS=(.5,1.,2.,3.)
GRID=np.arange(-12,14,2)


def energy(samples,truth):
    x=np.asarray(samples).reshape(len(samples),-1);y=np.asarray(truth).ravel();k=len(x)
    return float(np.linalg.norm(x-y,axis=-1).mean()-cdist(x,x).sum()/(2*k*(k-1)))


def coverage(samples,truth):
    x=np.asarray(samples).reshape(len(samples),-1);y=np.asarray(truth).ravel();center=x.mean(0)
    radii=np.linalg.norm(x-center,axis=-1);distance=np.linalg.norm(y-center)
    return float(distance<=np.quantile(radii,.9)),float(distance),float(np.quantile(radii,.9))


def summarize(rows,key,minimum_roots=100,minimum_scenes=30):
    values=collections.defaultdict(list);scenes={}
    for r in rows:
        if key in r and r[key] is not None and math.isfinite(float(r[key])):
            values[r['root_id']].append(float(r[key]));scenes[r['root_id']]=r['scene_id']
    by_scene=collections.defaultdict(list)
    for root,v in values.items():by_scene[scenes[root]].append(np.mean(v))
    means=np.array([np.mean(by_scene[s]) for s in sorted(by_scene)])
    if not len(means):return dict(mean=None,lo=None,hi=None,roots=0,scenes=0,sufficient=False)
    rng=np.random.default_rng(731);draw=means[rng.integers(len(means),size=(2000,len(means)))].mean(-1)
    return dict(mean=float(means.mean()),lo=float(np.quantile(draw,.025)),hi=float(np.quantile(draw,.975)),
        roots=len(values),scenes=len(means),sufficient=len(values)>=minimum_roots and len(means)>=minimum_scenes)


def paired(left,right,key,relative=False):
    def identity(r):return (r['root_id'],r['anchor'],r.get('physical',-1),r.get('horizon',-1),r.get('stream',41))
    l=[r for r in left if key in r];rr=[r for r in right if key in r]
    a={identity(r):r for r in l};b={identity(r):r for r in rr}
    if len(a)!=len(l) or len(b)!=len(rr):raise ValueError('duplicate matched score identity')
    rows=[]
    for k in sorted(a.keys()&b.keys()):
        x,y=a[k][key],b[k][key]
        if x is None or y is None:continue
        rows.append(dict(root_id=a[k]['root_id'],scene_id=a[k]['scene_id'],delta=x-y,candidate=x,baseline=y))
    if not relative:return summarize(rows,'delta')
    roots=collections.defaultdict(list);scenes={}
    for r in rows:roots[r['root_id']].append((r['candidate'],r['baseline']));scenes[r['root_id']]=r['scene_id']
    grouped=collections.defaultdict(list)
    for root,values in roots.items():grouped[scenes[root]].append(np.mean(values,axis=0))
    means=np.array([np.mean(grouped[s],axis=0) for s in sorted(grouped)])
    if not len(means):return dict(mean=None,lo=None,hi=None,roots=0,scenes=0,sufficient=False)
    pair=means.mean(0)
    if pair[1]<=0:raise ValueError('undefined relative aggregate with zero baseline score')
    rng=np.random.default_rng(731);draw=means[rng.integers(len(means),size=(2000,len(means)))].mean(1)
    relative_draw=(draw[:,0]-draw[:,1])/draw[:,1]
    return dict(mean=float((pair[0]-pair[1])/pair[1]),lo=float(np.quantile(relative_draw,.025)),hi=float(np.quantile(relative_draw,.975)),
        roots=len(roots),scenes=len(means),sufficient=len(roots)>=100 and len(means)>=30,
        candidate=float(pair[0]),baseline=float(pair[1]),method='ratio of matched root/scene aggregate scores in each paired scene bootstrap')


def occupancy(positions,present,robotxy):
    # A float32 +12 translation can move a representable negative boundary
    # offset to the adjacent grid cell. Score continuous world positions in
    # float64, independently of the neural feature storage dtype.
    positions=np.asarray(positions,np.float64);robotxy=np.asarray(robotxy,np.float64)
    k=positions.shape[0];bins=np.zeros((k,145),float)
    # 12x12 cells in [-12,12), plus a required outside bin.
    for w in range(k):
        p=positions[w,present[w]]-robotxy
        index=np.floor((p+12)/2).astype(int)
        inside=(index>=0).all(-1)&(index<12).all(-1)
        ids=index[inside,0]*12+index[inside,1]
        bins[w,ids]=1
        if (~inside).any():bins[w,-1]=1
    return bins.mean(0)


def frozen_jobs(split):
    """Use named campaign provenance instead of a moving/older train alias."""
    if split.startswith('dev'):return jobs_for('dev')
    manifest=json.loads((OUT/'prepared'/split/'manifest.json').read_text())
    return jobs_for(split) if manifest.get('snapshot') else jobs_for('train')


def poisson_binomial_poisson(p,rate,count):
    polynomial=np.array([1.])
    for probability in p:polynomial=np.convolve(polynomial,[1-probability,probability])
    return float(sum(mass*poisson.pmf(count-i,rate) for i,mass in enumerate(polynomial) if i<=count))


def count_log_probability(logits,age,rate,count,minimum_count=0,certain=None):
    return exact_count_log_probability(logits,age,rate,count,minimum_count,certain)


class Kinematic:
    def __init__(self,memory=True,graph=False):self.memory=memory;self.graph=graph;self.coefficients=None;self.diffusion=.35
    def fit(self,records):
        # Sufficient statistics retain the exact same Root-weighted ridge fit
        # without millions of Python objects on the full eligible snapshot.
        dimension=1+records[0]['x'].shape[-1]+records[0]['graph'].shape[-1]
        xx=np.zeros((dimension,dimension));xy=np.zeros((dimension,2));yy=0.;mass=0.
        for r in records:
            people,times=np.nonzero((r['target_mask']>0)&(r['times'][None]>=.4))
            if not len(people):continue
            seconds=r['times'][times,None]
            feature=np.c_[np.ones(len(r['keys'])),r['x'],r['graph'].mean(1)][people]
            residual=(r['target'][people,times]-r['origin'][people]-r['velocity'][people]*seconds)/seconds
            weight=r['weight']/max(1,r['target_mask'].sum())
            xx+=weight*feature.T@feature;xy+=weight*feature.T@residual
            yy+=weight*float((residual**2).sum());mass+=weight*len(people)
        if mass:
            if self.graph:
                self.coefficients=np.linalg.solve(xx+np.eye(dimension)*.1,xy)
                yy=yy-2*float((self.coefficients*xy).sum())+float((self.coefficients*(xx@self.coefficients)).sum())
            self.diffusion=max(.05,float(np.sqrt(max(0.,yy)/(2*mass))))
        return self

    def predict(self,r,times,seed=41):
        rng=np.random.default_rng(seed);n=len(r['keys']);k=32
        drift=r['velocity'].copy()
        if self.coefficients is not None:
            feature=np.c_[np.ones(n),r['x'],r['graph'].mean(1)]
            drift+=feature@self.coefficients
        z=rng.normal(size=(k,n,2));initial=rng.normal(size=(k,n,2))*r['age'][None,:,None]*self.diffusion
        samples=r['origin'][None,None]+initial[:,None]+times[None,:,None,None]*(drift[None,None]+z[:,None]*self.diffusion)
        support=np.ones(n,bool) if self.memory else r['age']<1e-7
        return samples,support


def run(name,split='dev',checkpoints=(),fitting_split=None,joint_set=False):
    records=load_records(split)
    fitting_split=fitting_split or ('train_pilot_topology' if split.endswith('_topology') else 'train_pilot')
    fit=load_records(fitting_split)
    fit_native_identity=None
    if fitting_split in ('train_pilot_native_v2','train_campaign_20261023'):
        from .future_population_targets import attach_targets as attach_native_targets
        fit_native_identity=attach_native_targets(fit,'train_campaign_20261023')
    dest=OUT/'evaluation'/name;dest.mkdir(parents=True,exist_ok=False)
    baselines={'cv':Kinematic(False).fit(fit),'memory_cv':Kinematic().fit(fit),'graph_ridge':Kinematic(graph=True).fit(fit)}
    for baseline,model in baselines.items():
        dump(dest/(baseline+'_parameters.json'),dict(schema='legal_kinematic_baseline_1',memory=model.memory,
            graph=model.graph,diffusion=model.diffusion,coefficients=None if model.coefficients is None else model.coefficients.tolist(),
            fitting_manifest_sha256=sha(OUT/'prepared'/fitting_split/'manifest.json')))
    import torch
    from .conditioned import HistoryConditionedWorldModel,ConditionedHumanWorlds
    models={Path(path).parent.name:(HistoryConditionedWorldModel(path) if 'conditioned_config' in torch.load(path,map_location='cpu') else Stage1WorldModel(path)) for path in checkpoints}
    if any(isinstance(m,HistoryConditionedWorldModel) for m in models.values()):
        from .positive_history import attach_histories
        from .last_event_metadata import attach_metadata
        attach_histories(records,'dev' if split.startswith('dev') else split)
        attach_metadata(records,'dev_native_v2' if split.startswith('dev') else split)
    if any(m.network.minimum_count_conditioned for m in models.values()):
        from .population import attach_targets
        attach_targets(records,split)
    independent_anchors=len(records)
    strata={}
    strata_path=DOC/'reviews/native_occlusion_development/anchors.jsonl.gz'
    if split.startswith('dev') and strata_path.exists():
        with gzip.open(strata_path,'rt') as stream:
            for line in stream:
                row=json.loads(line);strata[(row['root_id'],row['anchor_index'])]=row
    records=[dict(r,stream=stream) for r in records for stream in (41,107)]
    variants={**baselines,**models};path_rows={k:[] for k in variants};population_rows={k:[] for k in models};statuses=collections.defaultdict(collections.Counter)
    native={};folders={r['root_id']:Path(folder) for folder,r in frozen_jobs(split)}
    train_counts=collections.defaultdict(list)
    for r in fit:train_counts[r['root_id']].append(len(r['population_current_positions']) if 'population_current_positions' in r else len(r['unseen_positions'])+int(r['presence'].sum()))
    prior_rate=float(np.mean([np.mean(v) for v in train_counts.values()]))
    # Train-only constant occupancy prior from actual anchor truths. Future
    # evaluation is not used to fit these probabilities.
    prior_grid=np.zeros(145)
    training_folders={row['root_id']:Path(folder) for folder,row in frozen_jobs(fitting_split)}
    training_native={}
    for sample in fit:
        root=sample['root_id']
        if 'population_current_positions' in sample:
            p=sample['population_current_positions']
        elif 'set_positions' in sample and sample['set_valid']:
            p=np.r_[sample['set_positions'],sample['origin'][sample['age']<1e-7]]
        else:
            if root not in training_native:
                with np.load(training_folders[root]/'privileged_future.npz') as z:training_native[root]=(z['timestamps'].copy(),z['positions'][...,[0,2]].copy())
            ts,positions=training_native[root];p=positions[native_index(ts,sample['timestamp'])]
        prior_grid+=sample['weight']*occupancy(p[None],np.ones((1,len(p)),bool),sample['robotxy'])
    prior_grid/=max(1,len({r['root_id'] for r in fit}))
    start=time.perf_counter()
    for idx,r in enumerate(records):
        root=r['root_id'];meta=dict(root_id=root,scene_id=r['scene_id'],anchor=r['anchor_index'],stream=r['stream'])
        stratum=strata.get((root,r['anchor_index']),{})
        if stratum:
            if abs(stratum['timestamp']-r['timestamp'])>1e-7:raise ValueError('auxiliary native stratum timestamp mismatch')
            meta.update(native_occlusion=bool(stratum.get('has_native_occlusion')),
                fov_or_range_loss=bool(stratum.get('has_fov_or_range_loss')),
                native_stratum_supported=bool(stratum.get('cohort_supported')),
                empty_without_in_frustum_human=bool(stratum.get('cohort_supported') and stratum.get('valid_empty') and not stratum.get('has_native_occlusion')))
        if root not in native:
            with np.load(folders[root]/'privileged_future.npz',allow_pickle=False) as z:native[root]=(z['timestamps'].copy(),z['positions'][...,[0,2]].copy())
        native_times,truth=native[root];ti=native_index(native_times,r['timestamp'])
        requested=np.unique(np.r_[0,r['times'][r['times']>0]])
        features={k:r[k] for k in ('keys','x','graph','global_graph','origin','velocity','age','global_features','robotxy','timestamp')}
        if 'minimum_count' in r:features['minimum_count']=r['minimum_count']
        if 'positive_history' in r:
            features['positive_history']=r['positive_history']
            features['source_last_event_indices']=r['source_last_event_indices']
        outcomes={};count_estimates={}
        for variant,model in variants.items():
            try:
                seed=seed_for(root,r['stream'],r['anchor_index'])
                if isinstance(model,Kinematic):samples,support=model.predict(r,requested,seed);outcomes[variant]=(samples,support,None)
                else:
                    result=model.predict_features(features,worlds=32,seed=seed,times=requested)
                    if isinstance(result,ConditionedHumanWorlds):
                        ownership=result.association.event_person[:,result.source_event_indices]
                        tracked=result.positions[np.arange(32)[:,None,None],np.arange(len(requested))[None,:,None],ownership[:,None,:]]
                    else:tracked=result.positions
                    outcomes[variant]=(tracked,np.ones(len(r['keys']),bool),result)
                statuses[variant]['anchors_succeeded']+=1
            except Exception as e:
                statuses[variant]['anchors_failed']+=1
                path_rows[variant].append(dict(**meta,status='model_failed',failure_reason=repr(e)));continue
        for h in HORIZONS:
            hidx=int(np.flatnonzero(np.isclose(np.array([.25,.5,.75,1.,1.5,2.,2.5,3.]),h))[0])
            selected=np.flatnonzero((r['times']>0)&(r['times']<=h+.028001))
            endpoint=r['times'][hidx]
            if endpoint<=0:
                for v in variants:statuses[v]['horizon_censored']+=1
                continue
            selected=selected[r['times'][selected]<=endpoint+1e-7]
            timeidx=[int(np.flatnonzero(abs(requested-r['times'][j])<1e-7)[0]) for j in selected]
            valid=np.flatnonzero((r['physical']>=0)&(r['target_mask'][:,hidx]>0))
            for variant,(samples,support,result) in outcomes.items():
                joint_draw=[];joint_truth=[]
                for person in valid:
                    physical=int(r['physical'][person]);age=float(r['age'][person]);gap=.75-1e-7<=age<=1.25+1e-7 and r['anchor_index']>0
                    base=dict(**meta,physical=physical,horizon=h,actual_horizon=float(endpoint),age=age,
                        visible=age<1e-7,practical_gap=bool(gap),reappearance=bool(r['reappearance']))
                    if not support[person]:
                        path_rows[variant].append(dict(**base,status='missing_episode'));statuses[variant]['missing_episode']+=1;continue
                    draws=samples[:,timeidx,person];y=r['target'][person,selected]
                    cov,distance,radius=coverage(draws,y)
                    row=dict(**base,status='conditional_tracked_path',path_es=energy(draws,y)/np.sqrt(len(timeidx)),
                        coverage90=cov,error=distance/np.sqrt(len(timeidx)),radius90=radius/np.sqrt(len(timeidx)))
                    if result is not None:
                        row['presence_probability']=1. if isinstance(result,ConditionedHumanWorlds) else float(result.present[:,person].mean())
                        if isinstance(result,ConditionedHumanWorlds):row['status']='physical_history_lineage_path'
                    path_rows[variant].append(row);joint_draw.append(draws);joint_truth.append(y)
                if joint_draw:
                    draws=np.stack(joint_draw,axis=2);y=np.stack(joint_truth,axis=1)
                    cov,distance,radius=coverage(draws,y)
                    path_rows[variant].append(dict(**meta,physical=-1,horizon=h,visible=False,practical_gap=False,reappearance=bool(r['reappearance']),
                        status='joint_tracked_path',joint_es=energy(draws,y)/np.sqrt(len(timeidx)*len(joint_draw)),joint_coverage90=cov))
                if result is not None:
                    samples=result.positions
                    true_index=native_index(native_times,r['timestamp']+float(endpoint),1e-6);actual=truth[true_index]
                    pred=occupancy(samples[:,timeidx[-1]],result.present,r['robotxy']);target=occupancy(actual[None],np.ones((1,len(actual)),bool),r['robotxy'])
                    p=models[variant].parameters(features)
                    presence=1/(1+np.exp(-np.clip(p['presence_logits'][:len(r['keys'])],-50,50)))
                    certain=certain_sources(r['age'],r['minimum_count'],models[variant].network.compulsory_unique_episode,models[variant].network.compulsory_observed_clique)
                    presence=np.where(certain,1.,presence)
                    minimum=r['minimum_count'] if models[variant].network.minimum_count_conditioned else 0
                    if isinstance(result,ConditionedHumanWorlds):
                        # One independent fixed-success integration per anchor/
                        # stream. Reuse across horizons because the sampled
                        # complete population has persistent existence.
                        count_key=(variant,len(actual))
                        if count_key not in count_estimates:
                            count_estimates[count_key]=result.law.estimate_count_log_probability(len(actual),successes=32,
                                seed=seed_for(root,r['stream']+7000,r['anchor_index']))
                        log_countp=count_estimates[count_key]['log_probability']
                    else:log_countp=count_log_probability(p['presence_logits'][:len(r['keys'])],r['age'],float(p['rate']),len(actual),minimum,certain)
                    counts=result.present.sum(-1);max_count=max(int(counts.max()),len(actual))+8
                    cdf=np.mean(counts[:,None]<=np.arange(max_count)[None],axis=0)
                    count_rps=float(np.sum((cdf-(len(actual)<=np.arange(max_count)))**2))
                    population_row=dict(**meta,horizon=h,count_nll=-log_countp if math.isfinite(log_countp) else None,
                        count_support_status='supported' if math.isfinite(log_countp) else 'zero_probability_or_nonfinite',
                        constant_count_nll=float(-poisson.logpmf(len(actual),prior_rate)),count_rps=count_rps,
                        occupancy_brier=float(((pred-target)**2).mean()),constant_occupancy_brier=float(((prior_grid-target)**2).mean()),
                        valid_empty=not bool(np.any(r['age']<1e-7)),unresolved_prefix=r['unresolved_prefix'],
                        expected_count=float(counts.mean()) if minimum or isinstance(result,ConditionedHumanWorlds) else float(presence.sum()+p['rate']),true_count=len(actual),minimum_count=minimum)
                    if isinstance(result,ConditionedHumanWorlds):population_row['count_integration']=count_estimates[count_key]
                    if joint_set:
                        native_indices=[native_index(native_times,r['timestamp']+float(r['times'][j]),1e-6) for j in selected]
                        score=joint_future_set_kernel_score(samples[:,timeidx],result.present,truth[native_indices],times=r['times'][selected])
                        population_row.update(joint_set_kernel=score.score,joint_set_status=score.status)
                        # These controlled motion ablations retain exactly the
                        # same learned current population, counts and PPP paths.
                        # They are not independent unseen-population baselines.
                        for baseline in ('memory_cv','graph_ridge'):
                            if isinstance(result,ConditionedHumanWorlds):
                                population_row['joint_set_'+baseline]=None
                                population_row['joint_set_'+baseline+'_status']='not_defined_after_history_conditioning; use separately scored population baseline'
                                continue
                            alternative=samples[:,timeidx].copy()
                            alternative[:,:,:len(r['keys'])]=outcomes[baseline][0][:,timeidx]
                            ablation=joint_future_set_kernel_score(alternative,result.present,truth[native_indices],times=r['times'][selected])
                            population_row['joint_set_'+baseline]=ablation.score
                            population_row['joint_set_'+baseline+'_status']=ablation.status
                    population_rows[variant].append(population_row)
        if (idx+1)%500==0:print(json.dumps(dict(anchors=idx+1,seconds=time.perf_counter()-start)),flush=True)
    summary=dict(schema='two_stage_stage1_development_metrics_2',split=split,fitting_split=fitting_split,streams=[41,107],
        seed_rule='SHA256 two-stage-A:<stream>:<root>:<anchor_index> first4bytes little endian; exact independent historical runner match',roots=len({r['root_id'] for r in records}),anchors=independent_anchors,dependent_anchor_streams=len(records),
        statuses={k:dict(v) for k,v in statuses.items()},conditional_score_warning='Tracked trajectory ES conditions on source existence; population defects block promotion.',
        protocol_sha256=sha(DOC/'PROTOCOL.json'),
        joint_set_score=bool(joint_set),native_strata_sha256=sha(strata_path) if strata else None,
        data_manifest_sha256=sha(OUT/'prepared'/split/'manifest.json'),fitting_manifest_sha256=sha(OUT/'prepared'/fitting_split/'manifest.json'),
        fitting_native_target_identity=fit_native_identity,
        occupancy_numeric_policy='float64 world/robot arithmetic on unchanged2m grid; exact native current training prior where manifest-bound cache available',
        checkpoint_sha256={Path(p).parent.name:sha(p) for p in checkpoints},
        source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        variants={},comparisons={},seconds=time.perf_counter()-start)
    for name,rows in path_rows.items():
        groups={}
        for group,predicate in [('all',lambda r:True),('visible',lambda r:r.get('visible')),('practical_gap',lambda r:r.get('practical_gap')),('reappearance',lambda r:r.get('reappearance'))]:
            part=[r for r in rows if predicate(r)];groups[group]={key:summarize(part,key) for key in ('path_es','coverage90','joint_es','joint_coverage90','presence_probability','error','radius90')}
        groups['horizons']={str(h):{key:summarize([r for r in rows if r.get('horizon')==h],key) for key in ('path_es','coverage90','joint_es')} for h in HORIZONS}
        ages={'current':lambda a:a<1e-7,'(0,.75)':lambda a:1e-7<=a<.75-1e-7,
            '[.75,1.25]':lambda a:.75-1e-7<=a<=1.25+1e-7,'(1.25,3]':lambda a:1.25+1e-7<a<=3,'>3':lambda a:a>3}
        groups['age']={age:{key:summarize([r for r in rows if 'age' in r and predicate(r['age'])],key) for key in
            ('path_es','coverage90','presence_probability','error','radius90')} for age,predicate in ages.items()}
        groups['cohort_horizons']={group:{str(h):{key:summarize([r for r in rows if r.get('horizon')==h and r.get(group)],key)
            for key in ('path_es','coverage90','error','radius90')} for h in HORIZONS} for group in ('visible','practical_gap','reappearance','native_occlusion')}
        summary['variants'][name]=groups
        dump(dest/(name+'_paths.json'),rows)
    for name,rows in population_rows.items():
        summary['variants'][name]['population_failures']=dict(collections.Counter(r['count_support_status'] for r in rows))
        summary['variants'][name]['population']={group:{key:summarize([r for r in rows if group=='all' or r.get(group)],key) for key in
            ('count_nll','constant_count_nll','count_rps','occupancy_brier','constant_occupancy_brier','expected_count','true_count',
             'joint_set_kernel','joint_set_memory_cv','joint_set_graph_ridge')} for group in ('all','valid_empty','unresolved_prefix','native_occlusion','empty_without_in_frustum_human')}
        summary['variants'][name]['joint_set_statuses']=dict(collections.Counter(r.get('joint_set_status','not_requested') for r in rows))
        summary['variants'][name]['population_comparisons']={group:{} for group in ('all','valid_empty','native_occlusion','empty_without_in_frustum_human')}
        for group,comparisons in summary['variants'][name]['population_comparisons'].items():
            selected=[r for r in rows if group=='all' or r.get(group)]
            for key,baseline in [('occupancy_brier','constant_occupancy_brier'),('count_nll','constant_count_nll'),
                    ('joint_set_kernel','joint_set_memory_cv'),('joint_set_kernel','joint_set_graph_ridge')]:
                a=[dict(r,delta=r[key]-r[baseline]) for r in selected if r.get(key) is not None and r.get(baseline) is not None]
                comparisons[key+'_minus_'+baseline]=summarize(a,'delta')
        dump(dest/(name+'_population.json'),rows)
        for baseline in variants:
            if baseline==name:continue
            summary['comparisons'][name+'_minus_'+baseline]={group:paired([r for r in path_rows[name] if group=='all' or r.get('practical_gap')],
                [r for r in path_rows[baseline] if group=='all' or r.get('practical_gap')],'path_es',relative=True) for group in ('all','practical_gap')}
    dump(dest/'metrics.json',summary)
    print(json.dumps(dict(path=str(dest/'metrics.json'),seconds=summary['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--split',default='dev');p.add_argument('--fitting-split');p.add_argument('--checkpoint',action='append',default=[]);p.add_argument('--joint-set-score',action='store_true');a=p.parse_args();run(a.name,a.split,a.checkpoint,a.fitting_split,a.joint_set_score)
