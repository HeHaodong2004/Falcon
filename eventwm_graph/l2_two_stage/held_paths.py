"""Predeclared fitting-held native-path scaling diagnostic, never acceptance.

Only TRAIN campaign Roots may enter this diagnostic. The 2,048-Root selection
is sealed from metadata before any predictions; all their causal native anchors
are retained. Protected or scene-development payloads are never accessed here.
"""
import argparse
import collections
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .association import AssociationGraph
from .common import OUT, dump, sha, seed_for
from .conditioned import HistoryConditionedWorldModel
from .evaluate import HORIZONS, coverage, energy, paired, summarize
from .future_population_targets import attach_targets
from .last_event_metadata import attach_metadata
from .positive_history import attach_histories
from .set_score import joint_future_set_kernel_score
from .train import root_validation

SPLIT = 'train_campaign_20261023'
SELECTION_PREFIX = 'two-stage-fitting-held-path-diagnostic-20261023:'
ROOTS = 2048
STREAMS = (41, 107)
FEATURES = ('keys','x','graph','global_graph','origin','velocity','age','global_features',
            'robotxy','timestamp','minimum_count','positive_history','source_last_event_indices')
HORIZON_INDEX = {.5: 1, 1.: 3, 2.: 5, 3.: 7}
VERSION = 'predeclared_fitting_held_native_paths_1'


def source_hashes():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def paths():
    return dict(prepared=OUT/'prepared'/SPLIT/'manifest.json',
                history=OUT/'positive_history'/SPLIT/'manifest.json',
                events=OUT/'last_event_metadata'/SPLIT/'manifest.json',
                population=OUT/'population_future_targets'/SPLIT/'manifest.json')


def selection_rows(root_rows, count=ROOTS):
    """No scene/count/coverage/quality conditioning; deterministic hash order."""
    ids = [row['root_id'] for row in root_rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicated campaign Root metadata')
    held = [dict(row, selection_hash=hashlib.sha256(
        (SELECTION_PREFIX+row['root_id']).encode()).hexdigest())
        for row in root_rows if root_validation(row['root_id'])]
    ordered = sorted(held, key=lambda row: (row['selection_hash'], row['root_id']))
    if len(ordered) < count:
        raise ValueError('insufficient fitting-held Roots for fixed diagnostic')
    return ordered[:count], held


def prepare(name, training_experiment='full_held_o_guided_campaign'):
    if Path(name).name != name or Path(training_experiment).name != training_experiment:
        raise ValueError('diagnostic/experiment names must be single components')
    dest = OUT/'evaluation'/name
    if dest.exists():
        raise ValueError('preserve previous selection; do not overwrite')
    manifests = {key: json.loads(path.read_text()) for key,path in paths().items()}
    prepared = manifests['prepared']
    if prepared['split'] != 'train' or manifests['history']['split'] != 'train' or manifests['population']['split'] != 'train':
        raise ValueError('this diagnostic cannot open non-fitting payloads')
    data_hashes = {key: sha(path) for key,path in paths().items()}
    for key in ('history','events','population'):
        if manifests[key]['identity']['prepared_manifest_sha256'] != data_hashes['prepared']:
            raise ValueError('sidecar has another fitting-cohort identity')
    ledger_path = OUT/prepared['root_ledger']['path']
    if sha(ledger_path) != prepared['root_ledger']['sha256']:
        raise ValueError('campaign Root ledger changed')
    with gzip.open(ledger_path,'rt') as stream:
        ledger = json.load(stream)
    # This is metadata only: no observation/feature/privileged tape payload is
    # opened until the Root list and exact data identities are written below.
    selected, held = selection_rows(ledger)
    config_path = OUT/'experiments'/training_experiment/'config.json'
    config = json.loads(config_path.read_text())
    if config['split'] != SPLIT or config['sidecar'] != SPLIT or config['full_fit'] or config['data_manifest_sha256'] != data_hashes['prepared']:
        raise ValueError('diagnostic requires the declared campaign held fit')
    if len(held) != config['validation_roots'] or sum(r['anchors'] for r in held) != config['validation_anchors']:
        raise ValueError('fitting-held fold differs from trainer accounting')
    roots_from_history = {r['root_id']:r for shard in manifests['history']['shards'] for r in shard['roots']}
    for row in ledger:
        other = roots_from_history.get(row['root_id'])
        if other is None or (row['scene_id'],row['anchors']) != (other['scene_id'],other['anchors']):
            raise ValueError('Root ledger/history metadata differ')
    initial = Path(config['initial_checkpoint'])
    if sha(initial) != config['initial_sha256']:
        raise ValueError('initial O checkpoint changed')
    dest.mkdir(parents=True)
    dump(dest/'selection.json', selected)
    protocol = dict(schema=VERSION, name=name, training_experiment=training_experiment,
        training_config=str(config_path), training_config_sha256=sha(config_path),
        split=SPLIT, data_manifests={key:str(path) for key,path in paths().items()},
        data_manifest_sha256=data_hashes, root_ledger_sha256=sha(ledger_path),
        selection='smallest SHA256(prefix+root_id), then Root id; no outcome/support filter',
        selection_prefix=SELECTION_PREFIX, selected_roots=ROOTS,
        selected_anchors=sum(r['anchors'] for r in selected),
        selected_scenes=len({r['scene_id'] for r in selected}),
        selected_root_list_sha256=sha(dest/'selection.json'),
        full_fitting_held_roots=len(held), full_fitting_held_anchors=sum(r['anchors'] for r in held),
        initial_checkpoint=str(initial), initial_checkpoint_sha256=sha(initial),
        formal_comparison='initial O versus best.pt selected by full-held trainer proper risk after full anchor cycle; complete.json must certify selection',
        interim_policy='explicit interim role only; immutable copied completed-epoch checkpoint, no acceptance or diagnostic-based epoch selection',
        future_set_kernel=True, worlds=32, streams=list(STREAMS), horizons=list(HORIZONS),
        timestamp_censoring='identical native offsets/nearest supported endpoints and masks to evaluate.py; unsupported target entries never scored',
        cohorts='all, visible age<1e-7, native practical gap age[.75,1.25] and anchor>0, reappearance; same final evaluator definitions',
        scoring='each32-world stream separately; proper off-diagonal tracked/joint ES and complete future-set kernel; empirical90% trajectory coverage',
        uncertainty='same equal-scene/equal-Root within scene/equal supported original rows within Root and paired2000scene bootstrap as evaluate.py',
        failure_policy='every expected target row retained; any failed/missing score makes affected aggregate INCONCLUSIVE',
        interpretation='TRAINING-SCENE HELD-ROOT SCALING DIAGNOSTIC ONLY. Not scene-disjoint confirmation, protected evaluation, or Stage1 acceptance. Does not replace all15,982Root held proper risk or final2,110developmentRoot gates.',
        source_sha256=source_hashes(), declared_before_any_prediction=True)
    dump(dest/'protocol.json', protocol)
    print(json.dumps({key:protocol[key] for key in ('selected_roots','selected_anchors','selected_scenes','full_fitting_held_roots','full_fitting_held_anchors','selected_root_list_sha256')}),flush=True)
    return protocol


def load_selection(dest):
    protocol = json.loads((dest/'protocol.json').read_text())
    if protocol['schema'] != VERSION or protocol['split'] != SPLIT or protocol['selected_roots'] != ROOTS:
        raise ValueError('unsupported or changed diagnostic protocol')
    if sha(dest/'selection.json') != protocol['selected_root_list_sha256']:
        raise ValueError('predeclared Root list changed')
    for key,path in paths().items():
        if sha(path) != protocol['data_manifest_sha256'][key]:
            raise ValueError('predeclared fitting data identity changed')
    # Frozen executable sources make the later selected-checkpoint comparison
    # independent of unrelated edits while the long training job is running.
    if source_hashes() != protocol['source_sha256']:
        raise ValueError('diagnostic executable source changed since predeclaration; use its frozen package')
    selected = json.loads((dest/'selection.json').read_text())
    return protocol, selected


def load_selected_records(selected):
    wanted = {row['root_id']:row for row in selected}
    manifest = json.loads(paths()['prepared'].read_text())
    records=[];seen=set()
    # Stream, rather than retain the unselected243,502 feature records. Reading
    # one immutable prepared shard does not expose any protected data: all are
    # the explicitly authorized TRAIN campaign.
    for shard in manifest['shards']:
        data=(OUT/shard['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=shard['sha256']:
            raise ValueError('prepared fitting shard changed')
        for row in np.load(io.BytesIO(data),allow_pickle=True):
            rid=row['root_id']
            if rid not in wanted:continue
            key=(rid,int(row['anchor_index']))
            if key in seen or not root_validation(rid) or row['scene_id']!=wanted[rid]['scene_id']:
                raise ValueError('duplicate/nonheld/mismatched selected anchor')
            seen.add(key);records.append(row)
    actual=collections.Counter(r['root_id'] for r in records)
    if set(actual)!=set(wanted) or any(actual[rid]!=row['anchors'] for rid,row in wanted.items()):
        raise ValueError('selected fitting Roots omitted native anchors')
    records.sort(key=lambda r:(r['root_id'],r['anchor_index']))
    joins=dict(history=attach_histories(records,SPLIT),events=attach_metadata(records,SPLIT),future=attach_targets(records,SPLIT))
    return records,joins


def stable_checkpoint(source,dest):
    """Copy a completed stable file; never mutate or pause its training job."""
    source=Path(source);data=source.read_bytes();digest=hashlib.sha256(data).hexdigest()
    if sha(source)!=digest:
        raise RuntimeError('checkpoint changed while reading; retry after completed epoch')
    dest.write_bytes(data)
    if sha(dest)!=digest:raise ValueError('checkpoint copy differs')
    saved=torch.load(dest,map_location='cpu')
    if saved.get('resume_exact',True) is False:raise ValueError('mid-epoch failure checkpoint is not a stable diagnostic')
    return saved,digest


def expected_rows(record,stream):
    base=dict(root_id=record['root_id'],scene_id=record['scene_id'],anchor=int(record['anchor_index']),stream=stream,
        reappearance=bool(record['reappearance']))
    tracked=[];joint=[];population=[]
    for h in HORIZONS:
        hi=HORIZON_INDEX[h];endpoint=float(record['times'][hi])
        if endpoint<=0:continue
        selected=np.flatnonzero((record['times']>0)&(record['times']<=h+.028001))
        selected=selected[record['times'][selected]<=endpoint+1e-7]
        if not len(selected):raise ValueError('supported endpoint has no native path points')
        valid=np.flatnonzero((record['physical']>=0)&(record['target_mask'][:,hi]>0))
        for person in valid:
            if not np.all(record['target_mask'][person,selected]>0):
                raise ValueError('native supported path contains an invalid target')
            age=float(record['age'][person])
            tracked.append(dict(base,physical=int(record['physical'][person]),person=int(person),horizon=h,
                actual_horizon=endpoint,age=age,visible=age<1e-7,
                practical_gap=bool(.75-1e-7<=age<=1.25+1e-7 and record['anchor_index']>0),
                path_es=None,coverage90=None,error=None,radius90=None))
        if len(valid):joint.append(dict(base,physical=-1,horizon=h,joint_es=None,joint_coverage90=None))
        if not np.all(record['population_future_valid'][selected]):
            raise ValueError('declared native future population target is censored')
        population.append(dict(base,physical=-2,horizon=h,joint_set_kernel=None))
    return tracked,joint,population


def score_result(record,result,stream):
    """Exact final-evaluator path conventions; no oracle input to predictor."""
    tracked_rows,joint_rows,population_rows=expected_rows(record,stream)
    requested=np.asarray(result.times)
    ownership=result.association.event_person[:,result.source_event_indices]
    tracked=result.positions[np.arange(len(result.weights))[:,None,None],np.arange(len(requested))[None,:,None],ownership[:,None,:]]
    for h in HORIZONS:
        hi=HORIZON_INDEX[h];endpoint=float(record['times'][hi])
        if endpoint<=0:continue
        selected=np.flatnonzero((record['times']>0)&(record['times']<=h+.028001))
        selected=selected[record['times'][selected]<=endpoint+1e-7]
        timeidx=[int(np.flatnonzero(abs(requested-record['times'][j])<1e-7)[0]) for j in selected]
        here=[row for row in tracked_rows if row['horizon']==h]
        draws=[];truth=[]
        for row in here:
            person=row['person'];x=tracked[:,timeidx,person];y=record['target'][person,selected]
            cov,distance,radius=coverage(x,y)
            row.update(status='physical_history_lineage_path',path_es=energy(x,y)/np.sqrt(len(timeidx)),coverage90=cov,
                error=distance/np.sqrt(len(timeidx)),radius90=radius/np.sqrt(len(timeidx)))
            draws.append(x);truth.append(y)
        if draws:
            x=np.stack(draws,axis=2);y=np.stack(truth,axis=1);cov,_,_=coverage(x,y)
            row=next(row for row in joint_rows if row['horizon']==h)
            row.update(status='joint_tracked_path',joint_es=energy(x,y)/np.sqrt(len(timeidx)*len(draws)),joint_coverage90=cov)
        truth=record['population_future_positions'][selected]
        score=joint_future_set_kernel_score(result.positions[:,timeidx],result.present,truth,times=record['times'][selected])
        row=next(row for row in population_rows if row['horizon']==h)
        if score.status!='supported':raise ValueError('full-population score unsupported: '+score.status)
        row.update(status='complete_joint_future_set',joint_set_kernel=score.score)
    return tracked_rows,joint_rows,population_rows


def safe_summary(rows,key):
    valid=[r for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    answer=summarize(valid,key)
    answer.update(expected_rows=len(rows),scored_rows=len(valid),missing_rows=len(rows)-len(valid),
        expected_roots=len({r['root_id'] for r in rows}),expected_scenes=len({r['scene_id'] for r in rows}),
        status='INCONCLUSIVE_MISSING_SCORES' if len(valid)!=len(rows) else 'DIAGNOSTIC_SUPPORTED' if answer['sufficient'] else 'INCONCLUSIVE_EFFECTIVE_SUPPORT')
    return answer


def summaries(tracked,joint,population):
    answer={}
    for name,predicate in [('all',lambda r:True),('visible',lambda r:r.get('visible',False)),
            ('practical_gap',lambda r:r.get('practical_gap',False)),('reappearance',lambda r:r['reappearance'])]:
        subset=[r for r in tracked if predicate(r)]
        answer[name]={key:safe_summary(subset,key) for key in ('path_es','coverage90','error','radius90')}
    answer['horizons']={str(h):{key:safe_summary([r for r in tracked if r['horizon']==h],key)
        for key in ('path_es','coverage90')} for h in HORIZONS}
    answer['cohort_horizons']={cohort:{str(h):{key:safe_summary([r for r in tracked if r['horizon']==h and r.get(cohort,False)],key)
        for key in ('path_es','coverage90')} for h in HORIZONS} for cohort in ('visible','practical_gap','reappearance')}
    ages={'current':lambda a:a<1e-7,'(0,.75)':lambda a:1e-7<=a<.75-1e-7,
        '[.75,1.25]':lambda a:.75-1e-7<=a<=1.25+1e-7,'(1.25,3]':lambda a:1.25+1e-7<a<=3,'>3':lambda a:a>3}
    answer['age']={age:{key:safe_summary([r for r in tracked if predicate(r['age'])],key)
        for key in ('path_es','coverage90','error','radius90')} for age,predicate in ages.items()}
    answer['joint_tracked']={key:safe_summary(joint,key) for key in ('joint_es','joint_coverage90')}
    answer['complete_future_set']={'all':safe_summary(population,'joint_set_kernel'),
        'horizons':{str(h):safe_summary([r for r in population if r['horizon']==h],'joint_set_kernel') for h in HORIZONS}}
    return answer


def score(name,label,checkpoint,role):
    if Path(name).name!=name or Path(label).name!=label:raise ValueError('single-component names required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    dest=OUT/'evaluation'/name;protocol,selected=load_selection(dest)
    target=dest/'models'/label;target.mkdir(parents=True,exist_ok=False)
    saved,digest=stable_checkpoint(checkpoint,target/'checkpoint.pt');config=saved['config']
    if role=='initial':
        if digest!=protocol['initial_checkpoint_sha256']:raise ValueError('initial checkpoint differs from predeclared O')
    elif role=='selected':
        training=Path(protocol['training_config']).parent
        complete=json.loads((training/'complete.json').read_text())
        if (config['split']!=SPLIT or config['full_fit'] or digest!=complete['checkpoint_sha256']
                or complete['selected_checkpoint']!='best.pt' or saved['epoch']<config['minimum_selection_epoch']):
            raise ValueError('formal full-held selection has not completed as declared')
    elif role=='interim':
        if config['split']!=SPLIT or config['full_fit'] or saved['epoch']<=0:raise ValueError('interim requires completed held-fit epoch')
    else:raise ValueError('explicit initial/selected/interim role required')
    if role!='initial' and config['initial_sha256']!=protocol['initial_checkpoint_sha256']:
        raise ValueError('full-held fit used another initialization')
    dump(target/'checkpoint_identity.json',dict(checkpoint_source=str(checkpoint),sha256=digest,role=role,
        completed_epoch=int(saved['epoch']),steps=int(saved['steps']),configuration=config,
        inference_sources=source_hashes(),selection_sha256=protocol['selected_root_list_sha256'],
        decision='SCALING_DIAGNOSTIC_ONLY_NOT_STAGE1_ACCEPTANCE'))
    completed_epoch=int(saved['epoch'])
    del saved
    records,joins=load_selected_records(selected);dump(target/'data_join_audit.json',joins)
    model=HistoryConditionedWorldModel(target/'checkpoint.pt');started=time.perf_counter();previous={}
    tracked=[];joint=[];population=[];failures=[];counters=collections.Counter()
    with gzip.open(target/'rows.jsonl.gz','wt') as stream:
        for i,record in enumerate(records):
            rid=record['root_id'];history=record['positive_history']
            graph=previous[rid].append(history) if rid in previous else AssociationGraph.from_history(history);previous[rid]=graph
            requested=np.unique(np.r_[0,record['times'][record['times']>0]])
            for random_stream in STREAMS:
                wanted=expected_rows(record,random_stream)
                counters['anchor_streams']+=1;counters['horizon_censored']+=sum(record['times'][HORIZON_INDEX[h]]<=0 for h in HORIZONS)
                if len(requested)==1:
                    counters['wholly_future_censored_anchor_streams']+=1
                    if any(wanted):raise ValueError('wholly censored anchor has expected future score')
                    continue
                features={key:record[key] for key in FEATURES}
                try:
                    result=model.predict_features(features,worlds=32,seed=seed_for(rid,random_stream,record['anchor_index']),
                        times=requested,association_graph=graph)
                    rows=score_result(record,result,random_stream);counters['successful_anchor_streams']+=1
                except Exception as error:
                    rows=wanted;counters['failed_anchor_streams']+=1
                    failure=dict(root_id=rid,anchor=int(record['anchor_index']),stream=random_stream,error=repr(error));failures.append(failure)
                    for values in rows:
                        for row in values:row.update(status='MODEL_FAILED',failure_reason=repr(error))
                for kind,store,values in zip(('tracked','joint_tracked','complete_future_set'),(tracked,joint,population),rows):
                    store.extend(values)
                    for row in values:stream.write(json.dumps(dict(kind=kind,**row),allow_nan=False)+'\n')
            if (i+1)%200==0:
                progress=dict(anchors=i+1,total=len(records),seconds=time.perf_counter()-started,failures=len(failures))
                dump(target/'progress.json',progress);print(json.dumps(progress),flush=True)
    result=dict(schema=VERSION,role=role,checkpoint_sha256=digest,completed_epoch=completed_epoch,
        protocol_sha256=sha(dest/'protocol.json'),selected_root_list_sha256=sha(dest/'selection.json'),
        roots=len(selected),scenes=len({r['scene_id'] for r in records}),anchors=len(records),streams=list(STREAMS),
        counters={key:int(value) for key,value in counters.items()},failures=failures,summary=summaries(tracked,joint,population),
        source_sha256=source_hashes(),seconds=time.perf_counter()-started,
        decision='INCONCLUSIVE_MODEL_FAILURES' if failures else 'SCALING_DIAGNOSTIC_COMPLETE_NOT_ACCEPTANCE')
    if result['source_sha256']!=protocol['source_sha256']:raise ValueError('executed source changed during scoring')
    dump(target/'summary.json',result)
    print(json.dumps(dict(label=label,decision=result['decision'],all=result['summary']['all'],practical_gap=result['summary']['practical_gap'])),flush=True)
    return result


def compare(name,left,right):
    dest=OUT/'evaluation'/name;protocol,_=load_selection(dest)
    def read(label):
        summary=json.loads((dest/'models'/label/'summary.json').read_text());rows=collections.defaultdict(list)
        with gzip.open(dest/'models'/label/'rows.jsonl.gz','rt') as stream:
            for line in stream:
                row=json.loads(line);rows[row.pop('kind')].append(row)
        return summary,rows
    initial,a=read(left);candidate,b=read(right);answer={}
    for cohort,predicate in [('all',lambda r:True),('visible',lambda r:r.get('visible',False)),
            ('practical_gap',lambda r:r.get('practical_gap',False)),('reappearance',lambda r:r['reappearance'])]:
        aa=[r for r in a['tracked'] if predicate(r)];bb=[r for r in b['tracked'] if predicate(r)]
        answer[cohort]={}
        for key in ('path_es','coverage90','error','radius90'):
            if len(aa)!=len(bb) or any(r.get(key) is None for r in aa+bb):
                value=dict(status='INCONCLUSIVE_MISSING_SCORES')
            else:value=dict(status='SCALING_DIAGNOSTIC_ONLY',**paired(bb,aa,key,relative=key=='path_es'))
            answer[cohort][key]=value
    for kind,key in [('joint_tracked','joint_es'),('complete_future_set','joint_set_kernel')]:
        aa,bb=a[kind],b[kind]
        answer[kind]=dict(status='INCONCLUSIVE_MISSING_SCORES') if len(aa)!=len(bb) or any(r.get(key) is None for r in aa+bb) else dict(status='SCALING_DIAGNOSTIC_ONLY',**paired(bb,aa,key))
    result=dict(schema=VERSION,initial=left,candidate=right,initial_checkpoint=initial['checkpoint_sha256'],
        candidate_checkpoint=candidate['checkpoint_sha256'],selected_root_list_sha256=protocol['selected_root_list_sha256'],
        comparisons=answer,decision='FITTING_HELD_SCALING_ONLY_NOT_STAGE1_ACCEPTANCE')
    path=dest/('comparison_'+left+'_to_'+right+'.json')
    if path.exists():raise ValueError('preserve previous comparison')
    dump(path,result);return result


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--name',required=True);p.add_argument('--training-experiment',default='full_held_o_guided_campaign')
    p=sub.add_parser('score');p.add_argument('--name',required=True);p.add_argument('--label',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--role',choices=('initial','selected','interim'),required=True)
    p=sub.add_parser('compare');p.add_argument('--name',required=True);p.add_argument('--left',required=True);p.add_argument('--right',required=True)
    args=vars(parser.parse_args());command=args.pop('command');globals()[command](**args)


if __name__=='__main__':main()
