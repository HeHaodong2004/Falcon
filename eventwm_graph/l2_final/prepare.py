"""Full-cohort, compact root shards with unique/injective offline labels."""
import argparse
import collections
import gzip
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .common import OUT,OLD,PREVIOUS,sha,dump
from .common import OUT as CAMPAIGN_OUT

_CACHE=None
_CACHE_FILES=None
_STATIC=re.compile(rb'("static_updates"\s*:\s*)\[(?:\s*\{[^{}]*\}\s*,?)*\s*\]')

def previous_rows():
    global _CACHE,_CACHE_FILES
    if _CACHE is None:
        _CACHE={};paths=[OLD/'train.npy']
        _CACHE_FILES={r['root_id']:r['files'] for r in json.loads((PREVIOUS/'snapshot/manifest.json').read_text())['rows']}
        report=json.loads((PREVIOUS/'preparation.json').read_text())
        for row in report['shards']:
            path=PREVIOUS/'training_runtime/output/l2_graph_wm_120k_20261021'/('train_'+row['tag']+'.npy')
            assert sha(path)==row['cache_sha256'];paths.append(path)
        for path in paths:
            for row in np.load(path,allow_pickle=True):
                assert row['root_id'] not in _CACHE
                _CACHE[row['root_id']]=row
    return _CACHE


def shard(job):
    start,stop=job;manifest=json.loads((OUT/'snapshot/manifest.json').read_text());rows=manifest['rows'][start:stop]
    name='train_%06d_%06d'%(start,stop);existing=OUT/'shards'/(name+'.json')
    if existing.exists():
        report=json.loads(existing.read_text());assert report['start']==start and report['stop']==stop
        assert report['snapshot_sha256']==sha(OUT/'snapshot/manifest.json')
        contract=json.loads((CAMPAIGN_OUT/'preparation_contract.json').read_text())
        assert report['preparation_source_sha256'] in (sha(__file__),contract['original_replay_source_sha256'])
        assert sha(existing.with_suffix('.npy'))==report['sha256']
        return name
    from eventwm_graph.l2_120k.model import load_model
    wm=load_model(population=json.loads((OLD/'population_prior.json').read_text()),motion=json.loads((OLD/'motion.json').read_text()),
                  observation='docs/l2/graph_wm_120k_20261021/artifacts/checkpoints/frozen_observation_fit.json')
    result=[];ledger=[];counts=collections.Counter();begin=time.perf_counter()
    cached=previous_rows()
    for row in rows:
        folder=OUT/'snapshot/roots'/row['root_id']
        prior=cached.get(row['root_id'])
        if prior is not None:
            assert all(row['files'][name]==_CACHE_FILES[row['root_id']][name] for name in ('legal_root.json.gz','privileged_future.npz'))
            history_frames=prior['history_length'];legal=None;no_target=not prior['components']
        else:
            with gzip.open(folder/'legal_root.json.gz','rb') as f:raw=f.read()
            compact,removed=_STATIC.subn(rb'\1[]',raw)
            if removed!=raw.count(b'"static_updates"'):compact=raw
            match=re.search(rb'"history"\s*:\s*',compact);assert match is not None
            history,_=json.JSONDecoder().raw_decode(compact[match.end():].decode());history_frames=len(history)
            final=history[-1];now=float(final['timestamp'])
            current=[track for track in final.get('tracks',[]) if track.get('visible_now',False) and track.get('valid',True) and abs(float(track.get('last_seen_time',now))-now)<1e-7]
            no_target=not current and (len(history)==1 or float(history[-2]['timestamp'])<now)
            legal=None if no_target else json.loads(raw)
        if no_target:
            # Under the unchanged current-anchor objective this root has exactly
            # zero possible targets. Avoid an expensive replay whose gradient is
            # known to be absent from legal inputs alone; keep it in the ledger.
            ledger.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],status='no_valid_current_motion_target' if prior is not None else 'no_current_positive_motion_target',
                               history_frames=history_frames,legal_features='validated legal cache' if prior is not None else 'zero-loss eligibility certified from legal frame'))
            continue
        try:
            # Fixed network-independent legal estimator; same motion-training
            # inputs as the incumbent. Truth is opened only AFTER this replay.
            belief=wm._initialize_history(legal['history'],legal['partial_graph']) if prior is None else None
        except Exception as error:
            ledger.append(dict(root_id=row['root_id'],status='legal_filter_failed',error=repr(error)));continue
        with np.load(folder/'privileged_future.npz',allow_pickle=False) as f:
            times=f['timestamps'].copy();truth=f['positions'][...,[0,2]].copy()
        timestamp=belief.timestamp if belief is not None else prior['timestamp']
        assert abs(times[0]-timestamp)<1e-7 and np.all(np.diff(times)>0) and np.isfinite(truth).all()
        keep=times<=times[0]+5.+1e-8;times=times[keep];truth=truth[keep];components=[]
        hypotheses=belief.hypotheses if belief is not None else prior['filtered_hypotheses']
        for h in hypotheses:
            ids=np.full(h.n,-1,int)
            for i in np.flatnonzero(abs(h.last_seen-timestamp)<1e-7):
                distances=np.linalg.norm(truth[0]-h.mean[i,:2],axis=-1)
                close=np.flatnonzero(distances<2e-4)
                if len(close)==1 and distances[close[0]]<1e-5:ids[i]=close[0]
                elif len(close)>1:counts['ambiguous_target_slots']+=1
            repeated={j for j in ids if j>=0 and np.sum(ids==j)>1}
            for j in repeated:counts['duplicate_target_slots_censored']+=int(np.sum(ids==j));ids[ids==j]=-1
            selected=np.flatnonzero(ids>=0)
            if not len(selected) or len(times)<2:continue
            if prior is not None:
                matches=[c for c in prior['components'] if c['mean'].shape==h.mean.shape and np.array_equal(c['mean'],h.mean.astype(np.float32)) and np.array_equal(c['covariance'],h.covariance.astype(np.float32)) and abs(c['weight']-h.weight)<1e-12]
                assert matches,'lost cached legal component correspondence'
                components.append({**matches[0],'selected':selected,'truth':truth[:,ids[selected]].astype(np.float32)})
                continue
            components.append(dict(weight=float(h.weight),mean=h.mean.astype(np.float32),covariance=h.covariance.astype(np.float32),
                age=(timestamp-h.last_seen).astype(np.float32),graph=belief.graph.local_features(h.mean[:,:2]).astype(np.float32),
                selected=selected,truth=truth[:,ids[selected]].astype(np.float32),maneuver_mean=h.maneuver_mean,
                maneuver_covariance=h.maneuver_covariance,maneuver_probability=h.maneuver_probability,maneuver_until=h.maneuver_until))
        status='motion_supervised' if components else 'no_unique_current_motion_target'
        ledger.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],status=status,physical_count=int(truth.shape[1]),
                           native_horizon=float(times[-1]-times[0]),history_frames=history_frames,legal_features='hash_matched_validated_cache' if prior is not None else 'fresh_legal_replay'))
        if components:result.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],components=components,times=times,timestamp=timestamp))
    dest=OUT/'shards';dest.mkdir(exist_ok=True)
    name='train_%06d_%06d'%(start,stop);path=dest/(name+'.npy');assert not path.exists()
    np.save(path,np.asarray(result,dtype=object),allow_pickle=True)
    report=dict(start=start,stop=stop,eligible_roots=len(rows),motion_roots=len(result),scenes=sorted({r['scene_id'] for r in result}),
                counts=dict(counts),ledger=ledger,sha256=sha(path),seconds=time.perf_counter()-begin,
                snapshot_sha256=sha(OUT/'snapshot/manifest.json'),preparation_source_sha256=sha(__file__))
    dump(dest/(name+'.json'),report)
    print(json.dumps(dict(shard=name,roots=len(rows),motion=len(result),seconds=report['seconds'])),flush=True)
    return name


def run(workers=3):
    manifest=json.loads((OUT/'snapshot/manifest.json').read_text());n=len(manifest['rows'])
    jobs=[(start,min(n,start+500)) for start in range(0,n,500)]
    previous_rows() # One validated read; forked workers share immutable caches.
    with ProcessPoolExecutor(max_workers=workers) as pool:list(pool.map(shard,jobs))
    reports=[json.loads((OUT/'shards'/('train_%06d_%06d.json'%job)).read_text()) for job in jobs]
    assert sum(r['eligible_roots'] for r in reports)==n
    ledger=[v for r in reports for v in r['ledger']];assert len({v['root_id'] for v in ledger})==n
    supervised=sum(r['motion_roots'] for r in reports)
    summary=dict(eligible_roots=n,motion_roots=supervised,motion_scenes=len({s for r in reports for s in r['scenes']}),
                 statuses=dict(collections.Counter(v['status'] for v in ledger)),
                 target_hygiene=dict(sum((collections.Counter(r['counts']) for r in reports),collections.Counter())),
                 snapshot_sha256=sha(OUT/'snapshot/manifest.json'),shards=[dict(path='shards/'+('train_%06d_%06d.npy'%job),sha256=r['sha256'],motion_roots=r['motion_roots']) for job,r in zip(jobs,reports)],
                 interpretation='all eligible roots covered via validated legal feature caches or fresh legal replay; labels uniquely revalidated against immutable native truth; fixed population prior')
    dump(OUT/'preparation.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3);a=parser.parse_args();run(a.workers)
