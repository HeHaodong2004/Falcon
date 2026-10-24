"""Offline integrity audit of eligible immutable payloads; no model fitting."""
import collections
import gzip
import json
import time
import numpy as np
from .common import OUT,OLD,sha,dump

def run():
    snap=OUT/'snapshot';manifest=json.loads((snap/'manifest.json').read_text());old=json.loads((OLD/'snapshot/manifest.json').read_text())
    excluded={r['scene_id'] for r in old['rows'] if r['split']=='dev'}|{r['scene_id'] for r in old['protected']}
    assert not ({r['scene_id'] for r in manifest['rows']}&excluded)
    count=collections.Counter();states=collections.Counter();schemas=collections.Counter();durations=[];history_durations=[];errors=[];start=time.perf_counter()
    for i,row in enumerate(manifest['rows']):
        folder=snap/'roots'/row['root_id']
        with gzip.open(folder/'legal_root.json.gz','rt') as f:legal=json.load(f)
        try:
            assert set(legal)=={'history','history_actions','partial_graph','pointgoal','schema','sensor','version'}
            history=legal['history'];ts=np.array([o['timestamp'] for o in history]);assert len(ts)>0 and np.isfinite(ts).all() and np.all(np.diff(ts)>0)
            graph=legal['partial_graph'];assert graph['timestamp']<=ts[-1]+1e-8
            schemas[(legal['schema'],graph['schema'])]+=1;history_durations.append(float(ts[-1]-ts[0]))
            for c in graph['cells']:
                assert c['last_seen']<=ts[-1]+1e-8 and c['first_seen']<=c['last_seen']+1e-8
                states[str(c['state'])]+=1
            for ob in history:
                assert np.isfinite(ob['robot_pose']).all()
                for t in ob['tracks']:
                    assert t['last_seen_time']<=ob['timestamp']+1e-8
                    assert np.isfinite(t['position']).all()
                    count['anonymous_track_records']+=1
                    if t['visible_now'] and t['valid']:
                        assert abs(t['last_seen_time']-ob['timestamp'])<1e-7
                        count['genuine_positive_centers']+=1
            with np.load(folder/'privileged_future.npz') as f:
                native=f['timestamps'];positions=f['positions']
                assert positions.shape[0]==len(native) and np.all(np.diff(native)>0) and np.isfinite(positions).all()
                assert abs(native[0]-ts[-1])<1e-7
                durations.append(float(native[-1]-native[0]));count['native_truth_samples']+=len(native)
                count['native_people']+=positions.shape[1]
            count['legal_histories']+=1
        except Exception as e:errors.append(dict(root_id=row['root_id'],error=repr(e)))
        if (i+1)%5000==0:print(json.dumps(dict(roots=i+1,errors=len(errors),seconds=time.perf_counter()-start)),flush=True)
    dump(OUT/'data_integrity.json',dict(snapshot_sha256=sha(snap/'manifest.json'),counts=dict(count),errors=errors,
        scene_exclusion_disjoint=True,schemas=[dict(legal=k[0],graph=k[1],roots=v) for k,v in schemas.items()],
        stored_cell_states=dict(states),truth_duration_quantiles=np.quantile(durations,[0,.05,.5,.95,1]).tolist(),
        history_duration_quantiles=np.quantile(history_durations,[0,.05,.5,.95,1]).tolist(),seconds=time.perf_counter()-start,
        legality='Only eligible training payloads opened. Offline future arrays validate supervision timing; never passed to the online filter. Unknown cells are not filled, the graph is not used as an optical/human support map.'))
    if errors:raise ValueError('Legal-history/native-truth incompatibility; preserve and report, do not repair data')

if __name__=='__main__':run()
