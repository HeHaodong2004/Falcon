"""Full sealed legal-sequence gap audit. Physical correspondence is offline only."""
import collections
import json
import time
import numpy as np
from .common import SNAP, OUT, OLD, centers, read, truth_at, dump, sha, age_bin


def audit():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((SNAP/'manifest.json').read_text())
    counts = collections.Counter()
    support = collections.defaultdict(lambda: [set(), set()])
    values = collections.defaultdict(list)
    examples = []
    started = time.perf_counter()
    for ri, row in enumerate(manifest['rows']):
        folder = SNAP/'roots'/row['root_id']
        legal = read(folder/'legal_root.json.gz')
        nodes = read(folder/'future_observations.json.gz')
        with np.load(folder/'privileged_future.npz') as f:
            ts, xp = f['timestamps'], f['positions'][...,[0,2]]
        split = row['split']
        def add(key, amount=1):
            key = split+':'+key
            counts[key] += amount
            support[key][0].add(row['root_id']); support[key][1].add(row['scene_id'])
        # Source IDs are used ONLY for this separately named pre-root continuity
        # diagnostic, never online nor asserted to be physical identities.
        last = {}; freq = collections.Counter()
        for ob in legal['history']:
            if not ob.get('human_sensor_available',True) or not ob.get('observation_valid',True):
                continue
            for tr in ob.get('tracks', []):
                if not tr.get('visible_now') or not tr.get('valid', True):
                    continue
                if abs(tr['last_seen_time']-ob['timestamp']) > 1e-7:
                    continue
                key = tr['track_id']; freq[key] += 1
                if key in last:
                    values[split+':history_source_id_intermeasurement'].append(ob['timestamp']-last[key])
                last[key] = ob['timestamp']
            values[split+':history_last_seen_age'].extend(ob['timestamp']-t for t in last.values())
        add('history_source_tracks_multitimestamp', sum(n>1 for n in freq.values()))
        values[split+':history_duration'].append(legal['history'][-1]['timestamp']-legal['history'][0]['timestamp'])
        # All legal branch nodes, retaining ancestry. Count roots/scenes, never
        # regard siblings or DAG nodes as independent physical futures.
        states = {}
        root = legal['history'][-1]
        sequence = [(-1, None, root)]+[(n['node_id'], n['parent'], n['observation']) for n in nodes]
        for ni, parent, ob in sequence:
            if ni == -1:
                prev = {}; prevob = root
            else:
                if parent is not None and parent not in states:
                    raise ValueError('orphan legal DAG parent')
                prev, prevob = states[-1 if parent is None else parent]
            current = dict(prev)
            y = truth_at(ts, xp, ob['timestamp'])
            if y is None:
                add('nodes_without_native_truth'); states[ni] = (current, ob); continue
            z = centers(ob); mapping = []
            for zi in z:
                d = np.linalg.norm(y-zi, axis=-1)
                if not len(d) or d.min() > 2e-4:
                    add('unmatched_positive_centers'); continue
                j = int(d.argmin()); mapping.append(j)
                if np.count_nonzero(d<2e-4)>1:
                    add('ambiguous_exact_physical_matching')
                if len(d)>1 and np.partition(d,1)[1] < .5:
                    add('close_human_association_cases')
                if j in prev:
                    last_t, last_p, last_yaw, last_step = prev[j]
                    gap = float(ob['timestamp']-last_t)
                    # Missing INTERVENING legal sensor frames distinguish a gap
                    # from ordinary native sampling jitter.
                    if ob['timestamp']-prevob['timestamp'] < gap-1e-7:
                        add('reobservations:'+age_bin(gap))
                        values[split+':native_reobservation_gap'].append(gap)
                        yaw = abs(np.angle(np.exp(1j*(ob['robot_pose'][3]-last_yaw))))
                        add('reobservation_yaw_change_over15deg' if yaw>np.pi/12 else 'reobservation_yaw_change_under15deg')
                        for th in (.25,.5,1.,1.5,2.):
                            if gap>=th-1e-8:
                                add('reobservations_ge'+str(th))
                        k0 = int(np.searchsorted(ts,last_t-1e-8)); k1 = min(len(ts)-1,int(np.searchsorted(ts,ob['timestamp'])))
                        v = np.diff(xp[k0:k1+1,j],axis=0)/np.diff(ts[k0:k1+1])[:,None] if k1>k0 else np.empty((0,2))
                        if len(v):
                            s = np.linalg.norm(v,axis=-1)
                            if s.min()<.1: add('gap_contains_stop')
                            if s.min()<.1 and s.max()>.3: add('gap_stop_move_change')
                            if len(v)>1:
                                a = np.arctan2(v[:,1],v[:,0]); turn=np.abs(np.angle(np.exp(1j*np.diff(a))))
                                if np.any((turn>.3)&(s[1:]>.3)&(s[:-1]>.3)): add('gap_contains_turn')
                        add('reobservation_has_previous_positive')
                        if len(examples)<200 and gap>=.5:
                            examples.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],split=split,node_id=ni,physical_index=j,last_time=last_t,time=ob['timestamp'],gap=gap,yaw_change=yaw))
                current[j]=(ob['timestamp'],zi,ob['robot_pose'][3],ni)
            for j, item in current.items():
                age=float(ob['timestamp']-item[0]);add('native_age:'+age_bin(age))
                values[split+':native_age'].append(age)
                for h in (.5,1.,2.,3.):
                    add('age_horizon:'+age_bin(age)+':'+str(h)+('supported' if ob['timestamp']+h<=ts[-1]+1e-8 else 'censored'))
            states[ni]=(current,ob)
        add('roots'); add('future_nodes', len(nodes))
        if (ri+1)%1000==0:
            print(json.dumps(dict(roots=ri+1,seconds=round(time.perf_counter()-started,1))),flush=True)
    quant={k:dict(n=len(v),quantiles=np.quantile(v,[0,.25,.5,.75,.95,.99,1]).tolist() if v else []) for k,v in values.items()}
    result=dict(snapshot_sha256=sha(SNAP/'manifest.json'),counts={k:dict(events=n,roots=len(support[k][0]),scenes=len(support[k][1])) for k,n in counts.items()},distributions=quant,examples=examples,seconds=time.perf_counter()-started,
                caveats=['Pre-root source-ID continuity is a proxy, not verified physical identity.', 'Native future matching uses exact positive centers and offline truth only.', 'All-DAG event counts are dependent siblings; unique root and scene support governs.', 'Yaw change is measured camera/robot orientation change, not proof of occlusion cause.', 'No protected payload is opened; the sealed snapshot contains eligible train/dev only.'])
    dump(OUT/'data_and_gaps.json',result)


if __name__=='__main__':
    audit()
