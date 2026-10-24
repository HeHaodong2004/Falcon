"""OFFLINE pre/post context for the strict frozen native reappearance cohort."""
import collections
import json
import time
import numpy as np
from .common import BASE, OUT, SNAP, HORIZONS, read, canonical, centers, truth_at, dump, sha
from .evaluate import exact_ids


def run():
    target=OUT/'gap_context.json'
    if target.exists():raise ValueError('preserve previous native context evidence')
    ledger=OUT/'evaluation/reappearance_eligibility_verified.json'
    if not ledger.exists():raise ValueError('completed annotation-accounted native eligibility is required')
    events=json.loads(ledger.read_text());grouped=collections.defaultdict(list)
    keys=set()
    for event in events:
        key=(event['root_id'],event['physical'],event['index'])
        if key not in keys:grouped[event['root_id']].append(event);keys.add(key)
    records=[];start=time.perf_counter()
    for root,entries in grouped.items():
        folder=SNAP/'roots'/root;legal=read(folder/'legal_root.json.gz');_,observations=canonical(folder)
        with np.load(folder/'privileged_future.npz') as native:
            ts=native['timestamps'].copy();truth=native['positions'][...,[0,2]].copy()
        frames=[legal['history'][-1]]+observations
        matches=[exact_ids(ob,ts,truth) for ob in frames]
        def frame(index,person):
            ob=frames[index]
            return dict(timestamp=float(ob['timestamp']),sensor_available=bool(ob.get('human_sensor_available',True)),
                observation_valid=bool(ob.get('observation_valid',True)),raw_positive_centers=len(centers(ob)),
                focal_unique_offline_match=person in matches[index])
        for event in entries:
            person=event['physical'];index=event['index']+1
            previous=[i for i in range(index) if person in matches[i]]
            if not previous:raise AssertionError('reappearance has no genuine strict antecedent')
            last=previous[-1];begin=float(frames[last]['timestamp']);end=float(frames[index]['timestamp'])
            if abs(end-begin-event['gap'])>1e-7:raise AssertionError('eligibility gap differs from actual positive history')
            after=[i for i in range(index+1,len(frames)) if person in matches[i]]
            yaws=np.array([ob['robot_pose'][3] for ob in frames[last:index+1]])
            change=np.angle(np.exp(1j*(yaws-yaws[0])))
            native_times=np.r_[begin,ts[(ts>begin+1e-8)&(ts<end-1e-8)],end]
            positions=np.array([truth_at(ts,truth,float(t))[person] for t in native_times])
            velocity=np.diff(positions,axis=0)/np.diff(native_times)[:,None];speed=np.linalg.norm(velocity,axis=-1)
            turning=(np.abs(np.arctan2(velocity[:-1,0]*velocity[1:,1]-velocity[:-1,1]*velocity[1:,0],
                                     (velocity[:-1]*velocity[1:]).sum(-1))) if len(velocity)>1 else np.empty(0))
            records.append(dict(event,stream='deduplicated physical event',
                previous_positive_timestamps=[float(frames[i]['timestamp']) for i in previous],
                later_positive_timestamps=[float(frames[i]['timestamp']) for i in after],
                immediately_before=[frame(i,person) for i in range(max(0,last-1),last+1)],
                immediately_after=[frame(i,person) for i in range(index,min(len(frames),index+2))],
                intervening_frames=[frame(i,person) for i in range(last+1,index)],
                robot_yaw_net_change_rad=float(change[-1]),robot_yaw_max_excursion_rad=float(abs(change).max()),
                gap_has_stop=bool(np.any(speed<.1)),gap_has_stop_move_change=bool(np.any(speed<.1) and np.any(speed>.3)),
                gap_has_turn=bool(np.any((turning>.3)&(speed[:-1]>.3)&(speed[1:]>.3))) if len(turning) else False,
                native_intervals=len(speed),native_dt_quantiles=np.quantile(np.diff(native_times),[0,.5,.95,1]).tolist()))
    summary=[]
    for name,predicate in [(str(age),lambda r,a=age:r['gap']>=a-1e-8) for age in (.25,.5,1.,1.5,2.)]+[
            ('one_second',lambda r:.75-1e-8<=r['gap']<=1.25+1e-8)]:
        part=[r for r in records if predicate(r)]
        for flag in ('all','gap_has_stop','gap_has_stop_move_change','gap_has_turn','robot_yaw_net_over15deg','robot_yaw_excursion_over15deg'):
            selected=[r for r in part if flag=='all' or
                (abs(r['robot_yaw_net_change_rad'])>np.pi/12 if flag=='robot_yaw_net_over15deg' else
                 r['robot_yaw_max_excursion_rad']>np.pi/12 if flag=='robot_yaw_excursion_over15deg' else r[flag])]
            summary.append(dict(gap_group=name,feature=flag,events=len(selected),roots=len({r['root_id'] for r in selected}),
                scenes=len({r['scene_id'] for r in selected}),
                future_truth={str(h):dict(events=sum(r['future_available'][str(h)] for r in selected),
                    roots=len({r['root_id'] for r in selected if r['future_available'][str(h)]}),
                    scenes=len({r['scene_id'] for r in selected if r['future_available'][str(h)]})) for h in HORIZONS}))
    dump(target,dict(records=records,summary=summary,eligibility_sha256=sha(ledger),snapshot_sha256=sha(SNAP/'manifest.json'),
        source_sha256=sha(BASE/'eventwm_graph/l2_intermittent/gap_context.py'),seconds=time.perf_counter()-start,
        semantics='Annotation-accounted canonical first-person/gap-bin native reappearances, streams deduplicated; physical matches and motion changes OFFLINE only. '
                  'Pre-root physical correspondence is not inferred. Robot yaw/excursion does not prove optical cause. Raw empty frames are not spatial certificates.'))


if __name__=='__main__':run()
