#!/usr/bin/env python3
"""Privileged audit of frozen association on controlled native articulated humans.

Pose fixtures use a clear elevated volume in the real scene; native jaw camera,
actual articulated human base positions, native static raycasting and simulator
clock. These are sensor experiments, not controller/navigation trajectories.
"""
import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from collector.core import read, write
from collector.observation import HabitatHumanObserver, IdealHumanDetector, LegalHumanTracker, ObservationConfig


def audit(output):
    import habitat_sim
    import habitat
    from collector.runtime import build_config
    from habitat.config import read_write
    output.mkdir(parents=True,exist_ok=True)
    case=next(c for c in read(REPO/'evaluation/controlled50/cases.json') if c['tag']=='004_castle_h4')
    cfg=build_config(REPO/'evaluation/controlled50',case,output,2000,REPO/'pretrained_model/falcon_noaux_25.pth')
    with read_write(cfg): cfg.habitat.simulator.concur_render=False
    config=ObservationConfig.from_depth_sensor(cfg.habitat.simulator.agents.agent_0.sim_sensors.jaw_depth_sensor)
    checks=dict(one_to_one=True,declared_greedy_rule=True,legal_velocity=True,hidden_memory=True,
                identity_poison_and_order=True,valid_counts=True)
    results={}; all_rows={}
    fixtures={
        'separated_2': [[(-1.4+.015*i,4.),(1.4-.015*i,4.)] for i in range(20)],
        'crossing_2': [[(-.95+.1*i,4.),(.95-.1*i,4.06)] for i in range(20)],
        # Physically bounce after proximity: same observations also admit passing.
        'ambiguous_bounce_2': [[(-abs(-.95+.1*i),4.),(abs(.95-.1*i),4.01)] for i in range(20)],
        'loss_reacquire_2': [[(-1.2,4.),(1.2,4.)]]*3 + [[(-1.2,-4.),(1.2,4.)]]*8 +
                            [[(-1.2,4.),(1.2,4.)]]*3 + [[(-1.2,-4.),(1.2,4.)]]*25 + [[(-1.2,4.),(1.2,4.)]]*2,
        'stress_4': [[(-.95+.1*i,4.),(.95-.1*i,4.06),(-.8+.08*i,4.45),(.8-.08*i,4.51)] for i in range(20)]}
    with habitat.Env(config=cfg.habitat) as env:
        env.seed(case['seed']); env.reset(); sim=env.sim
        robot=sim.get_agent_data(0).articulated_agent
        original=np.asarray(robot.base_pos).copy(); original[1]+=30
        robot.base_pos=original; sim.maybe_update_articulated_agent()
        probe=HabitatHumanObserver(sim,config); origin,rotation=probe.camera_pose()
        forward=rotation @ np.array([0.,0.,-1.]); forward[1]=0; forward/=np.linalg.norm(forward)
        right=rotation @ np.array([1.,0.,0.]); right[1]=0; right/=np.linalg.norm(right)
        for name,poses in fixtures.items():
            observer=HabitatHumanObserver(sim,config); shadow=LegalHumanTracker(config)
            detector=observer.detector.detect; captured={}
            def detect(t,o,r,world,los):
                ds,diag=detector(t,o,r,world,los)
                poison=copy.deepcopy(world[::-1])
                for h in poison:
                    h.update(diagnostic_id=-999,gt_velocity=[999]*3,goal=[999]*3,future_position=[999]*3)
                other,_=IdealHumanDetector(config).detect(t,o,r,poison,los)
                checks['identity_poison_and_order'] &= ds==other
                captured.update(detections=ds,shadow=shadow.update(t,other))
                return ds,diag
            observer.detector.detect=detect
            previous={}; identities={}; switches=[]; rows=[]; reacquired=[]; expired=[]
            for index,pose in enumerate(poses):
                sim.step_world(.1)
                robot.base_pos=original; sim.maybe_update_articulated_agent()
                for slot,(x,depth) in enumerate(pose,1):
                    target=origin+right*x+forward*depth; target[1]=origin[1]-config.torso_height_m
                    sim.get_agent_data(slot).articulated_agent.base_pos=target
                tracks=observer.observe(sim.get_world_time(),len(pose)); t=float(sim.get_world_time())
                ds=captured['detections']; checks['identity_poison_and_order'] &= tracks==captured['shadow']
                diag=observer.diagnostics[-1]
                candidates=[]
                alive={tid:v for tid,v in previous.items() if t-v['last_seen_time']<=config.track_max_age_s}
                for tid,tr in alive.items():
                    pred=np.array(tr['position'])
                    if tr['velocity_valid']: pred+=np.array(tr['estimated_velocity'])*min(t-tr['last_seen_time'],.5)
                    for di,d in enumerate(ds):
                        distance=float(np.linalg.norm(pred-d.position))
                        if distance<=.8: candidates.append((distance,tid,di))
                selected=[]; used_t=set(); used_d=set()
                for distance,tid,di in sorted(candidates):
                    if tid not in used_t and di not in used_d:
                        selected.append(dict(track_id=tid,detection_index=di,distance_m=distance)); used_t.add(tid); used_d.add(di)
                actual=[]
                for tr in tracks:
                    tid=tr['track_id']; old=previous.get(tid)
                    checks['valid_counts'] &= tr['valid'] and 0<=tr['observation_age']<=2.000001
                    if tr['visible_now']:
                        matches=[di for di,d in enumerate(ds) if np.allclose(tr['position'],d.position,rtol=0,atol=1e-7)]
                        checks['one_to_one'] &= len(matches)==1
                        di=matches[0]; actual.append((tid,di))
                        if tid in alive:
                            checks['declared_greedy_rule'] &= any(s['track_id']==tid and s['detection_index']==di for s in selected)
                        if old and tid in alive:
                            dt=t-old['last_seen_time']; valid=1e-6<dt<=.5
                            checks['legal_velocity'] &= tr['velocity_valid']==valid
                            if valid: checks['legal_velocity'] &= np.allclose(tr['estimated_velocity'],(np.array(tr['position'])-old['position'])/dt)
                            if not old['visible_now']: reacquired.append(dict(timestamp=t,track_id=tid,gap=dt))
                        else: checks['legal_velocity'] &= not tr['velocity_valid']
                        physical=next(h['diagnostic_id'] for h in diag['humans'] if h['detected'] and np.allclose(h['gt_position'],tr['position'],atol=1e-7,rtol=0))
                        if tid in identities and identities[tid]!=physical: switches.append(dict(timestamp=t,track_id=tid,old_gt=identities[tid],new_gt=physical))
                        identities[tid]=physical
                    elif old:
                        checks['hidden_memory'] &= all(tr[k]==old[k] for k in ('position','estimated_velocity','last_seen_time','num_observations'))
                        checks['hidden_memory'] &= tr['observation_age']>old['observation_age']
                checks['one_to_one'] &= len(actual)==len(ds)==len({di for _,di in actual})==len({tid for tid,_ in actual})
                checks['declared_greedy_rule'] &= all((s['track_id'],s['detection_index']) in actual for s in selected)
                for tid in previous:
                    if tid not in {tr['track_id'] for tr in tracks}:
                        expired.append(dict(timestamp=t,track_id=tid,age=t-previous[tid]['last_seen_time']))
                        checks['hidden_memory'] &= t-previous[tid]['last_seen_time']>2
                rows.append(dict(timestamp=t,detections=[asdict(d) for d in ds],tracks=tracks,
                    candidate_distances=[dict(distance_m=d,track_id=tid,detection_index=i) for d,tid,i in sorted(candidates)],
                    selected_associations=selected,privileged_humans=diag['humans']))
                previous={tr['track_id']:copy.deepcopy(tr) for tr in tracks}
            results[name]=dict(humans=len(poses[0]),samples=len(rows),detections=sum(len(r['detections']) for r in rows),
                identity_switches=len(switches),switch_events=switches,reacquisitions=reacquired,expirations=expired,
                maximum_retained_tracks=max(len(r['tracks']) for r in rows))
            all_rows[name]=rows
    checks['separated_stable']=results['separated_2']['identity_switches']==0 and all(len(r['tracks'])==2 for r in all_rows['separated_2'])
    checks['stress_no_collapse']=all(len(r['tracks'])==len(r['detections'])==4 for r in all_rows['stress_4'])
    loss=results['loss_reacquire_2']
    checks['loss_exercised']=bool(loss['reacquisitions']) and bool(loss['expirations']) and loss['identity_switches']==0
    report=dict(status='PASS' if all(checks.values()) else 'FAIL',checks={k:bool(v) for k,v in checks.items()},
                config=asdict(config),cases=results,fixture='Native articulated humans in controlled elevated clear volume; pose-driven sensor fixtures, not navigation rollouts',
                privileged_audit_only=True,physical_identity_is_not_legal_input=True)
    write(output/'audit.json',report); write(output/'trajectories.json',all_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,name in zip(axes,['crossing_2','ambiguous_bounce_2','stress_4']):
        rows=all_rows[name]; t0=rows[0]['timestamp']
        for gt in range(1,results[name]['humans']+1):
            ax.plot([r['timestamp']-t0 for r in rows],
                    [np.dot(np.array(next(h['gt_position'] for h in r['privileged_humans'] if h['diagnostic_id']==gt))-origin,right) for r in rows],
                    'k--',alpha=.35)
        for tid in sorted({tr['track_id'] for r in rows for tr in r['tracks']}):
            values=[(r['timestamp']-t0,np.dot(np.array(tr['position'])-origin,right)) for r in rows for tr in r['tracks'] if tr['track_id']==tid]
            ax.plot(*zip(*values),'.-',label=tid)
        ax.set(title=f"{name}: {results[name]['identity_switches']} ID switches",xlabel='simulator seconds',ylabel='lateral position (m)')
        ax.legend(fontsize=7)
    fig.suptitle('PRIVILEGED AUDIT: dashed physical trajectories, coloured legal tracks')
    fig.tight_layout(); fig.savefig(output/'trajectories.png'); plt.close(fig)
    print(report,flush=True)
    if report['status']!='PASS': raise AssertionError('Multi-human audit failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args().output.absolute())
