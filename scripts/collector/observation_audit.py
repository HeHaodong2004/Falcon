#!/usr/bin/env python3
"""Native occluder fixtures and saved-data audits; never a WM input builder."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from collector.core import digest, read, write
from collector.observation import (ObservationConfig, HabitatHumanObserver,
    IdealHumanDetector, LegalHumanTracker, HumanDetection)


def native_scenarios(output, config):
    """Controlled sensor-world humans and an actual Bullet STATIC wall.

    These are sensor fixtures, not navigation episodes or training samples.
    Main collection independently exercises the observer on real native humans.
    """
    import habitat_sim
    import magnum as mn
    settings=habitat_sim.SimulatorConfiguration()
    settings.scene_id='NONE'; settings.create_renderer=False; settings.enable_physics=True
    agent=habitat_sim.agent.AgentConfiguration(); agent.sensor_specifications=[]
    sim=habitat_sim.Simulator(habitat_sim.Configuration(settings,[agent]))
    rows=[]; results={}
    def check(name,condition):
        results[name]={'status':'PASS' if condition else 'FAIL'}
    try:
        manager=sim.get_object_template_manager()
        template=manager.get_template_by_handle(manager.get_template_handles('cubeSolid')[0])
        template.scale=mn.Vector3(.5,2.,.1)
        handle='structured_observation_audit_wall'
        manager.register_template(template,handle)
        wall=sim.get_rigid_object_manager().add_object_by_template_handle(handle)
        wall.translation=[0.,1.,-1.]
        wall.motion_type=habitat_sim.physics.MotionType.STATIC
        # Use the exact production LOS implementation with an empty agent list.
        wrapper=SimpleNamespace(cast_ray=sim.cast_ray,
            agents_mgr=SimpleNamespace(articulated_agents_iter=[]),
            get_rigid_object_manager=sim.get_rigid_object_manager)
        observer=HabitatHumanObserver(wrapper,config)
        detector=IdealHumanDetector(config); tracker=LegalHumanTracker(config)
        origin=np.array([0.,1.,0.]); rotation=np.eye(3)
        def sample(name,t,p,occluded=False,track=None,poison=False):
            # Bullet intentionally ignores transform setters on STATIC objects.
            # Reposition this fixture as kinematic, then restore STATIC before LOS.
            wall.motion_type=habitat_sim.physics.MotionType.KINEMATIC
            location=[0.,1.,-1.] if occluded else [30.,1.,-1.]
            wall.translation=location
            wall.motion_type=habitat_sim.physics.MotionType.STATIC
            assert np.allclose(np.asarray(wall.translation),location)
            human=dict(active=True,position=list(p),diagnostic_id=7)
            if poison:
                human.update(gt_velocity=[999]*3,goal=[999]*3,waypoint_progress=999,
                             future_position=[999]*3,controller_state={'secret':999})
                human['diagnostic_id']=123456
            ds,diag=detector.detect(t,origin,rotation,[human],observer.line_of_sight)
            legal=(track if track is not None else tracker).update(t,ds)
            rows.append(dict(scenario=name,timestamp=t,camera_origin=origin.tolist(),
                camera_rotation=rotation.tolist(),humans=diag,legal_tracks=legal,
                wall_center=np.asarray(wall.translation).tolist(),wall_half_extents=[.5,2.,.1],
                privileged_overlay=True))
            return legal,ds,diag
        a,ds,_=sample('A_visible',0.,(0.,0.,-2.))
        check('A_visible',len(ds)==1 and a[0]['visible_now'] and a[0]['position']==[0.,0.,-2.] and not a[0]['velocity_valid'])
        b,ds,d=sample('B_range',.01,(0.,0.,-config.max_range_m-1),track=LegalHumanTracker(config))
        check('B_out_of_range',not b and not ds and d[0]['reason']=='out_of_range')
        c,ds,d=sample('C_fov',.02,(0.,0.,2.),track=LegalHumanTracker(config))
        check('C_out_of_fov',not c and not ds and d[0]['reason']=='outside_fov')
        d,ds,witness=sample('D_wall_F_never_seen',.03,(0.,0.,-2.),True,LegalHumanTracker(config))
        check('D_static_occlusion',not ds and witness[0]['reason']=='static_occlusion' and witness[0]['ray']['object_id']==wall.object_id)
        check('F_never_seen_hidden',not d)
        seen,_,_=sample('A_second_detection',.05,(.1,0.,-2.))
        hidden,_,_=sample('E_hidden',.13,(.3,0.,-2.),True)
        hidden2,_,_=sample('E_hidden_later',.22,(-.2,0.,-2.),True,poison=True)
        check('E_memory',all(h[0]['position']==seen[0]['position'] and
            h[0]['estimated_velocity']==seen[0]['estimated_velocity'] and
            h[0]['last_seen_time']==.05 and not h[0]['visible_now'] and h[0]['num_observations']==2
            for h in (hidden,hidden2)) and hidden2[0]['observation_age']>hidden[0]['observation_age'])
        again,_,_=sample('G_reappearance',.4,(.15,0.,-2.))
        check('G_reappearance',again[0]['track_id']==seen[0]['track_id'] and again[0]['observation_age']==0 and
              np.allclose(again[0]['estimated_velocity'],[(.15-.1)/(.4-.05),0,0]))
        clean_tracker,poison_tracker=LegalHumanTracker(config),LegalHumanTracker(config)
        equal=True
        for t,p,occ in [(0.,(0,0,-2),False),(.05,(.1,0,-2),False),(.13,(.3,0,-2),True),(.4,(.15,0,-2),False)]:
            x,_,_=sample('poison_clean',t,p,occ,clean_tracker)
            y,_,_=sample('poison_dirty',t,p,occ,poison_tracker,True)
            equal &= x==y
        check('GT_metadata_poison_native',equal)
        expired=tracker.update(2.400001,[])
        check('track_expiration',not expired)
        write(output/'audits/observation_scenarios.json',dict(config=config.__dict__,audits=results,scenarios=rows,
              fixture='Ideal sensor-world human samples; real Habitat-Sim Bullet STATIC cuboid; no learned detector'))
        plot_fixture(output,rows[6],config)
        print('NATIVE SCENARIOS',results,flush=True)
        if any(v['status']=='FAIL' for v in results.values()):
            raise AssertionError('Native visibility scenario failed')
    finally:
        sim.close()


def plot_fixture(output,row,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(6,5))
    origin=np.array(row['camera_origin']); p=np.array(row['humans'][0]['gt_position'])
    w=row['wall_center']; ext=row['wall_half_extents']
    ax.add_patch(Rectangle((w[0]-ext[0],w[2]-ext[2]),2*ext[0],2*ext[2],color='gray',label='STATIC Bullet wall'))
    ax.scatter(origin[0],origin[2],marker='^',color='black',label='sensor')
    ax.scatter(p[0],p[2],color='red',label='hidden human GT (PRIVILEGED)')
    for t in row['legal_tracks']:
        ax.scatter(t['position'][0],t['position'][2],facecolors='none',edgecolors='blue',s=100,label='legal remembered position')
    ax.plot([origin[0],p[0]],[origin[2],p[2]],'r:',label='blocked torso LOS')
    angle=np.deg2rad(config.horizontal_fov_deg/2)
    for sign in (-1,1): ax.plot([0,sign*3*np.sin(angle)],[0,-3*np.cos(angle)],'k--')
    ax.set(xlabel='world x (m)',ylabel='world z (m)',title='PRIVILEGED DIAGNOSTIC OVERLAY\ncontrolled static-occlusion fixture')
    ax.legend(fontsize=8); ax.axis('equal'); fig.tight_layout()
    fig.savefig(output/'diagnostics/observation_occlusion.png'); plt.close(fig)


def visible_prefix(output):
    """Two isolated 40-WAIT prefix audits; same initialized robot heading.

    Orientation is chosen once during controlled setup, before any history.
    This does not modify the primary collection or its replay recipe.
    """
    import math
    import random
    import habitat_sim
    import habitat
    import torch
    from omegaconf import OmegaConf
    from habitat.config import read_write
    from collector.runtime import Recorder, state
    from collector.core import legal_observation
    config=ObservationConfig(**read(output/'legal_observations/observation_config.json'))
    records=[]
    for enabled in (False,True):
        random.seed(2301); np.random.seed(2301); torch.manual_seed(2301); torch.set_num_threads(1)
        cfg=OmegaConf.load(output/'audits/workers/root/resolved_config.yaml')
        with read_write(cfg):
            cfg.habitat.task.controlled_audit_path=str(output/'audits'/f'visible_prefix_crowd_{enabled}.json')
        with habitat.Env(config=cfg.habitat) as env:
            env.seed(2301); obs=env.reset(); sim=env.sim
            observer=HabitatHumanObserver(sim,config)
            origin,rotation=observer.camera_pose()
            target=np.asarray(sim.get_agent_data(1).articulated_agent.base_pos)+[0,config.torso_height_m,0]
            forward=rotation @ np.array([0.,0.,-1.]); delta=target-origin
            turn=math.atan2(-delta[2],delta[0])-math.atan2(-forward[2],forward[0])
            robot=sim.get_agent_data(0).articulated_agent
            robot.base_rot=float(robot.base_rot)+turn
            sim.maybe_update_articulated_agent()
            obs=sim._sensor_suite.get_observations(sim.get_sensor_observations())
            recorder=Recorder(env,1); recorder.capture('initial')
            history=[]
            def observe(observations):
                r=state(robot)
                tracks=observer.observe(sim.get_world_time(),1) if enabled else None
                return legal_observation(observations,r['position'],r['yaw'],sim.get_world_time(),[0.,0.],tracks)[0]
            history.append(observe(obs))
            for i in range(40):
                recorder.index=i
                obs=env.step({'action':('agent_0_collector_velocity','agent_1_oracle_nav_randcoord_action_obstacle'),
                    'action_args':{'agent_0_base_vel':np.zeros(2),'agent_1_oracle_nav_randcoord_action_obstacle':np.ones(1)}})
                history.append(observe(obs))
                if env.episode_over: raise RuntimeError('Controlled visible-prefix audit terminated')
            records.append(dict(observer_enabled=enabled,truth=recorder.rows,history=history,
                                diagnostics=observer.diagnostics,initial_yaw=history[0]['robot_yaw']))
    same=digest(records[0]['truth'])==digest(records[1]['truth'])
    detections=sum(t['visible_now'] for r in records[1]['history'] for t in r['human_tracks'])
    write(output/'audits/visible_prefix.json',dict(physical_identical=same,legal_detections=detections,
          observer_on_off=records,setup='Robot yaw faces current human at initialization, before 40 real WAIT actions'))
    write(output/'legal_observations/visible_prefix_audit_history.json',records[1]['history'])
    print('VISIBLE PREFIX',dict(physical_identical=same,legal_detections=detections),flush=True)
    if not same or detections<2: raise AssertionError('Visible online prefix audit failed')


def validate(output, baseline):
    from eventwm_collect import branches
    rows,old=branches(output),branches(baseline)
    audits={}
    def check(name,condition,details):
        audits[name]=dict(status='PASS' if condition else 'FAIL',details=details)
    oldmap={b['branch_id']:b for b in old}
    physical=all(digest(b['truth'])==digest(oldmap[b['branch_id']]['truth']) and
                 b['controls']==oldmap[b['branch_id']]['controls'] for b in rows)
    check('physical_consequences_identical',len(rows)==len(old) and physical,
          'Every native state/contact/timestamp and issued control compared exactly to Phase 1 by branch ordinal')
    repeated=read(output/'audits/workers/observed_repeat/branch.json')
    original=next(b for b in rows if b['query_id']==repeated['query_id'])
    check('populated_tracker_reproducibility',all(digest(original[k])==digest(repeated[k]) for k in
          ('truth','controls','legal_observations')) and any(r['human_tracks'] for r in repeated['legal_observations']),
          'An additional branch with both visible and lost tracks replayed exactly in a fresh process')
    def normalized(q): return {k:v for k,v in q.items() if k not in ('query_id','root_id')}
    qs,oldqs=read(output/'queries/queries.json'),read(baseline/'queries/queries.json')
    check('query_behavior_unchanged',[normalized(q['query']) for q in qs]==[normalized(q['query']) for q in oldqs],
          'All frozen query fields identical except IDs derived from the enriched history fingerprint')
    labs,oldlabs=read(output/'labels/labels.json'),read(baseline/'labels/labels.json')
    check('labels_identical',[{k:v for k,v in x.items() if k!='query_id'} for x in labs]==
          [{k:v for k,v in x.items() if k!='query_id'} for x in oldlabs],'All offline labels exact; only query IDs excluded')
    config=ObservationConfig(**read(output/'legal_observations/observation_config.json'))
    workers=['root']+[b['branch_id'] for b in rows]
    history=read(output/'legal_observations/history.json')
    root_time=history[-1]['timestamp']
    counts=dict(root_detections=0,branch_detections=0,root_track_observations=0,
                branch_lost_track_observations=0,occluded_human_observations=0,
                never_seen_hidden_observations=0,reappearances=0)
    replay_ok,causal_ok,velocity_ok,memory_ok=True,True,True,True
    timeline=[]; best_rows=[]; best_score=-1
    for bid in workers:
        diagnostics=read(output/'audits/workers'/bid/'observation_diagnostics.json')
        tracker=LegalHumanTracker(config); previous={}; rebuilt=[]
        for r in diagnostics:
            t=r['timestamp']; ds=[HumanDetection(t,tuple(h['gt_position'])) for h in r['humans'] if h['detected']]
            tracks=tracker.update(t,ds); replay_ok &= tracks==r['legal_tracks']
            rebuilt.append((t,tracks))
            for track in tracks:
                prev=previous.get(track['track_id'])
                if prev and not track['visible_now']:
                    memory_ok &= all(track[k]==prev[k] for k in
                        ('position','estimated_velocity','last_seen_time','num_observations','velocity_valid'))
                    memory_ok &= abs(track['observation_age']-(t-track['last_seen_time']))<1e-9
                if prev and track['visible_now']:
                    dt=t-prev['last_seen_time']
                    if config.min_velocity_dt_s < dt <= config.velocity_max_gap_s:
                        velocity_ok &= track['velocity_valid'] and np.allclose(track['estimated_velocity'],
                            (np.array(track['position'])-prev['position'])/dt,rtol=0,atol=1e-12)
                    else:
                        velocity_ok &= not track['velocity_valid'] and track['estimated_velocity'] is None
                    if not prev['visible_now'] and bid!='root' and t>root_time: counts['reappearances']+=1
                if not prev: velocity_ok &= not track['velocity_valid']
                previous[track['track_id']]=copy.deepcopy(track)
            primary=bid=='root' or (bid.startswith('b') and t>root_time)
            if primary:
                if bid=='root':
                    counts['root_detections']+=len(ds); counts['root_track_observations']+=len(tracks)
                else:
                    counts['branch_detections']+=len(ds)
                    counts['branch_lost_track_observations']+=sum(not x['visible_now'] for x in tracks)
                counts['occluded_human_observations']+=sum(h['reason']=='static_occlusion' for h in r['humans'])
                counts['never_seen_hidden_observations']+=int(not ds and not tracks)
        prefix=[{'timestamp':t,'human_tracks':x} for t,x in rebuilt if t<=root_time]
        causal_ok &= prefix==[{'timestamp':r['timestamp'],'human_tracks':r['human_tracks']} for r in history]
        if bid!='root':
            legal=next(b['legal_observations'] for b in rows if b['branch_id']==bid)
            causal_ok &= [(x['timestamp'],x['human_tracks']) for x in legal]==[(t,x) for t,x in rebuilt if t>=root_time]
            score=sum(not tr['visible_now'] for r in diagnostics for tr in r['legal_tracks'])
            if score>best_score and any(r['legal_tracks'] for r in diagnostics):
                best_score=score; best_rows=diagnostics; best_bid=bid
    check('online_detection_tracker_replay',replay_ok,'Legal-only tracker reconstructed sequentially from gated detections matches every saved track')
    check('root_future_causal_boundary',causal_ok,'Each branch prefix equals root history; future observations occur only at/after root')
    check('velocity_from_legal_history',velocity_ok,'Every new and reassociated visible track checked against timestamped legal finite differences / gap invalidation')
    check('hidden_memory_not_refreshed',memory_ok,'Every remembered hidden track preserves last measurement, velocity and last-seen count/time')
    check('real_visible_humans_detected',counts['root_detections']+counts['branch_detections']>0,counts)
    scenarios=read(output/'audits/observation_scenarios.json')
    for key,value in scenarios['audits'].items(): audits[key]=value
    prefix=read(output/'audits/visible_prefix.json')
    check('real_visible_online_prefix',prefix['legal_detections']>=2 and prefix['physical_identical'],
          dict(legal_detections=prefix['legal_detections'],observer_on_off_physical_identical=prefix['physical_identical']))
    for r in best_rows:
        tracks=r['legal_tracks'] or [None]
        for tr in tracks:
            timeline.append(dict(privileged_diagnostic=True,branch_id=best_bid,timestamp=r['timestamp'],
                gt_visibility_reason=r['humans'][0]['reason'] if r['humans'] else 'no_active_human',
                track_id=tr['track_id'] if tr else '',visible_now=tr['visible_now'] if tr else False,
                x=tr['position'][0] if tr else '',z=tr['position'][2] if tr else '',
                vx=tr['estimated_velocity'][0] if tr and tr['velocity_valid'] else '',
                vz=tr['estimated_velocity'][2] if tr and tr['velocity_valid'] else '',
                age=tr['observation_age'] if tr else '',last_seen=tr['last_seen_time'] if tr else ''))
    with (output/'diagnostics/human_observation_timeline.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(timeline[0])); writer.writeheader(); writer.writerows(timeline)
    write(output/'audits/structured_observation_audits.json',dict(audits=audits,counts=counts,
          baseline=str(baseline),timeline_branch=best_bid))
    print(json.dumps(dict(audits={k:v['status'] for k,v in audits.items()},counts=counts),indent=2))
    if any(v['status']=='FAIL' for v in audits.values()): raise AssertionError('Structured observation audit failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['scenarios','prefix','validate'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=REPO/'datasets/raw/phase1_castle_smoke_v3')
    args=p.parse_args(); output=args.output.resolve()
    if args.command=='scenarios':
        report=output/'audits/observation_scenarios.json'
        failed=output/'audits/observation_scenarios_initial_failure.json'
        if report.exists() and not failed.exists() and any(v['status']=='FAIL' for v in read(report)['audits'].values()):
            write(failed,read(report))
        native_scenarios(output,ObservationConfig(**read(output/'legal_observations/observation_config.json')))
    elif args.command=='prefix': visible_prefix(output)
    else: validate(output,args.baseline.resolve())


if __name__=='__main__': main()
