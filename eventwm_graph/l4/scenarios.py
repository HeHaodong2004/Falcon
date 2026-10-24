"""Frozen analytic stress fixtures using verified native ego increments.

These are deterministic simulator-independent regressions, not Habitat scores.
Human trajectories and collision truth are environment-only; the executor sees
the existing anonymous legal tracker. Cases were declared before evaluation.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
from .reference import QuerySpec,compile_reference,ActionSpec
from .geometry import StaticGrid
from .executor import Executor,RuntimeContext,Status

CASES=('straight','corner','multiple_waypoints','orientation','head_on','crossing',
       'slow_leader','temporary_block','narrow_yield','limited_visibility','reobservation','multiple_humans')


def humans(case,t):
    if case in ('head_on','limited_visibility','reobservation'):return [[5.-t,0.,0.]]
    if case in ('crossing','narrow_yield'):return [[2.5,0.,-2.+1.2*t]]
    if case=='slow_leader':return [[2.+.65*t,0.,0.]]
    if case=='temporary_block':return [[2.8,0.,max(0.,t-1.5)*1.2]]
    if case=='multiple_humans':return [[5.5-.8*t,0.,0.],[3.,0.,-2.5+1.3*t]]
    return []


def fixture(case):
    free=np.ones((180,160),bool);height=np.zeros_like(free,dtype=float);origin=[-4,-8]
    route=[[0,0,0],[6,0,0]];yaw=0.
    if case=='corner':route=[[0,0,0],[3,0,0],[3,0,-3]];yaw=math.pi/2
    if case=='multiple_waypoints':route=[[0,0,0],[2,0,0],[2,0,-2],[4,0,-2]]
    if case=='orientation':route=[[0,0,0]];yaw=math.pi/2
    if case=='narrow_yield':
        free[:,:]=False;free[:,73:87]=True
    grid=StaticGrid(free,height,origin,.1,'analytic:'+case)
    q=QuerySpec('analytic:'+case,route,[0,0,0,0],yaw,execution_limit=300,duration_limit=16.8)
    return grid,q,compile_reference(q,grid)


def run(case,output=None,seed=221,speed=1.):
    from scripts.collector.observation import ObservationConfig,LegalHumanTracker,HumanDetection
    grid,q,ref=fixture(case);executor=Executor();handle=executor.start(q,ref,RuntimeContext(grid))
    tracker=LegalHumanTracker(ObservationConfig(90.,90.,10.));latency=[];records=[];collision=False
    rng=np.random.default_rng(seed)
    def observe(p,t):
        truth=humans(case,t*speed);detections=[]
        unavailable=case=='reobservation' and .25<t<1.1
        for h in truth:
            angle=math.atan2(-(h[2]-p[2]),h[0]-p[0])-p[3]
            angle=math.atan2(math.sin(angle),math.cos(angle))
            visible=not unavailable and (case!='limited_visibility' or abs(angle)<math.pi/4)
            if visible:detections.append(HumanDetection(t,tuple(h)))
        return dict(timestamp=t,robot_pose=list(p),tracks=tracker.update(t,detections),human_sensor_available=not unavailable),truth
    ob,truth=observe(q.start_pose,0.);start=time.perf_counter()
    while handle.status==Status.RUNNING:
        d=executor.act(handle,ob,ob);latency.append(d.control_ms)
        if d.action is None:break
        pose=handle.context.action_spec.step(ob['robot_pose'],d.action)
        t=ob['timestamp']+.048
        old=ob;ob,truth=observe(pose,t)
        # Offline continuous relative interpolation, independent .30m human disk.
        for u in np.linspace(0,1,21):
            robot=np.asarray(old['robot_pose'])*(1-u)+pose*u
            for h in humans(case,(old['timestamp']+u*.048)*speed):
                dx=h[0]-robot[0];dz=h[2]-robot[2];c=math.cos(robot[3]);s=math.sin(robot[3])
                x=c*dx-s*dz;z=s*dx+c*dz
                distance=math.hypot(max(-.646495-x,0.,x-.584202),max(-.242846-z,0.,z-.238225))
                collision=collision or distance<.30
        executor.on_transition(handle,dict(action=d.action,timestamp=t,robot_pose=pose,observation_ref=f'actual:{len(records)}',human_collision=collision,
            episode_termination=collision,episode_truncation=len(records)+1>=q.execution_limit))
        records.append(dict(observation=ob,decision=asdict(d),offline_humans=truth))
    report=dict(case=case,seed=seed,speed=speed,status=handle.status.value,human_collision=collision,
                native_environment=False,feedback=executor.feedback(handle),latency_ms=latency,
                wall_seconds=time.perf_counter()-start,reference=asdict(ref),records=records)
    if output is not None:Path(output).write_text(json.dumps(report,indent=2))
    return report


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--case');p.add_argument('--randomized',type=int,default=0)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cases=[a.case] if a.case else CASES
    manifest=dict(cases=list(cases),native_environment=False,ego_action_spec=asdict(ActionSpec()),
        fixture_version='analytic.native_increments.1',seeds=[221],human_truth_never_passed_to_executor=True)
    (a.output/'protocol.json').write_text(json.dumps(manifest,indent=2))
    results=[]
    for case in cases:
        r=run(case,a.output/(case+'.json'));results.append({k:r[k] for k in ('case','status','human_collision','wall_seconds')});print(results[-1],flush=True)
    for i in range(a.randomized):
        case=CASES[4+i%8];speed=float(np.random.default_rng(9000+i).uniform(.5,1.5))
        r=run(case,a.output/(f'random_{i}.json'),9000+i,speed)
        results.append({k:r[k] for k in ('case','status','human_collision','wall_seconds','seed','speed')})
    (a.output/'summary.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
