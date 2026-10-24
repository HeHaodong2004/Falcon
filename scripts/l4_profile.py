"""Bounded, single-thread control profiling; does not advance any environment."""
import argparse
import json
import os
import platform
from pathlib import Path
import resource
import time
import numpy as np
from eventwm_graph.l4.scenarios import fixture
from eventwm_graph.l4 import Executor,RuntimeContext
from eventwm_graph.l4.search import human_cost,path_score


def quantiles(values):
    return dict(zip(('p50','p95','p99','max'),map(float,[*np.percentile(values,[50,95,99]),max(values)])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);results={}
    grid,q,ref=fixture('straight');executor=Executor();context=RuntimeContext(grid)
    for name,positions in [('no_human',[]),('typical',[[3.,0,0], [4.,0,2.]]),
                           ('dense',[[3.,0,0],[4.,0,2.],[4.,0,-2.],[5.,0,3.],[5.,0,-3.],[-2.,0,2.]])]:
        ob=dict(timestamp=0.,robot_pose=q.start_pose,tracks=[dict(track_id=str(i),position=p,
            estimated_velocity=[-.8,0,0],velocity_valid=True,visible_now=True,last_seen_time=0.) for i,p in enumerate(positions)])
        first=time.perf_counter();decision=executor.act(executor.start(q,ref,context),ob,ob)
        startup_ms=(time.perf_counter()-first)*1000
        durations=[];counts=[]
        for _ in range(120):
            h=executor.start(q,ref,context)
            began=time.perf_counter();d=executor.act(h,ob,ob);durations.append((time.perf_counter()-began)*1000)
            counts.append(d.collision_check.get('expanded',0))
        results[name]=dict(control_ms=quantiles(durations),first_call_ms=startup_ms,
            candidate_counts=quantiles(counts),samples=len(durations),actions_checked=True)
    people=executor._people(ob,0.);before=np.array(q.start_pose);after=context.action_spec.step(before,'forward')
    components={}
    calls={'action_generation':lambda:[context.action_spec.step(before,x) for x in context.action_spec.actions],
           'static_swept_check':lambda:grid.motion_free(before,after),
           'human_swept_check_dense':lambda:human_cost(before,after,0,people,.056),
           'candidate_score':lambda:path_score(np.array([before,after]),np.array([0]),np.array(q.target),0.,True,0)}
    for name,call in calls.items():
        call();durations=[]
        for _ in range(1000):
            began=time.perf_counter();call();durations.append((time.perf_counter()-began)*1000)
        components[name]=quantiles(durations)
    report=dict(profile=results,component_microbench_ms=components,
        component_scope='isolated Python-to-kernel calls; not additive decomposition of the beam',
        hardware=dict(platform=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count(),
                      cpu_model=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'')),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        threads=1,horizon=32,beam_width=16,nominal_horizon_s=1.536,screening_horizon_s=1.792,
        extra_maneuver_templates=36,forecast='prediction-unavailable fallback; native B2 timings separate')
    (a.output/'profile.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
