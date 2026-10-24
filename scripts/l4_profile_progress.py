"""Profile deterministic replay of a native B1 trace, outside simulator cost."""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import time
import habitat_sim
import numpy as np
from eventwm_graph.benchmark.map import load_certified_map
from eventwm_graph.l4 import QuerySpec, CompiledReference, RuntimeContext, Executor
from eventwm_graph.l4.geometry import StaticGrid
from l4_summarize import percentiles


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace-folder',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    trace=json.loads((a.trace_folder/'actual.json').read_text());report=json.loads((a.trace_folder/'report.json').read_text())
    row=json.loads((a.trace_folder/'reference.json').read_text());row['query']=QuerySpec(**row['query']);ref=CompiledReference(**row)
    hmap=load_certified_map(report['case']['manifest']);g=StaticGrid.from_navigation(hmap.navigation,ref.nominal_waypoints);g.predictor=hmap.model
    ctx=RuntimeContext(g);executor=Executor();profile=cProfile.Profile();control=[];transition=[]
    def replay(timed):
        h=executor.start(ref.query,ref,ctx)
        events=report['feedback']['transition_events']
        for i,action in enumerate(report['feedback']['actual_actions']):
            ob=trace['observations'][i];then=trace['observations'][i+1]
            start=time.perf_counter();d=executor.act(h,ob,ob)
            if timed:control.append((time.perf_counter()-start)*1000)
            assert d.action==action, 'Trace action diverged at '+str(i)
            event=events[i]
            start=time.perf_counter();executor.on_transition(h,dict(action=action,timestamp=then['timestamp'],
                robot_pose=then['robot_pose'],observation_ref=f'actual:{i+1}',human_collision=event['human_collision'],
                static_collision=event['static_collision'],episode_termination=i==len(events)-1 and report['feedback']['episode_termination'],
                episode_truncation=i==len(events)-1 and report['feedback']['episode_truncation']))
            if timed:transition.append((time.perf_counter()-start)*1000)
    replay(False)
    replay(True)
    profile.enable();replay(False);profile.disable()
    profile.dump_stats(str(a.output/'native_replay.pstats'))
    with (a.output/'profile.txt').open('w') as f:pstats.Stats(profile,stream=f).sort_stats('cumulative').print_stats(35)
    result=dict(trace_folder=str(a.trace_folder),actual_trace_sha256=hashlib.sha256((a.trace_folder/'actual.json').read_bytes()).hexdigest(),
                exact_actions_reproduced=True,control_ms=percentiles(control),transition_ms=percentiles(transition),
                online_ms=percentiles(np.asarray(control)+transition),
                scope='Warm deterministic replay of retained actual native observations/poses. No simulator stepping; native trajectory distributions and throughput are separately reported.')
    (a.output/'profile.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
