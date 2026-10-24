"""Profile the mesh fallback at a retained actual observation; no oracle inputs."""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import time
import numpy as np
import habitat_sim
from eventwm_graph.benchmark.map import load_certified_map
from eventwm_graph.l4.geometry import StaticGrid
from eventwm_graph.l4.precise import attach,update,proposals
from eventwm_graph.l4 import Executor,ActionSpec


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);worst=(-1,None,None)
    for folder in a.root.glob('*_B1'):
        if not (folder/'actual.json').exists():continue
        trace=json.loads((folder/'actual.json').read_text())
        for i,d in enumerate(trace['decisions']):
            if d['collision_check'].get('precise_static_fallback') and d['control_ms']>worst[0]:worst=(d['control_ms'],folder,i)
    recorded_ms,folder,index=worst
    trace=json.loads((folder/'actual.json').read_text());r=json.loads((folder/'report.json').read_text())
    ref=json.loads((folder/'reference.json').read_text());ob=trace['observations'][index];d=trace['decisions'][index]
    hmap=load_certified_map(r['case']['manifest']);geometry=StaticGrid.from_navigation(hmap.navigation,ref['nominal_waypoints'])
    calibration=attach(geometry,hmap,trace['observations'][0]['robot_joint_positions'])
    try:
        update(geometry,calibration,ob['robot_joint_positions'],trace['actions'][:index].count('forward'))
        target=np.asarray(d['route_target']);terminal=np.allclose(target,ref['nominal_waypoints'][-1])
        args=(np.asarray(ob['robot_pose']),target,ref['query']['terminal_orientation'],terminal,
              Executor._people(ob,ob['timestamp']),geometry,ActionSpec(),32,0)
        proposals(*args) # resident JIT/cache before timing
        profiler=cProfile.Profile();durations=[];checks=[]
        profiler.enable()
        for _ in range(5):
            began=time.perf_counter();_,_,n=proposals(*args);durations.append((time.perf_counter()-began)*1000);checks.append(n)
        profiler.disable();profiler.dump_stats(str(a.output/'mesh.pstats'))
        stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumulative').print_stats(30)
        (a.output/'profile.txt').write_text(stream.getvalue())
        (a.output/'profile.json').write_text(json.dumps(dict(source=str(folder),actual_observation_index=index,
            recorded_control_ms=recorded_ms,mesh_proposal_ms=durations,static_checks=checks,
            scope='isolated bounded mesh proposals at an actual observed state; preference fixed to neutral; no live environment or hidden humans'),indent=2))
        print(stream.getvalue())
    finally:calibration.close()


if __name__=='__main__':main()
