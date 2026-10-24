"""Real Habitat Stage 0 traces; independent of production rendering/storage."""
from dataclasses import asdict
import math
from pathlib import Path
import time
import numpy as np
from ..config import read_json,write_json
from ..benchmark.map import load_certified_map
from ..stage1.runtime import NativeEpisode,episode_for_map,episode_record
from ..stage1.storage import write_gzip
from .map import PartialGraph
from .observer import GeometryObserver
from .query import generate_queries,compile_query,yaw_to,wrap
from .initialization import make_episode
from .storage import implementation


def scene_manifest(scene):
    base=Path('output/stage0_benchmark/certified')/scene
    paths=sorted(base.glob('*/manifest.json'))
    if paths:return paths[0]
    latest=Path('output/stage0_benchmark/corpus/certified')/scene/'latest.json'
    return Path(read_json(latest)['manifest'])


def prefix_actions(variant=0):
    # Independent installed primitive-action history, not a full-map route.
    # Translation supplies earlier camera views of the current jaw blind area.
    # No geometry, PointGoal shortest path, or human future selects this prefix.
    if variant==3:return ['forward']*15
    if variant==5:return ['forward']*13
    if variant==6:return ['forward']*17
    if variant==7:return ['forward']*13
    if variant==8:return ['forward']*13+['left']*12+['right']*24+['left']*12
    if variant==0:return ['left']*76+['forward']*10+['right']*19+['forward']*4+['left']*19
    if variant==1:return ['left']*38+['forward']*6+['left']*38+['forward']*5
    return ['forward']*10+['left']*19+['forward']*5+['right']*38+['forward']*4


def build_demo(scene,output,seed=310201001,node_index=4,humans=2,variant=0,probe=True,scenario='natural'):
    began=time.perf_counter();h=load_certified_map(scene_manifest(scene))
    node=h.valid_starts()[node_index%len(h.valid_starts())]['node_id']
    episode,initialization=make_episode(h,node,seed,humans,scenario,heading=0. if scenario=='natural' else None)
    history=[];snapshots=[];probe_result=None
    with NativeEpisode(h,episode,seed) as rt:
        sensor=GeometryObserver(rt,h.robot_config.collision_reference/6)
        first=sensor.observe();g=PartialGraph(sensor.resolution,first['robot_pose'][1]);g.update(first);history.append(first)
        prefix=[];queries=[];diagnostics={}
        frames=prefix_actions(variant)
        checkpoints={0,2,4,6,8,10,11,12,13,14,15,16,17,18,19,24,34,38,39,57,58,76,82,len(frames)}
        extension=False
        for step in range(len(frames)+181):
            if step in checkpoints:
                q,d=generate_queries(g,history[-1]['tracks'],history[-1],episode.goals[0].position)
                snapshots.append({'step':step,'graph':g.snapshot(),'queries':q,'diagnostics':d,
                                  'observation':history[-1]})
                queries,diagnostics=q,d
            if step==len(frames) and variant==7 and not extension:
                # A legal geometry-only demonstration selector: execute one
                # progress query whose frozen known graph also certifies a
                # return. The next decision then exposes genuine backtracking.
                extension=True;choices=[];start=history[-1]['robot_pose']
                for candidate in queries:
                    if candidate['route_distance']<1.:continue
                    end=candidate['compiled']['planned_poses'][-1]
                    d=np.asarray(start[:3])-end[:3]
                    if d[0]*np.cos(end[3])-d[2]*np.sin(end[3])>=-.3:continue
                    back=compile_query(g,{'waypoints':[start[:3]],'final_yaw':yaw_to(end[:3],start[:3])},{'robot_pose':end})
                    if back['ok']:choices.append(candidate)
                if choices:
                    chosen=max(choices,key=lambda q:(q['route_distance'],q['query_id']))
                    frames.extend(chosen['compiled']['actions']);checkpoints.update(range(step,len(frames)+1))
            if step==len(frames) or rt.env.episode_over:break
            rt.step(frames[step]);prefix.append(frames[step]);row=sensor.observe();g.update(row);history.append(row)
        # Static execution probe is Stage 0 action validity evidence, not a
        # production Stage 1 outcome. No root is admitted while a gate is closed.
        if probe and not rt.env.episode_over:
            choices=[q for q in queries if q['route_distance']>.4]
            chosen=choices[0] if choices else (queries[0] if queries else None)
            if chosen:
                from .metrics import contact_labels
                actual=[];contacts=[]
                for action in chosen['compiled']['actions']:
                    rt.step(action);row=sensor.observe();actual.append(row);contacts.append(contact_labels(rt))
                    if rt.env.episode_over:break
                expected=np.asarray(chosen['compiled']['planned_poses'])[1:len(actual)+1]
                observed=np.asarray([r['robot_pose'] for r in actual])
                probe_result={'query':chosen,'observations':actual,
                    'maximum_planar_error_m':float(np.linalg.norm((observed-expected)[:,[0,2]],axis=1).max()) if actual else 0.,
                    'final_yaw_error':abs(wrap(chosen['final_yaw']-observed[-1,3])) if actual else None,
                    'human_collision':bool(rt.env.get_metrics().get('human_collision',0)),
                    'static_obstacle_contact':any(c['static_obstacle_contact'] for c in contacts),
                    'contacts':contacts,
                    'complete_actions':len(actual)==len(chosen['compiled']['actions'])}
        trace={'schema':'partial_graph_stage0_trace_1','scene_id':scene,'seed':seed,
            'sensor':asdict(rt.observation_config),'resolution':sensor.resolution,'history':history,
            'prefix_actions':prefix,'snapshots':snapshots,'probe':probe_result,
            'private_initialization':episode_record(episode),'online_uses_private_initialization':False,
            'initialization_law':initialization,
            'implementation':implementation(),
            'performance':sensor.performance,'wall_seconds':time.perf_counter()-began,
            'root_terminated':bool(history and rt.env.episode_over)}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    suffix=f'{scene}_{seed}_h{humans}_v{variant}_{scenario}'
    path=output/f'{suffix}.json.gz';write_gzip(path,trace)
    summary={'scene':scene,'seed':seed,'trace':str(path.resolve()),'history_frames':len(history),
             'known_free_cells':sum(c['state']==1 for c in g.cells.values()),
             'reachable_cells':len(g.reachable(history[-1]['robot_pose'][:3])),
             'queries':len(queries),'query_sources':sorted({q['provenance'][0]['source'] for q in queries}),
             'human_bands':sorted({q['provenance'][0].get('intended_band','') for q in queries if q['provenance'][0]['source']=='human'}),
             'probe':{k:v for k,v in (probe_result or {}).items() if k not in ('query','observations','contacts')},
             'performance':trace['performance'],'wall_seconds':trace['wall_seconds']}
    write_json(output/f'{suffix}.summary.json',summary)
    return summary
