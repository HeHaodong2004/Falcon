"""Measured Stage 0 gates and real native static-query probes."""
import copy
from pathlib import Path
import time
import unittest
import numpy as np
from ..config import read_json,write_json,file_digest,digest
from ..stage1.storage import read_gzip,write_gzip
from ..benchmark.map import load_certified_map
from ..stage1.runtime import NativeEpisode
from .demo import scene_manifest
from .storage import implementation,graph_from_history,validate_global_graph
from .query import generate_queries,wrap
from .metrics import contact_labels
from .visualize import stage0 as render_stage0

STAGE0_SOURCES=('map.py','observer.py','query.py','initialization.py','metrics.py',
                'eventwm_graph/stage1/runtime.py','eventwm_graph/benchmark/contract.py',
                'eventwm_graph/benchmark/map.py','scripts/collector/observation.py',
                'falcon/additional_metric.py','falcon/additional_action.py')


def native_static_probe(trace_path,output,limit=3):
    """Actual installed actions with humans disabled ONLY for static review.

    These are explicitly labeled empty-human static references, not the social
    outcome associated with the original trace's human-conditioned proposal.
    The query/initial graph stay frozen from the real social observation root.
    """
    import habitat_sim
    from habitat.tasks.nav.nav import NavigationEpisode,NavigationGoal
    from ..robotmap.execution import signed_yaw
    trace=read_gzip(trace_path);root=trace['snapshots'][-1];queries=root['queries']
    h=load_certified_map(scene_manifest(trace['scene_id']))
    ep=trace['private_initialization'];info=copy.deepcopy(ep['info']);info['human_num']=0
    episode=NavigationEpisode(episode_id=ep['episode_id'],scene_id=ep['scene_id'],
        start_position=ep['start_position'],start_rotation=ep['start_rotation'],
        goals=[NavigationGoal(position=g['position']) for g in ep['goals']],info=info)
    chosen=[]
    for category in ('human:safe','human:intrusion','topology:behind','current_position:','frontier:','goal:','topology:'):
        candidates=[q for q in queries if q['provenance'][0]['source']+':'+q['provenance'][0].get('intended_band','')==category]
        if category=='topology:behind':candidates=[q for q in queries if q['provenance'][0].get('behind_robot')]
        if candidates:chosen.append(candidates[0])
        if len(chosen)>=limit:break
    results=[]
    for q in chosen:
        with NativeEpisode(h,episode,trace['seed']) as rt:
            for action in trace['prefix_actions']:rt.step(action)
            robot=rt.sim.get_agent_data(0).articulated_agent
            start=[*np.asarray(robot.base_pos,dtype=float).tolist(),signed_yaw(robot)]
            actual=[start];contacts=[]
            for action in q['compiled']['actions']:
                rt.step(action);actual.append([*np.asarray(robot.base_pos,dtype=float).tolist(),signed_yaw(robot)])
                contacts.append(contact_labels(rt))
                if rt.env.episode_over:break
            planned=np.asarray(q['compiled']['planned_poses']);observed=np.asarray(actual)
            difference=observed[:,:3]-planned[:len(actual),:3]
            error=float(np.linalg.norm(difference,axis=1).max())
            planar=float(np.linalg.norm(difference[:,[0,2]],axis=1).max())
            vertical=float(np.abs(difference[:,1]).max())
            terminal=float(np.linalg.norm(observed[-1,:3]-q['waypoints'][-1]))
            yaw=abs(wrap(q['final_yaw']-observed[-1,3]))
            body=any(c['static_obstacle_contact'] for c in contacts)
            # A cell stores its measured center height. Native support can
            # vary within that cell; .08m is an operational per-step cap, not
            # a proof of continuous height variation. Planar and terminal
            # route tolerances remain strict and contacts remain authoritative.
            ok=(len(actual)==len(planned) and planar<=.001 and vertical<=.080001 and
                terminal<=q['compiled']['position_tolerance']+.001 and
                yaw<=q['compiled']['yaw_tolerance']+1e-4 and not body)
            results.append({'query_id':q['query_id'],'source':q['provenance'],'passed':ok,
                'maximum_position_error_m':error,'maximum_planar_error_m':planar,
                'maximum_height_error_m':vertical,'terminal_position_error_m':terminal,
                'validation_limits':{'planar_m':.001,'height_m':.080001,'terminal_m':q['compiled']['position_tolerance']+.001},
                'final_yaw_error_rad':yaw,'static_obstacle_contact':body,
                'actual_poses':actual,'contacts':contacts,'actions':q['compiled']['actions']})
    result={'stage':'stage0_native_static_probe','trace':str(Path(trace_path).resolve()),
            'trace_sha256':file_digest(trace_path),'validation_tool_sha256':file_digest(__file__),
            'scene_id':trace['scene_id'],'seed':trace['seed'],'reference':'actual Env.step, humans disabled for static execution audit only',
            'implementation':{k:v for k,v in implementation().items() if k in STAGE0_SOURCES},
            'queries':results,'passed':bool(results) and all(r['passed'] for r in results)}
    path=Path(output)/(Path(trace_path).name.replace('.json.gz','.static_probe.json.gz'))
    path.parent.mkdir(parents=True,exist_ok=True);write_gzip(path,result)
    return result


def run_unit_review(output):
    suite=unittest.defaultTestLoader.discover('tests/partial_graph_query')
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    return {'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),
            'errors':len(result.errors),'log':str(path.resolve()),'log_sha256':file_digest(path)}


def accept_stage0(trace_paths,probe_paths,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    versions={k:v for k,v in implementation().items() if k in STAGE0_SOURCES}
    units=run_unit_review(output/'stage0_tests.log')
    scenes=set();sources=set();bands=set();headings=False;backtracking=False;causal=True;routes=True;reproducible=True
    frames=0;real_queries=0;visuals=[];traces=[]
    for path in trace_paths:
        trace=read_gzip(path)
        if any(trace.get('implementation',{}).get(k)!=v for k,v in versions.items()):
            raise ValueError('Stale Stage0 trace; regenerate '+str(path))
        graph=graph_from_history([],trace['resolution'],trace['history'][0]['robot_pose'][1])
        snapshot_index={s['step']:s for s in trace['snapshots']}
        for step,row in enumerate(trace['history']):
            from .storage import validate_observation
            validate_observation(row);graph.update(row)
            if step not in snapshot_index:continue
            saved=snapshot_index[step]
            q,d=generate_queries(graph,row['tracks'],row,trace['private_initialization']['goals'][0]['position'])
            reproducible &= [x['query_id'] for x in q]==[x['query_id'] for x in saved['queries']]
            validate_global_graph(graph,saved['graph'],row['robot_pose'])
            causal &= graph.cells=={tuple(c['cell']):{k:v for k,v in c.items() if k!='cell'} for c in saved['graph']['cells']}
            for query in q:
                real_queries+=1
                routes &= all(graph.segment_free(a,b) for a,b in zip(query['compiled']['route'],query['compiled']['route'][1:]))
                for p in query['provenance']:
                    sources.add(p['source'])
                    if p.get('behind_robot',False):
                        displacement=np.asarray(query['waypoints'][-1])-row['robot_pose'][:3]
                        yaw=row['robot_pose'][3]
                        backtracking |= displacement[0]*np.cos(yaw)-displacement[2]*np.sin(yaw)<-.1
                    if p['source']=='human':bands.add(p['intended_band'])
            groups={}
            for query in q:groups.setdefault(tuple(query['waypoints'][-1]),set()).add(round(query['final_yaw'],3))
            headings |= any(len(h)>1 for h in groups.values())
        if any(s['queries'] for s in trace['snapshots']):scenes.add(trace['scene_id'])
        frames+=len(trace['history']);visuals.append(render_stage0(path,output/'visualizations'))
        traces.append({'path':str(Path(path).resolve()),'sha256':file_digest(path),'scene':trace['scene_id']})
    probes=[read_gzip(p) for p in probe_paths]
    trace_lookup={t['path']:read_gzip(t['path']) for t in traces}
    native_scenes=set();native_bands=set();native=bool(probes);native_backtracking=False
    for p in probes:
        source=trace_lookup.get(p['trace']);native &= bool(source and p['passed'] and p['implementation']==versions and
            p.get('trace_sha256')==file_digest(p['trace']) and p.get('validation_tool_sha256')==file_digest(__file__))
        if not source:continue
        queries={q['query_id']:q for q in source['snapshots'][-1]['queries']}
        for result in p['queries']:
            q=queries.get(result['query_id']);native &= bool(q and result['passed'])
            if not q or not result['passed']:continue
            if q['route_distance']>.25:native_scenes.add(p['scene_id'])
            if q['provenance'][0].get('behind_robot'):
                origin=source['snapshots'][-1]['observation']['robot_pose']
                delta=np.asarray(q['waypoints'][-1])-origin[:3]
                native_backtracking |= bool(delta[0]*np.cos(origin[3])-delta[2]*np.sin(origin[3])<-.1)
            for provenance in q['provenance']:
                if provenance['source']=='human':native_bands.add(provenance['intended_band'])
    checks={'observation_causality':bool(causal and units['passed']),
        'graph_correctness':bool(routes and units['passed']),
        'query_correctness':bool(units['passed'] and bands=={'safe','intrusion'} and headings and backtracking and
                                 {'human','frontier','goal','topology','current_position'}.issubset(sources)),
        'determinism':bool(reproducible and units['passed']),
        'real_Habitat_demonstrations':bool(len(scenes)>=3 and len(native_scenes)>=2 and native and
            native_bands=={'safe','intrusion'} and native_backtracking and real_queries>=20)}
    gate={'stage':'stage0','status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'implementation':versions,'validation_tool_sha256':file_digest(__file__),
        'unit_review':units,'traces':traces,'visualizations':visuals,
        'native_probe_paths':[str(Path(p).resolve()) for p in probe_paths],
        'evidence':{'real_scenes':sorted(scenes),'native_static_probe_scenes':sorted(native_scenes),
                    'frames':frames,'accepted_queries_across_snapshots':real_queries,'sources':sorted(sources),
                    'human_bands':sorted(bands),'native_human_bands':sorted(native_bands),
                    'same_target_different_heading':headings,'backtracking':bool(backtracking),
                    'native_backtracking':native_backtracking},
        'time':time.time(),'claim':'Idealized visibility-limited geometry and exact legal detections; no depth-pixel or sensor-fairness claim'}
    write_json(output/'stage0_gate.json',gate)
    return gate


def reprocess_query_traces(source_directory,destination):
    """Reapply changed query code to immutable, unchanged legal observations.

    The private sensing producer must match exactly. Optimization changes must
    reproduce every old query identity, including the complete motion schedule.
    This avoids unnecessarily re-running native scene sensing experiments.
    Fresh native execution probes are still required for the derived traces.
    """
    from ..stage1.storage import write_gzip
    source_directory=Path(source_directory);destination=Path(destination)
    target=destination/'traces';target.mkdir(parents=True,exist_ok=True);versions=implementation();result=[]
    for path in sorted((source_directory/'traces').glob('*.json.gz')):
        old=read_gzip(path)
        for key in STAGE0_SOURCES:
            if key!='query.py' and old['implementation'].get(key)!=versions[key]:
                raise ValueError('Legal observation producer changed:'+key)
        graph=graph_from_history([],old['resolution'],old['history'][0]['robot_pose'][1])
        saved={s['step']:s for s in old['snapshots']};snapshots=[];began=time.perf_counter()
        for step,row in enumerate(old['history']):
            graph.update(row)
            if step not in saved:continue
            queries,diagnostics=generate_queries(graph,row['tracks'],row,old['private_initialization']['goals'][0]['position'])
            if [q['query_id'] for q in queries]!=[q['query_id'] for q in saved[step]['queries']]:
                raise ValueError('Query optimization changed decisions:'+str(path)+':'+str(step))
            snapshots.append({'step':step,'graph':graph.snapshot(),'queries':queries,'diagnostics':diagnostics,'observation':row})
        derived={**old,'snapshots':snapshots,'implementation':versions,
            'derivation':{'legal_trace':str(path.resolve()),'legal_trace_sha256':file_digest(path),
                'observation_producer_versions':old['implementation'],
                'all_query_identities_exactly_equal':True,'reprocess_seconds':time.perf_counter()-began}}
        output=target/path.name;write_gzip(output,derived);result.append(str(output.resolve()))
    write_json(destination/'trace_derivation.json',{'sources':result,'implementation':versions,
        'rule':'Unchanged legal observation producers; exact query identities and motion schedules required'})
    return result
