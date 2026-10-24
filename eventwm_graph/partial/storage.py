"""Versioned immutable roots with legal/privileged files and shared frame DAGs."""
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
from ..config import REPO,digest,file_digest,read_json,write_json
from ..stage1.storage import write_gzip,read_gzip,scene_split
from .map import PartialGraph

SCHEMA='observation_limited_query_conditioned_root_1'


def implementation():
    folder=Path(__file__).parent
    names=('__init__.py','map.py','observer.py','query.py','metrics.py','execution.py','initialization.py','storage.py','collect.py')
    result={n:file_digest(folder/n) for n in names if (folder/n).exists()}
    for p in ('eventwm_graph/stage1/runtime.py','eventwm_graph/benchmark/contract.py',
              'eventwm_graph/benchmark/map.py','scripts/collector/observation.py','falcon/additional_metric.py','falcon/additional_action.py'):
        path=folder.parents[1]/p
        if not path.exists():path=REPO/p
        result[p]=file_digest(path)
    return result


def graph_from_history(history,resolution,height):
    graph=PartialGraph(resolution,height)
    for row in history:graph.update(row)
    return graph


def validate_observation(row):
    t=float(row['timestamp'])
    if not np.isfinite(t) or not np.isfinite(np.asarray(row['robot_pose'])).all():raise ValueError('nonfinite_legal_observation')
    for h in row['tracks']:
        if (not np.isfinite(h['last_seen_time']) or h['last_seen_time']>t+1e-9 or
                h['first_seen_time']>h['last_seen_time'] or h['observation_age']<0 or
                abs(h['observation_age']-(t-h['last_seen_time']))>1e-6 or
                not np.isfinite(np.asarray(h['position'])).all()):
            raise ValueError('noncausal_legal_human_track')


def validate_global_graph(graph,snapshot,ego):
    if snapshot['timestamp']!=graph.timestamp or snapshot['revision']!=graph.revision:
        raise ValueError('root_graph_clock_or_revision_mismatch')
    reachable=graph.reachable(ego[:3]);nodes=snapshot.get('global_nodes',{})
    for node_id,node in nodes.items():
        key=tuple(node['cell'])
        if (node_id!=node['id'] or key not in reachable or graph.state(key)!=1 or
                np.linalg.norm(graph.center(key)-node['position'])>1e-8 or node['created_at']>graph.timestamp):
            raise ValueError('uncertified_global_node')
    adjacency={nid:set() for nid in nodes}
    for edge in snapshot.get('global_edges',[]):
        a,b=edge['source'],edge['target'];path=edge['path']
        if a not in nodes or b not in nodes or not path:raise ValueError('unknown_global_edge_endpoint')
        if (np.linalg.norm(np.asarray(path[0])-nodes[a]['position'])>1e-8 or
                np.linalg.norm(np.asarray(path[-1])-nodes[b]['position'])>1e-8 or
                not all(graph.segment_free(u,v) for u,v in zip(path,path[1:]))):
            raise ValueError('uncertified_global_edge')
        adjacency[a].add(b);adjacency[b].add(a)
    if nodes:
        seen={next(iter(nodes))};queue=list(seen)
        while queue:
            for nid in adjacency[queue.pop()]-seen:seen.add(nid);queue.append(nid)
        if len(seen)!=len(nodes):raise ValueError('disconnected_global_abstraction')


def check_legal(value):
    """Recursive denylist in addition to the serializer's explicit whitelist."""
    forbidden={'oracle_id','oracle_heading','human_num','private_initialization','oracle_future',
               'source_navmesh','source_scene','map_manifest','ground_truth','supervision_human_index',
               'official_center_distances_m','labels','private_cache','offset_transform','joints'}
    if isinstance(value,dict):
        for key,item in value.items():
            if key in forbidden:raise ValueError('Privileged field in model-facing record: '+key)
            check_legal(item)
    elif isinstance(value,list):
        for item in value:check_legal(item)


def online_input(folder,query_index=0):
    """Only two whitelisted files are opened; no private map reference exists."""
    folder=Path(folder);root=read_gzip(folder/'legal_root.json.gz')
    query=read_gzip(folder/'queries.json.gz')[query_index]
    result={'schema':SCHEMA,'history':root['history'],'history_actions':root['history_actions'],'partial_graph':root['partial_graph'],
            'belief':root['history'][-1]['tracks'],'pointgoal':root['pointgoal'],
            'sensor':root['sensor'],'query':query,'version':root['version']}
    check_legal(result)
    return result


def seal_root(output,record,tape,queries,outcomes,min_free_bytes=100*1024**3):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free<min_free_bytes:raise RuntimeError('disk_reserve_reached')
    versions=implementation()
    identity={'schema':SCHEMA,'scene_id':record['scene_id'],'seed':int(record['seed']),
              'initialization':record['private_initialization'],'version':versions,'collection_config':record['collection_config']}
    root_id=digest(identity);destination=output/'roots'/root_id[:2]/root_id
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():return validate_root(destination)
    staging=Path(tempfile.mkdtemp(prefix='.partial_',dir=destination.parent));began=time.perf_counter()
    try:
        legal={k:record[k] for k in ('history','history_actions','partial_graph','pointgoal','sensor')}
        legal.update(schema=SCHEMA,version=versions)
        check_legal(legal);check_legal(queries)
        write_gzip(staging/'legal_root.json.gz',legal)
        write_gzip(staging/'queries.json.gz',queries)
        nodes_legal=[{k:n[k] for k in ('node_id','parent','action','observation','graph_delta')} for n in tape.nodes]
        check_legal(nodes_legal)
        write_gzip(staging/'future_observations.json.gz',nodes_legal)
        write_gzip(staging/'privileged_step_labels.json.gz',[{'node_id':n['node_id'],**n['labels']} for n in tape.nodes])
        write_gzip(staging/'outcomes.json.gz',outcomes)
        write_gzip(staging/'privileged_initialization.json.gz',{
            'private':True,'online_input':False,'identity':identity,'episode':record['private_initialization'],
            'source_map_manifest':record['source_map_manifest'],'prefix_labels':record['prefix_labels'],
            'sampling':record['sampling']})
        positions=np.asarray([[h['position'] for h in frame] for frame in tape.truth],dtype=np.float64)
        if not tape.truth[0]:positions=np.empty((len(tape.truth),0,3),dtype=np.float64)
        np.savez_compressed(staging/'privileged_future.npz',timestamps=np.asarray(tape.times),positions=positions)
        files={p.name:file_digest(p) for p in sorted(staging.iterdir())}
        psc=[o['summary']['psc'] for o in outcomes if o['summary']['psc'] is not None]
        stats={'outcomes':len(outcomes),'human_collisions':sum(o['summary']['human_collision'] for o in outcomes),
               'static_obstacle_collisions':sum(o['summary']['static_obstacle_collision'] for o in outcomes),
               'psc_violations':sum(o['summary']['any_psc_violation'] for o in outcomes),
               'intrusive_without_collision':sum(o['summary']['any_psc_violation'] and not o['summary']['human_collision'] for o in outcomes),
               'query_completions':sum(o['summary']['query_complete'] for o in outcomes),
               'query_timeouts':sum(o['summary']['termination']=='query_timeout' for o in outcomes),
               'query_psc_mean':float(np.mean(psc)) if psc else 1.,
               'proposed_safe':sum(q['provenance'][0].get('intended_band')=='safe' for q in queries),
               'proposed_intrusion':sum(q['provenance'][0].get('intended_band')=='intrusion' for q in queries)}
        split={'train':'train','validation':'dev','test':'confirmation'}[scene_split(record['scene_id'])]
        manifest={'schema':SCHEMA,'root_id':root_id,'scene_id':record['scene_id'],'split':split,
                  'seed':record['seed'],'sealed':True,'files':files,'queries':len(queries),
                  'query_ids':[q['query_id'] for q in queries],'version':versions,
                  'statistics':stats,'sampling':record['sampling'],'performance':record['performance'],
                  'serialization_seconds':time.perf_counter()-began,
                  'bytes':sum(p.stat().st_size for p in staging.iterdir()),
                  'independent_futures':1,'future_id':digest(identity),
                  'collection_config':record['collection_config']}
        write_json(staging/'manifest.json',manifest)
        validation_started=time.perf_counter();validate_root(staging)
        manifest['validation_seconds']=time.perf_counter()-validation_started
        write_json(staging/'manifest.json',manifest)
        for p in staging.iterdir():
            with p.open('rb') as f:os.fsync(f.fileno())
        os.replace(staging,destination)
        descriptor=os.open(destination.parent,os.O_RDONLY);os.fsync(descriptor);os.close(descriptor)
        return {**manifest,'folder':str(destination.resolve())}
    except Exception as error:
        write_json(staging/'FAILED.json',{'error':repr(error),'sealed':False})
        raise


def validate_root(folder,reconstruct=True):
    folder=Path(folder);manifest=read_json(folder/'manifest.json')
    if manifest.get('schema')!=SCHEMA or not manifest.get('sealed'):raise ValueError('unknown_or_unsealed_root')
    for name,sha in manifest['files'].items():
        if file_digest(folder/name)!=sha:raise ValueError('corrupt_root_member:'+name)
    root=read_gzip(folder/'legal_root.json.gz');queries=read_gzip(folder/'queries.json.gz')
    nodes=read_gzip(folder/'future_observations.json.gz');labels=read_gzip(folder/'privileged_step_labels.json.gz')
    outcomes=read_gzip(folder/'outcomes.json.gz')
    private=read_gzip(folder/'privileged_initialization.json.gz')
    expected_split={'train':'train','validation':'dev','test':'confirmation'}[scene_split(manifest['scene_id'])]
    if manifest['split']!=expected_split:raise ValueError('scene_split_mismatch')
    if manifest['root_id']!=digest(private['identity']) or private['identity']['scene_id']!=manifest['scene_id']:
        raise ValueError('root_identity_mismatch')
    check_legal(root);check_legal(queries);check_legal(nodes)
    if len(queries)!=len(outcomes) or len(queries)!=manifest['queries']:raise ValueError('query_cardinality')
    if len({q['query_id'] for q in queries})!=len(queries):raise ValueError('duplicate_queries')
    if any(q['query_id']!=digest({k:v for k,v in q.items() if k!='query_id'}) for q in queries):raise ValueError('query_checksum')
    if len(nodes)!=len(labels) or any(n['node_id']!=i or labels[i]['node_id']!=i for i,n in enumerate(nodes)):
        raise ValueError('invalid_frame_DAG_index')
    with np.load(folder/'privileged_future.npz',allow_pickle=False) as f:
        times=f['timestamps'].copy();positions=f['positions']
        if (not np.isfinite(times).all() or np.any(np.diff(times)<=0) or not np.isfinite(positions).all() or
                positions.ndim!=3 or positions.shape[0]!=len(times) or positions.shape[2]!=3):
            raise ValueError('invalid_future_clock')
    for r in root['history']:validate_observation(r)
    initial=root['history'][-1];snapshot=root['partial_graph']
    if len(root['history_actions'])!=len(root['history'])-1:raise ValueError('missing_legal_history_actions')
    if abs(times[0]-initial['timestamp'])>1e-9:raise ValueError('root_clock_mismatch')
    graph=graph_from_history(root['history'],snapshot['resolution'],snapshot['height'])
    if graph.cells!=PartialGraph.restore(snapshot).cells:raise ValueError('history_cannot_reconstruct_graph')
    if (snapshot.get('ego_support')!=graph.ego_support.tolist() or snapshot['visited']!=graph.visited or
            root['version']!=manifest['version'] or len(private['prefix_labels'])!=len(root['history_actions'])):
        raise ValueError('history_state_or_version_mismatch')
    validate_global_graph(graph,snapshot,initial['robot_pose'])
    used=set();compiler_graph=PartialGraph.restore(snapshot)
    for q,o in zip(queries,outcomes):
        if q['query_id']!=o['query_id']:raise ValueError('outcome_query_identity')
        if q['decision_timestamp']!=initial['timestamp']:raise ValueError('query_decision_clock_mismatch')
        from .query import compile_query,physical_clearance
        compiled=compile_query(compiler_graph,q,initial)
        if compiled!=q['compiled']:raise ValueError('query_motion_not_reconstructable')
        if not physical_clearance(q['waypoints'][-1],q['final_yaw'],initial['tracks']):raise ValueError('proposal_human_physical_overlap')
        for waypoint,provenance in zip(q['waypoints'],q['provenance']):
            if provenance['source']=='human':
                tracks=[h for h in initial['tracks'] if h['track_id']==provenance['track_id'] and h['visible_now']]
                if (not tracks or tracks[0]['position']!=provenance['reference_position'] or
                        provenance['reference_timestamp']!=initial['timestamp']):raise ValueError('human_proposal_not_from_legal_detection')
                radius=float(np.linalg.norm(np.asarray(waypoint)-provenance['reference_position']))
                if (abs(radius-provenance['nominal_radius_m'])>1e-6 or
                        (provenance['intended_band']=='safe' and radius<=1.) or
                        (provenance['intended_band']=='intrusion' and radius>=1.)):
                    raise ValueError('candidate_intention_ring_mismatch')
        g=PartialGraph.restore(snapshot);last=None;t=initial['timestamp']
        actions=[];poses=[initial['robot_pose']]
        for j,i in enumerate(o['node_ids']):
            n=nodes[i];r=n['observation'];l=labels[i]
            used.add(i);validate_observation(r)
            if n['parent']!=last or r['timestamp']<=t:raise ValueError('invalid_branch_order')
            if abs(r['timestamp']-times[j+1])>1e-9 or r['timestamp']!=l['timestamp']:raise ValueError('frame_clock_mismatch')
            if j+1<len(o['node_ids']) and l['episode_over']:raise ValueError('post_termination_observation')
            if any(h['last_seen_time']>r['timestamp']+1e-9 for h in r['tracks']):raise ValueError('future_human_in_belief')
            ds=l['official_center_distances_m'];eu=l['euclidean_center_distances_m']
            if (not np.isfinite(np.asarray(ds)).all() or not np.isfinite(np.asarray(eu)).all() or
                    l['psc_violation']!=any(d<1. for d in ds) or
                    l['official_dmin_m']!=(min(ds) if ds else None) or
                    l['euclidean_dmin_m']!=(min(eu) if eu else None) or
                    l['euclidean_ps_violation']!=any(d<1. for d in eu)):
                raise ValueError('step_psc_inconsistent_with_distances')
            if reconstruct:
                delta=g.update(r)
                if delta!=n['graph_delta']:raise ValueError('graph_delta_not_reconstructable')
            actions.append(n['action']);poses.append(r['robot_pose']);t=r['timestamp'];last=i
        if actions!=q['compiled']['actions'][:len(actions)]:raise ValueError('executed_action_mismatch')
        if (o['successor']['robot_pose']!=poses[-1] or o['successor']['timestamp']!=t or
                (reconstruct and o['successor']['graph_revision']!=g.revision)):
            raise ValueError('successor_not_reconstructable')
        final_tracks=nodes[last]['observation']['tracks'] if last is not None else initial['tracks']
        if o['successor']['tracks']!=final_tracks:raise ValueError('successor_tracks_mismatch')
        if o['summary']['executed_steps']!=len(actions):raise ValueError('executed_steps_mismatch')
        from .metrics import summarize
        expected=summarize([labels[i] for i in o['node_ids']],poses,initial['timestamp'],q['compiled']['actions'],
            o['summary']['termination'],q,root['pointgoal'],
            sum(not l['psc_violation'] for l in private['prefix_labels']),len(root['history_actions']))
        if expected!=o['summary']:raise ValueError('outcome_summary_not_reconstructable')
    if used!=set(range(len(nodes))):raise ValueError('orphan_future_observation')
    online_input(folder,0)
    return {**manifest,'folder':str(folder.resolve())}
