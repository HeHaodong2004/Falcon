"""Independent actual-action certification of a benchmark surface hierarchy.

This does not promote schema-1 body-safety assertions. It creates a separate
versioned contract, directed evidence, and a fail-closed production gate.
"""
import copy
from dataclasses import asdict
import gzip
import os
from pathlib import Path
import time
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from ..config import digest, file_digest, read_json, write_json
from ..native import StaticGeometry
from ..map import load_hierarchical_map
from .contract import SCHEMA, installed_contract
from .motion import KinematicModel, replay_native

VERIFIED = "VERIFIED_TRAVERSABLE"
UNKNOWN = "UNKNOWN"
NON_TRAVERSABLE = "VERIFIED_NON_TRAVERSABLE"


def implementation_identity():
    return {n:file_digest(Path(__file__).with_name(n)) for n in ('contract.py','motion.py','certify.py')}


def graph_pairs(nodes,edges):
    ids={n:i for i,n in enumerate(nodes)}
    rows=[ids[e['source']] for e in edges];cols=[ids[e['target']] for e in edges]
    g=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(ids),len(ids)))
    distance=dijkstra(g,directed=True)
    return g,np.isfinite(distance)&~np.eye(len(ids),dtype=bool)


def independent_trace_geometry(h,poses):
    """Actual per-action surface connectivity, queried afresh at .025 m.

    Native swept surface, not physical articulated CCD. Floor identity is never
    inferred from top-down proximity. Sliding paths may bend around a boundary;
    the native geodesic of each step is retained/checked, not a chord through it.
    """
    import habitat_sim
    q=np.asarray(poses)
    pf=h.navigation.pf
    islands=[];maximum_geodesic=0.;height_transitions=0
    for i,p in enumerate(q[:,:3]):
        snap=np.asarray(pf.snap_point(p));diff=snap-p
        if not np.isfinite(snap).all() or np.linalg.norm(diff)>0.025:
            return {'valid':False,'reason':'actual_pose_outside_native_surface','step':i,'pose':p.tolist()}
        islands.append(int(pf.get_island(p)))
        if i and islands[-1]!=islands[-2]:
            return {'valid':False,'reason':'cross_island_step','step':i}
        if i and np.linalg.norm(p-q[i-1,:3])>1e-7:
            request=habitat_sim.ShortestPath();request.requested_start=q[i-1,:3];request.requested_end=p
            if not pf.find_path(request):return {'valid':False,'reason':'missing_3d_step_connection','step':i}
            # One forward command is at most 25/120 m horizontally; allow
            # native slope and a climb-sized vertical detail correction.
            if request.geodesic_distance>.55:
                return {'valid':False,'reason':'step_requires_long_detour','step':i,'geodesic':float(request.geodesic_distance)}
            maximum_geodesic=max(maximum_geodesic,float(request.geodesic_distance))
            height_transitions+=abs(p[1]-q[i-1,1])>.05
    return {'valid':True,'samples':len(q),'island':islands[0],
        'maximum_action_step_geodesic':maximum_geodesic,'height_change_steps':int(height_transitions),
        'scope':'native 3D support and bounded per-action connection; not articulated body safety'}


def _save_trace(path, prediction, actual):
    # Compact numeric evidence; no rendered data and no duplicated scene.
    t=actual['trace']
    import tempfile
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.npz',delete=False) as f:
        np.savez_compressed(f,predicted_poses=np.asarray(prediction['poses'],dtype=np.float64),
            actual_poses=np.asarray([x['pose'] for x in t],dtype=np.float64),
            timestamps=np.asarray([x['time'] for x in t],dtype=np.float64),
            joint_positions=np.asarray([x['joints'] for x in t],dtype=np.float32),
            gait_frames=np.asarray([x['gait_frame'] for x in t],dtype=np.int32),
            actions=np.asarray([{'forward':0,'left':1,'right':2,'stop':3}[a] for a in prediction['actions']],dtype=np.uint8))
        name=f.name
    os.replace(name,path)


def validate_operation(h,model,path,start_yaw,target_yaw,folder,name,max_actions=500,contact_diagnostics=True):
    began=time.perf_counter()
    p=model.compile(path,start_yaw,target_yaw,max_actions=max_actions)
    if not p['reached']:
        return {'status':UNKNOWN,'reason':p['reason'],'proposal':p,'seconds':time.perf_counter()-began}
    actual=replay_native(h,p,model.contract,diagnostics=contact_diagnostics)
    actual['contact_diagnostics_evaluated']=contact_diagnostics
    if not contact_diagnostics:actual['full_body_safe']=None
    native=independent_trace_geometry(h,[r['pose'] for r in actual['trace']])
    valid=actual['pose_parity'] and native['valid'] and not actual['benchmark_should_end']
    # Invalid prediction is never relabelled verified, nor is failed search
    # proof of globally blocked connectivity.
    trace_path=folder/(name+'.npz');_save_trace(trace_path,p,actual)
    return {'status':VERIFIED if valid else UNKNOWN,
        'reason':'actual_action_replay_and_surface_passed' if valid else 'independent_execution_disagrees',
        'trace':trace_path.name,'trace_sha256':file_digest(trace_path),
        'start_pose':p['poses'][0],'end_pose':actual['trace'][-1]['pose'],
        'actions':len(p['actions']),'goal_error':p['goal_error'],
        'within_default_episode_budget':len(p['actions'])<=model.contract['max_episode_steps'],
        'duration':actual['trace'][-1]['time'],'length':p['length'],
        'independent_surface':native, 'actual':{k:v for k,v in actual.items() if k!='trace'},
        'seconds':time.perf_counter()-began}


def certify_scene(source_manifest,output,multi_routes=20,resume=True,contact_diagnostics=True):
    start=time.perf_counter();h=load_hierarchical_map(source_manifest)
    if h.metadata['schema_version']!=1:raise ValueError('Use the native surface hierarchy, not the full-body diagnostic roadmap')
    contract=installed_contract('social_nav_v2/falcon_mp3d.yaml' if h.metadata['scene_id'].startswith('mp3d') else 'social_nav_v2/falcon_hm3d.yaml')
    identity={'schema_version':SCHEMA,'source_map_id':h.map_id,'robot':asdict(h.robot_config),
        'contract':contract,'certifier':implementation_identity(),'multi_route_count':multi_routes,
        'optional_contact_diagnostics':contact_diagnostics}
    map_id=digest(identity);folder=Path(output)/h.metadata['scene_id']/map_id
    folder.mkdir(parents=True,exist_ok=True);(folder/'traces').mkdir(exist_ok=True)
    manifest_path=folder/'manifest.json'
    if resume and manifest_path.exists():
        result=read_json(manifest_path)
        for f,sha in result['checksums'].items():
            if file_digest(folder/f)!=sha:raise ValueError('Certification cache checksum mismatch: '+f)
        return result
    write_json(folder/'identity.json',identity)
    progress_path=folder/'progress.json'
    rows=[]
    with StaticGeometry(h.metadata['scene'],h.robot_config) as geo:
        h.geometry=geo;model=KinematicModel(h.navigation,contract)
        for edge in h.edges.values():
            for reverse in (False,True):
                eid=edge['id']+('r' if reverse else 'f')
                path=np.asarray(edge['path'])[::-1] if reverse else np.asarray(edge['path'])
                source,target=(edge['target'],edge['source']) if reverse else (edge['source'],edge['target'])
                result_path=folder/(eid+'.json')
                if resume and result_path.exists():
                    row=read_json(result_path)
                    if row['map_id']!=map_id:raise ValueError('Stale edge record')
                    if row.get('trace') and file_digest(folder/'traces'/row['trace'])!=row['trace_sha256']:raise ValueError('Damaged edge evidence')
                else:
                    row={'id':eid,'map_id':map_id,'native_edge_id':edge['id'],'source':source,'target':target,
                        'path':path.tolist(),'floor_transition':edge['floor_transition'],
                        **validate_operation(h,model,path,0.,0.,folder/'traces',eid,contact_diagnostics=contact_diagnostics)}
                    write_json(result_path,row)
                rows.append(row)
                write_json(progress_path,{'phase':'edges','map_id':map_id,'completed':len(rows),'total':2*len(h.edges),
                    'verified':sum(r['status']==VERIFIED for r in rows),'elapsed':time.perf_counter()-start,'pid':os.getpid()})
        # Multi-edge paths are replayed from the actual evolving pose; no
        # teleportation/reset at graph nodes. Select long paths and both floors.
        nids=sorted(h.nodes);index={nid:i for i,nid in enumerate(nids)}
        rr=[];cc=[];weights=[]
        for edge in h.edges.values():
            a,b=index[edge['source']],index[edge['target']]
            rr.extend((a,b));cc.extend((b,a));weights.extend((edge['length'],edge['length']))
        # Candidate ordering uses graph distance only. Compile and validate
        # selected routes, avoiding O(nodes^2) endpoint/path compilations.
        distances=dijkstra(csr_matrix((weights,(rr,cc)),shape=(len(nids),len(nids))),directed=True)
        candidates=[]
        for i,nid in enumerate(nids):
            finite=np.where(np.isfinite(distances[i]),distances[i],-1)
            j=int(np.argmax(finite))
            if finite[j]>0:candidates.append((float(finite[j]),nid,nids[j]))
        candidates.sort(key=lambda x:(-x[0],x[1],x[2]));selected=[]
        for _,nid,target in candidates:
            try:r=h.get_graph_route(h.nodes[nid]['position'],target)
            except ValueError:continue
            if len(r['edge_ids'])>=2:selected.append((nid,target,r))
            if len(selected)>=multi_routes:break
        multi=[]
        for i,(nid,target,r) in enumerate(selected):
            name=f'route_{i:04d}';out=folder/(name+'.json')
            if resume and out.exists():row=read_json(out)
            else:
                path=h.compile_navmesh_path(r)
                row={'map_id':map_id,'start_node':nid,'target_node':target,'native_route':r,
                     **validate_operation(h,model,path,0.,0.,folder/'traces',name,max_actions=2000,contact_diagnostics=contact_diagnostics)}
                write_json(out,row)
            multi.append(row)
            write_json(progress_path,{'phase':'composition','map_id':map_id,'completed':len(multi),'total':len(selected),
                                     'elapsed':time.perf_counter()-start,'pid':os.getpid()})
    original=[{'source':e['source'],'target':e['target']} for e in h.edges.values()]
    original+= [{'source':e['target'],'target':e['source']} for e in h.edges.values()]
    _,before=graph_pairs(h.nodes,original);graph,after=graph_pairs(h.nodes,[r for r in rows if r['status']==VERIFIED])
    counts=Counter(r['status'] for r in rows)
    operational=sum(r['status']==VERIFIED for r in multi)
    losses=int((before&~after).sum())
    report={'map_id':map_id,'scene_id':h.metadata['scene_id'],'edges':len(rows),'status_counts':dict(counts),
        'known_false_positive_executable_edges':0,
        'independent_replay_disagreements':sum(r.get('reason')=='independent_execution_disagrees' for r in rows),
        'native_reachable_pairs':int(before.sum()),'certified_reachable_pairs':int(after.sum()),
        'lost_reachable_pairs':losses,'weak_components':int(connected_components(graph,directed=True,connection='weak')[0]),
        'native_area':float(h.surface.areas.sum()),'continuous_body_safe_area':None,
        'multi_edge_routes':len(multi),'multi_edge_passed':operational,
        'multi_edge_lengths':[r['native_route']['length'] for r in multi],
        'multi_edge_counts':[len(r['native_route']['edge_ids']) for r in multi],
        'scene_contact_edge_count':sum(bool(r.get('actual',{}).get('native_scene_collision_events')) for r in rows),
        'body_diagnostic_edge_count':sum(bool(r.get('actual',{}).get('contact_diagnostics')) for r in rows),
        'height_transition_edges':sum(r['floor_transition'] for r in rows),
        'verified_height_transition_edges':sum(r['floor_transition'] and r['status']==VERIFIED for r in rows),
        'construction_and_certification_seconds':time.perf_counter()-start,
        'composition_beyond_one_episode':sum(not r.get('within_default_episode_budget',True) for r in multi),
        'episode_budget_scope':'Long composition tests may span >500 actions; production compiler enforces remaining official episode budget',
        'coverage_semantics':'native surface retained; verified directed action connections; UNKNOWN proposals retained',
        'full_body_safety_required':False,'full_body_safety_certified':False,
        'optional_contact_diagnostics_evaluated':contact_diagnostics,
        'production_gate':operational==len(multi) and losses==0 and bool(rows) and counts[VERIFIED]>0,
        'gate_scope':'all edges actual-action checked; bounded long-route composition; no universal policy or CCD guarantee'}
    write_json(folder/'edges.json',rows);write_json(folder/'routes.json',multi);write_json(folder/'report.json',report)
    record={'schema_version':SCHEMA,'map_id':map_id,'scene_id':h.metadata['scene_id'],
        'source_manifest':str(Path(source_manifest).resolve()),'source_map_id':h.map_id,
        'robot_configuration_id':digest(asdict(h.robot_config)),'contract':contract,
        'identity':identity,'report':report,
        'checksums':{str(p.relative_to(folder)):file_digest(p) for p in sorted(folder.glob('traces/*.npz'))}}
    record['checksums'].update({n:file_digest(folder/n) for n in ('edges.json','routes.json','report.json','identity.json')})
    write_json(manifest_path,record);write_json(folder.parent/'latest.json',{'manifest':str(manifest_path.resolve()),'map_id':map_id})
    return record
