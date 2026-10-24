"""Versioned stronger-evidence revisions of the robot roadmap, before topology.

A finite sweep is not continuous collision detection. A later contact witness
must retire every stored primitive that crosses that configuration, even when
the original coarser samples passed. Unknown space and original artifacts are
retained. This is a map revision, not a query-time NavMesh filter.
"""
import copy
import fcntl
from dataclasses import asdict
import gzip
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from scipy.sparse.csgraph import connected_components

from ..config import digest, file_digest, read_json, write_json
from ..native import StaticGeometry
from .body import BodyOracle
from .config import TRAVERSABLE, NON_TRAVERSABLE, UNKNOWN, angle_delta
from .roadmap import state_adjacency
from .topology import build_robot_topology

REFINEMENT_VERSION = 'native-witness-revision-2'


def native_motion_check(navigation,poses,cache=None):
    """Validate published XYZ samples, including unprojected height bridges."""
    q=np.asarray(poses,dtype=float)[:,:3]
    mask=np.r_[True,np.linalg.norm(np.diff(q,axis=0),axis=1)>1e-8]
    points=q[mask];key=points.tobytes()
    if cache is not None and key in cache:return cache[key]
    result=navigation.validate_path(points,.04)
    if cache is not None:cache[key]=result
    return result


def _read_ledger(path,source_id):
    if not path.exists():return []
    data=read_json(path)
    if data.get('source_traversability')!=source_id or digest(data['witnesses'])!=data.get('checksum'):
        raise ValueError('Native-contact evidence ledger identity/checksum mismatch')
    return data['witnesses']


def _save_ledger(path,source_id,witnesses):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        merged={tuple(np.round(w['failed_pose'],8)):w for w in _read_ledger(path,source_id)+list(witnesses)}
        rows=[merged[k] for k in sorted(merged)]
        write_json(path,{'source_traversability':source_id,'witnesses':rows,'checksum':digest(rows)})


def crosses_witness(poses, witness):
    """Exact configuration on a specified piecewise stop/turn/forward path.

    Tolerances only account for float32 geometry. They do not inflate a failed
    configuration into a blocked area or assume a collision radius.
    """
    q=np.asarray(poses,dtype=float);w=np.asarray(witness['failed_pose'],dtype=float)
    if not len(q):return False
    d=np.linalg.norm(q[:,:3]-w[:3],axis=1)
    yaw=np.arctan2(np.sin(q[:,3]-w[3]),np.cos(q[:,3]-w[3]))
    if np.any((d<2e-5)&(np.abs(yaw)<2e-6)):return True
    if len(q)<2:return False
    a,b=q[:-1],q[1:];v=b[:,:3]-a[:,:3];length2=np.sum(v*v,axis=1)
    t=np.sum((w[:3]-a[:,:3])*v,axis=1)/np.maximum(length2,1e-30)
    projected=a[:,:3]+np.clip(t,0,1)[:,None]*v
    turn=np.arctan2(np.sin(b[:,3]-a[:,3]),np.cos(b[:,3]-a[:,3]))
    rel=np.arctan2(np.sin(w[3]-a[:,3]),np.cos(w[3]-a[:,3]))
    stationary=length2<1e-14
    on_turn=stationary&(np.linalg.norm(a[:,:3]-w[:3],axis=1)<2e-5)&(rel*turn>=-1e-10)&(np.abs(rel)<=np.abs(turn)+2e-6)
    angular_error=np.arctan2(np.sin(rel-np.clip(t,0,1)*turn),np.cos(rel-np.clip(t,0,1)*turn))
    on_line=(~stationary)&(t>=-1e-6)&(t<=1+1e-6)&(np.linalg.norm(projected-w[:3],axis=1)<2e-5)&(np.abs(angular_error)<2e-6)
    return bool(np.any(on_turn|on_line))


def blocking_witness(poses, witnesses):
    return next((w for w in witnesses if crosses_witness(poses,w)),None)


class WitnessOracle(BodyOracle):
    """Native checks plus persistent stronger contact evidence."""
    def __init__(self,*args,witnesses=(),**kwargs):
        super().__init__(*args,**kwargs);self.witnesses=witnesses

    def pose(self,position,yaw,use_cache=True):
        witness=blocking_witness([[*position,yaw]],self.witnesses)
        return copy.deepcopy(witness['pose_evidence']) if witness is not None else super().pose(position,yaw,use_cache)

    def motion(self,*args,**kwargs):
        result=super().motion(*args,**kwargs)
        if result['status']==TRAVERSABLE:
            native=native_motion_check(self.nav,result['swept_poses'])
            if not native['valid']:
                return {'status':NON_TRAVERSABLE,'reason':'published_native_transition_failed','native_evidence':native}
        witness=blocking_witness(result.get('swept_poses',[]),self.witnesses) if result['status']==TRAVERSABLE else None
        if witness is not None:
            return {'status':NON_TRAVERSABLE,'reason':'retained_native_contact_witness','failed_pose':witness['failed_pose'],
                    'pose_evidence':witness['pose_evidence']}
        return result


def refine_robot_map(source, witnesses=(), audit_count=40, progress=None):
    """Audit finite primitives, incorporate stronger witnesses, rebuild topology.

    Additional evidence is additive and source-qualified. All retired motions
    remain inspectable in the new artifact. The old artifact is untouched.
    """
    from .map import _write_compressed,load_robot_hierarchical_map
    began=time.perf_counter()
    cache=source.manifest_path.parent.parent
    base_id=source.metadata.get('refinement',{}).get('base_traversability',source.topology_id)
    ledger=cache/'contact_evidence'/(base_id+'.json')
    previous=source.roadmap.get('stronger_contact_witnesses',[])
    supplied=list(previous)+_read_ledger(ledger,base_id)+list(witnesses)
    for w in supplied:
        if (w.get('status')!=NON_TRAVERSABLE or w.get('reason')!='native_body_contact' or
                'contact' not in w.get('pose_evidence',{}) or len(w.get('failed_pose',[]))!=4):
            raise ValueError('Refinement requires an actual native body contact witness')
    # Stable evidence identity excludes history-sensitive manifold coordinates;
    # exact failed configurations and the native rejection remain authoritative.
    unique={tuple(np.round(w['failed_pose'],8)):copy.deepcopy(w) for w in supplied}
    supplied=[unique[k] for k in sorted(unique)]
    request={'version':REFINEMENT_VERSION,'implementation':file_digest(__file__),
             'source_traversability':source.topology_id,'base_traversability':base_id,'audit_count':audit_count,
             'additional_contact_poses':[w['failed_pose'] for w in supplied]}
    index=cache/'refinements'/(digest(request)+'.json')
    if index.exists():
        path=Path(read_json(index)['manifest']);m=read_json(path)
        m['robot']=source.metadata['robot'];m['local']=source.metadata['local']
        m['map_id']=digest({'traversability':m['traversability_id'],'robot':m['robot'],'local':m['local']})
        target=cache/'maps'/(m['map_id']+'.json')
        if target!=path:write_json(target,m)
        result=load_robot_hierarchical_map(target,geometry=source.geometry)
        result.build_metrics={'cache_hit':True,'elapsed_seconds':time.perf_counter()-began,
                              'source_build':getattr(source,'build_metrics',None)}
        return result
    if source.geometry is None:
        with StaticGeometry(source.metadata['scene'],source.robot_config) as geo:
            source.geometry=geo
            try:result=refine_robot_map(source,witnesses,audit_count,progress)
            finally:source.geometry=None
        result.geometry=None
        return result
    audit=[];cfg=source.traversability_config
    selected=np.linspace(0,len(source.motions)-1,min(audit_count,len(source.motions)),dtype=int)
    with BodyOracle(source.geometry,source.navigation,cfg) as oracle:
        for mid in selected:
            proof=oracle.motion(source.motions[int(mid)]['poses'],cfg.spatial_step/2,cfg.angular_step/2,
                                use_cache=False,project_to_surface=False)
            row={'source_motion_id':int(mid),**{k:v for k,v in proof.items() if k!='swept_poses'}}
            audit.append(row)
            if proof.get('reason')=='native_body_contact':
                row['source_traversability']=source.topology_id
                unique[tuple(np.round(row['failed_pose'],8))]=row
    all_witnesses=[unique[k] for k in sorted(unique)]
    _save_ledger(ledger,base_id,all_witnesses)
    roadmap=copy.deepcopy(source.roadmap)
    retired=[];kept=[];native_cache={};native_failures=[]
    failed_audits={r['source_motion_id']:r for r in audit if r['status']!=TRAVERSABLE}
    for motion in roadmap['motions']:
        witness=blocking_witness(motion['poses'],all_witnesses)
        native=native_motion_check(source.navigation,motion['poses'],native_cache)
        if not native['valid']:
            native_failures.append({'source_motion_id':motion['id'],'evidence':native})
            retired.append({**motion,'status':NON_TRAVERSABLE,'reason':'published_native_transition_failed','evidence':native})
        elif witness is not None or motion['id'] in failed_audits:
            retired.append({**motion,'status':NON_TRAVERSABLE if witness is not None else failed_audits[motion['id']]['status'],
                            'reason':'stronger_native_contact_witness' if witness is not None else 'stronger_audit_inconclusive',
                            'evidence':witness or failed_audits[motion['id']]})
        else:
            motion['source_motion_id']=motion['id'];motion['id']=len(kept);kept.append(motion)
    roadmap['motions']=kept
    for site in roadmap['sites']:
        for i,yaw in enumerate(site['yaws']):
            witness=blocking_witness([[*site['position'],yaw]],all_witnesses)
            if witness is not None:site['pose_evidence'][i]=copy.deepcopy(witness['pose_evidence'])
        for turn in site['turns']:
            if turn['status']==TRAVERSABLE and blocking_witness(turn['poses'],all_witnesses):
                turn['status']=NON_TRAVERSABLE;turn['reason']='stronger_native_contact_witness'
        for state_id in site['state_ids']:
            state=roadmap['states'][state_id];reachable={state['heading_index']}
            if site['pose_evidence'][state['heading_index']]['status']!=TRAVERSABLE:
                state['status']=site['pose_evidence'][state['heading_index']]['status']
            changed=True
            while changed:
                changed=False
                for turn in site['turns']:
                    if turn['status']==TRAVERSABLE and (turn['source'] in reachable or turn['target'] in reachable):
                        before=len(reachable);reachable.update((turn['source'],turn['target']));changed|=len(reachable)>before
            state['heading_indices']=sorted(reachable)
    adj=state_adjacency(roadmap['states'],kept)
    for name in ('strong','weak'):
        count,labels=connected_components(adj,directed=True,connection=name)
        for state,label in zip(roadmap['states'],labels):state[name+'_component']=int(label)
        roadmap['statistics'][name+'_components']=int(count)
    transit={roadmap['states'][i]['surface_id'] for m in kept for i in (m['source'],m['target'])}
    roadmap['statistics']['area_with_transit_witness']=float(sum(source.surface.areas[i] for i in transit))
    stopping={s['surface_id'] for s in roadmap['states'] if s['status']==TRAVERSABLE}
    roadmap['statistics']['area_with_stopping_witness']=float(sum(source.surface.areas[i] for i in stopping))
    roadmap['statistics']['motions']=len(kept)
    roadmap['statistics']['stronger_evidence_revision']={'source_motions':len(source.motions),'retired_motions':len(retired),
        'native_contact_witnesses':len(all_witnesses),'audited':len(audit),'audit_passed':sum(r['status']==TRAVERSABLE for r in audit),
        'native_published_motion_denominator':len(source.motions),'native_published_motion_failures':len(native_failures)}
    roadmap['native_published_motion_failures']=native_failures
    roadmap['stronger_contact_witnesses']=all_witnesses
    roadmap['retired_motions']=roadmap.get('retired_motions',[])+retired
    roadmap['refinement_audit']=audit
    # Reconstruct regions, reachable components and global edges from the
    # revised robot roadmap. No old global edges are copied or filtered.
    original_regions=source.surface.region_ids.copy()
    graph=build_robot_topology(source.surface,roadmap,cfg,source.geometry)
    arrays=source.surface.arrays()
    source.surface.region_ids=original_regions
    inputs=copy.deepcopy(read_json(source.artifacts/'manifest.json')['inputs'])
    inputs['refinement']={**request,'contact_poses':[w['failed_pose'] for w in all_witnesses]}
    identity=digest(inputs);folder=cache/'robot_traversability'/identity;folder.mkdir(parents=True,exist_ok=True)
    _write_compressed(folder/'roadmap.json.gz',roadmap);write_json(folder/'graph.json',graph)
    np.savez_compressed(folder/'surface.npz',**arrays)
    write_json(folder/'manifest.json',{'schema_version':source.metadata['schema_version'],'inputs':inputs,
        'checksums':{n:file_digest(folder/n) for n in ('roadmap.json.gz','graph.json','surface.npz')}})
    m=copy.deepcopy(source.metadata);m['traversability_id']=identity
    m['map_id']=digest({'traversability':identity,'robot':m['robot'],'local':m['local']})
    path=cache/'maps'/(m['map_id']+'.json');m['artifacts']=os.path.relpath(folder,path.parent)
    m['refinement']=inputs['refinement'];write_json(path,m);write_json(index,{'manifest':str(path)})
    result=load_robot_hierarchical_map(path,geometry=source.geometry)
    result.build_metrics={'cache_hit':False,'elapsed_seconds':time.perf_counter()-began,
                          'source_build':getattr(source,'build_metrics',None)}
    if progress:progress({'phase':'stronger_evidence_revision',**roadmap['statistics']['stronger_evidence_revision']})
    return result
