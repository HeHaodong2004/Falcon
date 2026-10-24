"""Causal continuous track/robot summaries and a spatial partial-graph encoder.

No simulator, target reader, query, physical identity, or future input imports.
All positive observations contribute to memory; nondetection != disappearance.
"""
import hashlib
import numpy as np
from collections.abc import Mapping
from .types import SealedObservation,packed_cell_keys
from .topology import topology_features

RADII = np.array([.2,.5,1.,2.,4.])
ANGLES = np.arange(8)*np.pi/4
OFFSETS = np.r_[np.zeros((1,2)), np.stack([np.repeat(RADII,8)*np.tile(np.cos(ANGLES),5),
    np.repeat(RADII,8)*np.tile(np.sin(ANGLES),5)],axis=-1)].astype(np.float32)
# Sparse radial/angular edges, self loops, and center-to-inner ring connections.
ADJ = np.eye(len(OFFSETS),dtype=np.float32)
for r in range(5):
    for a in range(8):
        i=1+r*8+a
        for j in (1+r*8+(a+1)%8, 0 if r==0 else i-8):
            ADJ[i,j]=ADJ[j,i]=1
ADJ /= ADJ.sum(-1,keepdims=True)
FEATURES = 28
GRAPH_FEATURES = 14


class OfflineViewFrame(Mapping):
    """Task-owned decoded frame with reused view facts; never a trusted online seal."""
    def __init__(self,ob):
        self.ob=ob
        self.free_support=frozenset(tuple(u['cell']) for u in ob['static_updates'] if u['state']==1 and u['source']!='occupied_ego_base_support')
        self.occupied_support=frozenset(tuple(u['cell']) for u in ob['static_updates'] if u['state']==2)
    def __getitem__(self,k):return self.ob[k]
    def __iter__(self):return iter(self.ob)
    def __len__(self):return len(self.ob)


def historical_view_evidence(history,points,height,resolution,view=None):
    if view is not None:
        view.validate_certificate()
        if view.frame_count!=len(history) or view.timestamp!=float(history[-1]['timestamp']):
            raise ValueError('view facts differ from arrived history')
        indices=np.flatnonzero(view.valid)
        if not len(indices):return np.zeros((*points.shape[:-1],3),np.float32)
        if not view.packed_supported[indices].all():return historical_view_reference(history,points,height,resolution)
        flat=np.asarray(points).reshape(-1,2)
        if not len(flat):return np.zeros((*points.shape[:-1],3),np.float32)
        floor=np.floor(flat/resolution)
        if np.any(floor<-(1<<31)) or np.any(floor>=(1<<31)):
            return historical_view_reference(history,points,height,resolution)
        keys=packed_cell_keys(floor.astype(np.int64))
        xyz=np.c_[flat[:,0],np.full(len(flat),height+1.),flat[:,1]]
        origins=view.origins[indices];rotations=view.rotations[indices]
        from .view import certified_camera_frustum
        frustum,ambiguous=certified_camera_frustum(xyz,origins,rotations)
        if ambiguous.any():
            # Boundary/nonfinite fallback keeps the ORIGINAL full point shape
            # for each entire ambiguous frame, not a reshaped subset of points.
            local=np.matmul(xyz[None]-origins[ambiguous,None],rotations[ambiguous])
            forward=-local[...,2]
            frustum[ambiguous]=(forward>1e-6)&(np.abs(local[...,0])<=forward)&(np.abs(local[...,1])<=forward)&(np.linalg.norm(local,axis=-1)<=10.)
        decay=np.exp(-(history[-1]['timestamp']-view.timestamps[indices]))
        from .view import reduce_view_evidence
        result=reduce_view_evidence(frustum,flat,keys,decay,indices,view.code_offsets,view.codes,view.positive_offsets,view.positive_xy)
        return result.reshape(*points.shape[:-1],3).astype(np.float32)
    valid=[ob for ob in history if ob.get('observation_valid',True) and ob.get('human_sensor_available',True)]
    if not valid:return np.zeros((*points.shape[:-1],3),np.float32)
    if not all(isinstance(ob,SealedObservation) and ob.free_codes is not None for ob in valid):
        return historical_view_reference(history,points,height,resolution)
    flat=np.asarray(points).reshape(-1,2)
    if not len(flat):return np.zeros((*points.shape[:-1],3),np.float32)
    floor=np.floor(flat/resolution)
    if np.any(floor<-(1<<31)) or np.any(floor>=(1<<31)):
        return historical_view_reference(history,points,height,resolution)
    keys=packed_cell_keys(floor.astype(np.int64))
    xyz=np.c_[flat[:,0],np.full(len(flat),height+1.),flat[:,1]]
    origins=np.asarray([ob['camera_origin'] for ob in valid])
    rotations=np.asarray([ob['camera_rotation'] for ob in valid])
    local=np.matmul(xyz[None]-origins[:,None],rotations)
    forward=-local[...,2]
    frustum=(forward>1e-6)&(np.abs(local[...,0])<=forward)&(np.abs(local[...,1])<=forward)&(np.linalg.norm(local,axis=-1)<=10.)
    decay=np.exp(-(history[-1]['timestamp']-np.asarray([ob['timestamp'] for ob in valid])))
    support=np.zeros(frustum.shape,bool);nearby=np.zeros(frustum.shape,bool)
    for i,ob in enumerate(valid):
        codes=ob.free_codes
        if len(codes):
            at=np.searchsorted(codes,keys);inside=at<len(codes)
            support[i,inside]=codes[at[inside]]==keys[inside]
        detections=np.array([np.asarray(tr['position'])[[0,2]] for tr in positives(ob)]).reshape(-1,2)
        if len(detections):nearby[i]=np.linalg.norm(flat[:,None]-detections[None],axis=-1).min(-1)<.5
    result=np.stack(((frustum*decay[:,None]).max(0),(nearby*decay[:,None]).max(0),
        ((frustum&support&(~nearby))*decay[:,None]).max(0)),axis=-1)
    return result.reshape(*points.shape[:-1],3).astype(np.float32)


def historical_view_reference(history,points,height,resolution):
    """Spatial, timestamped *opportunities*, never certified empty FOVs.

    Each actual camera frame contributes only its own arrived free-support
    cells, not the final map projected backwards. Human-mesh occlusion remains
    unknown. The network learns what these nondetections imply probabilistically.
    """
    flat=np.asarray(points).reshape(-1,2);now=history[-1]['timestamp'];out=np.zeros((len(flat),3),np.float32)
    xyz=np.c_[flat[:,0],np.full(len(flat),height+1.),flat[:,1]]
    cell_keys=[tuple(k) for k in np.floor(flat/resolution).astype(int)]
    for ob in history:
        if not ob.get('observation_valid',True) or not ob.get('human_sensor_available',True):continue
        local=(xyz-np.asarray(ob['camera_origin']))@np.asarray(ob['camera_rotation'])
        forward=-local[:,2];frustum=(forward>1e-6)&(np.abs(local[:,0])<=forward)&(np.abs(local[:,1])<=forward)&(np.linalg.norm(local,axis=-1)<=10.)
        decay=np.exp(-(now-ob['timestamp']))
        out[:,0]=np.maximum(out[:,0],frustum*decay)
        if isinstance(ob,(SealedObservation,OfflineViewFrame)):free=ob.free_support;blocked=ob.occupied_support
        else:
            free={tuple(u['cell']) for u in ob['static_updates'] if u['state']==1 and u['source']!='occupied_ego_base_support'}
            blocked={tuple(u['cell']) for u in ob['static_updates'] if u['state']==2}
        support=np.array([k in free and k not in blocked for k in cell_keys])
        detections=np.array([np.asarray(tr['position'])[[0,2]] for tr in positives(ob)]).reshape(-1,2)
        nearby=(np.linalg.norm(flat[:,None]-detections[None],axis=-1).min(-1)<.5) if len(detections) else np.zeros(len(flat),bool)
        out[:,1]=np.maximum(out[:,1],nearby*decay)
        out[:,2]=np.maximum(out[:,2],frustum*support*(~nearby)*decay)
    return out.reshape(*points.shape[:-1],3)


def positives(ob):
    if not ob.get('observation_valid',True) or not ob.get('human_sensor_available',True): return []
    return [tr for tr in ob['tracks'] if tr.get('visible_now') and tr.get('valid',True)
            and abs(float(tr['last_seen_time'])-float(ob['timestamp']))<1e-7]


def extract(context):
    history=context.history; graph=context.partial_graph; now=float(history[-1]['timestamp'])
    robot=np.asarray(history[-1]['robot_pose']); robotxy=robot[[0,2]]
    from .context import OnlineContext
    if isinstance(context,OnlineContext):
        summary=context.motion_summary
        if summary.frame_count!=len(history) or summary.timestamp!=now:
            raise ValueError('motion summary differs from actual arrived history')
        memory=summary.memory;valid=summary.valid;empty=summary.empty;invalid=summary.invalid
        distance=summary.distance;yaw_motion=summary.yaw_motion;minimum_count=summary.minimum_count
    else:
        memory={}; valid=0;empty=0;invalid=0;distance=0.;yaw_motion=0.;minimum_count=0
        for j,ob in enumerate(history):
            if j:
                prev=history[j-1]['robot_pose'];distance+=np.linalg.norm(np.asarray(ob['robot_pose'])[[0,2]]-np.asarray(prev)[[0,2]])
                yaw_motion+=abs(np.angle(np.exp(1j*(ob['robot_pose'][3]-prev[3]))))
            tracks=positives(ob)
            minimum_count=max(minimum_count,len(tracks))
            if ob.get('observation_valid',True) and ob.get('human_sensor_available',True): valid+=1;empty+=not bool(tracks)
            else: invalid+=1
            for tr in tracks:
                key=str(tr['track_id']);t=float(ob['timestamp']);pos=np.asarray(tr['position'])[[0,2]]
                velocity=np.zeros(2);acceleration=np.zeros(2)
                if key in memory:
                    old=memory[key];dt=t-old['last'];velocity=(pos-old['pos'])/dt
                    acceleration=(velocity-old['velocity'])/max(dt,.048)
                    # Record every actual observation through stable online moments.
                    count=old['count']+1;mean=old['mean']+(pos-old['mean'])/count
                    first=old['first'];observed_path=old['path']+np.linalg.norm(pos-old['pos'])
                else:
                    count=1;mean=pos.copy();first=t;observed_path=0.
                    if tr.get('velocity_valid') and tr.get('estimated_velocity') is not None:
                        velocity=np.asarray(tr['estimated_velocity'])[[0,2]]
                memory[key]=dict(pos=pos,last=t,velocity=velocity,acceleration=acceleration,count=count,
                                 mean=mean,first=first,path=observed_path)
    keys=tuple(memory);n=len(keys)
    x=np.zeros((n,FEATURES),np.float32);origin=np.zeros((n,2),np.float32);vel=np.zeros((n,2),np.float32)
    cells=context.certified_cells if hasattr(context,'certified_cells') else {tuple(c['cell']):c for c in graph['cells']}
    resolution=float(graph['resolution'])
    positions=np.array([memory[k]['pos']+memory[k]['velocity']*(now-memory[k]['last']) for k in keys]).reshape(n,2)
    graph_centers=np.r_[positions,robotxy[None]]
    graph_points=graph_centers[:,None]+OFFSETS[None]
    all_g=np.zeros((n+1,len(OFFSETS),GRAPH_FEATURES),np.float32)
    for i,key in enumerate(keys):
        m=memory[key];age=now-m['last'];origin[i]=m['pos']+m['velocity']*age;vel[i]=m['velocity']
        rel=m['pos']-robotxy;mean=m['mean']-robotxy
        other=np.delete(positions,i,axis=0);nearest=float(np.linalg.norm(other-origin[i],axis=-1).min()) if len(other) else 20.
        x[i]=np.r_[rel/10,m['velocity']/2,np.clip(m['acceleration'],-10,10)/10,age/3,
            (m['last']-m['first'])/5,np.log1p(m['count'])/5,mean/10,m['path']/10,
            np.sin(robot[3]),np.cos(robot[3]),distance/10,yaw_motion/10,
            empty/max(valid,1),invalid/len(history),valid/100,nearest/10,
            float(age==0),now-history[0]['timestamp'],
            np.asarray(history[-1]['camera_origin'])[[0,2]]-robotxy,
            np.asarray(history[-1]['camera_rotation'])[0,[0,2]],
            sum(a=='left' for a in context.actions)/100,sum(a=='right' for a in context.actions)/100]
    # Preserve the exact float64 points, floor boundaries and lookup order, while
    # avoiding a tiny ndarray/floor/astype allocation for every radial sample.
    coordinates=np.floor(graph_points.reshape(-1,2)/resolution).astype(int).tolist()
    local_cells=[cells.get(tuple(coordinate)) for coordinate in coordinates]
    local_states=np.asarray([0 if cell is None else cell['state'] for cell in local_cells])
    local_last=np.asarray([now if cell is None else cell['last_seen'] for cell in local_cells],np.float64)
    local_height=np.asarray([graph['height'] if cell is None else cell['height'] for cell in local_cells],np.float64)
    flat=all_g.reshape(-1,GRAPH_FEATURES)
    flat[:,0]=local_states==0;flat[:,1]=local_states==1;flat[:,2]=local_states==2
    flat[:,3]=np.minimum(10,now-local_last)/10
    all_g[...,4:6]=OFFSETS/4
    flat[:,6]=local_height-graph['height']
    all_g[...,7:10]=historical_view_evidence(history,graph_points,graph['height'],resolution,
        context.view_summary if isinstance(context,OnlineContext) else None)
    all_g[...,10:]=topology_features(graph,graph_points)
    g=all_g[:-1];global_graph=all_g[-1]
    if isinstance(context,OnlineContext):
        facts=context.cell_facts
        states=facts.states;last=facts.last_seen
    else:
        states=np.array([c['state'] for c in graph['cells']]);last=np.array([c['last_seen'] for c in graph['cells']])
    global_features=np.array([n/10,len(cells)/5000,np.mean(states==1) if len(states) else 0,
        np.mean(states==2) if len(states) else 0,empty/max(valid,1),invalid/len(history),
        now-history[0]['timestamp'],distance/10,yaw_motion/10,
        np.mean(now-last)/10 if len(last) else 0,len(graph['global_edges'])/100,
        len(graph['global_nodes'])/100],np.float32)
    return dict(x=x,graph=g,global_graph=global_graph,origin=origin,velocity=vel,keys=keys,global_features=global_features,minimum_count=minimum_count,
                timestamp=now,robotxy=robotxy.astype(np.float32),age=np.array([now-memory[k]['last'] for k in keys]),
                last_positions=np.array([memory[k]['pos'] for k in keys]).reshape(n,2),
                last_times=np.array([memory[k]['last'] for k in keys]),cells=cells,resolution=resolution)
