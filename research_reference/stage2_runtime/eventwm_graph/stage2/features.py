"""Only current legal map and executed XYZ/yaw/time enter these features.

The output domain is initially UNKNOWN cell centers within 11m of a recorded
pose. Sensor max range is10m; the extra meter covers base/camera offset and
cell extent. Domain misses are always counted during evaluation.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

RADIUS=11.
OFFSETS=np.array([(x,z) for x in (-3,0,3) for z in (-3,0,3)],np.int32)
FEATURES=47


def validate(graph,trajectory):
    if graph.get('schema')!='causal_partial_cspace_graph_1':raise ValueError('unsupported PartialGraph schema')
    resolution=float(graph['resolution'])
    if not np.isfinite(resolution) or not .02<=resolution<=1:raise ValueError('unsupported map resolution')
    p=np.asarray(trajectory,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=5 or not len(p) or not np.isfinite(p).all():raise ValueError('trajectory rows must be [time,x,y,z,yaw]')
    if len(p)>1 and np.any(np.diff(p[:,0])<=0):raise ValueError('executed trajectory timestamps must increase')
    if abs(p[0,0]-float(graph['timestamp']))>1e-7:raise ValueError('trajectory must begin at current graph time')
    if len({tuple(c['cell']) for c in graph['cells']})!=len(graph['cells']):raise ValueError('duplicate graph cells')
    if any(c['state'] not in (1,2) or not np.isfinite(c['height']) for c in graph['cells']):raise ValueError('invalid known cell')
    return p


class SceneFeatures:
    """Per-call scratch/cache of only the supplied initial graph."""
    def __init__(self,graph,trajectory):
        self.graph=graph;self.path=validate(graph,trajectory);self.resolution=float(graph['resolution'])
        self.known={tuple(c['cell']):c for c in graph['cells']}
        self.xy=self.path[:,[1,3]].astype(np.float32)
        self.tree=cKDTree(self.xy)
        # These values depend on poses, not candidate cells. Keep float64
        # arithmetic identical to the original feature definition.
        self.pose_cos=np.cos(self.path[:,4]);self.pose_sin=np.sin(self.path[:,4])
        self.duration=float(self.path[-1,0]-self.path[0,0])
        self.pose_timefrac=(self.path[:,0]-self.path[0,0])/max(self.duration,1e-4)
        self._candidate_keys=None;self._candidate_neighbors=None
        # Dense scratch is local to the executed segment, never full hidden map.
        lo=np.floor((self.xy.min(0)-RADIUS)/self.resolution).astype(np.int32)-4
        hi=np.floor((self.xy.max(0)+RADIUS)/self.resolution).astype(np.int32)+5
        shape=hi-lo
        if np.prod(shape)>4000000:raise ValueError('trajectory local domain exceeds four million scratch cells; split segment')
        self.lo=lo;self.shape=shape
        self.states=np.zeros(shape,np.uint8);self.heights=np.full(shape,float(graph['height']),np.float32)
        if graph['cells']:
            keys=np.array([c['cell'] for c in graph['cells']],np.int32)-lo
            ok=np.all((keys>=0)&(keys<shape),axis=1);keys=keys[ok]
            cells=[c for c,k in zip(graph['cells'],ok) if k]
            self.states[keys[:,0],keys[:,1]]=[c['state'] for c in cells]
            self.heights[keys[:,0],keys[:,1]]=[c['height'] for c in cells]
        self.dist=[]
        for state in (1,2):
            self.dist.append(distance_transform_edt(self.states!=state).astype(np.float32)*self.resolution
                if np.any(self.states==state) else np.full(shape,RADIUS,np.float32))
        if np.any(self.states):
            inds=distance_transform_edt(self.states==0,return_distances=False,return_indices=True)
            self.base_height=self.heights[inds[0],inds[1]]
        else:self.base_height=self.heights

    def candidates(self):
        ij=np.argwhere(self.states==0).astype(np.int32)
        keys=ij+self.lo
        xy=(keys.astype(np.float32)+.5)*self.resolution
        count=min(4,len(self.path));distance,index=self.tree.query(xy,k=count,workers=1)
        distance=np.asarray(distance).reshape(len(keys),count);index=np.asarray(index).reshape(len(keys),count)
        keep=distance[:,0]<=RADIUS
        self._candidate_keys=keys[keep]
        self._candidate_neighbors=(distance[keep].astype(np.float32),index[keep])
        return self._candidate_keys

    def transform(self,keys):
        reuse=keys is self._candidate_keys
        keys=np.asarray(keys,np.int32).reshape(-1,2);n=len(keys)
        if not n:return np.empty((0,FEATURES),np.float32),np.empty(0,np.float32)
        ij=keys-self.lo
        if np.any(ij<0) or np.any(ij>=self.shape):raise ValueError('candidate outside scratch domain')
        xy=(keys.astype(np.float32)+.5)*self.resolution
        # Four closest actual poses retain headings/time; the complete path
        # controls candidate support, duration, endpoint and length summaries.
        count=min(4,len(self.path))
        if reuse:d,idx=self._candidate_neighbors
        else:
            d,idx=self.tree.query(xy,k=count,workers=1)
            d=np.asarray(d).reshape(n,count).astype(np.float32);idx=np.asarray(idx).reshape(n,count)
        if count<4:
            d=np.pad(d,((0,0),(0,4-count)),mode='edge');idx=np.pad(idx,((0,0),(0,4-count)),mode='edge')
        rel_x=xy[:,0,None]-self.path[:,1][idx];rel_z=xy[:,1,None]-self.path[:,3][idx]
        cosine=self.pose_cos[idx];sine=self.pose_sin[idx]
        distance=np.maximum(d,1e-4)
        facing=(rel_x*cosine-rel_z*sine)/distance
        side=(rel_x*sine+rel_z*cosine)/distance
        # Assign directly to the final float32 matrix. The old concatenate
        # promoted every column to float64 before making this same cast.
        x=np.empty((n,FEATURES),np.float32)
        x[:,0:4]=d/10;x[:,4:8]=np.clip(facing,-1,1);x[:,8:12]=np.clip(side,-1,1)
        x[:,12:16]=self.pose_timefrac[idx]
        if np.all(ij>=3) and np.all(ij<self.shape-3):
            center=ij[:,0]*int(self.shape[1])+ij[:,1]
            offsets=OFFSETS[:,0]*int(self.shape[1])+OFFSETS[:,1]
            neighbor_states=self.states.ravel()[center[:,None]+offsets[None,:]]
        else:
            neighbor=ij[:,None,:]+OFFSETS[None,:,:]
            neighbor=np.clip(neighbor,0,self.shape-1)
            neighbor_states=self.states[neighbor[:,:,0],neighbor[:,:,1]]
        x[:,16:25]=neighbor_states==1;x[:,25:34]=neighbor_states==2
        baseline=self.base_height[ij[:,0],ij[:,1]]
        start=(xy-self.xy[0])/10;end=(xy-self.xy[-1])/10
        length=float(np.linalg.norm(np.diff(self.xy,axis=0),axis=1).sum())
        x[:,34]=self.dist[0][ij[:,0],ij[:,1]]/10;x[:,35]=self.dist[1][ij[:,0],ij[:,1]]/10
        x[:,36:38]=start;x[:,38:40]=end;x[:,40]=baseline-self.path[0,2]
        x[:,41:47]=np.array([self.duration/5,length/5,self.resolution,(self.path[-1,2]-self.path[0,2]),
            np.cos(self.path[-1,4]-self.path[0,4]),np.sin(self.path[-1,4]-self.path[0,4])],np.float32)
        if x.shape!=(n,FEATURES) or not np.isfinite(x).all():raise ValueError('invalid feature contract: '+str(x.shape))
        return x,baseline
