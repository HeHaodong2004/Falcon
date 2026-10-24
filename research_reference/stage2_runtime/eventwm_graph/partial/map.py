"""Causal, sparse, observed configuration-space grid and certified topology.

This module deliberately imports neither Habitat nor any full map API. A free
cell is a *robot-center configuration-space* certificate supplied by the sensor
boundary; it is not an uninflated material occupancy pixel. Missing cells are
UNKNOWN. Historical certificates and newly observed obstructions are causal.
"""
from collections import deque
import copy
import heapq
import math
import numpy as np

UNKNOWN, KNOWN_FREE, KNOWN_OCCUPIED = 0, 1, 2


def skeleton_cells(keys):
    """Deterministic Zhang–Suen thinning of observed free support only."""
    if not keys:return set()
    xmin=min(k[0] for k in keys)-1;zmin=min(k[1] for k in keys)-1
    xmax=max(k[0] for k in keys)+1;zmax=max(k[1] for k in keys)+1
    image=np.zeros((xmax-xmin+1,zmax-zmin+1),dtype=np.uint8)
    for x,z in keys:image[x-xmin,z-zmin]=1
    while True:
        changed=False
        for phase in (0,1):
            p=np.pad(image,1)
            ns=[p[:-2,1:-1],p[:-2,2:],p[1:-1,2:],p[2:,2:],p[2:,1:-1],p[2:,:-2],p[1:-1,:-2],p[:-2,:-2]]
            degree=sum(ns);transitions=sum((a==0)&(b==1) for a,b in zip(ns,ns[1:]+ns[:1]))
            if phase==0:condition=(ns[0]*ns[2]*ns[4]==0)&(ns[2]*ns[4]*ns[6]==0)
            else:condition=(ns[0]*ns[2]*ns[6]==0)&(ns[0]*ns[4]*ns[6]==0)
            remove=(image==1)&(degree>=2)&(degree<=6)&(transitions==1)&condition
            if remove.any():image[remove]=0;changed=True
        if not changed:break
    return {(int(x+xmin),int(z+zmin)) for x,z in np.argwhere(image)}


class PartialGraph:
    def __init__(self, resolution, height, clearance=.25):
        if resolution <= 0 or clearance <= 0:
            raise ValueError("Positive collision-scaled resolution/clearance required")
        self.resolution = float(resolution)
        self.height = float(height)
        self.clearance = float(clearance)
        self.cells = {}
        self.timestamp = None
        self.revision = 0
        self.visited = []
        self.global_nodes = {}
        self.global_edges = []
        self.ego_support = None
        self._cache_revision = -1
        self._adjacency = {}

    def key(self, position):
        p = np.asarray(position)
        return (math.floor(p[0]/self.resolution), math.floor(p[2]/self.resolution))

    def center(self, key):
        c = self.cells.get(tuple(key))
        return np.array([(key[0]+.5)*self.resolution,
                         c['height'] if c else self.height,
                         (key[1]+.5)*self.resolution], dtype=float)

    def state(self, key):
        return self.cells.get(tuple(key), {}).get('state', UNKNOWN)

    def point_free(self, position):
        k = self.key(position)
        if self.ego_support is not None and np.linalg.norm(np.asarray(position)-self.ego_support) < 1e-7:
            return True
        return (self.state(k) == KNOWN_FREE and
                abs(float(position[1])-self.cells[k]['height']) <= .22)

    def segment_cells(self, a, b):
        """Exact grid-line interval supercover, including corner-touch cells."""
        a,b=np.asarray(a,dtype=float)[[0,2]]/self.resolution,np.asarray(b,dtype=float)[[0,2]]/self.resolution
        delta=b-a;crossings={0.,1.}
        for axis in (0,1):
            if abs(delta[axis])<1e-14:continue
            lo,hi=sorted((a[axis],b[axis]))
            for line in range(math.ceil(lo),math.floor(hi)+1):
                t=(line-a[axis])/delta[axis]
                if 0.<=t<=1.:crossings.add(float(t))
        ts=sorted(crossings);samples=ts+[(u+v)/2 for u,v in zip(ts,ts[1:])]
        found=set()
        for t in samples:
            p=a+delta*t;choices=[]
            for coordinate in p:
                nearest=round(float(coordinate))
                choices.append((nearest-1,nearest) if abs(coordinate-nearest)<1e-10 else (math.floor(coordinate),))
            for x in choices[0]:
                for z in choices[1]:found.add((x,z))
        return found

    def segment_free(self, a, b):
        if np.linalg.norm(np.asarray(a)-b)<1e-8:return self.point_free(a)
        if not self.point_free(a) or not self.point_free(b):
            return False
        if abs(float(a[1])-float(b[1])) > .22:
            return False
        return all(self.state(k) == KNOWN_FREE for k in self.segment_cells(a, b))

    def update(self, observation):
        t = float(observation['timestamp'])
        if not math.isfinite(t) or (self.timestamp is not None and t <= self.timestamp):
            raise ValueError("Map observations must arrive in strict timestamp order")
        updates = observation['static_updates']
        if any(float(u['timestamp']) != t for u in updates):
            raise ValueError("Future/stale static evidence is forbidden")
        added, removed, changed = [], [], []
        occupied_keys={tuple(u['cell']) for u in updates if u['state']==KNOWN_OCCUPIED}
        for u in updates:
            k = tuple(u['cell'])
            if u['source']=='occupied_ego_base_support':
                # Proprioception certifies exactly the occupied configuration,
                # never every shifted robot center in its containing square.
                continue
            state = int(u['state'])
            if state not in (KNOWN_FREE, KNOWN_OCCUPIED):
                raise ValueError("Only measured free/occupied cells may be published")
            previous = self.state(k)
            old_height=self.cells.get(k,{}).get('height')
            if state==KNOWN_FREE and k in occupied_keys:
                continue
            # Occupied evidence wins within a frame. A subsequent explicit
            # free certificate can clear a formerly obstructed static cell.
            self.cells[k] = {'state':state, 'height':float(u['height']),
                             'first_seen':self.cells.get(k,{}).get('first_seen', t),
                             'last_seen':t, 'source':u['source']}
            if previous != state or (old_height is not None and abs(old_height-float(u['height']))>1e-8):
                changed.append(list(k))
                if state == KNOWN_FREE:added.append(list(k))
                elif previous == KNOWN_FREE:removed.append(list(k))
        pose = observation['robot_pose']
        self.ego_support=np.asarray(pose[:3],dtype=float).copy()
        if not self.visited or np.linalg.norm((np.asarray(pose[:3])-self.visited[-1]['position'])[[0,2]]) >= .55:
            self.visited.append({'position':list(pose[:3]),'timestamp':t})
        self.timestamp = t
        if changed:
            self.revision += 1
        if changed:
            self.global_nodes={k:n for k,n in self.global_nodes.items() if self.state(n['cell'])==KNOWN_FREE}
            self.global_edges=[e for e in self.global_edges if e['source'] in self.global_nodes and e['target'] in self.global_nodes
                and all(self.segment_free(a,b) for a,b in zip(e['path'],e['path'][1:]))]
        return {'timestamp':t, 'revision':self.revision, 'added_free_cells':added,
                'invalidated_free_cells':removed, 'changed_cells':changed}

    @property
    def adjacency(self):
        if self._cache_revision == self.revision:
            return self._adjacency
        result = {}
        for k, c in sorted(self.cells.items()):
            if c['state'] != KNOWN_FREE:continue
            neighbors = []
            for dx,dz in ((-1,0),(0,-1),(0,1),(1,0),(-1,-1),(-1,1),(1,-1),(1,1)):
                v = (k[0]+dx,k[1]+dz)
                if self.state(v) != KNOWN_FREE:continue
                if dx and dz and (self.state((k[0]+dx,k[1])) != KNOWN_FREE or
                                  self.state((k[0],k[1]+dz)) != KNOWN_FREE):continue
                if abs(c['height']-self.cells[v]['height']) > .12:continue
                neighbors.append((v,float(np.linalg.norm(self.center(k)-self.center(v)))))
            result[k] = neighbors
        self._adjacency, self._cache_revision = result, self.revision
        return result

    def reachable(self, position):
        start = self.key(position)
        if not self.point_free(position) or self.state(start)!=KNOWN_FREE:return set()
        found, queue = {start}, deque([start])
        while queue:
            for v,_ in self.adjacency.get(queue.popleft(), []):
                if v not in found:found.add(v);queue.append(v)
        return found

    def route(self, start, target):
        """A* over confirmed cells; endpoint attachments use the same certificate."""
        if not self.point_free(start) or not self.point_free(target):
            raise ValueError('unknown_or_occupied_endpoint')
        if np.linalg.norm(np.asarray(start)-target)<1e-7:return np.asarray([start],dtype=float)
        source, goal = self.key(start), self.key(target)
        if not self.segment_free(start,self.center(source)) or not self.segment_free(self.center(goal),target):
            raise ValueError('uncertified_endpoint_attachment')
        costs, previous = {source:0.}, {source:None}
        queue = [(float(np.linalg.norm(self.center(source)-self.center(goal))),0.,source)]
        while queue:
            _, cost, k = heapq.heappop(queue)
            if cost != costs[k]:continue
            if k == goal:break
            for v,length in self.adjacency.get(k, []):
                new = cost+length
                if new < costs.get(v,math.inf)-1e-10:
                    costs[v],previous[v] = new,k
                    heuristic = float(np.linalg.norm(self.center(v)-self.center(goal)))
                    heapq.heappush(queue,(new+heuristic,new,v))
        if goal not in previous:raise ValueError('disconnected_known_free_component')
        keys, k = [],goal
        while k is not None:keys.append(k);k=previous[k]
        path = [np.asarray(start), *[self.center(k) for k in reversed(keys)], np.asarray(target)]
        path = [p for j,p in enumerate(path) if j == 0 or np.linalg.norm(p-path[j-1]) > 1e-7]
        # Visibility-string pulling never consults the private geometry cache.
        simple, i = [path[0]],0
        while i < len(path)-1:
            j = len(path)-1
            while j > i+1 and not self.segment_free(path[i],path[j]):j -= 1
            simple.append(path[j]);i = j
        if not all(self.segment_free(a,b) for a,b in zip(simple,simple[1:])):
            raise ValueError('uncertified_grid_edge')
        return np.asarray(simple)

    def frontiers(self, position, limit=8):
        reachable = self.reachable(position)
        boundary = {k for k in reachable if any(self.state((k[0]+dx,k[1]+dz)) == UNKNOWN
                    for dx,dz in ((1,0),(-1,0),(0,1),(0,-1)))}
        components=[]
        while boundary:
            seed=min(boundary);boundary.remove(seed);component=[seed];queue=deque([seed])
            while queue:
                k=queue.popleft()
                for dx in (-1,0,1):
                    for dz in (-1,0,1):
                        v=(k[0]+dx,k[1]+dz)
                        if v in boundary:boundary.remove(v);component.append(v);queue.append(v)
            if len(component)<2:continue
            mean=np.mean([self.center(k) for k in component],axis=0)
            rep=min(component,key=lambda k:(np.linalg.norm(self.center(k)-mean),k))
            unknown=[self.center((rep[0]+dx,rep[1]+dz)) for dx,dz in ((1,0),(-1,0),(0,1),(0,-1))
                     if self.state((rep[0]+dx,rep[1]+dz)) == UNKNOWN]
            direction=np.mean(unknown,axis=0)-self.center(rep)
            components.append({'cell':list(rep),'position':self.center(rep).tolist(),
                               'component_cells':len(component),
                               'unknown_yaw':math.atan2(-direction[2],direction[0])})
        return sorted(components,key=lambda c:(-c['component_cells'],c['cell']))[:limit]

    def rebuild_global(self, position, limit=72):
        reachable=self.reachable(position)
        proposed=[]
        skeleton=skeleton_cells(reachable)
        for k in sorted(skeleton):
            ns=[(k[0]+dx,k[1]+dz) for dx in (-1,0,1) for dz in (-1,0,1)
                if (dx or dz) and (k[0]+dx,k[1]+dz) in skeleton]
            if len(ns)>=3:kind='junction'
            elif len(ns)==2 and (ns[0][0]+ns[1][0]!=2*k[0] or ns[0][1]+ns[1][1]!=2*k[1]):kind='corridor_turn'
            elif len(ns)==1:kind='corridor_transition'
            else:kind='visibility_support'
            proposed.append(({'junction':0,'corridor_turn':1,'corridor_transition':1,'visibility_support':3}[kind],k,kind))
        for v in self.visited:
            if self.key(v['position']) in reachable:proposed.append((-1,self.key(v['position']),'historical_decision'))
        for f in self.frontiers(position):proposed.append((0,tuple(f['cell']),'frontier_viewpoint'))
        retained={k:v for k,v in self.global_nodes.items() if tuple(v['cell']) in reachable}
        selected=sorted(retained.values(),key=lambda n:n['id'])
        for _,k,kind in sorted(proposed):
            p=self.center(k)
            if any(np.linalg.norm((p-np.asarray(n['position']))[[0,2]])<.85 for n in selected):continue
            nid=f'g_{k[0]}_{k[1]}'
            selected.append({'id':nid,'cell':list(k),'position':p.tolist(),'kind':kind,'created_at':self.timestamp})
            if len(selected)>=limit:break
        # Uniformly spaced visibility support is a fallback where an open-room
        # skeleton has few vertices. Farthest-point coverage is deterministic,
        # rooted at the historical ego node, and retains points behind it.
        remaining=sorted(reachable)
        positions=np.asarray([self.center(k)[[0,2]] for k in remaining])
        if len(remaining):
            nearest=(np.linalg.norm(positions[:,None,:]-np.asarray([n['position'] for n in selected])[:,[0,2]][None,:,:],axis=-1).min(axis=1)
                     if selected else np.full(len(remaining),np.inf))
        while len(selected)<limit and remaining:
            index=int(np.argmax(nearest));distance=float(nearest[index])
            if distance<1.25:break
            k=remaining[index];p=self.center(k)
            selected.append({'id':f'g_{k[0]}_{k[1]}','cell':list(k),'position':p.tolist(),
                             'kind':'visibility_support','created_at':self.timestamp})
            nearest=np.minimum(nearest,np.linalg.norm(positions-p[[0,2]],axis=1))
        self.global_nodes={n['id']:n for n in selected[:limit]}
        edges=[];seen=set()
        for a in self.global_nodes.values():
            nearby=sorted((float(np.linalg.norm(np.asarray(a['position'])-b['position'])),b['id'])
                          for b in self.global_nodes.values() if b['id']!=a['id'])[:4]
            for distance,bid in nearby:
                pair=tuple(sorted((a['id'],bid)))
                if pair in seen or distance>3.5:continue
                seen.add(pair)
                try:path=self.route(a['position'],self.global_nodes[bid]['position'])
                except ValueError:continue
                length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
                if a['id']!=pair[0]:path=path[::-1]
                edges.append({'source':pair[0],'target':pair[1], 'path':path.tolist(),
                              'length':length,'revision':self.revision,
                              'certificate':'known_free_cell_supercover'})
        # Sparse nearest-neighbor links alone can strand an ego/historical node
        # when the bounded node budget fills elsewhere. Join every abstract
        # component by an explicitly stored local A* path, never a straight
        # unverified shortcut or a complete-map connectivity lookup.
        parent={nid:nid for nid in self.global_nodes}
        def find(k):
            while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
            return k
        for e in edges:parent[find(e['source'])]=find(e['target'])
        pairs=sorted((float(np.linalg.norm(np.asarray(a['position'])-b['position'])),a['id'],b['id'])
                     for a in self.global_nodes.values() for b in self.global_nodes.values() if a['id']<b['id'])
        for _,a,b in pairs:
            if find(a)==find(b):continue
            path=self.route(self.global_nodes[a]['position'],self.global_nodes[b]['position'])
            edges.append({'source':a,'target':b,'path':path.tolist(),
                          'length':float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
                          'revision':self.revision,'certificate':'known_free_Astar_component_backbone'})
            parent[find(a)]=find(b)
        edges.sort(key=lambda e:(e['source'],e['target']))
        self.global_edges=edges
        return {'nodes':copy.deepcopy(list(self.global_nodes.values())), 'edges':copy.deepcopy(edges)}

    def snapshot(self):
        return {'schema':'causal_partial_cspace_graph_1','resolution':self.resolution,
                'height':self.height,'clearance':self.clearance,'timestamp':self.timestamp,'revision':self.revision,
                'ego_support':self.ego_support.tolist() if self.ego_support is not None else None,
                'cells':[{'cell':list(k),**copy.deepcopy(v)} for k,v in sorted(self.cells.items())],
                'visited':copy.deepcopy(self.visited),
                'global_nodes':copy.deepcopy(self.global_nodes),'global_edges':copy.deepcopy(self.global_edges)}

    @classmethod
    def restore(cls, snapshot):
        g=cls(snapshot['resolution'],snapshot['height'],snapshot['clearance'])
        g.timestamp,g.revision=snapshot['timestamp'],snapshot['revision']
        g.ego_support=np.asarray(snapshot['ego_support']) if snapshot.get('ego_support') is not None else None
        g.cells={tuple(c['cell']):{k:v for k,v in c.items() if k!='cell'} for c in snapshot['cells']}
        g.visited=copy.deepcopy(snapshot['visited'])
        g.global_nodes=copy.deepcopy(snapshot.get('global_nodes',{}));g.global_edges=copy.deepcopy(snapshot.get('global_edges',[]))
        return g
