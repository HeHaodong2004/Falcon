"""Public benchmark surface map. Production access always checks certification."""
from collections import defaultdict
from dataclasses import asdict
import copy
import heapq
import math
from pathlib import Path
import time
import numpy as np
from ..config import digest,file_digest,read_json,write_json
from ..map import load_hierarchical_map as load_source
from ..topology import concatenate
from .certify import VERIFIED
from .contract import installed_contract,SCHEMA
from .motion import KinematicModel


class CertificationError(ValueError):
    pass


class CertifiedSurfaceMap:
    def __init__(self,manifest,require_production=True):
        began=time.perf_counter();self.manifest_path=Path(manifest).resolve()
        self.metadata=read_json(self.manifest_path);meta=self.metadata
        if meta.get('schema_version')!=SCHEMA or digest(meta['identity'])!=meta['map_id']:
            raise CertificationError('Certification identity mismatch')
        self.folder=self.manifest_path.parent
        for name,sha in meta['checksums'].items():
            if file_digest(self.folder/name)!=sha:raise CertificationError('Damaged certification evidence: '+name)
        self.contract=installed_contract(meta['contract']['config_name'])
        if self.contract!=meta['contract']:raise CertificationError('Installed benchmark differs from certified contract')
        for name in ('motion.py','contract.py'):
            if file_digest(Path(__file__).with_name(name))!=meta['identity']['certifier'][name]:
                raise CertificationError('Certified motion implementation changed: '+name)
        self.source=load_source(meta['source_manifest'])
        if self.source.map_id!=meta['source_map_id']:raise CertificationError('Source map identity mismatch')
        self.robot_config=self.source.robot_config
        if digest(asdict(self.robot_config))!=meta['robot_configuration_id']:
            raise CertificationError('Robot identity mismatch')
        self.map_id=meta['map_id'];self.robot_configuration_id=meta['robot_configuration_id']
        self.navigation=self.source.navigation;self.surface=self.source.surface;self.local=self.source.local
        self.nodes=copy.deepcopy(self.source.nodes)
        for node in self.nodes.values():
            node['historical_body_diagnostic']={'pose_validation':node.get('pose_validation'),'visiting_yaws':node.get('visiting_yaws')}
            node['pose_validation']='native_surface_and_installed_kinematic_actions'
            node['visiting_yaws']=None;node['orientation_domain']='SE(2); discrete action compilation required'
        self.edges={r['id']:r for r in read_json(self.folder/'edges.json')}
        self.report=read_json(self.folder/'report.json')
        if self.report!=meta['report']:raise CertificationError('Gate report mismatch')
        valid=[e for e in self.edges.values() if e['status']==VERIFIED]
        for e in valid:
            independent=e.get('actual_acceptance',{}).get('valid',e['actual']['pose_parity'])
            if (e['map_id']!=self.map_id or not independent or not e['independent_surface']['valid']
                or e['actual']['benchmark_should_end'] or e['source'] not in self.nodes or e['target'] not in self.nodes):
                raise CertificationError('Executable edge lacks required evidence')
        if len(valid)!=self.report['status_counts'].get(VERIFIED,0):raise CertificationError('Evidence denominator mismatch')
        if require_production and not self.report['production_gate']:
            raise CertificationError('Stage 0 production gate is closed; inspect '+str(self.folder/'report.json'))
        self.adjacency=defaultdict(list)
        for e in valid:self.adjacency[e['source']].append((e['target'],e['id'],e['length']))
        self.model=KinematicModel(self.navigation,self.contract)
        self.load_seconds=time.perf_counter()-began

    def get_global_graph(self):
        return {'map_id':self.map_id,'schema_version':SCHEMA,'directed':True,
                'nodes':copy.deepcopy(list(self.nodes.values())),'edges':copy.deepcopy(list(self.edges.values())),
                'regions':copy.deepcopy(self.source.graph['regions']),
                'contract':'benchmark kinematic surface; body contacts remain diagnostics/reward events'}

    def get_local_patch(self,position,radius,floor_context=None):
        patch=self.source.get_local_patch(position,radius,floor_context)
        patch['benchmark_map_id']=self.map_id
        patch['human_occupancy_support_retained']=True
        patch['robot_execution_scope']='native support; executable operations require the action compiler'
        return patch

    def get_local_nodes_for_path(self,path,radius=None):
        patch=self.source.get_local_nodes_for_path(path,radius)
        patch['benchmark_map_id']=self.map_id;patch['human_occupancy_support_retained']=True
        return patch

    def get_associated_local_nodes(self,node_id):
        return self.source.get_associated_local_nodes(node_id)

    def local_graph_view(self,k):
        """Change local resolution without rebuilding/certifying navigation."""
        from ..config import LocalConfig
        from ..local import LocalGraph
        view=LocalGraph(self.surface,self.robot_config,LocalConfig(k=k),self.source.topology_id)
        view.certified_map_id=self.map_id
        view.support_identity=digest({'certified_map':self.map_id,'local':asdict(view.config) if hasattr(view,'config') else {'k':k},
                                     'collision_reference':self.robot_config.collision_reference})
        return view

    def valid_starts(self):
        # No human state, body-radius erosion or private simulator state.
        return [{'node_id':n,'position':self.nodes[n]['position'],'floor_id':self.nodes[n]['floor_id'],
                 'island':self.nodes[n]['island'],'orientation_domain':'SE(2); compile discrete actions'}
                for n in sorted(self.adjacency)]

    def _search(self,start_node,avoid_edges=()):
        costs={start_node:0.};previous={start_node:None};queue=[(0.,start_node)];avoid=set(avoid_edges)
        while queue:
            cost,n=heapq.heappop(queue)
            if cost!=costs[n]:continue
            for target,eid,length in self.adjacency[n]:
                if eid in avoid:continue
                new=cost+length
                if new<costs.get(target,math.inf)-1e-10:
                    costs[target]=new;previous[target]=(n,eid);heapq.heappush(queue,(new,target))
        return costs,previous

    def get_global_candidates(self,start_node,constraints=None):
        constraints=constraints or {};costs,_=self._search(start_node)
        rows=[{'node_id':n,'distance':d,'type':self.nodes[n]['type'],'floor_id':self.nodes[n]['floor_id']}
              for n,d in costs.items() if n!=start_node and constraints.get('min_distance',.3)<=d<=constraints.get('max_distance',math.inf)]
        return sorted(rows,key=lambda r:(r['distance'],r['node_id']))[:constraints.get('limit',len(rows))]

    def get_graph_route(self,start_node,target_node,avoid_edges=()):
        if start_node not in self.nodes or target_node not in self.nodes:raise KeyError('Unknown global node')
        costs,previous=self._search(start_node,avoid_edges)
        if target_node not in costs:raise CertificationError('UNKNOWN: no certified directed connection')
        nodes=[target_node];edges=[]
        while previous[nodes[-1]] is not None:
            parent,eid=previous[nodes[-1]];nodes.append(parent);edges.append(eid)
        nodes.reverse();edges.reverse()
        path=concatenate([self.edges[e]['path'] for e in edges]) if edges else np.asarray([self.nodes[start_node]['position']])
        record={'map_id':self.map_id,'robot_configuration_id':self.robot_configuration_id,
                'node_ids':nodes,'edge_ids':edges,'start_node':start_node,'target_node':target_node,
                'path':path.tolist(),'length':costs[target_node],
                'edge_validation_level':'actual_falcon_kinematic_actions_static_scene',
                'route_composition_status':'UNKNOWN_until_compiled_from_actual_start'}
        record['route_id']=digest(record);return record

    def compile_route(self,route,start_pose=None,target_yaw=None,max_actions=500):
        if route['map_id']!=self.map_id or digest({k:v for k,v in route.items() if k!='route_id'})!=route['route_id']:
            raise CertificationError('Changed route or map identity')
        if any(self.edges[e]['status']!=VERIFIED for e in route['edge_ids']):raise CertificationError('Uncertified route edge')
        nodes=route['node_ids'];eids=route['edge_ids']
        if len(nodes)!=len(eids)+1 or nodes[0]!=route['start_node'] or nodes[-1]!=route['target_node']:
            raise CertificationError('Malformed route composition')
        for a,b,eid in zip(nodes[:-1],nodes[1:],eids):
            e=self.edges[eid]
            if (a,b)!=(e['source'],e['target']):raise CertificationError('Route direction or transition mismatch')
        expected=concatenate([self.edges[e]['path'] for e in eids]) if eids else np.asarray([self.nodes[nodes[0]]['position']])
        path=np.asarray(route['path']).copy()
        if path.shape!=expected.shape or not np.array_equal(path,expected):raise CertificationError('Route geometry differs from certified edge sequence')
        start_pose=[*path[0],0.] if start_pose is None else list(start_pose)
        if np.linalg.norm(np.asarray(start_pose[:3])-path[0])>1e-4:
            # A root must use cached map support. Later recursive queries may
            # use a separately certified endpoint attachment, not an implicit snap.
            raise CertificationError('Start does not match this route support; explicit attachment required')
        operation=self.model.compile(path,float(start_pose[3]),target_yaw,max_actions=max_actions)
        if operation['status']!=VERIFIED:raise CertificationError('UNKNOWN compiled route: '+operation['reason'])
        operation.update(map_id=self.map_id,robot_configuration_id=self.robot_configuration_id,
            route=route,contract_id=self.contract['identity'],
            validation_level='native_kinematic_compilation_with_independent_actual_action_campaign',
            individually_replayed_in_social_environment=False)
        operation['operation_id']=digest(operation)
        return operation


def load_certified_map(manifest,require_production=True):
    return CertifiedSurfaceMap(manifest,require_production)
