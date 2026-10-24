"""Private Habitat sensing boundary for an idealized geometry observation.

Only this adapter may read native static support and currently present actors.
It publishes visible configuration-space certificates, never native graph IDs,
mesh bounds, hidden-human slots, navmesh paths, or future state. Human detection
is exact-position torso detection with static AND current human mesh occlusion.
The live articulated jaw camera determines the frustum (not a base yaw proxy).
"""
import copy
import math
import time
import numpy as np
from scripts.collector.observation import HabitatHumanObserver, HumanDetection, LegalHumanTracker
from .map import KNOWN_FREE, KNOWN_OCCUPIED

SENSOR_VERSION = 'ideal_frustum_observed_cspace_dynamic_occlusion_1'


def in_frustum(points, origin, rotation, config):
    local = (np.asarray(points)-origin) @ rotation
    forward = -local[...,2]
    return ((forward > 1e-6) &
            (np.abs(local[...,0]) <= forward*math.tan(math.radians(config.horizontal_fov_deg/2))) &
            (np.abs(local[...,1]) <= forward*math.tan(math.radians(config.vertical_fov_deg/2))) &
            (np.linalg.norm(local,axis=-1) <= config.max_range_m))


class GeometryObserver:
    def __init__(self, runtime, resolution, private_cache=None):
        self.rt = runtime
        self.sim = runtime.sim
        self.config = runtime.observation_config
        self.resolution = float(resolution)
        self.camera = HabitatHumanObserver(self.sim,self.config)
        self.tracker = LegalHumanTracker(self.config)
        self.cache = private_cache if private_cache is not None else {}
        self.cache_namespace=(str(runtime.episode.scene_id),str(runtime.episode.info['graph_navmesh']),self.resolution)
        self.seen = set()
        self.last_timestamp = None
        self.last_row = None
        self.performance = {'visibility_seconds':0.,'private_geometry_seconds':0.,'rays':0,'observations':0}
        self.robot_ids = self._actor_ids(self.sim.get_agent_data(0).articulated_agent)
        self.all_actor_ids = set(self.robot_ids)
        self.human_ids = []
        for i in range(1,runtime.count+1):
            ids = self._actor_ids(self.sim.get_agent_data(i).articulated_agent)
            self.human_ids.append(ids);self.all_actor_ids.update(ids)
        all_agents=getattr(getattr(self.sim,'agents_mgr',None),'articulated_agents_iter',[])
        present_ids=set()
        for actor in all_agents:present_ids.update(self._actor_ids(actor))
        self.inactive_ids=present_ids-self.all_actor_ids

    @staticmethod
    def _actor_ids(actor):
        return {actor.sim_obj.object_id,*actor.sim_obj.link_object_ids.keys()}

    def state(self):
        return {'tracker':copy.deepcopy(self.tracker),'seen':set(self.seen),
                'last_timestamp':self.last_timestamp,'last_row':copy.deepcopy(self.last_row)}

    def restore(self,state):
        self.tracker=copy.deepcopy(state['tracker']);self.seen=set(state['seen'])
        self.last_timestamp=state['last_timestamp'];self.last_row=copy.deepcopy(state['last_row'])

    def clear_ray(self, origin, target, ignore=()):
        import habitat_sim
        vector=np.asarray(target)-origin;length=float(np.linalg.norm(vector))
        if length<1e-7:return True
        self.performance['rays']+=1
        hits=self.sim.cast_ray(habitat_sim.geo.Ray(origin,vector/length),max_distance=length).hits
        ignored=self.robot_ids | getattr(self,'inactive_ids',set()) | set(ignore)
        return not any(h.object_id not in ignored and h.ray_distance<length-.015 for h in hits)

    def _support(self, key, base_height):
        """Private, local square certificate, never native global connectivity.

        Native mesh already carries the installed .25 m inflation. Nine samples
        and straight native transitions reject thin walls/holes and corner cuts.
        This is a sampled 2.5D certificate, independently checked at execution.
        Unsupported multi-level/slope cells remain unknown, not free.
        """
        layer=round(base_height/.1)
        cache_key=(getattr(self,'cache_namespace',('isolated',id(self.sim.pathfinder),self.resolution)),key[0],key[1],layer)
        if cache_key in self.cache:return self.cache[cache_key]
        began=time.perf_counter();r=self.resolution
        center=np.array([(key[0]+.5)*r,base_height,(key[1]+.5)*r])
        pf=self.sim.pathfinder
        snapped=np.asarray(pf.snap_point(center),dtype=float)
        if not np.isfinite(snapped).all() or abs(snapped[1]-base_height)>.18 or np.linalg.norm((snapped-center)[[0,2]])>.004:
            value=None
        elif (not hasattr(pf,'distance_to_closest_obstacle') or
              float(pf.distance_to_closest_obstacle(snapped,1.)) < r/math.sqrt(2)+.008):
            # A disk enclosing the entire square must lie inside native free
            # support. Star/vertex samples alone can miss an obstacle cap.
            value=None
        else:
            points=[];valid=True
            for dx,dz in ((0,0),(-.49,-.49),(-.49,.49),(.49,-.49),(.49,.49),(-.49,0),(.49,0),(0,-.49),(0,.49)):
                p=center+np.array([dx*r,0,dz*r]);p[1]=snapped[1]
                s=np.asarray(pf.snap_point(p),dtype=float)
                if not np.isfinite(s).all() or abs(s[1]-snapped[1])>.08 or np.linalg.norm((s-p)[[0,2]])>.004:
                    valid=False;break
                stepped=np.asarray(pf.try_step_no_sliding(snapped,s))
                if np.linalg.norm((stepped-s)[[0,2]])>.003:
                    valid=False;break
                points.append(s)
            value=(snapped,np.asarray(points)) if valid else None
        self.cache[cache_key]=value
        self.performance['private_geometry_seconds']+=time.perf_counter()-began
        return value

    def observe(self, timestamp=None):
        from ..robotmap.execution import signed_yaw
        t=float(self.sim.get_world_time() if timestamp is None else timestamp)
        if t == self.last_timestamp:return copy.deepcopy(self.last_row)
        if self.last_timestamp is not None and t<self.last_timestamp:raise ValueError('Future/rewound observer state')
        began=time.perf_counter()
        robot=self.sim.get_agent_data(0).articulated_agent
        pose=[*np.asarray(robot.base_pos,dtype=float).tolist(),signed_yaw(robot)]
        origin,rotation=self.camera.camera_pose()
        detections=[]
        for i,ids in enumerate(self.human_ids,1):
            p=np.asarray(self.sim.get_agent_data(i).articulated_agent.base_pos,dtype=float)
            target=p+[0,self.config.torso_height_m,0]
            if in_frustum(target[None],origin,rotation,self.config)[0] and self.clear_ray(origin,target,ids):
                detections.append(HumanDetection(t,tuple(p.tolist())))
        tracks=self.tracker.update(t,sorted(detections,key=lambda d:d.position))
        r=self.resolution;radius=self.config.max_range_m
        ix0,ix1=math.floor((origin[0]-radius)/r),math.floor((origin[0]+radius)/r)
        iz0,iz1=math.floor((origin[2]-radius)/r),math.floor((origin[2]+radius)/r)
        ix,iz=np.meshgrid(np.arange(ix0,ix1+1),np.arange(iz0,iz1+1),indexing='ij')
        keys=np.stack([ix.ravel(),iz.ravel()],axis=1)
        centers=np.stack([(keys[:,0]+.5)*r,np.full(len(keys),pose[1]+.55),(keys[:,1]+.5)*r],axis=1)
        keys=keys[in_frustum(centers,origin,rotation,self.config)]
        updates=[]
        for raw_key in keys:
            key=tuple(map(int,raw_key))
            if key in self.seen:continue
            support=self._support(key,pose[1])
            if support is None:continue
            center,points=support
            # The visible square and clearance halo must both be in the camera
            # frustum and unoccluded. Do not infer free cells behind a person.
            halo_radius=.25+r/math.sqrt(2)
            halo=np.array([center+[math.cos(a)*halo_radius,0.,math.sin(a)*halo_radius]
                           for a in np.linspace(0.,2*math.pi,16,endpoint=False)])
            probes=np.concatenate([points[[0,1,2,3,4]],halo])
            targets=np.concatenate([probes+[0,.025,0],probes+[0,.7,0],probes+[0,1.4,0]])
            if not in_frustum(targets,origin,rotation,self.config).all():continue
            if not all(self.clear_ray(origin,p) for p in targets):continue
            updates.append({'cell':list(key),'state':KNOWN_FREE,'height':float(center[1]),
                            'timestamp':t,'source':'visible_native_cspace_square_and_clearance_halo'})
            self.seen.add(key)
        # Sparse surface rays reveal actual obstructing material, including
        # newly exposed walls. Humans mask rays but never enter static topology.
        import habitat_sim
        for x in np.linspace(-.95,.95,35):
            for y in (-.5,0.,.5):
                d=rotation@np.array([x,y,-1.]);d/=np.linalg.norm(d)
                hits=self.sim.cast_ray(habitat_sim.geo.Ray(origin,d),max_distance=radius).hits
                self.performance['rays']+=1
                hits=sorted((h for h in hits if h.object_id not in self.robot_ids and h.object_id not in self.inactive_ids),key=lambda h:h.ray_distance)
                if not hits:continue
                hit=hits[0]
                if hit.object_id in self.all_actor_ids:continue
                p=np.asarray(hit.point)
                if not pose[1]+.2 < p[1] < pose[1]+1.4:continue
                key=(math.floor(p[0]/r),math.floor(p[2]/r))
                # A wall witness blocks nearby center cells by installed radius.
                span=int(math.ceil(.25/r))+1
                for dx in range(-span,span+1):
                    for dz in range(-span,span+1):
                        k=(key[0]+dx,key[1]+dz)
                        c=np.array([(k[0]+.5)*r,pose[1],(k[1]+.5)*r])
                        if np.linalg.norm((c-p)[[0,2]]) <= .25:
                            updates.append({'cell':list(k),'state':KNOWN_OCCUPIED,'height':pose[1],
                                            'timestamp':t,'source':'observed_static_wall_inflated_025m'})
                            self.seen.discard(k)
        # This one cell is known from proprioceptive base support, never a
        # seeded free disk or a complete local map. It does not bridge unknown.
        ego_key=(math.floor(pose[0]/r),math.floor(pose[2]/r))
        updates.append({'cell':list(ego_key),'state':KNOWN_FREE,'height':pose[1],
                        'timestamp':t,'source':'occupied_ego_base_support'})
        row={'timestamp':t,'robot_pose':pose,'tracks':tracks,'camera_origin':origin.tolist(),
             'camera_rotation':rotation.tolist(),'static_updates':updates,
             'sensor_protocol':SENSOR_VERSION,'observation_valid':True}
        self.last_timestamp,self.last_row=t,copy.deepcopy(row)
        self.performance['visibility_seconds']+=time.perf_counter()-began
        self.performance['observations']+=1
        return row
