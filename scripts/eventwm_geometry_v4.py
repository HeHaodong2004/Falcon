"""Explicit TRAIN-only static-geometry proposal distribution for V4 research.

This changes the V3 proposal law. It uses NavMesh geometry alone, never human
poses, labels, future observations or SR. Native full certification remains.
Natural-reference and unseen-evaluation episodes use untouched V3 sampling.
"""
import hashlib
import json
import random
from pathlib import Path
from collector.core import digest
import eventwm_mass_collection as mass

VERSION='geometry-aware-v4-clearance-mixture-v1'
# Measured frozen articulated Spot geometry; derived necessary clearance bound.
RADIUS=0.5996548533439636
CLEARANCE=RADIUS-.25+.03+.02/2


def spec(index,scenes,master_seed):
    root=mass.schedule_spec(index,scenes,master_seed)
    if root['role']!='train':return root
    # Fixed pre-outcome 1/8 V3 component preserves challenging coverage support.
    mixture='v3_support' if random.Random(root['seed'] ^ 0x4734A19).random()<.125 else 'geometry_feasible'
    root['recipe']['episode_sampler']=dict(version=VERSION,component=mixture,
        train_only=True,geometry_radius=RADIUS,required_navmesh_clearance=CLEARANCE,
        original_component_probability=.125,geometry_component_probability=.875,
        geometry_sources='frozen Spot link bounds plus prepared static NavMesh',
        sampling_law='V4; not V3 proposal-equivalent',proposal_seed=master_seed+index)
    root['root_id']='v4g_'+digest(dict(recipe=root['recipe'],physical_scene_id=root['physical_scene_id']))[:20]
    root['recipe']['root_id']=root['root_id'];root['split_group']=root['root_id']
    return root


def episode(scene,seed,human_count,covered,sampler):
    if sampler['version']!=VERSION:raise RuntimeError('Unknown geometry sampler')
    if sampler['component']=='v3_support':
        return mass.episode_for(scene,seed,human_count,covered),dict(version=VERSION,component='v3_support',filtered=False)
    import habitat_sim
    factory=habitat_sim.PathFinder
    audit=dict(version=VERSION,component='geometry_feasible',static_only=True,filtered=True,
               start_attempts=0,start_feasible=0,clearance_histogram={},threshold=CLEARANCE,
               navmesh_sha256=scene['navmesh_sha256'],geometry_radius=RADIUS)
    class GeometryFinder:
        def __init__(self):self.native=factory();self.starts=0
        def __getattr__(self,name):return getattr(self.native,name)
        def get_random_navigable_point(self,*args,**kwargs):
            if self.starts>=128:return self.native.get_random_navigable_point(*args,**kwargs)
            self.starts+=1
            for _ in range(4096):
                point=self.native.get_random_navigable_point(*args,**kwargs)
                clearance=float(self.native.distance_to_closest_obstacle(point,5.))
                audit['start_attempts']+=1
                bucket=str(int(max(-1.,min(5.,clearance))/.1))
                audit['clearance_histogram'][bucket]=audit['clearance_histogram'].get(bucket,0)+1
                if clearance>=CLEARANCE:
                    audit['start_feasible']+=1
                    return point
            # Explicit failure, never silently call a different sampler.
            raise RuntimeError('geometry_episode_unavailable: no feasible start in bounded search')
    try:
        habitat_sim.PathFinder=GeometryFinder
        value=mass.episode_for(scene,seed,human_count,covered)
    finally:habitat_sim.PathFinder=factory
    value['info']['episode_sampler']=sampler
    value['info']['geometry_preparation']=audit
    return value,audit
