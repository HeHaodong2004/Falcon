"""PRIVATE episode-initialization curriculum, never an online planning input.

This samples spawn configurations using cached native static geometry. It does
not certify the robot's map, choose a query, inspect humans/futures, or bypass
legal-history admission. A uniform component retains every certified spawn.
"""
import math
import time
import numpy as np
from ..config import digest

PRIOR_VERSION='private_bootstrap_spawn_mixture_1'


def geometry_table(h):
    began=time.perf_counter();rows=[];pf=h.navigation.pf
    angles=np.linspace(-math.pi,math.pi,32,endpoint=False)
    for node in h.valid_starts():
        start=np.asarray(h.nodes[node['node_id']]['position'],dtype=float);row=[]
        for angle in angles:
            direction=np.array([math.cos(angle),0.,-math.sin(angle)])
            far=np.asarray(pf.try_step_no_sliding(start,start+3.8*direction),dtype=float)
            vantage=np.asarray(pf.try_step_no_sliding(start,start+(13*25/120)*direction),dtype=float)
            progress=float(np.linalg.norm((far-start)[[0,2]])) if np.isfinite(far).all() else 0.
            finite=np.isfinite(vantage).all()
            clearance=float(pf.distance_to_closest_obstacle(vantage,1.)) if finite else 0.
            height=abs(float(vantage[1]-start[1])) if finite else 1000.
            row.append([progress,clearance,height])
        rows.append(row)
    return {'rows':np.asarray(rows),'seconds':time.perf_counter()-began,
            'law':PRIOR_VERSION,'native_geometry_private':True}


def eligible_indices(table,seed):
    result=[]
    for index,row in enumerate(table['rows']):
        maximum=float(row[:,0].max())
        if maximum<3.5:continue
        tied=np.flatnonzero(row[:,0]>=maximum-.05)
        # Exactly the private initial-heading tie break in make_episode.
        rng=np.random.default_rng(int(seed)^0x562010);direction=int(tied[int(rng.integers(len(tied)))])
        if row[direction,1]>=.20 and row[direction,2]<=.15:result.append(index)
    return result


def choose_start(h,seed,probability,cache):
    if not 0<=probability<1:raise ValueError('Curriculum probability must retain a positive uniform component')
    starts=h.valid_starts();n=len(starts)
    if not n:raise ValueError('no_certified_starts')
    if probability:
        if 'geometry_table' not in cache:cache['geometry_table']=geometry_table(h)
        eligible=eligible_indices(cache['geometry_table'],seed)
    else:eligible=[]
    rng=np.random.default_rng(int(digest({'seed':int(seed),'domain':PRIOR_VERSION})[:16],16))
    curriculum=bool(rng.random()<probability)
    if curriculum and not eligible:raise ValueError('no_private_bootstrap_start')
    if curriculum:
        index=int(eligible[int(rng.integers(len(eligible)))]);override=index
    else:
        prior=np.random.default_rng(int(digest({'seed':int(seed),'domain':'partial_root_prior_v1'})[:16],16))
        index=int(prior.integers(n));override=None
    effective=(1.-probability)/n+(probability/len(eligible) if index in eligible else 0.)
    metadata={'law':PRIOR_VERSION,'private_initialization_only':True,'uses_future':False,
        'component':'geometry_curriculum' if curriculum else 'uniform',
        'curriculum_probability':probability,'all_start_count':n,'eligible_start_count':len(eligible),
        'selected_start_index':index,'selected_start_node':starts[index]['node_id'],
        'unconditional_selected_start_probability':effective,
        'uniform_reference_probability':1./n,'uniform_reference_importance_ratio':(1./n)/effective,
        'eligibility':'private native straight progress>=3.5m; selected-heading13-step viewpoint clearance>=.20m,height change<=.15m',
        'admission':'Actual causal observations and unchanged geometric query checks still decide acceptance'}
    return override,metadata
