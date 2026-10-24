"""Private, declared episode sampling; never imported by online query logic."""
import math
import numpy as np
from ..stage1.runtime import episode_for_map


def make_episode(h,node,seed,count,scenario='natural',heading=None):
    episode=episode_for_map(h,node,seed,count)
    start=np.asarray(episode.start_position,dtype=float)
    rng=np.random.default_rng(int(seed)^0x562010)
    if heading is None:
        # Initial pose distribution can depend on static simulator geometry,
        # exactly as valid-spawn sampling does. It conveys no map to the agent.
        angles=np.linspace(-math.pi,math.pi,32,endpoint=False)
        scored=[]
        for a in angles:
            target=start+np.array([3.8*math.cos(a),0.,-3.8*math.sin(a)])
            stepped=np.asarray(h.navigation.pf.try_step_no_sliding(start,target),dtype=float)
            progress=float(np.linalg.norm((stepped-start)[[0,2]]))
            scored.append((progress,float(a)))
        maximum=max(p for p,_ in scored)
        angles=[a for p,a in scored if p>=maximum-.05]
        heading=angles[int(rng.integers(len(angles)))]
    # Installed quaternion_to_rad_angle ignores the axis sign and returns
    # [0,2pi]. Encode negative yaw with its positive modulo, rather than its
    # negative half-angle (which silently reflects the actual initial heading).
    encoded_heading=heading % (2*math.pi)
    episode.start_rotation=[math.cos(encoded_heading/2),0.,math.sin(encoded_heading/2),0.]
    if scenario!='natural' and count:
        forward=np.array([math.cos(heading),0.,-math.sin(heading)])
        left=np.array([-math.sin(heading),0.,-math.cos(heading)])
        for i in range(count):
            if scenario=='crossing':
                points=[start+forward*4.3+left*(2.5 if i%2==0 else -2.5),
                        start+forward*4.3-left*(2.5 if i%2==0 else -2.5),start-forward]
            else:
                points=[start+forward*(5.+.7*i)+left*(.3*(i%2)),start-forward*1.5,start+forward*5.]
            snapped=[np.asarray(h.navigation.pf.snap_point(p),dtype=float) for p in points]
            # Impossible initialization is an explicit sampler rejection, never
            # a geometric query made feasible using hidden support.
            if any(not np.isfinite(p).all() for p in snapped):raise ValueError('unsupported_encounter_initialization')
            if np.linalg.norm((snapped[0]-start)[[0,2]])<1.4:raise ValueError('encounter_spawn_too_close')
            for k,p in enumerate(snapped):episode.info[f'human_{i}_waypoint_{k}_position']=p.tolist()
            d=snapped[1]-snapped[0]
            episode.info[f'human_{i}_waypoint_0_rotation']=math.atan2(-d[2],d[0])
    return episode,{'law':'certified_spawn_native_heading_and_declared_encounter_v1',
                    'scenario':scenario,'root_heading':heading,'private_initialization':True}
