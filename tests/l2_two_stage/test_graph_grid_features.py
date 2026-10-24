"""Exact floor/UNKNOWN/age/height counterexamples for batched graph samples."""
from types import SimpleNamespace
import numpy as np
from eventwm_graph.l2_two_stage.features import OFFSETS,extract


def test_batched_graph_samples_match_independent_scalar_boundary_reference():
    generator=np.random.default_rng(851)
    for resolution in (.1,.125,1.):
        for origin in ((.1,-.1),(np.nextafter(.1,0.),np.nextafter(-.1,0.)),
                       (np.nextafter(.1,np.inf),np.nextafter(-.1,-np.inf)),
                       (100000.1,-100000.1)):
            now=21.125;height=-3.125
            points=np.asarray(origin)+OFFSETS
            cells={}
            for j,point in enumerate(points):
                if j%3==0:continue
                key=tuple(np.floor(point/resolution).astype(int))
                cells[key]=dict(cell=key,state=1+j%2,height=height+generator.normal(),
                                first_seen=0.,last_seen=now-generator.uniform(0,20))
            observation=dict(timestamp=now,robot_pose=(origin[0],height,origin[1],0.),
                camera_origin=(origin[0],height+1.5,origin[1]),camera_rotation=np.eye(3),
                observation_valid=False,human_sensor_available=False,tracks=[],static_updates=[])
            graph=dict(cells=tuple(cells.values()),height=height,resolution=resolution,
                       global_nodes={},global_edges=[])
            context=SimpleNamespace(history=(observation,),actions=(),partial_graph=graph)
            actual=extract(context)['global_graph'][:,:7]
            expected=np.empty((len(OFFSETS),7),np.float32)
            for j,point in enumerate(points):
                cell=cells.get(tuple(np.floor(point/resolution).astype(int)))
                state=0 if cell is None else cell['state']
                expected[j]=[state==0,state==1,state==2,
                    0 if cell is None else min(10,now-cell['last_seen'])/10,
                    OFFSETS[j,0]/4,OFFSETS[j,1]/4,
                    0 if cell is None else cell['height']-height]
            assert actual.dtype==expected.dtype
            assert actual.tobytes()==expected.tobytes()
