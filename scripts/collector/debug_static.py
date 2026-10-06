"""Opt-in native collision-ray reference. No plotting imports or simulation steps."""
import numpy as np
import magnum as mn
import habitat_sim
from collector.core import write


def capture_static_reference(world, position, output, resolution=100):
    p=np.asarray(position);xs=np.linspace(p[0]-2.7,p[0]+2.7,resolution)
    zs=np.linspace(p[2]-2.7,p[2]+2.7,resolution)
    heights=[]
    for z in zs:
        row=[]
        for x in xs:
            ray=habitat_sim.geo.Ray(mn.Vector3(float(x),float(p[1]+1.2),float(z)),mn.Vector3(0,-1,0))
            hits=world.sim.cast_ray(ray).hits
            static=[h for h in hits if h.object_id in world.static_ids and h.ray_distance <= 1.6]
            row.append(float(min(static,key=lambda h:h.ray_distance).point[1]-p[1]) if static else None)
        heights.append(row)
    write(output,dict(kind='native_collision_rays_downward',xs=xs.tolist(),zs=zs.tolist(),height_relative_to_root=heights,
        ray_start_height=1.2,ray_length=1.6,static_ids=sorted(world.static_ids),
        warning='DEBUG GT / PRIVILEGED; NOT MODEL INPUT; NOT QUERY SELECTION',
        limitations='First native STATIC/stage surface below root+1.2m; missing hits unknown; not a full 3D occupancy map'))
