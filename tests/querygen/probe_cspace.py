"""Privileged diagnostic: contact-free yaw intervals near the start pose.

Places the robot (transform only, legs at reset pose) at offsets along its start
heading and sweeps yaw in 5 degree steps; prints '.' for contact-free yaws and
'#' for static contact.  Never used by the generator/oracle.
"""
import math
import sys
import numpy as np
import magnum as mn
from eventwm_graph.querygen.native import EpisodeSpec, NativeTask, classify_contacts

key = sys.argv[1] if len(sys.argv) > 1 else 'hm3d/val/3t8DB4Uzvkt/1'
ds, sp, sc, ep = key.split('/')
t = NativeTask(EpisodeSpec(ds, sp, sc, ep, 1))
obj = t.robot.sim_obj
M0 = mn.Matrix4(obj.transformation)
pose0 = t.robot_pose()
print('start pose', np.round(pose0, 3))
print('M0 rows', np.round(np.asarray(M0), 4).tolist())
for a in ['right']*14:
    t.step(a)
    print(a, np.round(t.robot_pose(), 3).tolist(), 'static', classify_contacts(t, gate_humans=False)['static'])
Mr = np.asarray(t.robot.sim_obj.transformation)
print('after 14 right rows', np.round(Mr, 4).tolist())
obj.transformation = M0
pf = t.sim.pathfinder
for off in (0., .2, .4, .6, .8, 1.0, 1.2):
    row = ''
    for k in range(72):
        yaw = pose0[3]+math.radians(5*k)
        p = np.array([pose0[0]+off*math.cos(pose0[3]), M0.translation[1], pose0[2]-off*math.sin(pose0[3])])
        R = mn.Matrix4.rotation_y(mn.Rad(yaw))
        # local +x forward: rotation about +y by yaw maps x -> (cos, 0, -sin)
        obj.transformation = mn.Matrix4.from_(R.rotation(), mn.Vector3(*p))
        c = classify_contacts(t, gate_humans=False, need_static=True)
        row += '#' if c['static'] else '.'
    nav = pf.is_navigable(mn.Vector3(p[0], pose0[1], p[2]))
    print(f'offset {off:.1f} navigable {nav}  yaw +0..355 (CCW): {row}')
t.close()
