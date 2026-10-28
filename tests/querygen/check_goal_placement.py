"""Self-check for feasibility.certify_goal placement: re-placing the robot at its
reset pose with the same construction reproduces the reset transformation and
the reset static-contact state."""
import math, sys
import numpy as np
import magnum as mn
from eventwm_graph.querygen.suites import load
from eventwm_graph.querygen.native import NativeTask, BranchSimulator, classify_contacts, FAR_AWAY, matrix_yaw
for key in sys.argv[1:]:
    spec = [s for s in load('dev') if s.key == key][0]
    task = NativeTask(spec); sim = BranchSimulator(task, None)
    robot = task.robot; obj = robot.sim_obj; pf = task.sim.pathfinder
    c_reset = classify_contacts(task, gate_humans=False)
    M0 = mn.Matrix4(obj.transformation)
    base0 = np.asarray(robot.base_pos, dtype=float)
    off = np.asarray(M0.translation, dtype=float)-base0
    snap0 = np.asarray(pf.snap_point(base0.astype(np.float32)), dtype=float)
    dy = base0[1]-snap0[1]
    R0 = mn.Matrix4.from_(M0.rotation(), mn.Vector3(0, 0, 0))
    yaw0 = matrix_yaw(np.asarray(M0, dtype=np.float32))
    # same construction as certify_goal with yaw measured relative to R0's frame
    for yaw in (0.0,):
        obj.transformation = mn.Matrix4.translation(mn.Vector3(*FAR_AWAY))
        task.sim.perform_discrete_collision_detection()
        M = mn.Matrix4.translation(mn.Vector3(*(snap0+[0, dy, 0]+off))) @ mn.Matrix4.rotation_y(mn.Rad(yaw)) @ R0
        obj.transformation = M; robot.update()
        c = classify_contacts(task, gate_humans=False)
        d = float(np.abs(np.asarray(M, dtype=float)-np.asarray(M0, dtype=float)).max())
        print(key, 'dy', round(dy, 4), 'off', np.round(off, 4), 'max |M-M0|', f'{d:.2e}',
              'reset static', c_reset['static'], len(c_reset['static_witnesses']), 'replaced static', c['static'], len(c['static_witnesses']),
              'yaw0', round(yaw0, 3), 'yaw(M)', round(matrix_yaw(np.asarray(M, dtype=np.float32)), 3))
    # heading convention: rotation_y(+a) @ R0 should change matrix_yaw by +a
    M = mn.Matrix4.rotation_y(mn.Rad(0.5)) @ R0
    print('  yaw shift for +0.5 rad:', round(matrix_yaw(np.asarray(M, dtype=np.float32))-yaw0, 3))
    task.close()
