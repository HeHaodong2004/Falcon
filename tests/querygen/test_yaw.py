"""matrix_yaw must equal the project's signed_yaw on real robot states."""
import numpy as np
from eventwm_graph.querygen.native import EpisodeSpec, NativeTask, matrix_yaw
from eventwm_graph.robotmap.execution import signed_yaw

t = NativeTask(EpisodeSpec('hm3d', 'val', '3t8DB4Uzvkt', '1', 3))
worst = 0.
prev = None
for a in ['left']*30+['forward']*5+['right']*50:
    if t.episode_over:
        break
    t.step(a)
    y1 = matrix_yaw(np.asarray(t.robot.sim_obj.transformation, dtype=np.float32))
    y2 = signed_yaw(t.robot)
    worst = max(worst, abs(np.arctan2(np.sin(y1-y2), np.cos(y1-y2))))
    if a == 'left' and prev is not None:
        assert np.sin(y2-prev) > 0, 'left must increase yaw'
    prev = y2
print('max |matrix_yaw - signed_yaw|', worst)
assert worst < 1e-6
m = np.asarray(t.robot.sim_obj.transformation, dtype=np.float32)
assert np.allclose(m[3], [0, 0, 0, 1]), m
print('YAW_OK')
t.close()
