"""Explicit planner-trial camera adapter; frozen Stage 1 remains independent."""
from dataclasses import dataclass
import time
import numpy as np


@dataclass(frozen=True)
class TrialTrace:
    trajectory: np.ndarray
    camera_origins: np.ndarray
    camera_rotations: np.ndarray

    @classmethod
    def from_frames(cls, frames):
        # Read geometry channels only. Predicted/observed static_updates,
        # certificate labels and future graphs are never forwarded to Stage 2.
        rows=list(frames)
        if not rows:raise ValueError('trial requires its starting frame')
        p=np.array([[r['timestamp'],*r['robot_pose']]for r in rows],np.float64)
        origin=np.array([r['camera_origin']for r in rows],np.float64)
        rotation=np.array([r['camera_rotation']for r in rows],np.float64)
        for a in [p,origin,rotation]:a.flags.writeable=False
        return cls(p,origin,rotation)

    def camera_inputs(self):
        return dict(origins=self.camera_origins,rotations=self.camera_rotations)


def capture_trial_frame(runtime):
    """Read transforms after an existing sandbox step; no rendering/sensing.

    Trial simulation is upstream of Stage 2. This hook does not step the real
    environment or generate map labels and does not call an observer's observe.
    """
    from scripts.collector.observation import HabitatHumanObserver
    from eventwm_graph.robotmap.execution import signed_yaw
    robot=runtime.sim.get_agent_data(0).articulated_agent
    camera=HabitatHumanObserver(runtime.sim,runtime.observation_config)
    origin,rotation=camera.camera_pose()
    return dict(timestamp=float(runtime.sim.get_world_time()),robot_pose=[*np.asarray(robot.base_pos,dtype=float).tolist(),signed_yaw(robot)],camera_origin=origin.tolist(),camera_rotation=rotation.tolist())


class GeometryWorldModelSandbox:
    def __init__(self,stage1,stage2):
        self.stage1=stage1;self.stage2=stage2;self.last_profile=None

    def predict_decision(self,previous_context,observation,*,executed_action,trials,arrived_graph=None):
        begin=time.perf_counter()
        people=self.stage1.predict(previous_context,observation,partial_graph=arrived_graph,executed_action=executed_action,horizon=3.,worlds=40)
        first=time.perf_counter();actual=self.stage1.last_context;trials=list(trials)
        deltas=self.stage2.predict_segments(partial_graph=actual.partial_graph,trajectories=[t.trajectory for t in trials],camera_trajectories=[t.camera_inputs()for t in trials])
        finish=time.perf_counter();self.last_profile=dict(stage1_seconds=first-begin,stage2_complete_seconds=finish-first,total_seconds=finish-begin,queries=len(trials),stage1_calls=1)
        return dict(human_worlds=people,graph_deltas=deltas,hypothetical_graphs=[d['future_partial_graph']for d in deltas],actual_context=actual)
