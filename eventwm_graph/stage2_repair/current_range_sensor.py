"""ENVIRONMENT-ONLY ideal current-time range camera, never a WM predictor.

This explicitly extends the observation interface with measured current ranges.
The environment performs first-hit sensing; it does not expose its scene mesh,
navigation data, object identities, labels or future states to the world model.
Active humans occlude rays. Robot/inactive-agent handling matches the existing
accepted ideal observation boundary. This is NOT rendered RGB-D validation.
"""
import os
import numpy as np
from eventwm_graph.partial.observer import GeometryObserver
from eventwm_graph.partial.native_batch import NativeRayBatch


class CurrentRangeSensor:
    def __init__(self,runtime,resolution,size=128):
        os.environ.setdefault('EVENTWM_NATIVE_BATCH_BUILD','/home/haodong/workspace/projects/eventwm-falcon/output/stage01_acceleration_20261021/native_batch/build')
        self.runtime=runtime;self.boundary=GeometryObserver(runtime,resolution);self.batch=NativeRayBatch(runtime.sim);self.size=size
        xx,yy=np.meshgrid((np.arange(size)+.5)*2/size-1,1-(np.arange(size)+.5)*2/size,indexing='xy');v=np.stack([xx,yy,-np.ones_like(xx)],axis=-1);self.local=v.reshape(-1,3);self.local/=np.linalg.norm(self.local,axis=1)[:,None]

    def measure(self):
        origin,rotation=self.boundary.camera.camera_pose();directions=self.local@rotation.T;ignored=sorted(self.boundary.robot_ids|self.boundary.inactive_ids);hits=self.batch.nearest_hits(origin,directions,np.full(len(directions),10.),ignored);ranges=np.where(hits['valid'],hits['distances'],np.nan).astype(np.float32).reshape(self.size,self.size)
        return dict(timestamp=float(self.runtime.sim.get_world_time()),camera_origin=origin.copy(),camera_rotation=rotation.copy(),range_image=ranges,sensor_protocol='ideal_current_jaw_range_dynamic_occlusion_1',max_range_m=10.,horizontal_fov_deg=90.,vertical_fov_deg=90.)
