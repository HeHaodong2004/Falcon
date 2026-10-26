"""Bounded diagnostic: retain all supplied past range frames, same frozen model.

This changes mesh-derived features and must not reuse the pruned feature cache.
No observations later than the current graph timestamp are accepted.
"""
from .device_pipeline import ColumnPreparedPipelinePredictor
from .range_planar import PlanarRangeMesh


class FullHistoryPredictor(ColumnPreparedPipelinePredictor):
    def _make_mesh(self,ranges,origins,rotations,timestamps,now,resolution,tracks):
        mesh=PlanarRangeMesh(ranges,origins,rotations,timestamps,now,resolution,self.device,tracks=tracks,bvh='radix',keyframe_distance=0.,keyframe_rotation=0.)
        self.range_kernel.module.backend.prepare(mesh)
        return mesh

    def predict_segments(self,**kwargs):
        results=super().predict_segments(**kwargs)
        for d in results:d['provenance']['range_history_policy']='all supplied legal past frames; no motion keyframe pruning'
        return results
