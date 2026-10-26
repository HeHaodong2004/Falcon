"""Larger measured planar tiles, retaining every input pixel and depth edge."""
from .device_pipeline import ColumnPreparedPipelinePredictor
from .fast_point_api import FastPointPredictor
from .range_planar import PlanarRangeMesh


class Tile8Mesh:
    def _make_mesh(self,ranges,origins,rotations,timestamps,now,resolution,tracks):
        mesh=PlanarRangeMesh(ranges,origins,rotations,timestamps,now,resolution,self.device,tracks=tracks,bvh='radix',tile_size=8)
        self.range_kernel.module.backend.prepare(mesh)
        return mesh

    def predict_segments(self,**kwargs):
        results=super().predict_segments(**kwargs)
        for d in results:d['provenance']['range_mesh_approximation']='All original measured pixels. Merge8x8 only when ALL grid points measured and within0.00005m of a plane, all original quad edges under0.3m. Preserve every hole/depth edge.'
        return results


class Tile8Predictor(Tile8Mesh,ColumnPreparedPipelinePredictor):pass
class FastPointTile8Predictor(Tile8Mesh,FastPointPredictor):pass
