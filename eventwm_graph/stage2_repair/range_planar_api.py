"""Complete legal-range API with measured-plane mesh simplification."""
from .range_api import RangeEvidencePredictor
from .range_planar import PlanarRangeMesh


class PlanarRangePredictor(RangeEvidencePredictor):
    def _make_mesh(self,ranges,origins,rotations,timestamps,now,resolution,tracks):
        return PlanarRangeMesh(ranges,origins,rotations,timestamps,now,resolution,self.device,tracks=tracks)

    def predict_segments(self,**kwargs):
        results=super().predict_segments(**kwargs)
        for d in results:d['provenance']['range_mesh_approximation']='Full measured pixels; merge4x4 fully measured planar tiles within0.00005m. Holes/depth edges and original output cells retained.'
        return results
