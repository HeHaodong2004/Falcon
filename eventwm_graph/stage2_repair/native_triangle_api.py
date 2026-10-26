"""Direct hardware triangles built only from legal measured depth arrays.

Native watertight triangle intersection replaces the custom padded barycentric
test. This numerical change must be assessed on original reconstruction cells.
"""
from .device_pipeline import ColumnPreparedPipelinePredictor
from .range_planar import PlanarRangeMesh


class NativeTrianglePredictor(ColumnPreparedPipelinePredictor):
    primitive_mode='native'
    def __init__(self,*args,**kwargs):
        if kwargs.get('fast_features')is False:raise ValueError('direct hardware triangles require the legal fast feature path')
        super().__init__(*args,**kwargs)

    def _make_mesh(self,ranges,origins,rotations,timestamps,now,resolution,tracks):
        mesh=PlanarRangeMesh(ranges,origins,rotations,timestamps,now,resolution,self.device,tracks=tracks,bvh='native')
        self.range_kernel.module.backend.prepare(mesh)
        return mesh

    def predict_segments(self,**kwargs):
        results=super().predict_segments(**kwargs)
        self.last_batch_profile.update(mesh_triangles=self.active_mesh.triangles,mesh_prepare_seconds=self.active_mesh.prepare_seconds)
        for d in results:d['provenance']['measured_surface_intersection']='Native hardware triangles; all legal measured pixels/triangles, no redundant software BVH. Native edge tolerance differs from custom padded intersection.'
        return results
