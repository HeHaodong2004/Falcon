"""Exact static-geometry reuse pilot; not the default released predictor.

Current graph identity and the immutable future-map base are always rebuilt.
Only feature dependencies with identical typed values are eligible for reuse.
"""
from collections import OrderedDict
import hashlib
import numpy as np
from .api import SegmentPredictor


def geometry_key(prepared, graph):
    digest = hashlib.sha256()
    for value in (prepared.keys, prepared.values, prepared.heights):
        digest.update(str((value.dtype.str, value.shape)).encode('ascii'))
        digest.update(value.tobytes(order='C'))
    digest.update(np.asarray([prepared.resolution, prepared.height,
                              graph['clearance']], dtype='<f8').tobytes())
    return digest.hexdigest()


class TemporalReusePredictor(SegmentPredictor):
    """Synchronous, bounded, exact feature-cache reuse across legal clocks."""
    def __init__(self, *args, geometry_cache_size=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry_cache_size = int(geometry_cache_size)
        if self.geometry_cache_size < 0:
            raise ValueError('negative geometry cache size')
        self.geometry_cache = OrderedDict()
        self.geometry_hits = self.geometry_misses = 0

    def clear_cache(self):
        super().clear_cache()
        self.geometry_cache.clear()

    def prepare_graph(self, graph):
        hits = self.cache_hits
        prepared = super().prepare_graph(graph)
        if self.cache_hits != hits or not self.geometry_cache_size or not getattr(prepared, 'native_eligible', False):
            return prepared
        key = geometry_key(prepared, graph)
        if key in self.geometry_cache:
            owner = self.geometry_cache[key]
            self.geometry_cache.move_to_end(key)
            self.geometry_hits += 1
            # These values depend only on the precisely hashed static arrays.
            # Identity, height64, input_order and future_base remain current.
            prepared.crops = owner.crops
            prepared.integrals = owner.integrals
            prepared.cuda_grids = owner.cuda_grids
            prepared.cuda_geometry = owner.cuda_geometry
        else:
            self.geometry_misses += 1
            prepared.cuda_grids = OrderedDict()
            prepared.cuda_geometry = OrderedDict()
            self.geometry_cache[key] = prepared
            while len(self.geometry_cache) > self.geometry_cache_size:
                self.geometry_cache.popitem(last=False)
        return prepared
