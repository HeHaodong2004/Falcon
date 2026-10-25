"""Reuse only map-dependent ray support; future trial visibility is recomputed."""
from collections import OrderedDict
import hashlib
from .range_fast import RangeFeatureKernel
from .range_fast_proxy import ProxyKernel


class SupportReuse:
    def support(self,mesh,union,floors,uk,ff):
        digest=hashlib.sha256(memoryview(union).cast('B'))
        digest.update(memoryview(floors).cast('B'))
        identity=(union.shape,floors.shape,digest.digest())
        if not hasattr(mesh,'support_cache'):mesh.support_cache=OrderedDict()
        if identity not in mesh.support_cache:
            mesh.support_cache[identity]=super().support(mesh,union,floors,uk,ff)
            while len(mesh.support_cache)>4:mesh.support_cache.popitem(last=False)
        mesh.support_cache.move_to_end(identity)
        return mesh.support_cache[identity]


class ReusedProxyKernel(SupportReuse,ProxyKernel):pass
class ReusedExactKernel(SupportReuse,RangeFeatureKernel):pass
