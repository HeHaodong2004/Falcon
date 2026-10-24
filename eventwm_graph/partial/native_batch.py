"""Private, exact batch calls into the installed Habitat collision world.

This workstation's Habitat conda package statically links Bullet and hides its
C++ symbols, and ships no development headers.  The adapter therefore binds the
verified local ELF entry point, with the entire installed binary pinned by
SHA-256.  It fails closed on a different build; it does not infer struct offsets
or use a second geometry kernel.  Build explicitly before starting workers::

    python scripts/build_stage01_native_batch.py

Inputs mirror ``habitat_sim.geo.Ray``: a float32 origin[3], float32
directions[N,3], and float64 maximum distances[N].  Directions are NOT normalized
here: the caller supplies precisely the original observer's normalized values.
Only the native PyBind dispatch/all-hit Python object allocation is removed.
The live simulator (including all articulated human meshes) is unchanged.
"""
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

HABITAT_SHA256 = '892f4d32c5eb67559811aa3e78739016298b06706c8ba2bd1ea2895de866528b'
ABI_VERSION = 'habitat031_892f4d32_native_batch_v1'
DEFAULT_BUILD = (Path(__file__).resolve().parents[2] /
                 'output/stage01_acceleration_20261021/native_batch/build')


def _digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def load_backend():
    """Validate binary/source bindings once per process, then load extension."""
    import habitat_sim._ext.habitat_sim_bindings as native
    library=Path(native.__file__).resolve()
    if _digest(library)!=HABITAT_SHA256:
        raise RuntimeError('native_batch_unsupported_habitat_binary: rebuild/review required')
    directory=Path(os.environ.get('EVENTWM_NATIVE_BATCH_BUILD',str(DEFAULT_BUILD))).resolve()
    metadata=json.loads((directory/'build.json').read_text())
    source=Path(__file__).with_suffix('.cpp')
    extension=directory/metadata['extension']
    if metadata.get('source_sha256')!=_digest(source):
        raise RuntimeError('native_batch_stale_build: run scripts/build_stage01_native_batch.py')
    if (metadata.get('habitat_sha256')!=HABITAT_SHA256 or
            metadata.get('extension_sha256')!=_digest(extension)):
        raise RuntimeError('native_batch_build_integrity_failure')
    module_name='_eventwm_native_batch'
    spec=importlib.util.spec_from_file_location(module_name,extension)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.abi_version!=ABI_VERSION:
        raise RuntimeError('native_batch_wrong_abi')
    sys.modules[module_name]=module
    return module,library,metadata


class NativeRayBatch:
    """Immutable interface binding an exact native simulator collision world.

    ``clear_groups`` keeps each group's original short-circuit ray order and
    returns ``{'clear': bool[G], 'rays': actual_cast_count}``. Group offsets are
    int64[G+1], beginning at zero and ending at N. All methods retain the GIL:
    concurrent mutation of this simulator during an observation is forbidden.
    The owner must not close the simulator while a batch adapter remains live.
    """
    def __init__(self,simulator):
        module,library,metadata=load_backend()
        self._native=module.Engine(simulator,str(library))
        self.identity={'abi_version':ABI_VERSION,'habitat_sha256':HABITAT_SHA256,
                       'extension_sha256':metadata['extension_sha256'],
                       'source_sha256':metadata['source_sha256']}

    def clear_rays(self,origin,directions,max_distances,ignored_ids=()):
        """Return bool[N] using the accepted strict 0.015 m endpoint rule."""
        return self._native.clear_rays(origin,directions,max_distances,
                                       np.asarray(tuple(ignored_ids),dtype=np.int32))

    def clear_groups(self,origin,directions,max_distances,ignored_ids,group_offsets):
        return self._native.clear_groups(origin,directions,max_distances,
                                         np.asarray(tuple(ignored_ids),dtype=np.int32),
                                         np.asarray(group_offsets,dtype=np.int64))

    def nearest_hits(self,origin,directions,max_distances,ignored_ids=()):
        """First unignored native hit; preserve the native equal-distance order."""
        return self._native.nearest_hits(origin,directions,max_distances,
                                         np.asarray(tuple(ignored_ids),dtype=np.int32))

    def cast_rays(self,origin,directions,max_distances):
        """All-hit packed arrays for deterministic differential verification."""
        return self._native.cast_rays(origin,directions,max_distances)


def for_simulator(simulator):
    """Return no adapter for test fakes; native ABI/integrity errors fail closed."""
    try:
        import habitat_sim._ext.habitat_sim_bindings as native
    except ImportError:
        return None
    if not isinstance(simulator,native.Simulator):
        return None
    return NativeRayBatch(simulator)
