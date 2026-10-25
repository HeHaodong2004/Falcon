"""Fast128bit cache identity; file/checkpoint integrity still uses SHA256."""
import ctypes as ct
import ctypes.util
import hashlib
import numpy as np


class Digest128(ct.Structure):_fields_=[('low64',ct.c_uint64),('high64',ct.c_uint64)]


try:
    LIB=ct.CDLL(ctypes.util.find_library('xxhash'))
    LIB.XXH3_createState.restype=ct.c_void_p
    LIB.XXH3_freeState.argtypes=[ct.c_void_p]
    LIB.XXH3_128bits_reset.argtypes=[ct.c_void_p]
    LIB.XXH3_128bits_update.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t]
    LIB.XXH3_128bits_digest.argtypes=[ct.c_void_p];LIB.XXH3_128bits_digest.restype=Digest128
except (OSError,AttributeError,TypeError):LIB=None


def identity(ranges,origins,rotations,timestamps,tracks,resolution,stride):
    positions=[];offsets=[0]
    for rows in tracks:
        for row in rows:
            if not row.get('valid',True):continue
            p=np.asarray(row['position'],np.float64)
            if row.get('velocity_valid')and row.get('estimated_velocity')is not None:p=p+np.asarray(row['estimated_velocity'])*min(float(row.get('observation_age',0)),.5)
            positions.append(p)
        offsets.append(len(positions))
    arrays=[ranges,origins,rotations,timestamps,np.asarray(positions,np.float64).reshape(-1,3),np.asarray(offsets,np.int64),np.asarray([resolution,stride],np.float64)]
    state=LIB.XXH3_createState()if LIB else None;fallback=hashlib.sha256()if LIB is None else None
    if state:LIB.XXH3_128bits_reset(state)
    try:
        for a in arrays:
            a=np.ascontiguousarray(a);shape=np.asarray(a.shape,np.int64)
            for value in [shape,a]:
                if LIB:LIB.XXH3_128bits_update(state,ct.c_void_p(value.ctypes.data),value.nbytes)
                elif value.nbytes:fallback.update(memoryview(value).cast('B'))
        if LIB:
            d=LIB.XXH3_128bits_digest(state);return 'xxh3_128_v1:%016x%016x'%(d.high64,d.low64)
        return 'sha256_v1:'+fallback.hexdigest()
    finally:
        if state:LIB.XXH3_freeState(state)
