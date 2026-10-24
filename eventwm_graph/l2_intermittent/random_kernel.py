"""Exact NumPy1.23.5 SeedSequence specialization for 128-bit keyed entropy.

The mixing algorithm is adapted from NumPy's BSD-licensed bit_generator.pyx
(v1.23.5); attribution, license and source hashes accompany this release in
docs/l2/intermittent_wm_20261022/third_party/. PCG64 and its distributions still
execute through NumPy, with a fresh invocation-local generator workspace. No
global RNG or cross-call stochastic cache is used. Other NumPy versions retain
the original constructor rather than assuming seed compatibility.
"""
import hashlib
import sys
import numpy as np
from numba import njit


@njit(cache=True)
def seed_words(entropy):
    pool=np.empty(4,np.uint32);hc=np.uint32(0x43b0d7e5)
    for i in range(4):
        value=np.uint32(entropy[i]^hc);hc=np.uint32(hc*np.uint32(0x931e8875))
        value=np.uint32(value*hc);pool[i]=np.uint32(value^(value>>np.uint32(16)))
    for source in range(4):
        for destination in range(4):
            if source==destination:continue
            value=np.uint32(pool[source]^hc);hc=np.uint32(hc*np.uint32(0x931e8875))
            value=np.uint32(value*hc);value=np.uint32(value^(value>>np.uint32(16)))
            mixed=np.uint32(np.uint32(0xca01f9dd)*pool[destination]-np.uint32(0x4973f715)*value)
            pool[destination]=np.uint32(mixed^(mixed>>np.uint32(16)))
    words=np.empty(8,np.uint32);hc=np.uint32(0x8b51f9dd)
    for i in range(8):
        value=np.uint32(pool[i%4]^hc);hc=np.uint32(hc*np.uint32(0x58f38ded))
        value=np.uint32(value*hc);words[i]=np.uint32(value^(value>>np.uint32(16)))
    result=np.empty(4,np.uint64)
    for i in range(4):result[i]=np.uint64(words[2*i])|(np.uint64(words[2*i+1])<<np.uint64(32))
    return result


class KeyedWorkspace:
    """Reuse allocated Generator storage; reset its exact state for EVERY key."""
    def __init__(self):
        self.fast=np.__version__=='1.23.5' and sys.byteorder=='little'
        self.rng=np.random.Generator(np.random.PCG64(0)) if self.fast else None

    def generator(self,seed,slot,counter):
        raw=hashlib.blake2b(('%s:%s:%s'%(seed,slot,counter)).encode(),digest_size=16).digest()
        entropy=int.from_bytes(raw,'little')
        if not self.fast:return np.random.default_rng(entropy)
        values=seed_words(np.frombuffer(raw,dtype=np.uint32))
        initial=(int(values[0])<<64)|int(values[1])
        mask=(1<<128)-1
        increment=((((int(values[2])<<64)|int(values[3]))<<1)|1)&mask
        state=((initial+increment)*0x2360ed051fc65da44385df649fccf645+increment)&mask
        self.rng.bit_generator.state=dict(bit_generator='PCG64',state=dict(state=state,inc=increment),has_uint32=0,uinteger=0)
        return self.rng
