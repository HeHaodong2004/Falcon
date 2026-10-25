"""Compact legal-map sight-line proxies; no privileged visibility claims.

Inflated configuration-space BLOCKED cells are not optical geometry. These
descriptors summarize only the supplied legal map and are learned covariates.
Sixteen discrete map probes along each of three supplied-pose-to-cell lines
are feature sampling, never invented sensor observations or target boundaries.
"""
import numpy as np
from numba import njit
from eventwm_graph.stage2_quality.features import Features


@njit(cache=True)
def legal_lines_reference(keys, path, resolution, lo, states):
    out = np.zeros((len(keys), 12), np.float32)
    indices = (min(1, len(path)-1), len(path)//2, len(path)-1)
    for i in range(len(keys)):
        x = (keys[i, 0]+.5)*resolution; z = (keys[i, 1]+.5)*resolution
        for ray in range(3):
            j = indices[ray]; sx = path[j, 1]; sz = path[j, 3]
            free = 0.; blocked = 0.; first_unknown = 1.; first_blocked = 1.
            for step in range(1, 17):
                fraction = step/17.
                ix = int(np.floor((sx+(x-sx)*fraction)/resolution))-lo[0]
                iz = int(np.floor((sz+(z-sz)*fraction)/resolution))-lo[1]
                state = 0
                if 0 <= ix < states.shape[0] and 0 <= iz < states.shape[1]:
                    state = states[ix, iz]
                if state == 1:
                    free += 1.
                elif state == 2:
                    blocked += 1.; first_blocked = min(first_blocked, fraction)
                else:
                    first_unknown = min(first_unknown, fraction)
            out[i, ray*4] = free/16.
            out[i, ray*4+1] = blocked/16.
            out[i, ray*4+2] = first_unknown
            out[i, ray*4+3] = first_blocked
    return out


@njit(cache=True)
def legal_lines(keys, path, resolution, lo, states):
    out = np.zeros((len(keys), 12), np.float32)
    indices = (min(1, len(path)-1), len(path)//2, len(path)-1)
    for i in range(len(keys)):
        for ray in range(3):
            j = indices[ray]; sx = path[j, 1]; sz = path[j, 3]
            duplicate = -1
            for previous in range(ray):
                k = indices[previous]
                if path[k, 1] == sx and path[k, 3] == sz:
                    duplicate = previous; break
            if duplicate >= 0:
                for col in range(4):
                    out[i, ray*4+col] = out[i, duplicate*4+col]
                continue
            gx = sx/resolution; gz = sz/resolution
            dx = keys[i, 0]+.5-gx; dz = keys[i, 1]+.5-gz
            # Conservative roundoff guard preserves the reference grid side
            # when a probe lies on/near an exact cell boundary.
            bound_x = 64*2.220446049250313e-16*(abs(gx)+abs(keys[i, 0])+1)
            bound_z = 64*2.220446049250313e-16*(abs(gz)+abs(keys[i, 1])+1)
            x = (keys[i, 0]+.5)*resolution; z = (keys[i, 1]+.5)*resolution
            free = 0.; blocked = 0.; first_unknown = 1.; first_blocked = 1.
            for step in range(1, 17):
                fraction = step/17.
                px = gx+dx*fraction; pz = gz+dz*fraction
                if abs(px-round(px)) <= bound_x:
                    px = (sx+(x-sx)*fraction)/resolution
                if abs(pz-round(pz)) <= bound_z:
                    pz = (sz+(z-sz)*fraction)/resolution
                ix = int(np.floor(px))-lo[0]; iz = int(np.floor(pz))-lo[1]
                state = 0
                if 0 <= ix < states.shape[0] and 0 <= iz < states.shape[1]:
                    state = states[ix, iz]
                if state == 1:
                    free += 1.
                elif state == 2:
                    blocked += 1.; first_blocked = min(first_blocked, fraction)
                else:
                    first_unknown = min(first_unknown, fraction)
            out[i, ray*4] = free/16.; out[i, ray*4+1] = blocked/16.
            out[i, ray*4+2] = first_unknown; out[i, ray*4+3] = first_blocked
    return out


class SegmentFeatures(Features):
    def transform(self, keys, mode='both'):
        import time
        x, baseline = super().transform(keys, mode)
        tick = time.perf_counter()
        rays = legal_lines(np.asarray(keys, np.int32), self.path, self.resolution, self.lo, self.states)
        out = np.empty((len(keys), 104), np.float32)
        out[:, :92] = x; out[:, 92:] = rays
        self.last_transform_profile['legal_line_seconds'] = time.perf_counter()-tick
        return out, baseline


class SegmentEndFeatures(Features):
    def transform(self, keys, mode='both'):
        import time
        x,baseline=super().transform(keys,mode);tick=time.perf_counter()
        lines=legal_lines(np.asarray(keys,np.int32),self.path[-1:],self.resolution,self.lo,self.states)
        out=np.empty((len(keys),96),np.float32);out[:,:92]=x;out[:,92:]=lines[:,8:12]
        self.last_transform_profile['legal_line_seconds']=time.perf_counter()-tick
        return out,baseline
