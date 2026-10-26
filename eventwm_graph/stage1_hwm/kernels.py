"""Numba kernels for the CPU deployment path (exact re-implementations, fused loops).

``draw_paths`` fuses TrajectoryHead post-processing (``infer.mixture``) and mixture sampling
(``infer.draw``) given externally drawn uniforms/normals, so its output equals the NumPy
reference bit-for-bit up to floating-point association. ``raycast`` is
``LegalState._visibility``'s per-ray first-BLOCKED search.
"""
import math
import numpy as np
from numba import njit

S = 13
STEP = .25


@njit(cache=True)
def _softplus(x):
    return max(x, 0.) + math.log1p(math.exp(-abs(x)))


@njit(cache=True)
def draw_paths(raw, modes, base_v, age, zero_first, u, z, eps, max_speed, temp=1., scale=1.):
    """raw [N, M*(1+3S+3)], base_v [N,2], age [N], zero_first [N] bool,
    u [W,N] uniforms, z [W,N,2], eps [W,N,S,2] standard normals -> knots [W,N,S,2]."""
    W = u.shape[0];N = raw.shape[0];F = raw.shape[1] // modes
    out = np.empty((W, N, S, 2))
    prob = np.empty(modes)
    for n in range(N):
        mx = -1e300
        for m in range(modes):
            mx = max(mx, raw[n, m * F] / temp)
        tot = 0.
        for m in range(modes):
            prob[m] = math.exp(raw[n, m * F] / temp - mx);tot += prob[m]
        for w in range(W):
            # mode: number of cumulative probabilities strictly below u (same as infer.draw)
            c = 0.;mode = 0
            for m in range(modes):
                c += prob[m] / tot
                if u[w, n] > c:mode = m + 1
            if mode > modes - 1:mode = modes - 1
            o = mode * F
            l00 = (_softplus(raw[n, o + 1 + 3 * S]) * .5 + 1e-3) * scale
            l11 = (_softplus(raw[n, o + 2 + 3 * S]) * .5 + 1e-3) * scale
            l10 = raw[n, o + 3 + 3 * S] * .3 * scale
            ox = l00 * z[w, n, 0];oz = l10 * z[w, n, 0] + l11 * z[w, n, 1]
            px = 0.;pz = 0.
            for s in range(S):
                d = age[n] if s == 0 else STEP
                mux = (base_v[n, 0] + raw[n, o + 1 + 2 * s] * 1.5) * d
                muz = (base_v[n, 1] + raw[n, o + 2 + 2 * s] * 1.5) * d
                sig = (_softplus(raw[n, o + 1 + 2 * S + s] - 1.) + .01) * math.sqrt(d / STEP + 1e-4) * scale
                dx = mux + d * ox + sig * eps[w, n, s, 0]
                dz = muz + d * oz + sig * eps[w, n, s, 1]
                if s == 0 and zero_first[n]:
                    dx = 0.;dz = 0.
                norm = math.sqrt(dx * dx + dz * dz);limit = max_speed * d
                if norm > limit:
                    f = limit / max(norm, 1e-12);dx *= f;dz *= f
                px += dx;pz += dz
                out[w, n, s, 0] = px;out[w, n, s, 1] = pz
    return out


@njit(cache=True)
def raycast(blocked, last_visible, ox, oz, cos_a, sin_a, steps, coarse, origin0, origin1, t):
    """Mark coarse cells visible before the first BLOCKED cell along each ray (in place).

    cos/sin are computed by NumPy by the caller so sample coordinates equal the reference."""
    G = blocked.shape[0]
    for a in range(cos_a.shape[0]):
        ca = cos_a[a];sa = sin_a[a]
        for k in range(steps.shape[0]):
            x = ox + ca * steps[k];z = oz + sa * steps[k]
            i = math.floor(x / coarse) - origin0;j = math.floor(z / coarse) - origin1
            if i < 0 or i >= G or j < 0 or j >= G:
                continue
            if blocked[i, j] > 0:
                break
            last_visible[i, j] = t
