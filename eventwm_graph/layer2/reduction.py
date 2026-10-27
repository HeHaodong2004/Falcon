"""Bit-identical, allocation-free form of the L2.5 joint-world candidate reduction.

``aggregate_joint(positions, robots, lengths, weights)`` returns exactly the dictionary of
``eventwm_graph.l25.numeric.aggregate_joint``. Only ``joint_events`` (the [K, W, T, N] body-envelope
and distance pass, ~85 % of the reference cost) is replaced by one numba loop that performs the same
IEEE float64 operations in the same order (cos/sin come from NumPy as in the reference) and releases
the GIL, so it overlaps with Stage 2. Equality with the reference is tested on recorded candidates.
"""
import numpy as np
from numba import njit

from eventwm_graph.partial.query import HUMAN_ENVELOPE, ROBOT_BOUNDS, NOMINAL_STEP_SECONDS
from eventwm_graph.l25.numeric import weighted_quantiles


@njit(cache=True, nogil=True)
def _events(px, pz, rx, rz, c, s, xmin, xmax, zmin, zmax, envelope2):
    Q, T = rx.shape
    W, _, N = px.shape
    # [Q, T, W] memory (returned as a [Q, W, T] view): the layout NumPy gives the reference's broadcast
    # temporaries, so the later einsum/matmul reductions sum in the same order (bit-identical results)
    collision = np.zeros((Q, T, W), np.bool_)
    distance = np.full((Q, T, W), np.inf)
    for q in range(Q):
        for w in range(W):
            for t in range(T):
                hit = False
                best = np.inf
                for n in range(N):
                    x = px[w, t, n]; z = pz[w, t, n]
                    if not (np.isfinite(x) and np.isfinite(z)):
                        continue                      # absent person: no event, distance +inf (as the reference)
                    dx = x-rx[q, t]; dz = z-rz[q, t]
                    lx = c[q, t]*dx-s[q, t]*dz
                    lz = s[q, t]*dx+c[q, t]*dz
                    ex = max(max(xmin-lx, lx-xmax), 0.)
                    ez = max(max(zmin-lz, lz-zmax), 0.)
                    if ex*ex+ez*ez <= envelope2:
                        hit = True
                    d = np.sqrt(dx*dx+dz*dz)
                    if d < best:
                        best = d
                collision[q, t, w] = hit
                distance[q, t, w] = best
    return collision.transpose(0, 2, 1), distance.transpose(0, 2, 1)


def joint_events(positions, robots):
    q, t, _ = robots.shape
    w, _, n, _ = positions.shape
    if not n:
        return np.zeros((q, w, t), bool), np.full((q, w, t), np.inf)
    yaw = robots[:, :, 3]
    xmin, xmax, zmin, zmax = ROBOT_BOUNDS
    return _events(np.ascontiguousarray(positions[..., 0]), np.ascontiguousarray(positions[..., 1]),
                   np.ascontiguousarray(robots[:, :, 0]), np.ascontiguousarray(robots[:, :, 2]),
                   np.cos(yaw), np.sin(yaw), float(xmin), float(xmax), float(zmin), float(zmax), float(HUMAN_ENVELOPE**2))


def aggregate_joint(positions, robots, lengths, weights):
    """Same outputs as ``l25.numeric.aggregate_joint`` (one whole-prefix union per world)."""
    collision, distance = joint_events(positions, robots)
    q, w, t = collision.shape
    active = np.arange(t)[None, :] < np.asarray(lengths)[:, None]
    collision &= active[:, None, :]
    distance = np.where(active[:, None, :], distance, np.inf)
    social = (distance < 1.) & active[:, None, :]
    any_collision = collision.any(-1)
    before = np.cumsum(collision, axis=-1) == 1
    first = collision & before
    minimum = distance.min(-1) if t else np.full((q, w), np.inf)
    return dict(p_collision=any_collision@weights, collision_world_event=any_collision,
                first_collision_mass=np.einsum('qwt,w->qt', first, weights),
                p_social=social.any(-1)@weights,
                expected_social_duration=np.einsum('qwt,w->q', social, weights)*NOMINAL_STEP_SECONDS,
                expected_social_severity=np.einsum('qwt,w->q', np.maximum(1-distance, 0.), weights)*NOMINAL_STEP_SECONDS,
                social_endpoint_probability=np.einsum('qwt,w->qt', social, weights),
                minimum=minimum,
                minimum_quantiles=[weighted_quantiles(d, weights) for d in minimum],
                p_no_humans=np.isinf(minimum)@weights)


def evaluate_candidates(bank, robots, lengths):
    """``HumanWorlds.evaluate_candidates`` with the fast reduction (same checks, same outputs)."""
    robots = np.asarray(robots, np.float64)
    K, T = robots.shape[:2]
    lengths = np.full(K, T) if lengths is None else np.asarray(lengths, int)
    rel = np.arange(1, T+1)*NOMINAL_STEP_SECONDS
    if rel[-1] > float(bank.times[-1])+1e-9:
        raise ValueError('candidates exceed the bank horizon')
    return aggregate_joint(bank.sandbox_positions(rel), robots, lengths, np.asarray(bank.weights))
