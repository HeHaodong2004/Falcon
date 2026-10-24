"""Joint-world reductions; native endpoints, never independent occupancy."""
import numpy as np
from eventwm_graph.partial.query import HUMAN_ENVELOPE, ROBOT_BOUNDS, NOMINAL_STEP_SECONDS


def weighted_quantiles(values, weights, quantiles=(.1, .5, .9)):
    order = np.argsort(values, kind='stable')
    cdf = np.cumsum(weights[order])
    indices = np.searchsorted(cdf, quantiles).clip(0, len(values)-1)
    return [float(values[order[i]]) if np.isfinite(values[order[i]]) else None for i in indices]


def joint_events(positions, robots):
    """Return per Q,W,T envelope events and horizontal center separation.

    Reuses audited all-animation robot bounds and all-phase human disk from
    Stage 0/1 physical_clearance (with zero extra proposal margin). This is an
    enclosing-body approximation, not native mesh-contact probability.
    """
    q, t, _ = robots.shape
    w, _, n, _ = positions.shape
    if not n:
        return np.zeros((q, w, t), bool), np.full((q, w, t), np.inf)
    delta = positions[None]-robots[:, None, :, None, [0, 2]]
    yaw = robots[:, None, :, None, 3]
    local_x = np.cos(yaw)*delta[..., 0]-np.sin(yaw)*delta[..., 1]
    local_z = np.sin(yaw)*delta[..., 0]+np.cos(yaw)*delta[..., 1]
    xmin, xmax, zmin, zmax = ROBOT_BOUNDS
    dx = np.maximum(np.maximum(xmin-local_x, local_x-xmax), 0.)
    dz = np.maximum(np.maximum(zmin-local_z, local_z-zmax), 0.)
    collision = (dx*dx+dz*dz <= HUMAN_ENVELOPE**2) & np.isfinite(positions[None, ..., 0])
    distance = np.sqrt(np.sum(delta*delta, axis=-1))
    distance = np.where(np.isfinite(distance), distance, np.inf)
    return collision.any(-1), distance.min(-1)


def aggregate_joint(positions, robots, lengths, weights):
    """One whole-prefix union per world, preserving humans/time correlations."""
    collision, distance = joint_events(positions, robots)
    q, w, t = collision.shape
    active = np.arange(t)[None, :] < np.asarray(lengths)[:, None]
    collision &= active[:, None, :]
    distance = np.where(active[:, None, :], distance, np.inf)
    social = (distance < 1.) & active[:, None, :]
    any_collision = collision.any(-1)
    # First mass sums to query risk, even repeated hits by several humans.
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


def optical_batch(model, future, robots, lengths):
    """Vectorization of L2 predict_observations, tested against that API.

    Excludes root, uses exact same optical law and hidden-count channel. No
    static revelation, identity labels, centroids or future actual measurements.
    """
    q, t, _ = robots.shape
    # EndpointSequence.worlds avoids materializing identical root carriers Q times.
    worlds = getattr(future.endpoints, 'worlds', None)
    if worlds is None:
        worlds = future.endpoints[0]
    regimes = np.array([c.visibility_mode for c in worlds])
    tracked = future.positions[:, 1:t+1]
    pd = model.observation_law.detection(tracked[None], robots[:, None], np.broadcast_to(regimes, (q, len(regimes))))
    hidden = future.hidden_positions[:, 1:t+1]
    hd = np.asarray(model.observation_law.hidden_detection)[regimes, None, None]*np.isfinite(hidden[..., 0])
    endpoint_none = np.prod(1-pd, axis=-1)*np.prod(1-hd, axis=-1)[None]
    active = np.arange(t)[None, :] < np.asarray(lengths)[:, None]
    endpoint_none = np.where(active[:, None, :], endpoint_none, 1.)
    survival = np.cumprod(endpoint_none, axis=-1)
    before = np.concatenate([np.ones((q, len(regimes), 1)), survival[:, :, :-1]], axis=-1)
    none = survival[:, :, -1] if t else np.ones((q, len(regimes)))
    first = before*(1-endpoint_none)
    w = future.weights
    return dict(per_world_none=none, p_any=(1-none)@w,
                first_detection_mass=np.einsum('qwt,w->qt', first, w),
                endpoint_any=np.einsum('qwt,w->qt', 1-endpoint_none, w))


def binary_entropy(p):
    p = np.asarray(p, dtype=float)
    clipped = np.clip(p, 1e-15, 1-1e-15)
    return -p*np.log(clipped)-(1-p)*np.log1p(-clipped)


def finite_event_mi(likelihood, weights):
    return float(max(0., binary_entropy(weights@likelihood)-weights@binary_entropy(likelihood)))
